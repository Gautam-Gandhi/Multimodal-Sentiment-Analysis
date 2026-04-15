"""
Full model implementation of:
  "Multimodal sentiment analysis based on multi-layer feature fusion and multi-task learning"

Components
----------
UFEN  - Unimodal Feature Extraction Network (one instance per modality)
CrossModalAttention - pairwise cross-modal multi-head attention
MTFN  - Multi-Task Fusion Network (6 cross-attentions → encoder-decoder)
MultiTaskModel - top-level module (BERT + 3×UFEN + MTFN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def masked_mean(x, mask=None):
    """
    Mean-pool a padded sequence over the time dimension.

    x    : (batch, T, dim)
    mask : (batch, T) bool, True = padding position  [optional]
    returns (batch, dim)
    """
    if mask is None:
        return x.mean(dim=1)
    valid = (~mask).float().unsqueeze(-1)          # (batch, T, 1)
    lengths = valid.sum(dim=1).clamp(min=1.0)       # (batch, 1)
    return (x * valid).sum(dim=1) / lengths         # (batch, dim)


# ---------------------------------------------------------------------------
# UFEN
# ---------------------------------------------------------------------------

class UFEN(nn.Module):
    """
    Unimodal Feature Extraction Network.

    All modalities use the same pipeline:
        BiGRU (D_i → d_m)  →  N parallel [Conv1D → Self-Att → Unpool(→d_m)]  →  sum

    BERT is a pre-processor / feature extractor that produces input embeddings for the
    text modality *before* UFEN (same role as FACET for visual, COVAREP for audio).

    Output
    ------
    fusion : (batch, T, d_m)   sequence representation
    pred   : (batch,)          unimodal regression prediction  (Y_i)
    """

    def __init__(self, input_dim, d_m, n_layers=2, kernel_sizes=None,
                 conv_dim=64, n_att_heads=1, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        if kernel_sizes is None:
            kernel_sizes = [1, 3]

        assert len(kernel_sizes) == n_layers, "kernel_sizes length must equal n_layers"

        # ---- Temporal encoder: BiGRU for all modalities ----
        self.bigru = nn.GRU(
            input_dim, d_m // 2,
            batch_first=True, bidirectional=True,
        )
        branch_in_dim = d_m

        # ---- N parallel Conv1D branches ----
        self.convs = nn.ModuleList([
            nn.Conv1d(branch_in_dim, conv_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # ---- Self-attention per branch (Eq 5) ----
        self.self_atts = nn.ModuleList([
            nn.MultiheadAttention(conv_dim, n_att_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n_layers)
        ])

        # ---- Unpool: project conv_dim back to d_m (Eq 6) ----
        self.unpools = nn.ModuleList([
            nn.Linear(conv_dim, d_m) for _ in range(self.n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_m)
        self.dropout    = nn.Dropout(dropout)

        # ---- Unimodal prediction head (Eq 8) ----
        self.pred_head = nn.Linear(d_m, 1)

    # ------------------------------------------------------------------
    def forward(self, x, mask=None):
        """
        x    : (batch, T, input_dim)
        mask : (batch, T) bool, True = padding          [optional]
        """
        # ---- Temporal encoding ----
        x, _ = self.bigru(x)                            # (batch, T, d_m)

        # Zero-out padded positions after temporal encoder
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        base = x   # shared input to all branches

        # ---- N parallel branches: Conv → Self-Att gate → Unpool ----
        branch_outputs = []
        for conv, self_att, unpool in zip(self.convs, self.self_atts, self.unpools):

            # Conv1D  (expects channels-first)
            c = conv(base.transpose(1, 2)).transpose(1, 2)   # (batch, T, conv_dim)
            c = F.relu(c)
            if mask is not None:
                c = c.masked_fill(mask.unsqueeze(-1), 0.0)

            # Self-attention gating: X * Attention(X)
            att_out, _ = self_att(c, c, c, key_padding_mask=mask)  # (batch, T, conv_dim)
            c = c * att_out                                          # element-wise gate

            # Unpool → d_m
            c = unpool(c)                                            # (batch, T, d_m)
            branch_outputs.append(c)

        # ---- Element-wise sum over branches (Eq 7) ----
        fusion = sum(branch_outputs)                    # (batch, T, d_m)
        fusion = self.dropout(self.layer_norm(fusion))

        if mask is not None:
            fusion = fusion.masked_fill(mask.unsqueeze(-1), 0.0)

        # ---- Unimodal prediction from mean-pooled fusion ----
        pooled = masked_mean(fusion, mask)              # (batch, d_m)
        pred   = self.pred_head(pooled).squeeze(-1)     # (batch,)

        return fusion, pred


# ---------------------------------------------------------------------------
# Cross-Modal Attention
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """
    i → j:  Q from modality i,  K/V from modality j.
    Output is layer-normalised (Eq 10-11 + LayerNorm).
    """

    def __init__(self, d_m, n_heads, att_dropout=0.2):
        super().__init__()
        self.mha        = nn.MultiheadAttention(d_m, n_heads, dropout=att_dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x_i, x_j, mask_i=None, mask_j=None):
        """
        x_i : (batch, T_i, d_m)  — query modality
        x_j : (batch, T_j, d_m)  — key/value modality
        returns (batch, T_i, d_m)
        """
        out, _ = self.mha(x_i, x_j, x_j, key_padding_mask=mask_j)
        return self.layer_norm(out)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks (one layer each, as in the paper)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Nm_en     = LN(Input + MhAtt(Input))
    Output_en = Nm_en + FF(Nm_en)
    """
    def __init__(self, d_m, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_m, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_m, d_ff), nn.ReLU(), nn.Linear(d_ff, d_m))
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn))   # Nm_en
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Nm_de      = LN(Input + MhAtt(Input, Output_en))   ← cross-attention
    Output_de  = Nm_de + FF(Nm_de)
    """
    def __init__(self, d_m, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_m, n_heads, dropout=dropout, batch_first=True)
        self.ff  = nn.Sequential(nn.Linear(d_m, d_ff), nn.ReLU(), nn.Linear(d_ff, d_m))
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, memory):
        """x: decoder input (= stacked 6 tokens),  memory: encoder output"""
        attn, _ = self.cross_attn(x, memory, memory)
        x = self.norm1(x + self.drop(attn))    # Nm_de
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# MTFN
# ---------------------------------------------------------------------------

class MTFN(nn.Module):
    """
    Multi-Task Fusion Network.

    Steps
    -----
    1. Linear-project each modality's UFEN output (Eq 9 preamble).
    2. 6 cross-modal attentions: t→v, t→a, v→t, v→a, a→t, a→v.
    3. Mean-pool each output → (batch, d_m); stack → sequence of 6 tokens.
    4. Y_m  = Linear(6*d_m → 1) applied to the flattened 6-token input.
    5. Encoder-Decoder on the 6-token sequence → Y_m' from decoder output.
    """

    def __init__(self, d_m, n_cross_heads=4, d_ff=256, dropout=0.1, att_dropout=0.2):
        super().__init__()

        # Linear projections before cross-attention (W_i X_i + b_i, Eq 9)
        self.proj_t = nn.Linear(d_m, d_m)
        self.proj_v = nn.Linear(d_m, d_m)
        self.proj_a = nn.Linear(d_m, d_m)

        # 6 cross-modal attention modules
        make_ca = lambda: CrossModalAttention(d_m, n_cross_heads, att_dropout)
        self.ca_t2v = make_ca()
        self.ca_t2a = make_ca()
        self.ca_v2t = make_ca()
        self.ca_v2a = make_ca()
        self.ca_a2t = make_ca()
        self.ca_a2v = make_ca()

        # Encoder-Decoder (single layer each, as described in paper)
        self.encoder = EncoderBlock(d_m, n_cross_heads, d_ff, dropout)
        self.decoder = DecoderBlock(d_m, n_cross_heads, d_ff, dropout)

        # Y_m  : from flattened 6-token input (Eq 14)
        self.pred_fusion = nn.Linear(6 * d_m, 1)
        # Y_m' : from mean-pooled decoder output (Eq 14)
        self.pred_recon  = nn.Linear(d_m, 1)

    # ------------------------------------------------------------------
    def forward(self, feat_t, feat_v, feat_a, mask_t=None, mask_v=None, mask_a=None):
        """
        feat_x : (batch, T_x, d_m)
        Returns y_m, y_m_prime  — both (batch,)
        """
        # ---- Linear projections ----
        xt = self.proj_t(feat_t)
        xv = self.proj_v(feat_v)
        xa = self.proj_a(feat_a)

        # ---- 6 cross-modal attentions, then mean-pool to (batch, d_m) ----
        ca_t2v = masked_mean(self.ca_t2v(xt, xv, mask_t, mask_v), mask_t)
        ca_t2a = masked_mean(self.ca_t2a(xt, xa, mask_t, mask_a), mask_t)
        ca_v2t = masked_mean(self.ca_v2t(xv, xt, mask_v, mask_t), mask_v)
        ca_v2a = masked_mean(self.ca_v2a(xv, xa, mask_v, mask_a), mask_v)
        ca_a2t = masked_mean(self.ca_a2t(xa, xt, mask_a, mask_t), mask_a)
        ca_a2v = masked_mean(self.ca_a2v(xa, xv, mask_a, mask_v), mask_a)

        # ---- Stack as 6-token sequence: (batch, 6, d_m) ----
        tokens = torch.stack([ca_t2v, ca_t2a, ca_v2t, ca_v2a, ca_a2t, ca_a2v], dim=1)

        # ---- Y_m: linear on flattened tokens ----
        y_m = self.pred_fusion(tokens.reshape(tokens.size(0), -1)).squeeze(-1)  # (batch,)

        # ---- Encoder-Decoder ----
        enc_out = self.encoder(tokens)           # (batch, 6, d_m)
        dec_out = self.decoder(tokens, enc_out)  # (batch, 6, d_m)

        # ---- Y_m': mean pool decoder output → linear ----
        y_m_prime = self.pred_recon(dec_out.mean(dim=1)).squeeze(-1)  # (batch,)

        return y_m, y_m_prime


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Full model with BERT (text) + 3×UFEN + MTFN.

    Returns five predictions: y_t, y_v, y_a, y_m, y_m_prime
    Final sentiment output for evaluation: y_m_prime  (Y_m')
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # BERT for text pre-processing / feature extraction. Its last_hidden_state
        # (batch, 52, 768) feeds into ufen_t as the input sequence — same role as
        # FACET features for visual and COVAREP features for audio.
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Text UFEN: BERT output (seq, 768) → full BiGRU→Conv→Self-Att pipeline
        self.ufen_t = UFEN(
            input_dim=768,
            d_m=config.d_m,
            n_layers=config.n_layers,
            kernel_sizes=config.kernel_sizes,
            conv_dim=config.conv_dim,
            n_att_heads=config.self_att_heads,
            dropout=config.dropout,
        )
        # Visual UFEN
        self.ufen_v = UFEN(
            input_dim=config.visual_size,
            d_m=config.d_m,
            n_layers=config.n_layers,
            kernel_sizes=config.kernel_sizes,
            conv_dim=config.conv_dim,
            n_att_heads=config.self_att_heads,
            dropout=config.dropout,
        )
        # Audio UFEN
        self.ufen_a = UFEN(
            input_dim=config.acoustic_size,
            d_m=config.d_m,
            n_layers=config.n_layers,
            kernel_sizes=config.kernel_sizes,
            conv_dim=config.conv_dim,
            n_att_heads=config.self_att_heads,
            dropout=config.dropout,
        )

        # MTFN
        self.mtfn = MTFN(
            d_m=config.d_m,
            n_cross_heads=config.cross_att_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            att_dropout=config.att_dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, bert_ids, bert_mask, bert_type_ids, visual, v_mask, acoustic, a_mask):
        """
        bert_ids, bert_mask, bert_type_ids : (batch, 52)
        visual    : (batch, T_v, 47)   padded, float
        v_mask    : (batch, T_v)       bool, True = padding
        acoustic  : (batch, T_a, 74)   padded, float
        a_mask    : (batch, T_a)       bool, True = padding

        Returns y_t, y_v, y_a, y_m, y_m_prime  — all (batch,) float
        """
        # ---- Text: BERT encoding ----
        bert_out = self.bert(
            input_ids=bert_ids,
            attention_mask=bert_mask,
            token_type_ids=bert_type_ids,
        ).last_hidden_state                      # (batch, 52, 768)
        t_mask = (bert_mask == 0)                # True where BERT pads (batch, 52)

        # ---- UFEN per modality ----
        feat_t, y_t = self.ufen_t(bert_out, t_mask)
        feat_v, y_v = self.ufen_v(visual,   v_mask)
        feat_a, y_a = self.ufen_a(acoustic, a_mask)

        # ---- MTFN ----
        y_m, y_m_prime = self.mtfn(feat_t, feat_v, feat_a, t_mask, v_mask, a_mask)

        return y_t, y_v, y_a, y_m, y_m_prime
