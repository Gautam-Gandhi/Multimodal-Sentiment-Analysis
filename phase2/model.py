"""
Phase 2 model: UFEN + MTFN for 7-class emotion classification on MELD.

Text modality supports two modes (config.use_bert):
  - False: GloVe embeddings (300-dim) — Exp 1-5
  - True:  BERT contextual embeddings (768-dim) — Exp 6+

Visual: ResNet-101 features (2048-dim)
Audio:  Wave2Vec2.0 features (32-dim)
All prediction heads output num_classes (7).

Supports bimodal ablation via config.modalities (list of active modalities).
Default: ['text', 'audio', 'video']
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
# UFEN — Unimodal Feature Extraction Network
# ---------------------------------------------------------------------------

class UFEN(nn.Module):
    """
    Unimodal Feature Extraction Network.

    Pipeline:  BiGRU (input_dim → d_m) → N parallel [Conv1D → Self-Att → Unpool(→d_m)] → sum

    Output
    ------
    fusion : (batch, T, d_m)           sequence representation
    pred   : (batch, num_classes)      unimodal classification logits
    """

    def __init__(self, input_dim, d_m, num_classes, n_layers=2, kernel_sizes=None,
                 conv_dim=64, n_att_heads=1, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        if kernel_sizes is None:
            kernel_sizes = [1, 3]

        assert len(kernel_sizes) == n_layers

        # BiGRU: input_dim → d_m
        self.bigru = nn.GRU(
            input_dim, d_m // 2,
            batch_first=True, bidirectional=True,
        )
        branch_in_dim = d_m

        # N parallel Conv1D branches
        self.convs = nn.ModuleList([
            nn.Conv1d(branch_in_dim, conv_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Self-attention per branch
        self.self_atts = nn.ModuleList([
            nn.MultiheadAttention(conv_dim, n_att_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n_layers)
        ])

        # Unpool: project conv_dim back to d_m
        self.unpools = nn.ModuleList([
            nn.Linear(conv_dim, d_m) for _ in range(self.n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_m)
        self.dropout = nn.Dropout(dropout)

        # Unimodal classification head: d_m → num_classes
        self.pred_head = nn.Linear(d_m, num_classes)

    def forward(self, x, mask=None):
        """
        x    : (batch, T, input_dim)
        mask : (batch, T) bool, True = padding
        """
        x, _ = self.bigru(x)                            # (batch, T, d_m)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        base = x

        branch_outputs = []
        for conv, self_att, unpool in zip(self.convs, self.self_atts, self.unpools):
            c = conv(base.transpose(1, 2)).transpose(1, 2)   # (batch, T, conv_dim)
            c = F.relu(c)
            if mask is not None:
                c = c.masked_fill(mask.unsqueeze(-1), 0.0)

            att_out, _ = self_att(c, c, c, key_padding_mask=mask)
            c = c * att_out

            c = unpool(c)                                     # (batch, T, d_m)
            branch_outputs.append(c)

        fusion = sum(branch_outputs)                          # (batch, T, d_m)
        fusion = self.dropout(self.layer_norm(fusion))

        if mask is not None:
            fusion = fusion.masked_fill(mask.unsqueeze(-1), 0.0)

        pooled = masked_mean(fusion, mask)                    # (batch, d_m)
        pred = self.pred_head(pooled)                         # (batch, num_classes)

        return fusion, pred


# ---------------------------------------------------------------------------
# Cross-Modal Attention
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """Q from modality i, K/V from modality j."""

    def __init__(self, d_m, n_heads, att_dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_m, n_heads, dropout=att_dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x_i, x_j, mask_i=None, mask_j=None):
        out, _ = self.mha(x_i, x_j, x_j, key_padding_mask=mask_j)
        return self.layer_norm(out)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_m, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_m, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_m, d_ff), nn.ReLU(), nn.Linear(d_ff, d_m))
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_m, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_m, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_m, d_ff), nn.ReLU(), nn.Linear(d_ff, d_m))
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        attn, _ = self.cross_attn(x, memory, memory)
        x = self.norm1(x + self.drop(attn))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# MTFN — Multi-Task Fusion Network (modality-agnostic)
# ---------------------------------------------------------------------------

class MTFN(nn.Module):
    """
    Modality-agnostic cross-modal fusion.

    Accepts any subset of modalities (e.g. ['text','audio'] for bimodal ablation).
    Builds one directed cross-attention pair per ordered (i, j) modality pair.

    For 3 modalities: 6 pairs → pred_fusion(6*d_m)
    For 2 modalities: 2 pairs → pred_fusion(2*d_m)
    """

    def __init__(self, d_m, num_classes, modalities, n_cross_heads=4, d_ff=256,
                 dropout=0.1, att_dropout=0.2):
        super().__init__()
        self.modalities = modalities
        self.d_m = d_m

        # One input projection per modality
        self.projs = nn.ModuleDict({m: nn.Linear(d_m, d_m) for m in modalities})

        # One CrossModalAttention per directed pair (i→j)
        self.cross_atts = nn.ModuleDict()
        for mi in modalities:
            for mj in modalities:
                if mi != mj:
                    self.cross_atts[f'{mi}2{mj}'] = CrossModalAttention(d_m, n_cross_heads, att_dropout)

        n_pairs = len(modalities) * (len(modalities) - 1)

        self.encoder = EncoderBlock(d_m, n_cross_heads, d_ff, dropout)
        self.decoder = DecoderBlock(d_m, n_cross_heads, d_ff, dropout)

        self.pred_fusion = nn.Linear(n_pairs * d_m, num_classes)
        self.pred_recon = nn.Linear(d_m, num_classes)

    def forward(self, feats, masks):
        """
        feats : dict[str → (batch, T, d_m)]   — UFEN outputs per modality
        masks : dict[str → (batch, T) bool]   — padding masks per modality
        """
        # Project each modality
        proj = {m: self.projs[m](feats[m]) for m in self.modalities}

        # Compute each directed cross-attention pair, mean-pool to (batch, d_m)
        tokens = []
        for mi in self.modalities:
            for mj in self.modalities:
                if mi != mj:
                    ca_out = self.cross_atts[f'{mi}2{mj}'](
                        proj[mi], proj[mj], masks.get(mi), masks.get(mj)
                    )
                    tokens.append(masked_mean(ca_out, masks.get(mi)))  # (batch, d_m)

        tokens = torch.stack(tokens, dim=1)                            # (batch, n_pairs, d_m)

        y_m = self.pred_fusion(tokens.reshape(tokens.size(0), -1))    # (batch, num_classes)

        enc_out = self.encoder(tokens)
        dec_out = self.decoder(tokens, enc_out)

        y_m_prime = self.pred_recon(dec_out.mean(dim=1))               # (batch, num_classes)

        return y_m, y_m_prime


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Full model: text encoder + UFEN per active modality + MTFN.

    Active modalities are controlled by config.modalities (default: ['text','audio','video']).
    This enables bimodal ablation without changing training code.

    Returns dict with keys = active modality names + 'fusion' + 'recon'.
    All values are (batch, num_classes) logits.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_bert = getattr(config, 'use_bert', False)
        num_classes = config.num_classes
        self.modalities = getattr(config, 'modalities', ['text', 'audio', 'video'])

        # ---- Text encoder (only if 'text' is active) ----
        if 'text' in self.modalities:
            if self.use_bert:
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                text_input_dim = 768
            else:
                vocab_size, emb_dim = config.pretrained_emb.shape
                self.text_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.text_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_emb))
                self.text_embedding.weight.requires_grad = True
                text_input_dim = emb_dim
            self.norm_t = nn.LayerNorm(text_input_dim)
            self.ufen_t = UFEN(
                input_dim=text_input_dim, d_m=config.d_m, num_classes=num_classes,
                n_layers=config.n_layers, kernel_sizes=config.kernel_sizes,
                conv_dim=config.conv_dim, n_att_heads=config.self_att_heads,
                dropout=config.dropout,
            )

        # ---- Video encoder (only if 'video' is active) ----
        if 'video' in self.modalities:
            visual_proj_dim = getattr(config, 'visual_proj_dim', None)
            if visual_proj_dim is not None:
                self.video_proj = nn.Linear(config.visual_size, visual_proj_dim)
            else:
                self.video_proj = None
                visual_proj_dim = config.visual_size
            self.norm_v = nn.LayerNorm(visual_proj_dim)
            self.ufen_v = UFEN(
                input_dim=visual_proj_dim, d_m=config.d_m, num_classes=num_classes,
                n_layers=config.n_layers, kernel_sizes=config.kernel_sizes,
                conv_dim=config.conv_dim, n_att_heads=config.self_att_heads,
                dropout=config.dropout,
            )

        # ---- Audio encoder (only if 'audio' is active) ----
        if 'audio' in self.modalities:
            self.norm_a = nn.LayerNorm(config.acoustic_size)
            self.ufen_a = UFEN(
                input_dim=config.acoustic_size, d_m=config.d_m, num_classes=num_classes,
                n_layers=config.n_layers, kernel_sizes=config.kernel_sizes,
                conv_dim=config.conv_dim, n_att_heads=config.self_att_heads,
                dropout=config.dropout,
            )

        # ---- MTFN: only when >=2 modalities (cross-attention needs pairs) ----
        if len(self.modalities) >= 2:
            self.mtfn = MTFN(
                d_m=config.d_m, num_classes=num_classes, modalities=self.modalities,
                n_cross_heads=config.cross_att_heads, d_ff=config.d_ff,
                dropout=config.dropout, att_dropout=config.att_dropout,
            )
        else:
            self.mtfn = None

    def forward(self, token_ids, audio, video, av_mask,
                bert_ids=None, bert_mask=None, bert_type_ids=None):
        """
        Returns:
            preds  : dict  {modality_name: logits, ..., 'fusion': y_m, 'recon': y_m_prime}
            pooled : dict  {modality_name: (batch, d_m) mean-pooled UFEN output}
        All logits are (batch, num_classes).
        """
        feats, masks, preds, pooled = {}, {}, {}, {}

        # ---- Text ----
        if 'text' in self.modalities:
            if self.use_bert:
                bert_out = self.bert(
                    input_ids=bert_ids,
                    attention_mask=bert_mask,
                    token_type_ids=bert_type_ids,
                ).last_hidden_state                          # (batch, S, 768)
                text_emb = self.norm_t(bert_out)
                t_mask = (bert_mask == 0)                    # True where BERT pads
            else:
                text_emb = self.norm_t(self.text_embedding(token_ids))
                t_mask = av_mask
            feat_t, y_t = self.ufen_t(text_emb, t_mask)
            feats['text'] = feat_t
            masks['text'] = t_mask
            preds['text'] = y_t
            pooled['text'] = masked_mean(feat_t, t_mask)

        # ---- Video ----
        if 'video' in self.modalities:
            v = video
            if self.video_proj is not None:
                v = self.video_proj(v)
            v = self.norm_v(v)
            feat_v, y_v = self.ufen_v(v, av_mask)
            feats['video'] = feat_v
            masks['video'] = av_mask
            preds['video'] = y_v
            pooled['video'] = masked_mean(feat_v, av_mask)

        # ---- Audio ----
        if 'audio' in self.modalities:
            a = self.norm_a(audio)
            feat_a, y_a = self.ufen_a(a, av_mask)
            feats['audio'] = feat_a
            masks['audio'] = av_mask
            preds['audio'] = y_a
            pooled['audio'] = masked_mean(feat_a, av_mask)

        # ---- MTFN (skipped for single-modality — reuse unimodal pred) ----
        if self.mtfn is not None:
            y_m, y_m_prime = self.mtfn(feats, masks)
            preds['fusion'] = y_m
            preds['recon'] = y_m_prime
        else:
            # Single modality: fusion and recon are the same as the unimodal head
            sole = preds[self.modalities[0]]
            preds['fusion'] = sole
            preds['recon'] = sole

        return preds, pooled
