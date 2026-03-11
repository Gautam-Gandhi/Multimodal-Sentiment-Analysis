# Architecture Assumptions & Discrepancies

Decisions made where the paper was ambiguous or silent, with reasoning.

---

## 1. BERT role in UFEN for text modality

**Paper says (Experiments section):** "For text modality, we adopt the BERT pre-trained model as the **feature extractor**. BERT aims to learn word embeddings by pretraining deep bidirectional representations from unlabeled text."

**Paper says (UFEN section):** The UFEN pipeline (BiGRU → N parallel Conv1D → Self-Att → Unpool → sum) is described uniformly for all modalities. No text-specific exception appears anywhere in the methods section.

**Clarification on a prior confusion:** The quote "we replace the operations of ATCN in Eq (2)–(4) with Z_t = BERT(X_t; θ_bert_t)" comes from a **cited paper** referenced in our paper, NOT from our paper itself.

**Interpretation:** BERT is a **pre-processor / feature extractor** for text — it plays the same upstream role as FACET (visual features) and COVAREP (acoustic features). BERT outputs contextual word embeddings of shape (batch, seq, 768), which become the *input* to the text UFEN. The full UFEN pipeline (BiGRU 768→d_m, then N parallel Conv1D → Self-Att gate → Unpool, then sum) runs identically on all three modalities.

**Implementation:** `ufen_t` uses `UFEN(input_dim=768, ...)` with the same BiGRU + Conv architecture as `ufen_v` and `ufen_a`. BERT is called in `MultiTaskModel.forward` to obtain `last_hidden_state`, which is then passed directly into `ufen_t`.

---

## 2. Unimodal hidden dimension d_m

**Paper says:** Not specified explicitly.

**Assumption:** d_m = 128.

**Reason:** (a) With 4 cross-attention heads (Table 2), each head gets 128/4 = 32 dimensions — a standard choice. (b) MISA (the reference repo whose data we use) also uses 128. (c) Keeps the model from being too large for the small MOSI dataset (1283 training samples).

---

## 3. Conv1D filter count (conv_dim)

**Paper says:** Not specified.

**Assumption:** conv_dim = 64 per branch. Unpooling then maps 64 → d_m = 128.

**Reason:** Keeps per-branch capacity lower than d_m so the unpooling layer has something meaningful to learn, rather than being an identity. Two branches × 64 filters = 128 after summation, matching d_m naturally.

---

## 4. Conv1D kernel sizes

**Paper says:** Not specified, only that N=2 parallel branches exist (ablation: "when number of layers is 2, model performs best").

**Assumption:** kernel_sizes = [1, 3].

**Reason:** kernel=1 captures per-token (pointwise) features; kernel=3 captures local trigram context. Short utterances (avg 5–14 words) benefit from small kernels. Padding = k//2 ensures all kernels output the same time length T.

---

## 5. Feedforward dimension in Encoder-Decoder (d_ff)

**Paper says:** Not specified.

**Assumption:** d_ff = 256 = 2 × d_m.

**Reason:** Standard transformer practice is d_ff = 2–4× d_model. Given the small dataset, 2× keeps the model regularised.

---

## 6. Number of Encoder-Decoder layers

**Paper says:** Describes a single encoder block and a single decoder block (Fig 5 shows one of each).

**Assumption:** 1 encoder layer + 1 decoder layer.

**Reason:** The paper's figure and equations describe a single block, consistent with the limited dataset size (overfitting risk with deeper transformers).

---

## 7. Decoder design

**Paper says (Eq 13):** Nm_de = LN(Input + MhAtt(Input, Output_en)), meaning cross-attention only (Q from Input, K/V from encoder output). No explicit self-attention in the decoder.

**Discrepancy:** Standard transformer decoders include a self-attention sub-layer before the cross-attention.

**Assumption:** Follow the paper literally — decoder has only cross-attention (no self-attention).

**Reason:** The paper's formulas are explicit. A pure cross-attention decoder also makes sense for a short 6-token sequence.

---

## 8. Flattening strategy for Y_m

**Paper says:** Input ∈ R^(d_b × 6d_m); Y_m = softmax(W_m Input + b_m).

**Discrepancy:** Treating Input as flat 6d_m-dim vector vs. a 6-token sequence.

**Assumption:** Flatten the six mean-pooled cross-attention outputs into a single vector of size 6 × d_m, then apply Linear(6*d_m → 1).

**Reason:** The paper's dimension notation R^(d_b × 6d_m) treats d_b as batch and 6d_m as the feature dimension, consistent with flattening.

---

## 9. Softmax vs. linear output

**Paper says:** Y_i = softmax(W_i X + b_i) for all prediction heads.

**Discrepancy:** Labels are continuous regression targets in [−3, 3]. Applying softmax would constrain outputs to (0, 1), making MSE loss meaningless.

**Assumption:** Use a plain Linear(d → 1) with no activation on all five prediction heads.

**Reason:** This is a regression task with MSE loss (Eq 18). Softmax in the paper is a misprint / carried over from a classification framework. All comparable papers (MISA, Self-MM) use linear outputs for regression on MOSI.

---

## 10. BERT learning rate

**Paper says:** lr = 5e-3 for MOSI.

**Discrepancy:** 5e-3 is far too large for fine-tuning BERT (110M parameters); it would destroy the pre-trained weights.

**Assumption:** Use a separate learning rate: BERT params at 1e-5, all other params at 5e-3.

**Reason:** Standard practice for BERT fine-tuning. The paper likely applied 5e-3 only to the non-BERT parameters and used an implicit or separate low LR for BERT, consistent with how MISA and Self-MM handle this.

---

## 11. Mean-pooling step between cross-attention and Encoder-Decoder

**Paper says:** Cross-modal attention outputs are concatenated as Input ∈ R^(d_b × 6d_m).

**Discrepancy:** Each cross-attention output is a sequence of shape (batch, T_query, d_m). Since each query modality has a different T, they cannot be directly concatenated in the time dimension.

**Assumption:** Mean-pool each cross-attention output over its temporal dimension (using padding masks for correctness) → (batch, d_m) per output → stack 6 of them → (batch, 6, d_m) as the encoder-decoder input sequence.

**Reason:** This is the only way to reconcile variable-length cross-attention outputs with a fixed-size concatenation. The 6-token sequence view also makes the encoder-decoder's self/cross-attention semantically meaningful.

---

## 12. Gradient clipping

**Paper does not mention** gradient clipping.

**Assumption:** Apply gradient clipping with max_norm=1.0.

**Reason:** Standard training stability practice, especially with BERT and large initial lr for non-BERT parameters. Low risk of changing final results.

---

## Summary of key hyperparameters used

| Parameter        | Value          | Source               |
|------------------|----------------|----------------------|
| d_m              | 128            | Assumption #2        |
| conv_dim         | 64             | Assumption #3        |
| n_layers         | 2              | Paper ablation Table 7 |
| kernel_sizes     | [1, 3]         | Assumption #4        |
| d_ff             | 256            | Assumption #5        |
| self_att_heads   | 1              | Paper Table 2        |
| cross_att_heads  | 4              | Paper Table 2        |
| att_dropout      | 0.2            | Paper Table 2        |
| dropout          | 0.1            | Paper Table 2        |
| lr (non-BERT)    | 5e-3           | Paper Table 2        |
| lr (BERT)        | 1e-5           | Assumption #10       |
| batch_size       | 32             | Paper Table 2        |
| epochs           | 30             | Paper Table 2        |
| early_stop       | 8              | Paper Table 2        |
