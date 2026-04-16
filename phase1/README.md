# Multimodal Sentiment Analysis — UFEN + MTFN (Implementation v2)

Replication of **Cai et al., "Multimodal sentiment analysis based on multi-layer feature fusion and multi-task learning"**, *Scientific Reports* (2025) 15:2126.  
DOI: [10.1038/s41598-025-85859-6](https://doi.org/10.1038/s41598-025-85859-6)

This is a from-scratch implementation of the UFEN + MTFN architecture evaluated on the **CMU-MOSI** dataset, with systematic hyperparameter tuning across 25+ experiments.

---

## Architecture Overview

The model has three stages:

### 1. Feature Extraction
- **Text**: `bert-base-uncased` — fine-tuned during training; outputs contextual word embeddings (batch × 52 × 768) fed into the text UFEN.
- **Visual**: 47-dim FACET features, padded per-utterance.
- **Audio**: 74-dim COVAREP features, padded per-utterance.

### 2. UFEN — Unimodal Feature Extraction Network (×3)
One UFEN instance per modality. Each runs identically:

```
Input (batch, T, D_i)
  └─ BiGRU  (D_i → d_m)
       └─ N parallel branches, each:
            Conv1D(d_m → conv_dim, kernel=k_i)  → ReLU
            Self-Attention gate: X * MHA(X, X, X)
            Unpool: Linear(conv_dim → d_m)
       └─ Element-wise sum over branches
       └─ LayerNorm + Dropout
       └─ Mean-pool → Linear(d_m → 1)   → unimodal prediction Y_i
  Output: (batch, T, d_m),  Y_i (batch,)
```

### 3. MTFN — Multi-Task Fusion Network
```
3 × UFEN outputs (feat_t, feat_v, feat_a)
  └─ Linear projection per modality (W_i X_i + b_i)
  └─ 6 directed cross-modal attention pairs:
       t→v,  t→a,  v→t,  v→a,  a→t,  a→v
  └─ Mean-pool each → stack → (batch, 6, d_m)   [6-token sequence]
  └─ Y_m = Linear(6×d_m → 1)                    [direct fusion prediction]
  └─ Transformer Encoder (Self-Att + FFN + LayerNorm)
  └─ Transformer Decoder (Cross-Att + FFN + LayerNorm)
  └─ Mean-pool decoder output → Linear(d_m → 1) → Y_m'  [final prediction]
```

### 4. Multi-Task Loss
Five MSE losses summed with equal weight:

$$\mathcal{L} = \text{MSE}(Y_t, Y) + \text{MSE}(Y_v, Y) + \text{MSE}(Y_a, Y) + \text{MSE}(Y_m, Y) + \text{MSE}(Y'_m, Y)$$

Final evaluation uses $Y'_m$ (the encoder-decoder output head).

**Total parameters: ~110.8M** (dominated by `bert-base-uncased` at 110M).

---

## Project Structure

```
phase1/
├── data/MOSI/                 Preprocessed CMU-MOSI splits (embedding_and_mapping.pt)
├── src/
│   ├── model.py               Full model: UFEN, CrossModalAttention, MTFN, MultiTaskModel
│   ├── train.py               Training + evaluation pipeline with all hyperparameters
│   ├── data_loader.py         DataLoader with padding-mask generation
│   └── create_dataset.py      Dataset preprocessing / pkl builder
├── docs/
│   ├── assumptions.md         Architecture decisions where the paper was ambiguous
│   └── experiments.md         Full experiment log (25+ experiments across 3 phases)
├── report/
│   ├── main.tex               Mid-submission report source
│   └── arch.png               Architecture figure
└── README.md
```

The preprocessed CMU-MOSI data is expected at `../data/MOSI/embedding_and_mapping.pt` (top-level `data/` directory, not tracked in git).

---

## Setup

### Requirements

```bash
pip install torch transformers numpy scikit-learn
```

Tested with: Python 3.11.9 · PyTorch 2.6.0+cu124 · transformers 5.2.0

### Data

The preprocessed CMU-MOSI data (`data/MOSI/embedding_and_mapping.pt`) is included in the repository. It contains aligned text/visual/acoustic features and train/dev/test splits.

---

## Running

```bash
cd phase1
python src/train.py
```

Training runs with the best-found configuration (P3-2, see below). Outputs epoch-level dev metrics and final test results. Best checkpoint is saved to `best_model.pt`.

GPU is used automatically if available. On an RTX 3050, training takes ~3 minutes.

---

## Results

### Best Configuration (P3-2)

The best test results were achieved with the following config, found after systematic search across 3 phases and 25+ experiments:

| Hyperparameter | Value | Note |
|---|---|---|
| `d_m` | 128 | unimodal hidden dim |
| `conv_dim` | 64 | Conv1D filters per branch |
| `kernel_sizes` | `[1, 5]` | multi-scale local features |
| `d_ff` | 128 | encoder-decoder FFN dim |
| `lr` (non-BERT) | 5e-3 | paper Table 2 |
| `lr_bert` | 2e-5 | separate low LR for BERT |
| `use_bert_warmup` | True | 5-epoch linear ramp for BERT lr |
| `use_lr_scheduler` | True | cosine annealing for both LRs |
| `epochs` | 50 | `early_stop=15` |
| `batch_size` | 32 | paper Table 2 |
| `dropout` | 0.1 | paper Table 2 |
| `att_dropout` | 0.2 | paper Table 2 |

### Test Performance vs. Paper

| Metric | **Ours (P3-2)** | Paper | Gap |
|---|---|---|---|
| MAE ↓ | **0.812** | 0.728 | +0.084 |
| Corr ↑ | **0.745** | 0.792 | −0.047 |
| Acc-7 ↑ | **43.0** | 46.7 | −3.7 pp |
| Acc-2 (neg/non-neg) ↑ | **78.6** | 85.2 | −6.6 pp |
| Acc-2 (neg/pos) ↑ | **80.3** | 86.6 | −6.3 pp |
| F1 (neg/pos) ↑ | **80.4** | 86.7 | −6.3 pp |

> Dev MAE reached **0.7434** at epoch 42 (within 0.015 of the paper), indicating the gap is primarily a dev/test generalisation issue rather than a capacity one.

---

## Experiment Summary

Full experiment details are in [`docs/experiments.md`](docs/experiments.md).

### Phase 1 — Single-parameter search (14 experiments)

Varied `lr_bert`, `d_m`, `conv_dim`, `d_ff`, `kernel_sizes`, `grad_clip` one at a time from a paper-matching baseline.

| Key finding | Best value | Impact |
|---|---|---|
| `d_ff=128` (1×d_m, not 2×d_m) | 128 | MAE: 0.903 → 0.827 |
| `kernel_sizes=[1,5]` | [1, 5] | better than [1,3] default |
| `lr_bert=2e-5` (vs 1e-5) | 2e-5 | faster BERT adaptation |

### Phase 2 — Combination experiments (7 experiments)

Combined Phase 1 winners. Key discovery: **cosine LR scheduler + early_stop=15** was the single biggest unlock.

| Exp | Key addition | MAE | Corr |
|---|---|---|---|
| P2-1 | d_ff=128 + lr_bert=2e-5 | 0.916 | 0.706 |
| P2-4 | + kernel=[1,5] | 0.901 | 0.696 |
| **P2-6.2** | **+ cosine LR + early_stop=15** | **0.828** | **0.749** |

### Phase 3 — Regularization (5 experiments)

Targeted the dev/test gap revealed in Phase 2.

| Exp | Change | MAE | Corr | Acc-7 | Verdict |
|---|---|---|---|---|---|
| P3-1 | weight decay 0.01 | 1.017 | 0.672 | 33.7 | ❌ BERT collapse at ep4 |
| **P3-2** | **BERT lr warmup** | **0.812** | **0.745** | **43.0** | **✅ best overall** |
| P3-3 | dropout 0.2/att 0.3 | 0.880 | 0.711 | 39.4 | ❌ large dev/test gap |
| P3-4 | weight decay + warmup | 1.015 | 0.650 | 31.8 | ❌ WD dominates |
| P3-5 | AdamW + WD | 0.939 | 0.694 | 33.7 | ❌ worse than Adam |

**Key insight:** BERT warmup (linear ramp from 0 → `lr_bert` over 5 epochs) eliminated the training instability in epochs 1–5 and produced the cleanest convergence curve. Weight decay, even moderate amounts, was catastrophic for BERT fine-tuning on this small dataset.

---

## Key Architecture Decisions

Several aspects of the paper were underspecified. Full details are in [`docs/assumptions.md`](docs/assumptions.md). Major decisions:

1. **BERT as upstream feature extractor** — BERT outputs `last_hidden_state` (batch × 52 × 768) which feeds directly into the text UFEN, playing the same role as FACET (visual) and COVAREP (acoustic).
2. **Separate BERT learning rate** — The paper's stated `lr=5e-3` applies only to non-BERT parameters. BERT is fine-tuned at `lr_bert=2e-5` (standard practice).
3. **Output heads are linear, not softmax** — The paper describes `softmax(Wx+b)` for regression; this is clearly a misprint. All prediction heads are plain `Linear(d→1)` for continuous regression against MSE loss.
4. **Mean-pool before encoder-decoder** — Each cross-modal attention output is mean-pooled over the query modality's time dimension to create a fixed-size (batch, d_m) vector; six such vectors are stacked into a 6-token sequence for the encoder-decoder.
5. **No self-attention in decoder** — The paper's Eq. 13 describes only cross-attention in the decoder (no masked self-attention sub-layer). Implementation follows the paper literally.

---

## Reference

```bibtex
@article{cai2025multimodal,
  title   = {Multimodal sentiment analysis based on multi-layer feature fusion and multi-task learning},
  author  = {Cai, Yujian and Li, Xingguang and Zhang, Yingyu and Li, Jinsong and Zhu, Fazheng and Rao, Lin},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {2126},
  year    = {2025},
  doi     = {10.1038/s41598-025-85859-6}
}
```
