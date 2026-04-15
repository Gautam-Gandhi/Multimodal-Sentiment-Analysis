# Phase 2 — Multimodal Emotion Recognition on MELD

Extension of the UFEN+MTFN architecture from Phase 1 to the **MELD** dataset (7-class conversational emotion recognition: Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger).

The phase consisted of nine experiments that progressively improved weighted-F1 from the initial port (46.0) to the best ensemble model at **64.4 F1w / 49.0 F1m** — with only **3.58M parameters**, 30× fewer than a BERT-based baseline.

---

## Results Summary

| Exp | Description | F1w | F1m | Params |
|---|---|---|---|---|
| 1 | Direct GloVe port (class collapse) | 24.3 | 7.0 | 10.6M |
| 2 | + LayerNorm + lower LR + label smoothing | 46.0 | 25.5 | 10.6M |
| 3–5 | GloVe hyperparameter search | ~51 | ~29 | 10.6M |
| 6 | Fine-tuned **BERT** backbone | 60.3 | 44.2 | 110.9M |
| 7 | Ablations (text-only vs T+A+V) | 57.9→60.3 | — | 110.9M |
| 8 | OGM-GE + InfoNCE + Curriculum | 59.4 | 43.8 | 110.9M |
| 9 | **MHFT** + MultiEMO features (RoBERTa+openSMILE+DenseNet) | 61.5 | 45.7 | 3.58M |
| **9.1** | **+ speaker-disjoint dev + capped class weights + top-5 ensemble** | **64.4** | **49.0** | **3.58M** |

Full experiment log (training curves, per-class breakdowns, confusion matrices) is in [`docs/experiments.md`](docs/experiments.md).

---

## Project Structure

```
phase2/
├── src/
│   ├── model.py              UFEN + MTFN for MELD (GloVe/BERT variants)
│   ├── train.py              Baseline training loop
│   ├── train_enhanced.py     Exp 8: OGM-GE + InfoNCE + scheduled curriculum
│   ├── data_loader.py        MELD data loading and padding
│   └── config.py             Hyperparameter config
├── notebooks/
│   ├── 01_baseline_bert.ipynb       Exp 6: fine-tuned BERT backbone
│   ├── 02_multiemo_features.ipynb   Exp 9 prep: MultiEMO feature extraction
│   ├── 03_mhft_ufen_best.ipynb      Exp 9 / 9.1: MHFT + UFEN (BEST MODEL)
│   └── 04_mlp_baseline.ipynb        Exp 9 sanity check: MLP on MultiEMO features
├── docs/
│   ├── experiments.md               Full experiment log (Exp 1 → 9.1)
│   └── implementation_plan.md       Phase 2 planning document
└── README.md
```

MELD data is expected at `../data/MELD/` (not tracked in git).

---

## Key Architectural Additions

### MHFT — Multi-Head Feature Tokenization (Exp 9)

Utterance-level features (single vector per clip) cannot be consumed directly by UFEN, which expects a sequence. MHFT bridges the gap by projecting each utterance feature vector into **K=8 synthetic tokens**:

```
x ∈ (B, D)  →  Wx + b ∈ (B, K·d_m)  →  reshape (B, K, d_m)  →  + positional_embedding
```

This lets the unchanged UFEN+MTFN backbone operate on pre-extracted features from RoBERTa, openSMILE, and DenseNet (the MultiEMO feature set).

### Exp 9.1 Fixes (the leap from 61.5 → 64.4 F1w)

1. **Speaker-disjoint dev split.** The original random trainVid split leaked speaker context; Exp 9 had a 10-point dev→test gap. Building dev from held-out speakers reduced the gap to 5.7 and lifted test F1w by ~2.9.
2. **Capped class weights (max 1.5).** Uncapped weights over-penalised the majority class (Neutral) and hurt overall F1w.
3. **Top-5 checkpoint ensemble.** Averaging logits from the top-5 dev-F1w checkpoints gave a robust final predictor.

### Exp 8 (Enhanced Training) — What Did Not Work

`src/train_enhanced.py` implements:
- **OGM-GE** (on-the-fly gradient modulation — rescales weak-branch gradients)
- **InfoNCE** (CLIP-style cross-modal contrastive loss on UFEN pooled embeddings)
- **Phase-scheduled curriculum** (warm up weak modalities first, then joint training)

Net result: F1w -0.9 vs Exp 6. The Phase A (A+V only) warm-up collapsed to 2% accuracy — weak modalities cannot bootstrap alone on MELD. Preserved for reproducibility and as a negative result.

---

## Running

### Exp 9.1 (Best Model)

```bash
cd phase2
jupyter notebook notebooks/03_mhft_ufen_best.ipynb
```

Requires MultiEMO pre-extracted features (RoBERTa, openSMILE, DenseNet). The notebook documents the exact feature layout and splits.

### Exp 6 (BERT Baseline)

```bash
jupyter notebook notebooks/01_baseline_bert.ipynb
```

### Exp 8 (Enhanced Training)

```bash
python src/train_enhanced.py
```

---

## Requirements

```bash
pip install torch transformers numpy scikit-learn pandas
```

Tested with: Python 3.11 · PyTorch 2.6 · transformers 4.x · CUDA 12.4
