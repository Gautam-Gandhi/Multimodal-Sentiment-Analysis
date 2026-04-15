# Multimodal Sentiment & Emotion Analysis

**INLP Spring 2026 · Team TokeNization**

A two-phase study of multimodal sentiment/emotion analysis using the **UFEN + MTFN** architecture of Cai et al. (2025):

- **Phase 1 — CMU-MOSI** (regression, sentiment intensity ∈ [-3, +3]): from-scratch replication with BERT + COVAREP + FACET features.
- **Phase 2 — MELD** (7-class emotion classification): extension of the same backbone to conversational emotion, with a new **Multi-Head Feature Tokenization (MHFT)** front-end that lets UFEN consume utterance-level features (RoBERTa, openSMILE, DenseNet). The best model beats a 110M-param BERT baseline at **30× fewer parameters**.

---

## Headline Results

| Phase | Dataset | Best Model | Key Metric | Score |
|---|---|---|---|---|
| 1 | CMU-MOSI | UFEN+MTFN (P3-2) | MAE ↓ / Acc-2 (neg/pos) ↑ | **0.812 / 80.3%** |
| 2 | MELD | MHFT + UFEN+MTFN (Exp 9.1) | F1w / F1m | **64.4 / 49.0** |

Phase 2's best model uses just **3.58M parameters** (vs 110.9M for the fine-tuned BERT baseline in Exp 6).

---

## Repository Structure

```
.
├── README.md                 This file
├── Project_Details.md        Course-provided project brief
│
├── proposal/
│   └── INLP_Project_Proposal_Team_TokeNization.pdf
│
├── phase1/                   CMU-MOSI — UFEN+MTFN replication
│   ├── README.md
│   ├── src/                  model.py, train.py, data_loader.py, create_dataset.py
│   ├── docs/                 assumptions.md, experiments.md, research_paper.md
│   ├── report/               Mid-submission LaTeX report + architecture figure
│   └── assets/               Dataset stats + reference-paper figures
│
├── phase2/                   MELD — MHFT + UFEN+MTFN
│   ├── README.md
│   ├── src/                  model.py, train.py, train_enhanced.py, data_loader.py, config.py
│   ├── notebooks/            01_baseline_bert, 02_multiemo_features, 03_mhft_ufen_best, 04_mlp_baseline
│   └── docs/                 experiments.md, implementation_plan.md
│
├── report/                   Final submission (ACL 9-page format)
│   ├── main.tex
│   ├── main.pdf
│   └── figures/
│
├── presentation/             Beamer slides (Metropolis theme, 16:9)
│   ├── presentation.tex
│   ├── presentation.pdf
│   └── figures/
│
└── data/                     Not tracked in git
    ├── MOSI/                 Preprocessed CMU-MOSI (embedding_and_mapping.pt)
    └── MELD/                 MELD utterances and features
```

Checkpoints for both phases are hosted separately (see each phase's README for paths).

---

## Deliverables

| File | Contents |
|---|---|
| [`report/main.pdf`](report/main.pdf) | 9-page ACL-style final report covering both phases |
| [`presentation/presentation.pdf`](presentation/presentation.pdf) | Beamer slides for the final walkthrough |
| [`phase1/docs/experiments.md`](phase1/docs/experiments.md) | Phase 1 experiment log (25+ runs) |
| [`phase2/docs/experiments.md`](phase2/docs/experiments.md) | Phase 2 experiment log (Exp 1 → 9.1) |

---

## Quick Start

### Setup

```bash
pip install torch transformers numpy scikit-learn pandas
```

Tested with Python 3.11, PyTorch 2.6, CUDA 12.4.

### Reproduce Phase 1 (CMU-MOSI)

```bash
cd phase1
python src/train.py
```

Preprocessed data must be at `data/MOSI/embedding_and_mapping.pt`. Training takes ~3 minutes on an RTX 3050 and reproduces MAE 0.812 / Acc-2 80.3%.

### Reproduce Phase 2 Best Model (MELD, Exp 9.1)

```bash
cd phase2
jupyter notebook notebooks/03_mhft_ufen_best.ipynb
```

Requires MultiEMO pre-extracted features (RoBERTa + openSMILE + DenseNet). See [`phase2/README.md`](phase2/README.md) for details.

### Build the Report

```bash
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Build the Presentation

```bash
cd presentation
pdflatex presentation.tex && pdflatex presentation.tex
```

---

## Architecture at a Glance

Both phases share the same two-stage backbone:

1. **UFEN (Unimodal Feature Extraction Network)** — one per modality. Bi-GRU → parallel Conv1D branches (kernels [1,5]) with self-attention gates → unpool → element-wise sum. Produces both a sequence representation and a unimodal prediction.
2. **MTFN (Multi-task Transformer Fusion Network)** — six directed cross-modal attention pairs (t↔v, t↔a, v↔a) feed a Transformer encoder–decoder. Two fusion predictions (direct linear `Y_m` + encoder–decoder `Y'_m`).

Training uses a **5-head multi-task loss** (3 unimodal + 2 multimodal).

**Phase 2 additions:**
- **MHFT** front-end: maps utterance-level features (B, D) → (B, K=8, d_m) so UFEN can operate unchanged.
- **Weighted cross-entropy + label smoothing** (replaces MSE).
- **Exp 9.1** adds a speaker-disjoint dev split, capped class weights, and top-5 checkpoint ensembling.

See [`report/main.pdf`](report/main.pdf) §4 (Methodology) for full details and [`phase1/report/arch.png`](phase1/report/arch.png) for the architecture figure.

---

## Team

**TokeNization**

- Parth Tokekar
- (team members)

Course: Introduction to NLP (S26), IIIT Hyderabad.

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
