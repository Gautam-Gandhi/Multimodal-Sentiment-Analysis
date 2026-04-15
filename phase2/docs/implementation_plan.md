# Phase 2 Implementation Plan — MELD Emotion Classification

## Overview

**Goal:** Adapt the UFEN+MTFN architecture (validated on CMU-MOSI regression in Phase 1) to perform **7-class emotion classification** on the **MELD dataset**, as proposed in our project proposal (Section 4.2).

**Final Deadline:** April 8, 2026

---

## 1. MELD Dataset: Key Differences from CMU-MOSI

| Aspect | CMU-MOSI (Phase 1) | MELD (Phase 2) |
|---|---|---|
| Task | Regression (sentiment intensity) | 7-class classification (emotion) |
| Labels | Continuous [-3, 3] | Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise |
| Label distribution | Roughly balanced | Heavily imbalanced (Neutral ~47%, Fear ~2%) |
| Source | YouTube monologues | Friends TV show dialogues |
| Samples | 2,199 (1283/229/686) | ~13,708 (9,989/1,109/2,610) |
| Pre-extracted features | FACET 47-dim (visual), COVAREP 74-dim (audio), word-aligned | **Not word-aligned** — raw video/audio/text per utterance |
| Text format | Word-level tokens | Full utterance transcripts |
| Modalities included | Text + Audio + Video (word-aligned) | Text + Audio + Video (utterance-level) |

### MELD Label Distribution (approximate)
| Emotion | Train | % |
|---|---|---|
| Neutral | ~4,710 | 47.2% |
| Joy | ~1,743 | 17.5% |
| Surprise | ~1,205 | 12.1% |
| Anger | ~1,109 | 11.1% |
| Sadness | ~683 | 6.8% |
| Disgust | ~271 | 2.7% |
| Fear | ~268 | 2.7% |

---

## 2. Feature Extraction Strategy for MELD

Since MELD does not provide pre-extracted word-aligned FACET/COVAREP features like MOSI, we need a feature extraction pipeline.

### Option A: Utterance-Level Feature Extraction (Recommended)

Extract a **single feature vector per utterance** for audio and visual, then treat each utterance as a sequence of length 1 for audio/visual (while text remains a token sequence via BERT).

| Modality | Extractor | Output Dim | Notes |
|---|---|---|---|
| **Text** | `bert-base-uncased` | (batch, seq_len, 768) | Same as Phase 1; tokenize utterance transcript |
| **Audio** | `librosa` MFCC + prosodic features, OR pre-trained `wav2vec2` / `openSMILE` | ~40–768 dim | Extract from audio track of each video clip |
| **Visual** | Pre-trained CNN (ResNet/FaceNet) on sampled frames, OR `OpenFace` AU features | ~35–512 dim | Extract from video frames, average over frames |

### Option B: Use Pre-Extracted MELD Features (If Available)

Several papers provide pre-extracted MELD features. Check for:
- **MMGCN** / **DialogueRNN** repos — often include pre-extracted text (RoBERTa), audio, visual features
- **M3ED** / **MELD** official repo — may have feature files

**Recommendation:** Start with Option B if pre-extracted features are available (faster iteration). Fall back to Option A if not.

### Key Decision: Sequence Length for Audio/Visual

In Phase 1, audio/visual were word-aligned (same T as text). In MELD:
- **If utterance-level features:** Audio/visual have T=1 → UFEN's BiGRU and Conv1D become trivial. Consider either:
  - (a) Skip UFEN for audio/visual, feed features directly to MTFN after a linear projection
  - (b) Use frame-level visual features (sampled at N fps) and segment-level audio features to create short sequences
- **If using pre-extracted sequential features:** Feed directly into UFEN as before

---

## 3. Architecture Changes

### 3.1 Output Head Changes (Regression → Classification)

This is the core change described in Section 4.2 of our proposal.

#### Current (Phase 1 — Regression)
```
UFEN pred_head:    Linear(d_m, 1)     → scalar prediction per modality
MTFN pred_fusion:  Linear(6*d_m, 1)   → Y_m  scalar
MTFN pred_recon:   Linear(d_m, 1)     → Y_m' scalar
```

#### New (Phase 2 — 7-Class Classification)
```
UFEN pred_head:    Linear(d_m, 7)     → logits over 7 emotion classes
MTFN pred_fusion:  Linear(6*d_m, 7)   → Y_m  logits (7 classes)
MTFN pred_recon:   Linear(d_m, 7)     → Y_m' logits (7 classes)
```

**All 5 prediction heads** change from outputting shape `(batch,)` to `(batch, 7)`.

### 3.2 Tensor Shape Changes Summary

| Component | Phase 1 Shape | Phase 2 Shape | Change |
|---|---|---|---|
| BERT output | (batch, 52, 768) | (batch, seq_len, 768) | seq_len may differ; use dynamic length |
| Visual input | (batch, T, 47) | (batch, T_v, D_v) | D_v depends on extractor (e.g., 512 for ResNet) |
| Acoustic input | (batch, T, 74) | (batch, T_a, D_a) | D_a depends on extractor (e.g., 40 for MFCC) |
| UFEN output | (batch, T, d_m) + scalar | (batch, T, d_m) + (batch, 7) | pred_head now 7-dim |
| MTFN Y_m | (batch,) scalar | (batch, 7) logits | pred_fusion output 7-dim |
| MTFN Y_m' | (batch,) scalar | (batch, 7) logits | pred_recon output 7-dim |
| Labels | (batch,) float [-3,3] | (batch,) long [0-6] | Integer class indices |

### 3.3 Model Constructor Changes

```python
# model.py — UFEN.__init__
# OLD:
self.pred_head = nn.Linear(d_m, 1)
# NEW:
self.pred_head = nn.Linear(d_m, num_classes)  # num_classes = 7

# model.py — UFEN.forward
# OLD:
pred = self.pred_head(pooled).squeeze(-1)     # (batch,)
# NEW:
pred = self.pred_head(pooled)                 # (batch, num_classes)

# model.py — MTFN.__init__
# OLD:
self.pred_fusion = nn.Linear(6 * d_m, 1)
self.pred_recon  = nn.Linear(d_m, 1)
# NEW:
self.pred_fusion = nn.Linear(6 * d_m, num_classes)
self.pred_recon  = nn.Linear(d_m, num_classes)

# model.py — MTFN.forward
# OLD:
y_m = self.pred_fusion(...).squeeze(-1)       # (batch,)
y_m_prime = self.pred_recon(...).squeeze(-1)  # (batch,)
# NEW:
y_m = self.pred_fusion(...)                   # (batch, num_classes)
y_m_prime = self.pred_recon(...)              # (batch, num_classes)
```

### 3.4 Input Dimension Configuration

```python
# Phase 1 config (CMU-MOSI):
visual_size   = 47    # FACET features
acoustic_size = 74    # COVAREP features

# Phase 2 config (MELD) — depends on feature extractor choice:
visual_size   = ???   # e.g., 512 (ResNet), 35 (OpenFace AUs), 709 (DenseNet)
acoustic_size = ???   # e.g., 40 (MFCC), 768 (wav2vec2), 6373 (openSMILE)
num_classes   = 7     # NEW parameter
```

---

## 4. Loss Function Changes

### 4.1 Replace MSE with Cross-Entropy

```python
# Phase 1 (Regression):
mse_loss = nn.MSELoss()
loss = (mse_loss(y_t, labels) + mse_loss(y_v, labels) +
        mse_loss(y_a, labels) + mse_loss(y_m, labels) +
        mse_loss(y_m_prime, labels))

# Phase 2 (Classification):
ce_loss = nn.CrossEntropyLoss(weight=class_weights)  # handle imbalance
loss = (ce_loss(y_t, labels) + ce_loss(y_v, labels) +
        ce_loss(y_a, labels) + ce_loss(y_m, labels) +
        ce_loss(y_m_prime, labels))
```

- `y_t, y_v, y_a, y_m, y_m_prime`: each `(batch, 7)` — raw logits (no softmax needed; `CrossEntropyLoss` applies it internally)
- `labels`: `(batch,)` — `torch.long`, values in `{0, 1, 2, 3, 4, 5, 6}`

### 4.2 Class Imbalance Handling

MELD is heavily imbalanced. Options (try in order):

1. **Weighted Cross-Entropy** (baseline): Compute inverse-frequency weights
   ```python
   # Example weights (inverse frequency, normalized)
   class_counts = [1109, 271, 268, 1743, 4710, 683, 1205]  # Ang, Dis, Fear, Joy, Neu, Sad, Sur
   weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
   weights = weights / weights.sum() * len(weights)  # normalize to mean=1
   ce_loss = nn.CrossEntropyLoss(weight=weights.to(device))
   ```

2. **Focal Loss** (if weighted CE isn't enough): Down-weights easy examples
   ```python
   # gamma=2.0 is standard; reduces loss for well-classified examples
   ```

3. **Oversampling** minority classes in the DataLoader via `WeightedRandomSampler`

---

## 5. Evaluation Metrics Changes

### Phase 1 Metrics (Remove)
- ~~MAE~~, ~~Corr~~, ~~Acc-2 (nn/np)~~, ~~Acc-7 (rounded regression)~~, ~~F1 (binary)~~

### Phase 2 Metrics (Add)
```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def compute_metrics_classification(preds: np.ndarray, labels: np.ndarray):
    """
    preds:  (N,) predicted class indices (argmax of logits)
    labels: (N,) ground truth class indices
    """
    acc = accuracy_score(labels, preds) * 100
    f1_weighted = f1_score(labels, preds, average='weighted') * 100
    f1_macro = f1_score(labels, preds, average='macro') * 100  # treats all classes equally
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds,
                target_names=['Anger','Disgust','Fear','Joy','Neutral','Sadness','Surprise'])
    return {
        'Accuracy': acc,
        'F1_weighted': f1_weighted,
        'F1_macro': f1_macro,
        'confusion_matrix': cm,
        'report': report,
    }
```

### Model Selection Criterion
- Phase 1: Best dev MAE (lower is better)
- **Phase 2: Best dev weighted F1 (higher is better)**

---

## 6. Training Pipeline Changes

### 6.1 Data Pipeline — New MELD Dataset Class

Create a new class `MELD` in `create_dataset.py` (or a new file `create_meld_dataset.py`):

```python
class MELD:
    """
    MELD dataset loader.
    
    Expected directory structure:
    data/MELD/
        train.pkl  (or train_sent_emo.csv + feature files)
        dev.pkl
        test.pkl
    
    Each sample: ((text_tokens, visual_features, acoustic_features, words), label, utterance_id)
    label: integer in {0..6} mapping to emotion classes
    """
```

### 6.2 Label Mapping
```python
EMOTION_MAP = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'neutral': 4,
    'sadness': 5,
    'surprise': 6,
}
```

### 6.3 Collate Function Updates

The collate function in `data_loader.py` needs updates:
- Labels become `torch.LongTensor` instead of `torch.FloatTensor`
- Visual/acoustic dimensions change based on new feature extractors
- BERT tokenization stays the same (tokenize utterance text)

### 6.4 Config Changes for Phase 2

```python
config = SimpleNamespace(
    # --- data ---
    data_dir      = 'data/MELD',
    dataset_dir   = 'data/MELD',
    batch_size    = 32,

    # --- model dimensions ---
    d_m          = 128,        # keep from Phase 1 best (P3-2)
    conv_dim     = 64,         # keep
    n_layers     = 2,          # keep
    kernel_sizes = [1, 5],     # keep from Phase 1 best
    d_ff         = 128,        # keep from Phase 1 best
    num_classes  = 7,          # NEW

    # --- attention ---
    self_att_heads  = 1,       # keep
    cross_att_heads = 4,       # keep
    att_dropout     = 0.2,     # keep
    dropout         = 0.1,     # may increase for larger dataset

    # --- training ---
    lr               = 5e-3,   # may need tuning — MELD is ~8x larger than MOSI
    lr_bert          = 2e-5,   # keep from Phase 1 best
    epochs           = 30,     # MELD is larger, may converge faster
    early_stop       = 10,
    grad_clip        = 1.0,
    use_lr_scheduler = True,
    use_bert_warmup  = True,
    bert_warmup_epochs = 3,    # fewer epochs needed (more data per epoch)

    # --- new for Phase 2 ---
    use_class_weights = True,  # weighted cross-entropy
    
    # set by DataLoader:
    visual_size   = None,
    acoustic_size = None,
)
```

---

## 7. Hyperparameters: What to Keep vs. Re-Tune

### Keep from Phase 1 (architecture validated)
- `d_m = 128`, `conv_dim = 64`, `kernel_sizes = [1, 5]`, `d_ff = 128`
- `self_att_heads = 1`, `cross_att_heads = 4`
- `use_bert_warmup = True`, `use_lr_scheduler = True`

### Potentially Re-Tune for MELD
| Parameter | Phase 1 | Consider | Why |
|---|---|---|---|
| `lr` | 5e-3 | 1e-3 to 5e-3 | Larger dataset may benefit from slightly lower lr |
| `lr_bert` | 2e-5 | 2e-5 to 5e-5 | More data = safer to fine-tune BERT more aggressively |
| `batch_size` | 32 | 32–64 | MELD is ~8x larger; larger batch may help |
| `dropout` | 0.1 | 0.1–0.3 | 7-class task is harder; may need more regularization |
| `epochs` | 50 | 20–30 | More data per epoch means fewer epochs needed |
| `early_stop` | 15 | 8–10 | Faster convergence expected |
| `bert_warmup_epochs` | 5 | 2–3 | More data warms up faster |

---

## 8. Ablation Studies (Proposal Section: Mar 26 – Apr 1)

As specified in the proposal, we should run ablation experiments:

### 8.1 Modality Ablation
| Experiment | Modalities Used | Purpose |
|---|---|---|
| Full model | Text + Audio + Visual | Baseline |
| Text-only | Text only | How much do A/V help? |
| Text + Audio | Text + Audio | Visual contribution |
| Text + Visual | Text + Visual | Audio contribution |
| Audio + Visual | Audio + Visual | Text contribution |

**Implementation:** Add a `modality_mask` config option. When a modality is disabled, replace its UFEN output with zeros and skip its loss term.

### 8.2 Architecture Ablation
| Experiment | Change | Purpose |
|---|---|---|
| No MTFN | Concatenate UFEN outputs → Linear(3*d_m, 7) | Is cross-modal attention necessary? |
| No Encoder-Decoder | Use only Y_m (skip enc-dec) | Is reconstruction useful? |
| No multi-task loss | Only train on Y_m' | Is multi-task learning helpful? |

---

## 9. Qualitative Analysis (Proposal Section 5)

For the final report:
1. **Attention weight visualization:** Extract cross-modal attention weights from MTFN on test samples where text is sarcastic/ambiguous
2. **Confusion matrix analysis:** Which emotions are most confused? (e.g., Anger↔Disgust)
3. **Per-emotion F1 breakdown:** Which emotions benefit most from multimodal features?
4. **Error case studies:** Select specific utterances and analyze why the model fails

---

## 10. Implementation Order (Prioritized)

### Step 1: MELD Data Pipeline (Critical Path)
1. Download MELD dataset (CSV files + video/audio files)
2. Extract audio features from video clips (librosa MFCC or openSMILE)
3. Extract visual features from video frames (OpenFace or ResNet)
4. Create `MELD` class in `create_dataset.py` matching the existing data format
5. Update `data_loader.py` to handle MELD (integer labels, new feature dims)
6. Verify data loading works: print shapes, label distribution, sample batch

### Step 2: Model Architecture Updates
1. Add `num_classes` parameter to `UFEN`, `MTFN`, and `MultiTaskModel`
2. Change all 5 prediction heads from `Linear(*, 1)` to `Linear(*, num_classes)`
3. Remove `.squeeze(-1)` calls on prediction outputs
4. Verify forward pass works with dummy MELD-shaped input

### Step 3: Training Pipeline Updates
1. Replace `MSELoss` with `CrossEntropyLoss` (with class weights)
2. Update `compute_metrics` → `compute_metrics_classification`
3. Update `evaluate()` to use argmax on logits
4. Change model selection from best MAE to best weighted F1
5. Update print statements for new metrics

### Step 4: Initial Training Run
1. Train on MELD with Phase 1 hyperparameters as starting point
2. Evaluate baseline performance
3. Compare with published MELD baselines (DialogueRNN, MMGCN, etc.)

### Step 5: Hyperparameter Tuning
1. Tune learning rates (lr, lr_bert)
2. Tune regularization (dropout, class weights)
3. Try focal loss if class imbalance is problematic

### Step 6: Ablation Studies
1. Run modality ablation experiments (5 configs)
2. Run architecture ablation experiments (3 configs)
3. Record all results in `logs/experiments_meld.md`

### Step 7: Analysis & Report
1. Generate confusion matrices and per-class F1 scores
2. Visualize attention weights on interesting test cases
3. Write qualitative analysis section
4. Prepare final report and presentation

---

## 11. Expected Challenges

1. **Feature extraction for MELD** — MELD doesn't ship word-aligned features like MOSI. This is the biggest engineering effort. Look for pre-extracted feature repos first.
2. **Class imbalance** — Neutral dominates (~47%); Fear/Disgust are rare (~2.7% each). Weighted CE or focal loss is essential.
3. **Dialogue context** — MELD is dialogue-based; utterances have conversational context. Our current model treats each utterance independently. This is a known limitation (could mention as future work).
4. **Different feature dimensions** — Visual/acoustic feature dimensions will differ from MOSI. The model handles this (UFEN's input_dim is configurable), but we need to verify tensor shapes throughout.
5. **Larger dataset, longer training** — MELD has ~10K training samples vs. MOSI's 1.3K. Training will take longer but should converge more reliably.

---

## 12. MELD Baseline Comparisons (for Final Report)

| Model | Accuracy | Weighted F1 | Source |
|---|---|---|---|
| bc-LSTM | 56.7 | 56.4 | Poria et al. 2017 |
| DialogueRNN | 62.75 | 59.54 | Majumder et al. 2019 |
| MMGCN | 58.65 | 58.18 | Hu et al. 2021 |
| UniMSE | 65.09 | 65.51 | Hu et al. 2022 |
| **Ours (target)** | **~60–65** | **~58–63** | — |

> Note: Some baselines use dialogue context (previous utterances), which our model does not. Our results without dialogue context are expected to be competitive but may not match context-aware models.

---

## 13. File Changes Summary

| File | Change Type | Description |
|---|---|---|
| `model.py` | **Modify** | Add `num_classes` param; change all pred heads to output 7-dim |
| `train.py` | **Modify** | Replace MSE→CE loss; new metrics; new config for MELD |
| `data_loader.py` | **Modify** | Support MELD data format; integer labels |
| `create_dataset.py` | **Add class** | New `MELD` class for dataset loading |
| `create_meld_features.py` | **New file** | Feature extraction script for MELD audio/visual |
| `logs/experiments_meld.md` | **New file** | MELD experiment tracking |

---

## Quick Reference: The 3 Core Changes

If you want the absolute minimum to get Phase 2 running:

1. **Heads:** `Linear(*, 1)` → `Linear(*, 7)` in UFEN + MTFN (5 heads total)
2. **Loss:** `MSELoss` → `CrossEntropyLoss(weight=class_weights)`  
3. **Data:** New MELD dataset class with integer labels and appropriate feature extraction
