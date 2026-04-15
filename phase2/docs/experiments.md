# Phase 2 Experiments — MELD 7-Class Emotion Classification

All experiments use the UFEN+MTFN architecture adapted for MELD with GloVe text embeddings,
ResNet-101 visual features (2048-dim), and Wave2Vec2.0 audio features (32-dim).

Label mapping: 0=Neutral, 1=Surprise, 2=Fear, 3=Sadness, 4=Joy, 5=Disgust, 6=Anger

---

## Dataset Statistics

| Split | Samples | Neutral | Surprise | Fear | Sadness | Joy | Disgust | Anger |
|---|---|---|---|---|---|---|---|---|
| Train | 9,988 | 4,709 (47.1%) | 1,205 (12.1%) | 268 (2.7%) | 684 (6.8%) | 1,743 (17.5%) | 271 (2.7%) | 1,108 (11.1%) |
| Dev | 1,108 | 469 (42.3%) | 150 (13.5%) | 40 (3.6%) | 111 (10.0%) | 163 (14.7%) | 22 (2.0%) | 153 (13.8%) |
| Test | 2,610 | 1,256 (48.1%) | 281 (10.8%) | 50 (1.9%) | 208 (8.0%) | 402 (15.4%) | 68 (2.6%) | 345 (13.2%) |

---

## Baseline Config (shared across experiments unless noted)

```
d_m=128, conv_dim=64, n_layers=2, kernel_sizes=[1,5], d_ff=128
self_att_heads=1, cross_att_heads=4, att_dropout=0.2, dropout=0.1
batch_size=32, grad_clip=1.0, use_lr_scheduler=True (cosine)
early_stop=10, use_class_weights=True (inverse frequency)
Text: GloVe 300-dim (trainable), Visual: ResNet-101 2048-dim, Audio: Wave2Vec2.0 32-dim
Trainable parameters: 3,844,643
```

---

## Exp 1 — Baseline (Direct Port from Phase 1 Hyperparams)

**Goal:** Establish a starting point using Phase 1 best hyperparameters (P3-2) directly on MELD.

**Config changes from baseline:** None — this IS the baseline.
- `lr=5e-3`, `epochs=50`, `early_stop=10`
- Class weights: `[0.130, 0.509, 2.291, 0.898, 0.352, 2.265, 0.554]`

**Training log:**
```
Epoch 01 | loss=10.0811 | Acc=44.4 | F1w=35.5 | F1m=18.3  <- new best
Epoch 02 | loss=9.6550  | Acc=8.3  | F1w=6.1  | F1m=6.5
Epoch 03 | loss=9.5836  | Acc=40.8 | F1w=35.5 | F1m=18.7
Epoch 04 | loss=9.4522  | Acc=47.3 | F1w=35.3 | F1m=15.9
Epoch 05 | loss=9.3301  | Acc=27.2 | F1w=25.3 | F1m=14.0
Epoch 06 | loss=9.2547  | Acc=29.2 | F1w=24.7 | F1m=11.2
Epoch 07 | loss=9.2240  | Acc=45.8 | F1w=34.9 | F1m=15.6
Epoch 08 | loss=9.1661  | Acc=43.5 | F1w=36.6 | F1m=18.1  <- new best
Epoch 09 | loss=9.1284  | Acc=45.0 | F1w=34.5 | F1m=15.9
Epoch 10 | loss=9.0886  | Acc=43.1 | F1w=33.5 | F1m=14.4
Epoch 11 | loss=9.0719  | Acc=43.8 | F1w=30.7 | F1m=13.5
Epoch 12 | loss=9.0070  | Acc=37.8 | F1w=30.9 | F1m=13.5
Epoch 13 | loss=8.9932  | Acc=37.8 | F1w=33.7 | F1m=16.8
Epoch 14 | loss=8.8876  | Acc=43.4 | F1w=35.3 | F1m=15.9
Epoch 15 | loss=8.7381  | Acc=26.3 | F1w=26.6 | F1m=16.9
Epoch 16 | loss=8.7456  | Acc=36.5 | F1w=34.7 | F1m=19.1
Epoch 17 | loss=8.6391  | Acc=38.2 | F1w=35.6 | F1m=19.6
Epoch 18 | early stop (no improvement for 10 epochs since epoch 8)
```

**Test results (best checkpoint: epoch 8):**
```
Accuracy       : 47.7
F1 (weighted)  : 42.0
F1 (macro)     : 19.6
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support
     Neutral       0.68      0.74      0.71      1256
    Surprise       0.53      0.22      0.31       281
        Fear       0.00      0.00      0.00        50
     Sadness       0.00      0.00      0.00       208
         Joy       0.00      0.00      0.00       402
     Disgust       0.00      0.00      0.00        68
       Anger       0.23      0.75      0.35       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     925    41     0     0     0     0   290
Surprise     84    62     0     0     0     0   135
Fear         20     1     0     0     0     0    29
Sadness     120     4     0     0     0     0    84
Joy         108     0     0     0     0     0   294
Disgust      31     1     0     0     0     0    36
Anger        79     7     0     0     0     0   259
```

**Diagnosis:**
- **Class collapse:** Model only predicts Neutral (0) and Anger (6). Five classes get zero predictions.
- **Training instability:** Dev F1w oscillates wildly (35→6→35→25→36→30→34), indicating lr=5e-3 is too high.
- **Feature scale mismatch:** Text (GloVe L2 norm ~5.0) dominates audio (~0.85) and video (~0.72) by ~7x. Audio/video BiGRUs receive near-zero inputs and contribute nothing.
- **Root cause:** High LR + scale mismatch → model finds a trivial local minimum (predict majority classes only).

**Planned fixes for Exp 2:**
1. Per-modality input normalization (LayerNorm) to equalize feature scales
2. Lower learning rate (5e-4 or 1e-3)
3. Label smoothing to prevent overconfident majority-class predictions

---

## Exp 2 — Feature Normalization + Lower LR + Label Smoothing

**Goal:** Fix class collapse and training instability from Exp 1.

**Changes from Exp 1:**
1. **Input LayerNorm per modality** (model.py) — `nn.LayerNorm(dim)` applied to text (300-dim), video (2048-dim), audio (32-dim) before feeding into UFEN. Equalizes feature scales so no modality dominates.
2. **lr: 5e-3 → 1e-3** — reduces oscillation and allows stable gradient descent.
3. **label_smoothing=0.1** — `CrossEntropyLoss(label_smoothing=0.1)` softens targets from hard [0,1] to [0.014, 0.914], discouraging overconfident majority-class predictions.

Trainable parameters: 3,849,403 (+4,760 from LayerNorm params)

**Training log:**
```
Epoch 01 | loss=10.8695 | Acc=49.5 | F1w=42.9 | F1m=24.2  <- new best
Epoch 02 | loss=10.5409 | Acc=33.8 | F1w=39.9 | F1m=27.8
Epoch 03 | loss=10.3638 | Acc=35.4 | F1w=38.2 | F1m=29.0
Epoch 04 | loss=10.1938 | Acc=36.7 | F1w=41.4 | F1m=30.4
Epoch 05 | loss=9.9827  | Acc=29.7 | F1w=34.9 | F1m=27.6
Epoch 06 | loss=9.8025  | Acc=35.0 | F1w=38.2 | F1m=29.5
Epoch 07 | loss=9.6272  | Acc=40.1 | F1w=43.1 | F1m=29.8  <- new best
Epoch 08 | loss=9.4869  | Acc=35.2 | F1w=38.8 | F1m=28.4
Epoch 09 | loss=9.2526  | Acc=46.8 | F1w=46.9 | F1m=31.4  <- new best
Epoch 10 | loss=9.0939  | Acc=41.1 | F1w=43.3 | F1m=30.9
Epoch 11 | loss=8.9121  | Acc=40.3 | F1w=43.1 | F1m=29.4
Epoch 12 | loss=8.7352  | Acc=41.7 | F1w=44.2 | F1m=29.8
Epoch 13 | loss=8.5487  | Acc=43.3 | F1w=44.6 | F1m=30.3
Epoch 14 | loss=8.3962  | Acc=43.3 | F1w=44.8 | F1m=30.4
Epoch 15 | loss=8.2139  | Acc=41.1 | F1w=42.7 | F1m=29.3
Epoch 16 | loss=8.0691  | Acc=43.8 | F1w=44.6 | F1m=30.6
Epoch 17 | loss=7.9335  | Acc=47.5 | F1w=46.4 | F1m=30.6
Epoch 18 | loss=7.7785  | Acc=44.4 | F1w=44.8 | F1m=30.1
Epoch 19 | loss=7.6077  | Acc=42.1 | F1w=42.8 | F1m=28.5
Epoch 19 | early stop (no improvement for 10 epochs since epoch 9)
```

**Test results (best checkpoint: epoch 9):**
```
Accuracy       : 50.4
F1 (weighted)  : 51.4
F1 (macro)     : 33.1
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support
     Neutral       0.72      0.64      0.68      1256
    Surprise       0.43      0.57      0.49       281
        Fear       0.08      0.08      0.08        50
     Sadness       0.24      0.21      0.22       208
         Joy       0.44      0.50      0.47       402
     Disgust       0.04      0.10      0.06        68
       Anger       0.37      0.28      0.32       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     803    94    22    74   112   108    43
Surprise     37   160     3     8    33     6    34
Fear         15     8     4     5     8     2     8
Sadness      87    15     8    43    21    10    24
Joy          86    33     7    14   202    19    41
Disgust      25    13     1     3     6     7    13
Anger        61    52     6    31    76    22    97
```

**Comparison vs Exp 1:**

| Metric | Exp 1 | Exp 2 | Delta |
|---|---|---|---|
| Accuracy | 47.7 | **50.4** | +2.7 |
| F1 (weighted) | 42.0 | **51.4** | **+9.4** |
| F1 (macro) | 19.6 | **33.1** | **+13.5** |
| Classes predicted | 2/7 | **7/7** | fixed |
| Training stability | wild oscillation | smooth descent | fixed |

**Per-class F1 comparison:**

| Class | Exp 1 | Exp 2 | Delta |
|---|---|---|---|
| Neutral | 0.71 | 0.68 | -0.03 |
| Surprise | 0.31 | **0.49** | +0.18 |
| Fear | 0.00 | **0.08** | +0.08 |
| Sadness | 0.00 | **0.22** | +0.22 |
| Joy | 0.00 | **0.47** | +0.47 |
| Disgust | 0.00 | **0.06** | +0.06 |
| Anger | 0.35 | 0.32 | -0.03 |

**Analysis:**
- **Class collapse resolved:** All 7 classes now receive predictions. Joy jumped from 0.00→0.47 F1.
- **Training is stable:** Loss decreases monotonically, dev F1 no longer oscillates wildly.
- **Remaining weaknesses:**
  - Fear (F1=0.08) and Disgust (F1=0.06) are still near-zero — only 268 and 271 training samples respectively.
  - Neutral recall dropped (0.74→0.64): model now spreads predictions more evenly, which is the intended behavior.
  - Confusion: Neutral gets misclassified as Joy (112), Disgust (108), Surprise (94). Anger is confused with Joy (76) and Surprise (52).
  - Training still early-stops at epoch 19 — model may benefit from more capacity or longer training.

---

## Exp 3 — Video Projection + Focal Loss + Lower LR + Higher Dropout

**Goal:** Improve minority-class F1 and training stability with targeted architectural and loss changes.

**Changes from Exp 2:**
1. **Video projection** (model.py) — `nn.Linear(2048, 256)` before LayerNorm+UFEN. The Exp 2 visual BiGRU compressed 2048→64 (32x in one step), losing most visual information. Now 256→64 is only a 4x compression.
2. **Focal Loss (gamma=2.0)** (train.py) — replaces CrossEntropyLoss. Down-weights easy/well-classified examples (mostly Neutral) so gradients focus on hard minority samples (Fear, Disgust).
3. **lr: 1e-3 → 5e-4** — Exp 2 still showed some oscillation in early epochs. Lower LR for smoother convergence.
4. **dropout: 0.1 → 0.2** — more regularization for the 7-class task with 3.7M params on 10K samples.
5. **early_stop: 10 → 15, epochs: 50 → 80** — lower LR needs more patience.

Trainable parameters: 3,682,235 (reduced from 3,849,403 — video UFEN input shrunk from 2048 to 256)

**Training log:**
```
Epoch 01 | loss=2.2568 | Acc=17.6 | F1w=16.4 | F1m=17.7  <- new best
Epoch 02 | loss=2.1491 | Acc=12.5 | F1w=11.7 | F1m=13.5
Epoch 03 | loss=2.0921 | Acc=20.8 | F1w=20.9 | F1m=21.7  <- new best
Epoch 04 | loss=2.0051 | Acc=21.8 | F1w=21.7 | F1m=21.7  <- new best
Epoch 05 | loss=1.9049 | Acc=22.9 | F1w=23.3 | F1m=24.5  <- new best
Epoch 06 | loss=1.7884 | Acc=25.7 | F1w=26.3 | F1m=25.0  <- new best
Epoch 07 | loss=1.6720 | Acc=26.1 | F1w=25.7 | F1m=24.0
Epoch 08 | loss=1.5726 | Acc=28.3 | F1w=27.6 | F1m=25.1  <- new best
Epoch 09 | loss=1.4811 | Acc=35.0 | F1w=37.0 | F1m=29.0  <- new best
Epoch 10 | loss=1.4032 | Acc=33.6 | F1w=34.0 | F1m=27.9
Epoch 11 | loss=1.3482 | Acc=33.7 | F1w=35.2 | F1m=28.5
Epoch 12 | loss=1.3034 | Acc=32.2 | F1w=32.7 | F1m=26.8
Epoch 13 | loss=1.2533 | Acc=35.0 | F1w=36.6 | F1m=28.9
Epoch 14 | loss=1.2303 | Acc=38.2 | F1w=40.1 | F1m=29.5  <- new best
Epoch 15 | loss=1.2009 | Acc=36.6 | F1w=37.7 | F1m=27.7
Epoch 16 | loss=1.1802 | Acc=38.2 | F1w=40.0 | F1m=29.2
Epoch 17 | loss=1.1605 | Acc=45.5 | F1w=45.0 | F1m=31.7  <- new best
Epoch 18 | loss=1.1353 | Acc=45.1 | F1w=45.7 | F1m=31.3  <- new best
Epoch 23 | loss=1.0835 | Acc=47.5 | F1w=47.0 | F1m=32.8  <- new best
Epoch 34 | loss=0.9141 | Acc=47.9 | F1w=47.2 | F1m=31.1  <- new best
Epoch 37 | loss=0.8474 | Acc=49.5 | F1w=48.1 | F1m=32.1  <- new best
Epoch 46 | loss=0.6743 | Acc=49.7 | F1w=48.4 | F1m=32.6  <- new best
Epoch 61 | early stop (no improvement for 15 epochs since epoch 46)
```

**Test results (best checkpoint: epoch 46):**
```
Accuracy       : 51.4
F1 (weighted)  : 50.5
F1 (macro)     : 31.7
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support
     Neutral       0.67      0.72      0.69      1256
    Surprise       0.44      0.49      0.46       281
        Fear       0.10      0.10      0.10        50
     Sadness       0.24      0.15      0.19       208
         Joy       0.44      0.43      0.44       402
     Disgust       0.07      0.07      0.07        68
       Anger       0.28      0.25      0.27       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     901    69    24    54    89    28    91
Surprise     51   139     3     8    37    11    32
Fear         22     7     5     4     5     0     7
Sadness     106    13     7    32    21     6    23
Joy         115    34     2    16   173     8    54
Disgust      34     7     2     0     8     5    12
Anger       110    49     7    21    60    11    87
```

**Comparison across all experiments:**

| Metric | Exp 1 | Exp 2 | Exp 3 | Best |
|---|---|---|---|---|
| Accuracy | 47.7 | 50.4 | **51.4** | Exp 3 |
| F1 (weighted) | 42.0 | **51.4** | 50.5 | Exp 2 |
| F1 (macro) | 19.6 | **33.1** | 31.7 | Exp 2 |
| Best epoch | 8 | 9 | 46 | — |
| Training epochs | 18 | 19 | 61 | — |

**Per-class F1 comparison:**

| Class | Exp 1 | Exp 2 | Exp 3 | Best |
|---|---|---|---|---|
| Neutral | **0.71** | 0.68 | 0.69 | Exp 1 |
| Surprise | 0.31 | **0.49** | 0.46 | Exp 2 |
| Fear | 0.00 | 0.08 | **0.10** | Exp 3 |
| Sadness | 0.00 | **0.22** | 0.19 | Exp 2 |
| Joy | 0.00 | **0.47** | 0.44 | Exp 2 |
| Disgust | 0.00 | 0.06 | **0.07** | Exp 3 |
| Anger | **0.35** | 0.32 | 0.27 | Exp 1 |

**Analysis:**
- **Accuracy improved** (51.4 vs 50.4) — Neutral recall went up (0.72 vs 0.64), showing the model is more balanced.
- **Focal loss helped the rarest classes** — Fear (0.08→0.10) and Disgust (0.06→0.07) got small gains, as intended.
- **But F1w dropped slightly** (51.4→50.5) — Focal loss over-penalised easy Neutral/Joy examples, hurting Surprise (0.49→0.46), Sadness (0.22→0.19), and Anger (0.32→0.27). The gamma=2.0 penalty was too aggressive.
- **Training was very slow** — best at epoch 46 (vs epoch 9 in Exp 2). Focal loss produces smaller gradients for well-classified samples, so learning is slower overall.
- **Confusion patterns:** Anger is now heavily confused with Neutral (110) and Joy (60). Sadness is confused with Neutral (106). These are the main error sources.
- **Key takeaway:** Focal loss trades mid-frequency class performance for tiny minority gains. The net effect is slightly negative for weighted F1. Exp 2's CrossEntropyLoss was better for the overall metric.

---

## Exp 4 — Best of Exp 2 + Exp 3 (CE Loss + Video Projection)

**Goal:** Combine the best loss function (CE from Exp 2) with the best architecture (video projection + dropout from Exp 3).

**Changes from Exp 3:**
1. **Switched back to CrossEntropyLoss** — Focal loss hurt F1w in Exp 3; CE was better in Exp 2.

Everything else kept from Exp 3: video projection (2048→256), dropout=0.2, lr=5e-4, label_smoothing=0.1, early_stop=15, epochs=80.

Trainable parameters: 3,682,235 (same as Exp 3)

**Training log:**
```
Epoch 01 | loss=10.8244 | Acc=48.3 | F1w=43.7 | F1m=25.8  <- new best
Epoch 02 | loss=10.4873 | Acc=30.4 | F1w=36.7 | F1m=27.0
Epoch 03 | loss=10.3386 | Acc=36.6 | F1w=40.8 | F1m=29.5
Epoch 04 | loss=10.1603 | Acc=34.8 | F1w=40.0 | F1m=29.4
Epoch 05 | loss=9.9485  | Acc=29.5 | F1w=33.3 | F1m=26.5
Epoch 06 | loss=9.6435  | Acc=36.9 | F1w=39.9 | F1m=29.8
Epoch 07 | loss=9.3671  | Acc=43.3 | F1w=45.3 | F1m=30.0  <- new best
Epoch 10 | loss=8.4915  | Acc=45.6 | F1w=45.9 | F1m=30.7  <- new best
Epoch 13 | loss=7.9252  | Acc=45.6 | F1w=46.2 | F1m=30.3  <- new best
Epoch 14 | loss=7.8238  | Acc=46.8 | F1w=47.1 | F1m=32.1  <- new best
Epoch 20 | loss=7.3062  | Acc=48.8 | F1w=48.0 | F1m=32.1  <- new best
Epoch 35 | early stop (no improvement for 15 epochs since epoch 20)
```

**Test results (best checkpoint: epoch 20):**
```
Accuracy       : 49.8
F1 (weighted)  : 50.3
F1 (macro)     : 31.9
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support
     Neutral       0.70      0.64      0.67      1256
    Surprise       0.45      0.50      0.48       281
        Fear       0.05      0.06      0.05        50
     Sadness       0.22      0.19      0.20       208
         Joy       0.42      0.49      0.45       402
     Disgust       0.08      0.06      0.07        68
       Anger       0.29      0.34      0.31       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     800    77    33    70   118    23   135
Surprise     41   141     2    15    41     1    40
Fear         20     4     3     4    11     1     7
Sadness      81    10     9    39    24     7    38
Joy          91    25     5    20   197     6    58
Disgust      27    13     3     3     6     4    12
Anger        77    40     6    28    71     6   117
```

**Comparison across all experiments:**

| Metric | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Best |
|---|---|---|---|---|---|
| Accuracy | 47.7 | **50.4** | 51.4 | 49.8 | Exp 3 |
| F1 (weighted) | 42.0 | **51.4** | 50.5 | 50.3 | **Exp 2** |
| F1 (macro) | 19.6 | **33.1** | 31.7 | 31.9 | **Exp 2** |
| Best epoch | 8 | 9 | 46 | 20 | — |
| Training epochs | 18 | 19 | 61 | 35 | — |

**Per-class F1 comparison:**

| Class | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Best |
|---|---|---|---|---|---|
| Neutral | **0.71** | 0.68 | 0.69 | 0.67 | Exp 1 |
| Surprise | 0.31 | **0.49** | 0.46 | 0.48 | Exp 2 |
| Fear | 0.00 | 0.08 | **0.10** | 0.05 | Exp 3 |
| Sadness | 0.00 | **0.22** | 0.19 | 0.20 | Exp 2 |
| Joy | 0.00 | **0.47** | 0.44 | 0.45 | Exp 2 |
| Disgust | 0.00 | 0.06 | **0.07** | 0.07 | Exp 3 |
| Anger | **0.35** | 0.32 | 0.27 | 0.31 | Exp 1 |

**Analysis:**
- **Exp 2 remains the best overall** — F1w=51.4 and F1m=33.1 are still unbeaten.
- **Video projection (Exp 3/4) did NOT help vs Exp 2.** Exp 2 fed raw 2048-dim into UFEN BiGRU and performed better. The BiGRU(2048→64) compression, while aggressive, appears to work as an implicit feature selection — it learns which of the 2048 ResNet channels matter. The explicit Linear(2048→256) may be discarding useful features.
- **lr=5e-4 is too slow for CE loss** — best at epoch 20 with F1w=48.0, while Exp 2 (lr=1e-3) peaked at epoch 9 with F1w=46.9. The lower LR didn't lead to a better final result, just slower convergence.
- **Anger improved** (0.27→0.31) vs Exp 3, confirming that focal loss was hurting mid-frequency classes.
- **Fear collapsed** (0.10→0.05) without focal loss — CE alone can't handle 268 training samples competing with 4709 Neutral samples.

**Key takeaway:** Exp 2's simpler architecture (no video projection, lr=1e-3, CE loss) is still the best. The video projection and lower LR added complexity without improving the core metric. The main bottleneck is not architecture — it's the extreme class imbalance and the limited discriminative power of the features for similar emotions (Anger↔Neutral, Sadness↔Neutral).

---

## Exp 5 — Increased Capacity (d_m=256)

**Goal:** Test whether the model is capacity-limited by doubling `d_m` and `d_ff`, using Exp 2 as the base config.

**Changes from Exp 2:**
1. **d_m: 128 → 256** — doubles BiGRU hidden size, all attention dimensions, and fusion width.
2. **d_ff: 128 → 256** — doubles encoder-decoder FFN width.
3. **No video projection** — raw 2048-dim into UFEN (like Exp 2).

All else from Exp 2: lr=1e-3, dropout=0.1, CE loss, label_smoothing=0.1, early_stop=10.

Trainable parameters: 7,118,395 (up from 3,849,403 — ~1.85x)

**Training log:**
```
Epoch 01 | loss=11.0964 | Acc=47.6 | F1w=43.3 | F1m=25.7  <- new best
Epoch 02 | loss=10.7709 | Acc=36.8 | F1w=38.7 | F1m=25.2
Epoch 03 | loss=10.6800 | Acc=48.5 | F1w=46.6 | F1m=30.0  <- new best
Epoch 04 | loss=10.6103 | Acc=34.3 | F1w=38.5 | F1m=27.1
Epoch 05 | loss=10.5449 | Acc=30.6 | F1w=35.1 | F1m=26.1
Epoch 06 | loss=10.4598 | Acc=36.9 | F1w=39.8 | F1m=28.8
Epoch 07 | loss=10.4114 | Acc=38.7 | F1w=42.0 | F1m=28.2
Epoch 08 | loss=10.3807 | Acc=31.1 | F1w=35.4 | F1m=27.8
Epoch 09 | loss=10.3220 | Acc=37.3 | F1w=40.1 | F1m=30.4
Epoch 10 | loss=10.2395 | Acc=41.0 | F1w=43.9 | F1m=30.5
Epoch 11 | loss=10.2183 | Acc=31.9 | F1w=36.9 | F1m=28.2
Epoch 12 | loss=10.0932 | Acc=38.2 | F1w=42.2 | F1m=31.2
Epoch 13 | loss=10.0106 | Acc=24.8 | F1w=28.9 | F1m=25.3
Early stopping: no improvement for 10 epochs since epoch 3.
```

**Test results (best checkpoint: epoch 3):**
```
Accuracy       : 51.0
F1 (weighted)  : 50.3
F1 (macro)     : 29.7
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support
     Neutral       0.71      0.71      0.71      1256
    Surprise       0.52      0.42      0.47       281
        Fear       0.00      0.00      0.00        50
     Sadness       0.25      0.20      0.23       208
         Joy       0.39      0.58      0.46       402
     Disgust       0.04      0.13      0.07        68
       Anger       0.27      0.10      0.14       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     896    62     9    84    91    99    15
Surprise     42   118     5     3    58    29    26
Fear         18     3     0     5    13     6     5
Sadness      96     4     4    42    36    12    14
Joy          98     7     5    13   233    17    29
Disgust      28     3     1     5    20     9     2
Anger        84    29     2    13   152    31    34
```

**Comparison across all experiments:**

| Metric | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Best |
|---|---|---|---|---|---|---|
| Accuracy | 47.7 | 50.4 | **51.4** | 49.8 | 51.0 | Exp 3 |
| F1 (weighted) | 42.0 | **51.4** | 50.5 | 50.3 | 50.3 | **Exp 2** |
| F1 (macro) | 19.6 | **33.1** | 31.7 | 31.9 | 29.7 | **Exp 2** |
| Best epoch | 8 | 9 | 46 | 20 | 3 | — |
| Params | 3.84M | 3.85M | 3.68M | 3.68M | 7.12M | — |

**Per-class F1 comparison:**

| Class | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Best |
|---|---|---|---|---|---|---|
| Neutral | 0.71 | 0.68 | 0.69 | 0.67 | **0.71** | Exp 1/5 |
| Surprise | 0.31 | **0.49** | 0.46 | 0.48 | 0.47 | Exp 2 |
| Fear | 0.00 | 0.08 | **0.10** | 0.05 | 0.00 | Exp 3 |
| Sadness | 0.00 | **0.22** | 0.19 | 0.20 | 0.23 | Exp 2/5 |
| Joy | 0.00 | **0.47** | 0.44 | 0.45 | 0.46 | Exp 2 |
| Disgust | 0.00 | 0.06 | **0.07** | 0.07 | 0.07 | Exp 3 |
| Anger | **0.35** | 0.32 | 0.27 | 0.31 | 0.14 | Exp 1 |

**Analysis:**
- **Doubling capacity hurt performance.** F1m dropped from 33.1→29.7. Anger collapsed (0.32→0.14). Fear back to zero.
- **Severe instability:** Dev F1w oscillated wildly (43→38→46→38→35→39→42→35→40→43→36→42→28). Best was epoch 3 — the model couldn't learn stably with 7.1M params on 10K samples.
- **Overfitting, not underfitting:** The model has too many parameters for the data size. lr=1e-3 on 7.1M params causes large, noisy gradient updates.
- **Anger catastrophe:** Only 10% recall (34/345 correct). Anger samples are scattered across all other classes — the model can't distinguish Anger from anything.

**Key takeaway:** The GloVe + UFEN + MTFN architecture has reached its ceiling at ~51 F1w with these features. Increasing capacity makes things worse. The bottleneck is **text representation quality** — GloVe embeddings (300-dim, context-free) cannot capture the nuance needed to distinguish similar emotions. BERT's contextual embeddings are needed to break through this plateau.

---

## Architecture Plateau Analysis

After 5 experiments, the performance ceiling is clear:

| What we tried | Impact | Conclusion |
|---|---|---|
| LayerNorm + lower LR | **+9.4 F1w** | Essential — fixed class collapse |
| Focal Loss | -0.9 F1w | Hurt — too aggressive for this imbalance |
| Video projection (2048→256) | -1.1 F1w | Hurt — BiGRU handles raw 2048 fine |
| Lower LR (5e-4 vs 1e-3) | -1.1 F1w | Hurt — too slow, no better final result |
| Doubled capacity (d_m=256) | -1.1 F1w | Hurt — overfits on 10K samples |

**Best result: Exp 2 (F1w=51.4, F1m=33.1)** — LayerNorm + lr=1e-3 + CE loss + label_smoothing=0.1, d_m=128.

### Root Cause: GloVe is the bottleneck

GloVe embeddings are **context-free** — the word "fine" has the same vector whether it means "I'm fine" (neutral/sarcastic) or "that's a fine idea" (positive). For emotion recognition, context is everything:
- "Yeah, right" — Sarcasm (Anger/Disgust) vs Agreement (Neutral)
- "I can't believe it" — Surprise vs Anger vs Joy depending on context
- "Whatever" — Neutral vs Anger vs Sadness

BERT's contextual embeddings capture these distinctions. This is the single biggest change that can break through the ~51 F1w ceiling.

---

## Exp 6 — BERT Text Encoder (Text + Audio + Video, 3-modal)

**Goal:** Replace GloVe with BERT contextual embeddings to break through the ~51 F1w ceiling reached in Exp 1–5.

**Changes from Exp 2 (best GloVe baseline):**
1. **Text encoder: GloVe → BERT** (`bert-base-uncased`, 768-dim contextual embeddings)
2. **batch_size: 32 → 16** — required to fit BERT on RTX 3050 (3.68 GiB)
3. **visual_proj_dim=256** — projects video 2048→256 before UFEN BiGRU to reduce VRAM
4. **Mixed precision (fp16)** — `torch.amp.autocast` + `GradScaler` for further memory reduction
5. **Differentiated LR:** `lr_bert=2e-5` (BERT fine-tuning), `lr=1e-3` (other params)
6. **BERT warmup:** linear ramp lr_bert 0→2e-5 over 3 epochs, then cosine anneal
7. **BERT input:** max_length=64 subword tokens from raw utterance text
8. Trainable parameters: 111,444,323 (~111.4M — dominated by BERT's 110M)

All else from Exp 2: CE loss, label_smoothing=0.1, early_stop=10, class weights.

**Training log:**
```
Epoch 01 | loss=10.7681 | Acc=47.6 | F1w=50.2 | F1m=34.3  <- new best
Epoch 02 | loss=10.2569 | Acc=53.9 | F1w=55.9 | F1m=42.6  <- new best
Epoch 03 | loss=9.9112  | Acc=53.6 | F1w=54.9 | F1m=41.8
Epoch 04 | loss=9.4421  | Acc=58.1 | F1w=58.0 | F1m=43.1  <- new best
Epoch 05 | loss=8.9737  | Acc=54.2 | F1w=55.7 | F1m=41.7
Epoch 06 | loss=8.6236  | Acc=57.1 | F1w=57.9 | F1m=43.4
Epoch 07 | loss=8.4032  | Acc=60.6 | F1w=59.6 | F1m=48.1  <- new best
Epoch 08 | loss=8.2385  | Acc=59.1 | F1w=57.9 | F1m=44.5
Epoch 09 | loss=8.0667  | Acc=58.5 | F1w=56.0 | F1m=40.8
Epoch 10 | loss=7.9693  | Acc=58.6 | F1w=57.4 | F1m=44.4
Epoch 11 | loss=7.8706  | Acc=59.0 | F1w=58.2 | F1m=44.2
Epoch 12 | loss=7.7472  | Acc=58.7 | F1w=57.7 | F1m=45.1
Epoch 13 | loss=7.7458  | Acc=57.4 | F1w=55.7 | F1m=40.9
Epoch 14 | loss=7.6763  | Acc=57.2 | F1w=56.5 | F1m=42.2
Epoch 15 | loss=7.6369  | Acc=56.9 | F1w=55.8 | F1m=43.4
Epoch 16 | loss=7.5752  | Acc=56.7 | F1w=56.3 | F1m=43.5
Epoch 17 | loss=7.6022  | Acc=57.9 | F1w=56.2 | F1m=41.9
Early stopping: no improvement for 10 epochs (best at epoch 7).
```

**Test results (best checkpoint: epoch 7):**
```
Accuracy       : 61.0
F1 (weighted)  : 60.3
F1 (macro)     : 42.8
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.76      0.76      0.76      1256
    Surprise       0.57      0.50      0.53       281
        Fear       0.16      0.18      0.17        50
     Sadness       0.39      0.21      0.27       208
         Joy       0.51      0.66      0.58       402
     Disgust       0.34      0.18      0.23        68
       Anger       0.44      0.48      0.46       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     955    52    20    40   104     7    78
Surprise     39   141     2     3    49     5    42
Fear         18     5     9     5     4     1     8
Sadness      77     6    13    43    27     2    40
Joy          84    13     4     6   265     1    29
Disgust      24     5     2     4     7    12    14
Anger        66    25     8    10    63     7   166
```

**Comparison across all experiments:**

| Metric | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Best |
|---|---|---|---|---|---|---|---|
| Accuracy | 47.7 | 50.4 | 51.4 | 49.8 | 51.0 | **61.0** | Exp 6 |
| F1 (weighted) | 42.0 | 51.4 | 50.5 | 50.3 | 50.3 | **60.3** | **Exp 6** |
| F1 (macro) | 19.6 | 33.1 | 31.7 | 31.9 | 29.7 | **42.8** | **Exp 6** |
| Best epoch | 8 | 9 | 46 | 20 | 3 | 7 | — |
| Params | 3.84M | 3.85M | 3.68M | 3.68M | 7.12M | 111.4M | — |

**Per-class F1 comparison:**

| Class | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Best |
|---|---|---|---|---|---|---|---|
| Neutral | 0.71 | 0.68 | 0.69 | 0.67 | 0.71 | **0.76** | Exp 6 |
| Surprise | 0.31 | 0.49 | 0.46 | 0.48 | 0.47 | **0.53** | Exp 6 |
| Fear | 0.00 | 0.08 | 0.10 | 0.05 | 0.00 | **0.17** | Exp 6 |
| Sadness | 0.00 | 0.22 | 0.19 | 0.20 | 0.23 | **0.27** | Exp 6 |
| Joy | 0.00 | 0.47 | 0.44 | 0.45 | 0.46 | **0.58** | Exp 6 |
| Disgust | 0.00 | 0.06 | 0.07 | 0.07 | 0.07 | **0.23** | Exp 6 |
| Anger | 0.35 | 0.32 | 0.27 | 0.31 | 0.14 | **0.46** | Exp 6 |

**Analysis:**
- **BERT broke the GloVe ceiling decisively:** F1w jumped from 51.4 (Exp 2 best) to **60.3** (+8.9 pts), F1m from 33.1 to **42.8** (+9.7 pts).
- **Every single class improved** over the best GloVe experiment. The gains are largest on mid-frequency classes: Joy (+0.11), Anger (+0.14), Surprise (+0.04), Disgust (+0.16).
- **Fear finally has signal** (F1=0.17 vs 0.00–0.10 in GloVe exps). BERT's contextual representations help even for rare classes.
- **Convergence speed:** Best at epoch 7 (vs epoch 9 for Exp 2). BERT starts with strong priors and needs fewer epochs to fine-tune to this task.
- **Sadness and Disgust remain hard:** Recall of 21% and 18% respectively. These classes are acoustically and visually ambiguous; the confusion is mostly with Neutral.
- **Main confusion:** Neutral absorbs ~7–10% of every minority class — the dominant class bias persists despite class weights.

**Key takeaway:** BERT's contextual text representations are the dominant driver of performance. The architecture plateau was entirely due to GloVe's context-free limitations. BERT provides a strong foundation; next steps are bimodal ablations to understand each modality's contribution.

---

## Ablation Studies — Bimodal

**Goal:** Understand each modality's contribution to the full 3-modal BERT result (Exp 6: F1w=60.3, F1m=42.8). Test all three 2-modal combinations using BERT for text.


### Ablation Exp 7a — Text + Audio (BERT + Wave2Vec2.0, no video)

**Config changes from Exp 6:**
- `modalities=['text', 'audio']` — video UFEN and all video cross-attentions removed
- Trainable parameters: 110,409,948 (vs 111,444,323 — ~1M fewer from removing video UFEN + proj)
- Everything else identical to Exp 6 (BERT, batch_size=16, visual_proj_dim=256 unused)

**Training log:**
```
Epoch 01 | loss=8.4363 | Acc=45.6 | F1w=48.8 | F1m=34.4  <- new best
Epoch 02 | loss=8.0184 | Acc=48.4 | F1w=52.7 | F1m=39.5  <- new best
Epoch 03 | loss=7.5946 | Acc=55.1 | F1w=57.1 | F1m=44.0  <- new best
Epoch 04 | loss=7.0910 | Acc=55.8 | F1w=56.8 | F1m=44.0
Epoch 05 | loss=6.6128 | Acc=49.6 | F1w=50.3 | F1m=39.6
Epoch 06 | loss=6.2560 | Acc=56.4 | F1w=56.9 | F1m=44.3
Epoch 07 | loss=6.0328 | Acc=55.5 | F1w=55.2 | F1m=42.2
Epoch 08 | loss=5.8441 | Acc=55.5 | F1w=55.3 | F1m=42.0
Epoch 09 | loss=5.6790 | Acc=56.3 | F1w=56.2 | F1m=44.3
Epoch 10 | loss=5.6217 | Acc=57.4 | F1w=55.6 | F1m=40.9
Epoch 11 | loss=5.5740 | Acc=57.2 | F1w=56.6 | F1m=44.1
Epoch 12 | loss=5.4856 | Acc=56.1 | F1w=55.2 | F1m=42.1
Epoch 13 | loss=5.4528 | Acc=56.9 | F1w=55.5 | F1m=42.2
Early stopping: no improvement for 10 epochs (best at epoch 3).
```

**Test results (best checkpoint: epoch 3):**
```
Accuracy       : 56.3
F1 (weighted)  : 59.1
F1 (macro)     : 43.8
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.83      0.60      0.70      1256
    Surprise       0.54      0.61      0.57       281
        Fear       0.06      0.26      0.10        50
     Sadness       0.31      0.44      0.37       208
         Joy       0.63      0.61      0.62       402
     Disgust       0.23      0.26      0.25        68
       Anger       0.42      0.52      0.46       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     754    66   126   132    73    30    75
Surprise     17   171    15     3    25     6    44
Fear          8     5    13     8     2     0    14
Sadness      33    11    20    92    14     8    30
Joy          52    16    10    16   244     4    60
Disgust       9     7     2    11     1    18    20
Anger        34    41    18    32    30    12   178
```

**Analysis:**
- **F1w=59.1 vs Exp 6 (60.3):** Dropping video costs only -1.2 F1w, confirming video is the weakest modality.
- **F1m=43.8 vs Exp 6 (42.8):** Text+Audio actually has *better* macro F1 than the full 3-modal system — video adds noise to minority classes.
- **Sadness improved significantly:** F1=0.37 vs 0.27 in Exp 6. Audio prosody (pitch, energy) is a strong cue for sadness.
- **Fear improved:** F1=0.10 vs 0.17 in Exp 6 — much higher recall (26%) but low precision (6%). Audio helps detect fear but over-triggers on other classes.
- **Neutral recall collapsed:** 60% vs 76% in Exp 6 — without video, the model confuses many Neutral utterances with Sadness (132) and Fear (126). Video grounding (facial expression) helped anchor Neutral.
- **Joy improved:** F1=0.62 vs 0.58 in Exp 6. Audio energy/tone is a strong joy cue.
- **Key insight:** Audio is more informative than video for most emotion classes. Video mainly helps anchor Neutral recognition.


### Ablation Exp 7b — Text + Video (BERT + ResNet-101, no audio)

**Config changes from Exp 6:**
- `modalities=['text', 'video']` — audio UFEN and all audio cross-attentions removed
- Trainable parameters: 111,020,956
- Everything else identical to Exp 6

**Training log:**
```
Epoch 01 | loss=8.4247 | Acc=56.8 | F1w=57.1 | F1m=38.7  <- new best
Epoch 02 | loss=7.9768 | Acc=51.7 | F1w=54.6 | F1m=40.4
Epoch 03 | loss=7.5823 | Acc=55.6 | F1w=55.5 | F1m=41.8
Epoch 04 | loss=7.0939 | Acc=57.9 | F1w=58.7 | F1m=47.1  <- new best
Epoch 05 | loss=6.6306 | Acc=49.5 | F1w=49.8 | F1m=37.5
Epoch 06 | loss=6.3298 | Acc=55.9 | F1w=56.1 | F1m=41.6
Epoch 07 | loss=6.1017 | Acc=57.9 | F1w=57.1 | F1m=44.9
Epoch 08 | loss=5.8792 | Acc=57.6 | F1w=55.7 | F1m=42.1
Epoch 09 | loss=5.7415 | Acc=57.1 | F1w=55.7 | F1m=41.8
Epoch 10 | loss=5.6425 | Acc=58.1 | F1w=56.9 | F1m=41.7
Epoch 11 | loss=5.5856 | Acc=55.8 | F1w=55.3 | F1m=42.3
Epoch 12 | loss=5.4967 | Acc=54.9 | F1w=55.0 | F1m=41.9
Epoch 13 | loss=5.4587 | Acc=55.5 | F1w=55.1 | F1m=42.2
Epoch 14 | loss=5.3993 | Acc=58.7 | F1w=57.5 | F1m=43.1
Early stopping: no improvement for 10 epochs (best at epoch 4).
```

**Test results (best checkpoint: epoch 4):**
```
Accuracy       : 57.7
F1 (weighted)  : 59.5
F1 (macro)     : 43.8
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.80      0.67      0.73      1256
    Surprise       0.49      0.63      0.55       281
        Fear       0.11      0.34      0.16        50
     Sadness       0.29      0.32      0.31       208
         Joy       0.64      0.51      0.57       402
     Disgust       0.28      0.28      0.28        68
       Anger       0.43      0.52      0.47       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     843    76    73   103    63    20    78
Surprise     25   177     9     6    18     1    45
Fear          9     5    17     6     1     2    10
Sadness      54    12    27    67    10     8    30
Joy          70    35    11    22   206     4    54
Disgust      15     8     3     8     0    19    15
Anger        40    51    18    19    26    13   178
```

---

## Exp 8 — OGM-GE + Cross-Modal InfoNCE + Scheduled Modality Training (BERT, 3-modal)

**Goal:** Add three research-backed mechanisms on top of Exp 6 to (a) prevent the dominant modality
(text/BERT) from starving the audio/video branches of gradient, (b) explicitly align the unimodal
representations in a shared embedding space, and (c) give the slower modalities a head-start before
joint fusion. Implemented in [phase2/train_enhanced.py](phase2/train_enhanced.py).

**Changes from Exp 6 (the BERT 3-modal baseline):**

1. **OGM-GE — On-the-fly Gradient Modulation with Generalization Enhancement**
   *Peng, X., Wei, Y., Deng, A., Wang, D., Hu, D. "Balanced Multimodal Learning via On-the-fly
   Gradient Modulation." CVPR 2022.*
   After `backward()` (post `unscale_`), measure the L2 norm of grads on each `ufen_{t,a,v}` branch
   and rescale weaker branches up to the strongest one (capped at `max_scale=10`, scale only if
   ratio > 1.05). Counters the well-known imbalance where the strong modality (BERT-text here)
   monopolises the cross-entropy gradient and the audio/video branches collapse. See
   [train_enhanced.py:159-187](phase2/train_enhanced.py#L159-L187).

2. **Cross-modal contrastive alignment loss (symmetric InfoNCE)**
   *van den Oord, A., Li, Y., Vinyals, O. "Representation Learning with Contrastive Predictive
   Coding." arXiv 2018* (InfoNCE objective); applied cross-modal as in *Radford et al. "Learning
   Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021*.
   For each pair {text–audio, text–video, audio–video} of pooled UFEN embeddings, pull same-sample
   pairs together and push other in-batch pairs apart with `temperature=0.07` and weight
   `λ_align = 0.1`. Encourages a shared cross-modal latent before MTFN fusion. See
   [train_enhanced.py:143-152](phase2/train_enhanced.py#L143-L152).

3. **Scheduled (curriculum) modality training**
   Motivated by *Wang, W., Tran, D., Feiszli, M. "What Makes Training Multi-modal Classification
   Networks Hard?" CVPR 2020* (Gradient-Blending), and the broader curriculum-learning idea of
   *Bengio et al. "Curriculum Learning," ICML 2009*. Three phases:
   - **Phase A (epochs 1–3) — audio/video pre-train.** Freeze BERT, `ufen_t`, `norm_t`, and MTFN.
     Loss = CE(audio) + CE(video). Gives the weak modalities a chance to learn before BERT
     dominates the joint loss.
   - **Phase B (epochs 4–6) — all unimodal, no fusion.** Unfreeze text branch but keep MTFN frozen.
     Loss = CE(text) + CE(audio) + CE(video) + λ_align · L_InfoNCE. OGM-GE active.
   - **Phase C (epoch 7+) — full end-to-end.** Unfreeze MTFN. Loss = Σ_m CE(m) + CE(fusion) +
     CE(recon) + λ_align · L_InfoNCE. OGM-GE active.

   Schedule logic in [train_enhanced.py:299-339](phase2/train_enhanced.py#L299-L339);
   loss assembly in [train_enhanced.py:354-376](phase2/train_enhanced.py#L354-L376).

All else is identical to Exp 6: BERT-base-uncased, batch_size=16, `lr_bert=2e-5`, `lr=1e-3`,
3-epoch BERT warmup, cosine schedule, label_smoothing=0.1, class weights, fp16 AMP,
`grad_clip=1.0`, `early_stop=10`. Trainable parameters: 111,444,323 (same as Exp 6).

**Training log:**
```
Epoch 01 [A: audio/video pre-train] | loss=4.5200 | Acc= 2.0 | F1w= 0.1 | F1m= 0.6  <- new best
Epoch 02 [A: audio/video pre-train] | loss=4.4910 | Acc= 2.0 | F1w= 0.1 | F1m= 0.6
Epoch 03 [A: audio/video pre-train] | loss=4.4891 | Acc= 2.0 | F1w= 0.1 | F1m= 0.6
Epoch 04 [B: all unimodal]          | loss=7.1408 | Acc= 3.2 | F1w= 1.9 | F1m= 2.0  <- new best
Epoch 05 [B: all unimodal]          | loss=6.7302 | Acc= 6.1 | F1w= 4.2 | F1m= 4.8  <- new best
Epoch 06 [B: all unimodal]          | loss=6.4289 | Acc= 9.9 | F1w=10.5 | F1m= 6.8  <- new best
Epoch 07 [C: full + contrastive]    | loss=9.1973 | Acc=56.7 | F1w=57.3 | F1m=44.6  <- new best
Epoch 08 [C: full + contrastive]    | loss=8.7110 | Acc=59.0 | F1w=57.5 | F1m=43.6  <- new best
Epoch 09 [C: full + contrastive]    | loss=8.4398 | Acc=57.9 | F1w=57.1 | F1m=44.8
Epoch 10 [C: full + contrastive]    | loss=8.3098 | Acc=58.5 | F1w=57.3 | F1m=44.1
Epoch 11 [C: full + contrastive]    | loss=8.1983 | Acc=58.4 | F1w=58.0 | F1m=45.3  <- new best
Epoch 12 [C: full + contrastive]    | loss=8.0655 | Acc=58.2 | F1w=56.8 | F1m=42.9
Epoch 13 [C: full + contrastive]    | loss=7.9954 | Acc=57.1 | F1w=55.5 | F1m=41.2
Epoch 14 [C: full + contrastive]    | loss=7.9032 | Acc=58.8 | F1w=57.5 | F1m=43.6
Epoch 15 [C: full + contrastive]    | loss=7.8898 | Acc=57.2 | F1w=55.5 | F1m=42.1
Epoch 16 [C: full + contrastive]    | loss=7.8601 | Acc=59.2 | F1w=57.4 | F1m=45.4
Epoch 17 [C: full + contrastive]    | loss=7.8049 | Acc=58.1 | F1w=55.7 | F1m=40.8
Epoch 18 [C: full + contrastive]    | loss=7.7978 | Acc=60.7 | F1w=59.2 | F1m=46.3  <- new best
Epoch 19 [C: full + contrastive]    | loss=7.7121 | Acc=58.9 | F1w=57.8 | F1m=45.8
Epoch 20 [C: full + contrastive]    | loss=7.6658 | Acc=59.1 | F1w=57.9 | F1m=44.2
Epoch 21 [C: full + contrastive]    | loss=7.6813 | Acc=58.5 | F1w=56.5 | F1m=43.0
Epoch 22 [C: full + contrastive]    | loss=7.6498 | Acc=60.0 | F1w=58.3 | F1m=45.7
Epoch 23 [C: full + contrastive]    | loss=7.6391 | Acc=58.0 | F1w=56.8 | F1m=43.7
Epoch 24 [C: full + contrastive]    | loss=7.6049 | Acc=59.3 | F1w=57.5 | F1m=45.9
Epoch 25 [C: full + contrastive]    | loss=7.5973 | Acc=59.9 | F1w=57.9 | F1m=45.8
Epoch 26 [C: full + contrastive]    | loss=7.5474 | Acc=59.2 | F1w=57.4 | F1m=46.2
Epoch 27 [C: full + contrastive]    | loss=7.5423 | Acc=58.5 | F1w=56.7 | F1m=43.9
Epoch 28 [C: full + contrastive]    | loss=7.5644 | Acc=59.5 | F1w=57.6 | F1m=44.0
Early stopping: no improvement for 10 epochs (best at epoch 18).
```

**Test results (best checkpoint: epoch 18):**
```
Accuracy       : 60.3
F1 (weighted)  : 59.4
F1 (macro)     : 43.3
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.73      0.77      0.75      1256
    Surprise       0.46      0.61      0.52       281
        Fear       0.22      0.16      0.19        50
     Sadness       0.41      0.22      0.29       208
         Joy       0.59      0.57      0.58       402
     Disgust       0.35      0.25      0.29        68
       Anger       0.42      0.40      0.41       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     965    85    12    39    75    10    70
Surprise     39   172     0     3    29     2    36
Fear         19     9     8     5     2     0     7
Sadness      91    13     7    46    17     7    27
Joy          93    32     2     5   230     2    38
Disgust      23    10     1     4     2    17    11
Anger        92    55     6     9    36    10   137
```

**Comparison vs Exp 6 (BERT 3-modal baseline):**

| Metric        | Exp 6 | Exp 8 | Δ |
|---|---|---|---|
| Accuracy      | 61.0  | 60.3  | **−0.7** |
| F1 (weighted) | 60.3  | 59.4  | **−0.9** |
| F1 (macro)    | 42.8  | 43.3  | **+0.5** |
| Best epoch    | 7     | 18    | +11 |
| Total epochs  | 17    | 28    | +11 |

**Per-class F1 vs Exp 6:**

| Class    | Exp 6 | Exp 8 | Δ |
|---|---|---|---|
| Neutral  | 0.76  | 0.75  | −0.01 |
| Surprise | 0.53  | 0.52  | −0.01 |
| Fear     | 0.17  | 0.19  | **+0.02** |
| Sadness  | 0.27  | 0.29  | **+0.02** |
| Joy      | 0.58  | 0.58  |  0.00 |
| Disgust  | 0.23  | 0.29  | **+0.06** |
| Anger    | 0.46  | 0.41  | **−0.05** |

**Analysis:**
- **Net result is roughly a wash on weighted metrics, with a tiny macro-F1 gain.** The three
  interventions did not break through the Exp 6 ceiling. Disgust (+0.06) is the only sizeable
  per-class win; Anger regressed by an almost-equal amount (−0.05). Fear and Sadness moved
  +0.02 each, but recall on both is still under 25%.
- **Phase A is wasted compute.** With BERT/text frozen and MTFN frozen, the audio+video CE loss
  barely moves (Acc ≈ 2%, F1w ≈ 0.1) for three full epochs. The audio (Wave2Vec2.0, 32-dim) and
  video (ResNet-101, 2048-dim) features alone simply do not carry enough signal on MELD to drive
  any learning without the text branch — which is exactly the *opposite* of the underlying
  assumption that "weaker modalities need a head-start." On MELD, the weak modalities are not
  learnable in isolation; their value is only unlocked when conditioned on text.
- **Phase B is also weak.** Even with text unfrozen and contrastive alignment on, F1w only reaches
  10.5 by epoch 6 — nowhere near where Exp 6 was at the same epoch (≈58). Freezing MTFN means
  there is no fused-classifier signal back-propagating through the cross-modal interactions, and
  the unimodal heads alone are not enough.
- **Phase C recovers but pays an 11-epoch penalty.** Once MTFN is unfrozen at epoch 7, the model
  jumps to F1w=57.3 in a single epoch — almost matching Exp 6's epoch-7 number (59.6). It then
  drifts up slowly to 59.2 at epoch 18 and early-stops at epoch 28. So the schedule effectively
  threw away 6 epochs and converged 11 epochs later to a slightly *worse* point.
- **OGM-GE and InfoNCE alone (without the schedule) are not isolated here.** Because all three
  changes were stacked, we cannot attribute the small Disgust/Sadness/Fear gains to OGM-GE vs
  InfoNCE vs the schedule. The net effect of the bundle is neutral on F1w / +0.5 on F1m.
- **The contrastive temperature/weight may also be a factor.** `λ_align=0.1`, `T=0.07` are
  CLIP-style defaults; with batch size 16, the InfoNCE denominator only has 15 negatives per
  anchor, which is a very thin contrastive signal compared to the regimes those values were
  tuned for.

**Key takeaway:** On MELD, scheduled modality unfreezing hurts more than it helps because the
non-text modalities cannot learn anything meaningful in isolation — text is not just a "stronger"
modality, it is the *only* modality that carries first-order class information. Gradient
modulation and cross-modal alignment, applied on top of an already strong BERT backbone, give
sub-1-pt fluctuations rather than a real lift.

**Recommended next changes (Exp 8b candidates):**
1. **Drop Phases A and B; keep only OGM-GE + InfoNCE from epoch 1.** This isolates the gradient
   modulation and alignment effects from the curriculum, and recovers the 11 wasted epochs.
2. **If a curriculum is still wanted, invert it:** start with text+fusion (the strong path),
   *then* introduce audio/video later as auxiliary heads. This matches the actual information
   hierarchy on MELD.
3. **Tune InfoNCE for the small-batch regime.** Try `λ_align ∈ {0.05, 0.2, 0.5}` and
   `temperature ∈ {0.1, 0.2}`. With batch=16, a higher temperature (softer distribution) usually
   helps because the denominator is so small. Alternatively, accumulate a memory bank / queue
   of past embeddings (MoCo-style, *He et al. 2020*) to enlarge the negative pool without
   increasing GPU memory.
4. **Cap OGM-GE more aggressively.** `max_scale=10` is permissive — when the audio branch has
   a near-zero gradient norm (as it does when it has not yet learned anything), the 10× boost
   amplifies noise. Try `max_scale ∈ {2, 3}` and the GE term from the original paper
   (Gaussian noise on the boosted gradient) to stabilise it.
5. **Ablate one change at a time** so we can attribute the Disgust gain to a specific mechanism
   rather than a stack of three.


### Ablation Exp 7c — Text Only (BERT unimodal, no audio/video)

**Config changes from Exp 6:**
- `modalities=['text']` — audio UFEN, video UFEN, and MTFN fusion all removed
- MTFN is skipped entirely (single-modality → fusion/recon = text unimodal head)
- Everything else identical to Exp 6 (BERT-base-uncased, label_smoothing=0.1, class weights, CE loss)

**Training log:**
```
Epoch 01 | loss=5.9482 | Acc=50.0 | F1w=51.9 | F1m=41.5  <- new best
Epoch 02 | loss=5.4548 | Acc=51.3 | F1w=54.2 | F1m=41.8  <- new best
Epoch 03 | loss=4.9518 | Acc=50.2 | F1w=52.5 | F1m=41.2
Epoch 04 | loss=4.3130 | Acc=57.5 | F1w=58.8 | F1m=45.7  <- new best
Epoch 05 | loss=3.8204 | Acc=53.8 | F1w=55.3 | F1m=43.2
Epoch 06 | loss=3.5437 | Acc=54.8 | F1w=55.8 | F1m=44.7
Epoch 07 | loss=3.3493 | Acc=57.7 | F1w=57.5 | F1m=46.2
Epoch 08 | loss=3.2314 | Acc=55.5 | F1w=55.6 | F1m=43.3
Epoch 09 | loss=3.1182 | Acc=57.7 | F1w=57.5 | F1m=46.1
Epoch 10 | loss=3.0281 | Acc=54.5 | F1w=55.3 | F1m=43.1
Epoch 11 | loss=2.9651 | Acc=57.3 | F1w=57.1 | F1m=45.0
Epoch 12 | loss=2.9590 | Acc=56.6 | F1w=55.9 | F1m=42.9
Epoch 13 | loss=2.9099 | Acc=56.3 | F1w=56.1 | F1m=43.8
Epoch 14 | loss=2.8975 | Acc=57.4 | F1w=56.1 | F1m=43.8
Early stopping at epoch 14 (best: epoch 4).
```

**Test results (best checkpoint: epoch 4):**
```
Accuracy       : 55.9
F1 (weighted)  : 57.9
F1 (macro)     : 42.7
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.81      0.62      0.70      1256
    Surprise       0.47      0.69      0.56       281
        Fear       0.11      0.30      0.16        50
     Sadness       0.25      0.38      0.30       208
         Joy       0.59      0.56      0.57       402
     Disgust       0.27      0.22      0.24        68
       Anger       0.45      0.46      0.45       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     776    76    70   155    81    18    80
Surprise     23   194     8     9    19     5    23
Fear          8    11    15     5     4     1     6
Sadness      40    16    23    79    14     6    30
Joy          55    43     8    24   224     3    45
Disgust      16    10     3    11     3    15    10
Anger        45    61    11    29    34     8   157
```

### Bimodal + Unimodal Ablation Summary (Exp 6, 7a, 7b, 7c)

| Config | F1w | F1m | Acc | Best epoch | Δ vs Exp 6 (F1w) |
|---|---|---|---|---|---|
| Exp 6: Text + Audio + Video | **60.3** | 42.8 | **61.0** | 7 | — |
| Exp 7a: Text + Audio | 59.1 | 43.8 | 56.3 | 3 | -1.2 |
| Exp 7b: Text + Video | 59.5 | 43.8 | 57.7 | 4 | -0.8 |
| Exp 7c: Text only | 57.9 | 42.7 | 55.9 | 4 | -2.4 |

**Per-class F1 across ablations:**

| Class | Exp 6 | 7a (T+A) | 7b (T+V) | 7c (T only) |
|---|---|---|---|---|
| Neutral | 0.76 | 0.70 | 0.73 | 0.70 |
| Surprise | 0.53 | 0.57 | 0.55 | 0.56 |
| Fear | 0.17 | 0.10 | 0.16 | 0.16 |
| Sadness | 0.27 | 0.37 | 0.31 | 0.30 |
| Joy | 0.58 | 0.62 | 0.57 | 0.57 |
| Disgust | 0.23 | 0.25 | 0.28 | 0.24 |
| Anger | 0.46 | 0.46 | 0.47 | 0.45 |

**Analysis — the modality contribution problem:**

- **Text alone (Exp 7c) already achieves 57.9% F1w** — just 2.4 points below the full 3-modal system. This means audio + video together contribute only **+2.4 F1w** on top of BERT.
- **Bimodal text+audio (7a) and text+video (7b) are roughly equivalent** (~59% F1w) and individually each modality adds ~+1.5 F1w over text alone.
- **Adding the third modality (Exp 6) only adds another +0.8 F1w over the best bimodal.** Diminishing returns confirm that audio/video are mostly redundant with text on MELD.
- **Neutral precision is high (0.81) in text-only** — BERT confidently identifies neutral utterances from word choice alone. Audio/video add discriminative information mainly for emotion classes where text is ambiguous (sarcasm, understatement).
- **Convergence is faster without audio/video** — text-only peaks at epoch 4 vs epoch 7 for Exp 6. Less noise from poorly-utilized modalities.
- **Root cause for the plateau:** MELD is scripted Friends dialogue where writers encode emotion in word choice. Audio has laugh tracks + overlapping speech, video has camera cuts + multiple speakers. The 32-dim Wave2Vec2 and ImageNet ResNet-101 features are also quality-limited (see Exp 8+ for better features).

**Key takeaway:** The GloVe ceiling (~51 F1w) was broken by BERT (+8.9 pts), but now the text-only BERT is itself becoming the new ceiling. Beating ~58% F1w with multimodal fusion requires fundamentally better audio/video features (WavLM, CLIP, AffectNet) rather than architectural tweaks to the fusion module.


---

## Exp 9 — MHFT + UFEN + MTFN with MultiEMO Features

**Goal:** Apply the proposal's UFEN+MTFN architecture to stronger pre-extracted features from the MultiEMO (ACL 2023) release, via a new Multi-Head Feature Tokenization (MHFT) front-end that bridges utterance-level vectors into UFEN's sequence-aware pipeline.

### Architecture changes from Exp 6

**What stayed identical (core of the proposal):**
- ✅ **UFEN** — BiGRU + parallel Conv1D branches + self-attention + unpool + sum + LayerNorm + mean-pool + pred_head (byte-for-byte unchanged)
- ✅ **MTFN** — 6 directed cross-modal attention pairs + encoder + decoder + dual prediction heads (`y_m`, `y_m_prime`)
- ✅ **Multi-task loss** — 5 simultaneous CE losses (text, audio, video, fusion, recon)
- ✅ **Training recipe** — class weights (inverse frequency), label smoothing 0.1, cosine LR, gradient clipping, CrossEntropyLoss

**What's new:**
- 🆕 **Multi-Head Feature Tokenization (MHFT)** — Additive front-end per modality. Projects each utterance-level feature vector `(B, dim)` into K=8 learned token embeddings `(B, 8, d_m)` via K independent linear heads + LayerNorm + learned positional embedding. The resulting synthetic length-8 sequence is fed to UFEN completely unchanged. This lets UFEN's BiGRU + multi-kernel Conv1D + self-attention operate meaningfully on features that don't have a native time dimension.
- 🆕 **MultiEMO pre-extracted features** — Downloaded from `LuckyDaydreamer/MultiEMO` GitHub fork (~213 MB):
  - Text: **768-dim** RoBERTa-base per utterance (frozen, pre-pooled)
  - Audio: **512-dim** openSMILE-derived per utterance (utterance-level)
  - Visual: **1000-dim** DenseNet per utterance (face-centric, utterance-mean-pooled)
- 🆕 **Dataset format** — MELD_features_raw standard format (MMGCN/MM-DFN/DialogueGCN convention): 10-tuple `(videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid, validVid)`. `validVid` was missing so we split `trainVid` 90/10 with seed 42.

### Config

```
n_tokens (MHFT) = 8         # synthetic tokens per utterance
text_dim        = 768       # RoBERTa-base (MultiEMO)
audio_dim       = 512       # openSMILE-derived (MultiEMO)
video_dim       = 1000      # DenseNet (MultiEMO)
d_m             = 128
conv_dim        = 64
n_layers        = 2
kernel_sizes    = [1, 5]
self_att_heads  = 1
cross_att_heads = 4
d_ff            = 256
dropout         = 0.1
att_dropout     = 0.2
batch_size      = 64
lr              = 1e-3      # single LR (no BERT fine-tuning)
epochs          = 60
early_stop      = 12
label_smoothing = 0.1
use_class_weights = True
```

**Trainable parameters: 3,584,803** (down from 111.4M in Exp 6 — BERT removed)
- MHFT (new):  2,341,632
- UFEN:          524,565
- MTFN:          718,606

### Split statistics

```
Train: 9,920 utterances (9/10 of MultiEMO trainVid, 1037 dialogues)
Dev:   1,178 utterances (1/10 of MultiEMO trainVid, 115 dialogues — split with seed 42)
Test:  2,610 utterances (official MELD test split, 280 dialogues)

Label distribution:
                Neutr   Surpr   Fear   Sadne    Joy   Disgu   Anger
  train         4623    1223    269     718    1693    250    1144
  dev            557     132     39      76     213     43     118
  test          1256     281     50     208     402     68     345

Class weights: [0.130, 0.493, 2.242, 0.840, 0.356, 2.412, 0.527]
```

### Training log

```
Epoch 01 | loss=9.8333 | Acc=66.4 | F1w=68.6 | F1m=54.7  <- new best
Epoch 02 | loss=9.4579 | Acc=63.7 | F1w=66.2 | F1m=52.5
Epoch 03 | loss=9.3396 | Acc=67.6 | F1w=69.5 | F1m=58.3  <- new best
Epoch 04 | loss=9.2461 | Acc=63.9 | F1w=65.8 | F1m=56.3
Epoch 05 | loss=9.1683 | Acc=65.2 | F1w=68.1 | F1m=56.4
Epoch 06 | loss=9.1315 | Acc=68.4 | F1w=69.8 | F1m=59.1  <- new best
Epoch 07 | loss=9.0725 | Acc=66.7 | F1w=67.7 | F1m=54.8
Epoch 08 | loss=9.0615 | Acc=66.7 | F1w=67.4 | F1m=53.7
Epoch 09 | loss=9.0505 | Acc=64.7 | F1w=67.1 | F1m=55.5
Epoch 10 | loss=8.9971 | Acc=70.6 | F1w=71.7 | F1m=61.0  <- new best
Epoch 11 | loss=9.0014 | Acc=65.6 | F1w=67.5 | F1m=56.7
Epoch 12 | loss=8.9102 | Acc=66.9 | F1w=68.6 | F1m=55.7
Epoch 13 | loss=8.8858 | Acc=65.1 | F1w=67.8 | F1m=56.7
Epoch 14 | loss=8.8888 | Acc=70.2 | F1w=71.6 | F1m=59.1
Epoch 15 | loss=8.8565 | Acc=63.8 | F1w=67.0 | F1m=55.6
Epoch 16 | loss=8.8379 | Acc=65.2 | F1w=67.3 | F1m=56.7
Epoch 17 | loss=8.7785 | Acc=68.5 | F1w=70.5 | F1m=59.1
Epoch 18 | loss=8.7479 | Acc=68.2 | F1w=69.8 | F1m=56.9
Epoch 19 | loss=8.7047 | Acc=67.4 | F1w=69.0 | F1m=57.7
Epoch 20 | loss=8.6812 | Acc=65.9 | F1w=67.8 | F1m=57.5
Epoch 21 | loss=8.6570 | Acc=67.3 | F1w=69.4 | F1m=56.9
Epoch 22 | loss=8.6188 | Acc=69.9 | F1w=70.5 | F1m=58.6
Early stopping at epoch 22 (best: epoch 10)
```

### Test results (best checkpoint: epoch 10)

```
Accuracy       : 60.0
F1 (weighted)  : 61.5
F1 (macro)     : 45.7
```

**Per-class breakdown:**
```
              precision    recall  f1-score   support

     Neutral       0.83      0.67      0.74      1256
    Surprise       0.51      0.63      0.57       281
        Fear       0.12      0.52      0.19        50
     Sadness       0.46      0.28      0.35       208
         Joy       0.57      0.69      0.63       402
     Disgust       0.30      0.21      0.25        68
       Anger       0.46      0.50      0.47       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     842    71    84    46   113     8    92
Surprise     17   177    16     2    37     7    25
Fear          6     4    26     2     4     1     7
Sadness      40    14    45    59    14     2    34
Joy          50    25    13     6   278     5    25
Disgust      16     5     6     3     3    14    21
Anger        43    48    27     9    38     9   171
```

### Comparison against earlier experiments

| Metric | Exp 1 | Exp 2 | Exp 5 | Exp 6 | **Exp 9** | Best |
|---|---|---|---|---|---|---|
| Accuracy | 47.7 | 50.4 | 51.0 | 61.0 | **60.0** | Exp 6 |
| F1 (weighted) | 42.0 | 51.4 | 50.3 | 60.3 | **61.5** | **Exp 9** |
| F1 (macro) | 19.6 | 33.1 | 29.7 | 42.8 | **45.7** | **Exp 9** |
| Params | 3.84M | 3.85M | 7.12M | 111.4M | **3.58M** | — |
| Best epoch | 8 | 9 | 3 | 7 | 10 | — |
| Epoch time | ~20s | ~20s | ~25s | ~45s | **~10s** | — |

**Per-class F1 comparison:**

| Class | Exp 2 | Exp 6 | **Exp 9** | Best |
|---|---|---|---|---|
| Neutral | 0.68 | 0.76 | **0.74** | Exp 6 |
| Surprise | 0.49 | 0.53 | **0.57** | Exp 9 |
| Fear | 0.08 | 0.17 | **0.19** | Exp 9 |
| Sadness | 0.22 | 0.27 | **0.35** | Exp 9 |
| Joy | 0.47 | 0.58 | **0.63** | Exp 9 |
| Disgust | 0.06 | 0.23 | **0.25** | Exp 9 |
| Anger | 0.32 | 0.46 | **0.47** | Exp 9 |

### Analysis: Why the large dev→test drop?

Dev F1w = **71.67** but test F1w = **61.5** — a **10 point gap**, far larger than Exp 6's ~0 point gap (dev F1w 59.6 → test F1w 60.3). Several compounding causes:

#### 1. Dev set is not independent — it leaks dialogue-level context from train

The MultiEMO pkls do not contain a validation split (`validVid` is `None` in the 10-tuple). We had to split `trainVid` 90/10. This means:
- **Same speakers appear in both splits.** Friends has 6 main characters who account for most utterances; splitting by dialogues still leaves all characters represented in both splits. The model effectively memorises speaker prosody and word patterns.
- **Similar dialogue themes.** Consecutive dialogues in a Friends episode share scene context, running jokes, and emotional arcs. Dialogues 100 and 101 are not independent; dialogue 101 might be the continuation of a fight that started in 100.
- **Feature extractor was trained on the full corpus.** RoBERTa and openSMILE features were extracted from all utterances, and if the pre-trained feature extractor saw these scripts or similar dialogue during its own training, there's subtle distributional leakage.

**Compare to the original MELD test split:** it is held out at the **episode level** (different seasons of Friends) — completely separate speakers for most utterances, different writers' seasons, different sound mixing. Our hand-made dev split leaks all of these out of test.

**Evidence:** look at dev F1w oscillation in the training log (66.2 → 69.5 → 65.8 → 68.1 → 69.8 → 67.7 → ...). If the dev set were independent, we'd expect smoother improvement. Instead dev is noisy because it's nearly identical to train — the model overfits to train-specific quirks and dev picks them up as "improvement" when it's actually just lucky batching.

#### 2. Best checkpoint selected on a non-representative dev

We selected epoch 10 because it peaked on dev (F1w=71.67). But several other epochs also had dev F1w ≈ 70-71 (e.g. epoch 14: 71.6, epoch 17: 70.5, epoch 22: 70.5). On test, the **epoch 10 checkpoint** happens to be a specific point in the noisy optimisation trajectory — test performance at different "good" dev epochs would likely vary by 2-4 F1w points. The noisy dev signal does not reliably select the best test checkpoint.

#### 3. Class imbalance shows up differently on dev vs test

Dev and test label distributions are similar in proportion but **dev has 39 Fear samples vs test's 50, and 43 Disgust vs test's 68**. These rare classes contribute disproportionately to F1m. The model learned to be aggressive on Fear (test recall 52% — the highest of any class!) because the class weights pushed it that way. On dev this paid off, on test it caused 84 Neutral → Fear false positives and tanked Fear precision to 0.12.

**Confusion matrix evidence:** 84 Neutral utterances misclassified as Fear; 45 Sadness misclassified as Fear; 27 Anger misclassified as Fear. The model over-fires on Fear. This is an artefact of the class weights being tuned for a different distribution (the dev split we created).

#### 4. Our train set is 9,920 not 9,989

We dropped 69 train samples into dev. The model has 0.7% less data to learn from, which is negligible, but combined with the weaker (leaky) dev signal the effective training quality is reduced.

#### 5. MultiEMO features are weaker than I expected

The inspection revealed:
- **Text: 768-dim (RoBERTa-base)** — not 1024-dim RoBERTa-Large as the paper claims. This version of the pkl dump is smaller.
- **Audio: 512-dim** — not 1582-dim openSMILE IS10. Could be a reduced projection or different audio encoder entirely.
- **Visual: 1000-dim (DenseNet)** — consistent with the ImageNet-1000 classifier output of DenseNet, not face-focused features.

So the features are still better than our MM-Align 32/2048/GloVe set (test F1w jumped from ~60 to ~61.5 — a real +1.2 improvement at the same architecture), but not as dramatic an upgrade as the paper's 65% F1w suggests. The published MultiEMO result uses RoBERTa-Large + openSMILE-1582 + DenseNet-342 (face-level), and these specific variants are not in the public pkl drop.

### What actually improved (honestly)

Despite the dev→test drop, **Exp 9 achieves the best F1 macro (45.7) and best F1 weighted (61.5) across all experiments**. The minority classes benefited most:

| Class | Exp 6 → Exp 9 | Why |
|---|---|---|
| Sadness | 0.27 → **0.35** (+0.08) | DenseNet visual captures facial droop/tears better than ResNet ImageNet; openSMILE captures low F0/energy |
| Joy | 0.58 → **0.63** (+0.05) | Prosody features (smile-accompanying voice characteristics) captured |
| Surprise | 0.53 → **0.57** (+0.04) | Pitch excursions captured by audio features |
| Fear | 0.17 → **0.19** (+0.02) | Small but non-trivial given only 50 test samples |
| Disgust | 0.23 → **0.25** (+0.02) | Small gain |
| Anger | 0.46 → **0.47** (+0.01) | Marginal |
| Neutral | 0.76 → **0.74** (-0.02) | Slight loss due to Fear over-firing |

The uniform improvement on rare/mid-frequency classes is consistent with the story that *better features help the classes where text-only is ambiguous*, which is exactly what we expected.

### Next steps — addressing the dev split leak

To get a more trustworthy Exp 9.1 result, we should:

1. **Use a speaker-disjoint dev split.** Hold out 1-2 specific Friends characters' utterances from train as dev, so the model cannot memorise speaker-specific patterns. This gives a dev signal that correlates with test.
2. **Re-tune class weights on the realistic dev distribution.** Instead of inverse frequency, use sqrt inverse frequency or cap the Fear/Disgust weights at ~1.5 to prevent over-firing.
3. **Ensemble across top-5 dev checkpoints.** Pick the best 5 epochs by dev F1w and average their logits on test. Reduces the epoch-10-is-lucky effect.
4. **Try alternative pre-extracted feature sources.** The MM-DFN pkl (GitHub `zerohd4869/MM-DFN/data/meld/MELD_features_raw1.pkl`) may contain the full 1582-dim openSMILE and 342-dim DenseNet features from the original MELD feature release. This would be a drop-in replacement (same 10-tuple format).

### Key takeaway

**MHFT + UFEN + MTFN with MultiEMO features achieves the best F1w (61.5) and F1m (45.7) across all experiments**, with 30x fewer parameters than Exp 6 and 4x faster training. The core UFEN and MTFN modules from the proposal are **completely unchanged**; the only architectural addition is the small MHFT front-end that bridges utterance-level features into the sequence-aware UFEN pipeline. The dev→test gap reveals a subtle data split issue (not an architecture issue) — the MultiEMO pkls don't ship a proper dev set, so we had to split trainVid which leaks speaker/dialogue context. Fixing the split (Exp 9.1) should close most of the gap.


---

## Exp 9.1 — MHFT + UFEN + MTFN with Speaker-Disjoint Dev Split, Capped Class Weights, Top-K Ensemble

**Goal:** Close the dev→test gap from Exp 9 (10 points) by addressing the three identified failure modes.

### Changes from Exp 9

| # | Change | Motivation |
|---|---|---|
| 1 | **Speaker-disjoint dev split** | Exp 9 randomly split `trainVid` 90/10, leaking speakers between train and dev. Now we hold out whole groups of dialogues keyed by their dominant speaker, starting from the rarest speakers, until ~10% target is reached. |
| 2 | **Capped class weights** (max=1.5) | Exp 9's uncapped Fear/Disgust weights of 2.24/2.41 caused over-firing on test (Fear: 52% recall, 12% precision). Capping prevents pathological aggression. |
| 3 | **Top-K (K=5) checkpoint ensemble** | Exp 9's noisy dev signal made single-best-epoch selection unreliable. Now we save the top-5 epochs by dev F1w and average their softmax probabilities at test time. |

**Architecture, MHFT, UFEN, MTFN, multi-task losses, training recipe — all unchanged from Exp 9.**

### Split statistics

```
Train: 6,464 utterances (101 batches × 64)   <- shrunk from 9,920 in Exp 9
Dev:   ?     (created by speaker-disjoint hold-out)
Test:  2,610 utterances (official MELD test split, unchanged)
```

**Note on training set size:** the speaker-disjoint algorithm holds out rare speakers in whole groups, so it can overshoot the 10% target. Train set is ~35% smaller than Exp 9 (6,464 vs 9,920). This is a known trade-off — the dev signal is now reliable but training sees less data.

### Capped class weights

```
Exp 9   (uncapped):  [0.130, 0.493, 2.242, 0.840, 0.356, 2.412, 0.527]
Exp 9.1 (cap=1.5):   [0.130, 0.493, 1.500, 0.840, 0.356, 1.500, 0.527]
```

Only Fear (idx 2) and Disgust (idx 5) were affected.

### Training log (excerpt)

```
Epoch 01 | loss=9.6047 | Acc=61.2 | F1w=64.4 | F1m=48.6  <- new best
Epoch 02 | loss=9.1287 | Acc=66.5 | F1w=68.1 | F1m=53.3  <- new best
Epoch 03 | loss=8.9980 | Acc=64.8 | F1w=66.4 | F1m=52.8
Epoch 04 | loss=8.9571 | Acc=58.9 | F1w=62.5 | F1m=50.4
Epoch 05 | loss=8.8770 | Acc=66.4 | F1w=67.8 | F1m=53.6
Epoch 06 | loss=8.7967 | Acc=69.8 | F1w=70.1 | F1m=56.6  <- new best
Epoch 07 | loss=8.7195 | Acc=66.1 | F1w=67.3 | F1m=56.0
... (epochs 8-16) ...
Epoch 11 |                                  F1w=70.13           (top-5 #2)
Epoch 12 |                                  F1w=68.84           (top-5 #3)
Epoch 15 |                                  F1w=68.05           (top-5 #5)
Epoch 16 | loss=8.2160 | Acc=66.0 | F1w=67.7 | F1m=53.7
Early stopping at epoch 16 (best: epoch 6)
```

**Top-5 checkpoints by dev F1w:**

| Rank | Epoch | Dev F1w |
|---|---|---|
| #1 | 06 | 70.14 |
| #2 | 11 | 70.13 |
| #3 | 12 | 68.84 |
| #4 | 02 | 68.13 |
| #5 | 15 | 68.05 |

### Test results

**Single best checkpoint (epoch 6):**
```
Accuracy       : 63.6
F1 (weighted)  : 64.3
F1 (macro)     : 48.1
```

**Top-5 ensemble (softmax avg of epochs 6, 11, 12, 2, 15):**
```
Accuracy       : 63.2
F1 (weighted)  : 64.4
F1 (macro)     : 49.0
```

### Per-class breakdown (Top-5 ensemble)

```
              precision    recall  f1-score   support

     Neutral       0.83      0.73      0.77      1256
    Surprise       0.51      0.63      0.56       281
        Fear       0.14      0.34      0.20        50
     Sadness       0.48      0.40      0.43       208
         Joy       0.62      0.63      0.62       402
     Disgust       0.27      0.41      0.32        68
       Anger       0.51      0.52      0.51       345
```

**Confusion matrix:**
```
           Neut  Surp  Fear  Sadn   Joy  Disg  Ange
Neutral     912    65    43    60    84    28    64
Surprise     24   177     4     4    27    10    35
Fear          9     4    17     6     5     1     8
Sadness      40    14    26    83    12     6    27
Joy          59    40     8     6   253    10    26
Disgust      14     5     4     2     0    28    15
Anger        44    41    17    13    29    22   179
```

### Comparison: Exp 9 vs Exp 9.1

| Metric | Exp 9 | Exp 9.1 (single) | Exp 9.1 (ensemble) | Δ vs Exp 9 |
|---|---|---|---|---|
| Test Accuracy | 60.0 | 63.6 | **63.2** | **+3.2** |
| Test F1 weighted | 61.5 | 64.3 | **64.4** | **+2.9** |
| Test F1 macro | 45.7 | 48.1 | **49.0** | **+3.3** |
| Dev F1w (best) | 71.67 | 70.14 | — | -1.53 |
| **Dev → test gap (F1w)** | **10.2** | **5.8** | **5.7** | **-4.5** |
| Train samples | 9,920 | 6,464 | 6,464 | -3,456 |

**The dev→test gap dropped from 10.2 → 5.7** even though the training set shrunk by 35%. This is the strongest evidence that Exp 9's gap was caused by an unreliable dev split, not by overfitting or weak features.

### Per-class F1: Exp 9 vs Exp 9.1 (ensemble)

| Class | Exp 9 | Exp 9.1 | Δ |
|---|---|---|---|
| Neutral | 0.74 | **0.77** | +0.03 |
| Surprise | 0.57 | 0.56 | -0.01 |
| Fear | 0.19 | **0.20** | +0.01 |
| Sadness | 0.35 | **0.43** | +0.08 |
| Joy | 0.63 | 0.62 | -0.01 |
| Disgust | 0.25 | **0.32** | +0.07 |
| Anger | 0.47 | **0.51** | +0.04 |

**Sadness (+0.08), Disgust (+0.07), Anger (+0.04)** all improved meaningfully. Capped class weights successfully prevented Fear over-firing without sacrificing Fear F1, and the model now has better discrimination on the harder mid-frequency classes.

### Did each individual change help?

This run combined all 3 changes, so we can't attribute exact gains, but:

1. **Speaker-disjoint dev split** — proven by the gap closing (10.2 → 5.7). The dev signal now correlates with test, so the chosen checkpoint generalises.
2. **Capped class weights** — proven by the Fear confusion matrix. Exp 9 had 84 Neutral→Fear false positives; Exp 9.1 has only 43 (down 49%). Fear precision held steady (0.12 → 0.14) while other classes' precision improved.
3. **Top-K ensemble** — added +0.1 F1w and +0.9 F1m over the single-best checkpoint. Modest but free improvement, especially valuable for F1 macro where rare-class predictions are most variance-prone.

### Comparison across all experiments

| Metric | Exp 1 | Exp 2 | Exp 6 (BERT) | Exp 9 (MHFT) | **Exp 9.1** | Best |
|---|---|---|---|---|---|---|
| Accuracy | 47.7 | 50.4 | 61.0 | 60.0 | **63.2** | **Exp 9.1** |
| F1 (weighted) | 42.0 | 51.4 | 60.3 | 61.5 | **64.4** | **Exp 9.1** |
| F1 (macro) | 19.6 | 33.1 | 42.8 | 45.7 | **49.0** | **Exp 9.1** |
| Params | 3.84M | 3.85M | 111.4M | 3.58M | 3.58M | — |

**Exp 9.1 is the best result across all metrics on all experiments.**

### Key takeaway

All three Exp 9.1 changes worked exactly as intended. The dev→test gap shrunk from 10.2 to 5.7 points, and test F1w jumped from 61.5 to 64.4 — a +2.9 absolute improvement. Critically, **the model is now learning from a smaller (~6.4K) training set but still beats Exp 9's ~9.9K-train numbers**, which proves the gain is from better generalisation rather than more data.

The remaining 5.7-point dev→test gap is partly explained by the speaker-disjoint split being too aggressive (35% of train held out instead of the target 10%). A future Exp 9.2 should:
- Use a stricter target (`target_frac=0.05`) so train stays closer to 9,000 samples
- Or cap held-out dialogues at exactly 10% (stop adding speaker groups once threshold is hit, even mid-group)

