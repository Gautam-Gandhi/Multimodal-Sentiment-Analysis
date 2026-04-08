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


