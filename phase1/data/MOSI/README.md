# CMU-MOSI Dataset — Preprocessed PKL Files

## Overview

This folder contains the **CMU-MOSI** (Multimodal Opinion Sentiment and Subjectivity) dataset in preprocessed pickle format, ready for use with the multimodal sentiment analysis model described in:

> *"Multimodal sentiment analysis based on multi-layer feature fusion and multi-task learning"* (Scientific Reports, 2025)

---

## Files

| File | Samples | Size (approx.) |
|---|---|---|
| `train.pkl` | 1283 | ~7.7 MB |
| `dev.pkl` | 229 | ~1.3 MB |
| `test.pkl` | 686 | ~4.7 MB |

> Note: The paper reports 1284 training samples; one sample appears to have been dropped in this preprocessed version.

---

## Feature Dimensions

| Modality | Tool | Dimensions | Notes |
|---|---|---|---|
| Text | BERT-base-uncased | 768-dim (after encoding) | Raw words stored; BERT tokenization done at load time |
| Visual | Facet 4.1 | **47-dim** | Word-level aligned |
| Audio | COVAREP | **74-dim** | Word-level aligned |

Visual and acoustic features are **word-level aligned** — the sequence length equals the number of words in the utterance, and both modalities always share the same sequence length within a sample.

---

## File Format

Each `.pkl` file is a Python `list` of tuples. Every tuple has exactly three elements:

```
(features_tuple, label, segment_id)
```

### `features_tuple` — a 4-element tuple

| Index | Variable | Type | Shape | Description |
|---|---|---|---|---|
| `[0]` | `words` | `np.ndarray` | `(seq_len,)` | GloVe word indices (legacy, not used by BERT model) |
| `[1]` | `visual` | `np.ndarray` | `(seq_len, 47)` | Facet 4.1 visual features, `float32` |
| `[2]` | `acoustic` | `np.ndarray` | `(seq_len, 74)` | COVAREP audio features, `float32` |
| `[3]` | `actual_words` | `list[str]` | `(seq_len,)` | Raw word strings (used for BERT tokenization) |

### `label`

- Type: `np.ndarray`, shape `(1, 1)`, dtype `float32`
- Sentiment score in the range **[-3, 3]**
  - Strongly negative → -3
  - Neutral → 0
  - Strongly positive → +3

### `segment_id`

- Type: `str`
- Example: `'03bSnISJMiM[0]'`
- YouTube video ID + clip index; useful for debugging but not required for training

---

## Sequence Length Statistics (Train Set)

| Modality | Min | Max | Mean |
|---|---|---|---|
| Visual / Acoustic | 1 | ~102 | ~11.7 |

Both modalities always have the same `seq_len` per sample (word-aligned).

---

## How to Load Manually

```python
import pickle

with open('data/MOSI/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Access one sample
(words, visual, acoustic, actual_words), label, segment = train_data[0]

print(visual.shape)       # e.g. (5, 47)
print(acoustic.shape)     # e.g. (5, 74)
print(actual_words)       # ['it', "'s", 'a', 'good', 'movie']
print(label)              # [[2.4]]
print(segment)            # '03bSnISJMiM[0]'
```

---

## How to Use with the Project DataLoader

The `data_loader.py` file wraps these pkl files via `create_dataset.py` and exposes a standard PyTorch `DataLoader`.

### Minimal Config

```python
from types import SimpleNamespace

config = SimpleNamespace(
    data_dir  = 'data/MOSI',   # must contain "mosi" (case-insensitive)
    mode      = 'train',        # 'train', 'dev', or 'test'
    batch_size = 32,
)
```

### Getting a DataLoader

```python
from data_loader import get_loader

train_loader = get_loader(config, shuffle=True)
dev_loader   = get_loader(SimpleNamespace(**vars(config), mode='dev'),   shuffle=False)
test_loader  = get_loader(SimpleNamespace(**vars(config), mode='test'),  shuffle=False)
```

> After calling `get_loader`, the config object is **mutated** to add:
> - `config.visual_size` → `47`
> - `config.acoustic_size` → `74`
> - `config.word2id` → word-to-index dict
> - `config.pretrained_emb` → pretrained embeddings array (or `None` if no GloVe path given)
> - `config.data_len` → number of samples in the split

### Batch Structure

Each batch returned by the DataLoader is an **8-tuple**:

```python
sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask = batch
```

| Variable | Shape | dtype | Description |
|---|---|---|---|
| `sentences` | `[max_seq, batch]` | `int64` | GloVe word indices, padded |
| `visual` | `[max_seq, batch, 47]` | `float32` | Visual features, padded to longest seq in batch |
| `acoustic` | `[max_seq, batch, 74]` | `float32` | Audio features, padded to longest seq in batch |
| `labels` | `[batch]` | `float32` | Sentiment scores in [-3, 3] |
| `lengths` | `[batch]` | `int64` | Actual (unpadded) sequence lengths, sorted descending |
| `bert_sentences` | `[batch, 52]` | `int64` | BERT input token IDs (CLS + up to 50 tokens + SEP) |
| `bert_sentence_types` | `[batch, 52]` | `int64` | BERT token type IDs (all zeros for single-sentence) |
| `bert_sentence_att_mask` | `[batch, 52]` | `int64` | BERT attention mask (1=real token, 0=padding) |

> `visual` and `acoustic` are `[seq, batch, feat]` (sequence-first, not batch-first). Transpose to `[batch, seq, feat]` before passing to your model:
> ```python
> visual   = visual.permute(1, 0, 2)    # [batch, max_seq, 47]
> acoustic = acoustic.permute(1, 0, 2)  # [batch, max_seq, 74]
> ```

### BERT Tokenization Details

- Tokenizer: `bert-base-uncased`
- Fixed length: **52** tokens (= 50 content tokens + `[CLS]` + `[SEP]`)
- Longer sequences are truncated; shorter ones are padded with `[PAD]` tokens (mask=0)
- Input to BERT: `bert_sentences`, `bert_sentence_att_mask`, `bert_sentence_types`

---

## Integration with the Model (UFEN + MTFN)

The three modalities flow into the model as:

```
bert_sentences ──► BERT ──────────────────► T  [batch, 52, 768]
visual ───────────────────────────────────► V  [batch, max_seq, 47]
acoustic ─────────────────────────────────► A  [batch, max_seq, 74]
```

All three are then processed by UFEN (Bi-GRU + conv + self-attention) and fused by MTFN (cross-modal attention + Encoder-Decoder).

During training, pass `lengths` to `pack_padded_sequence` inside the Bi-GRU to avoid computing over padding tokens.

---

## Label Thresholds for Evaluation

The paper evaluates on multiple metrics using the following thresholds:

| Metric | Threshold |
|---|---|
| Acc-2 (non-neg vs neg) | `label >= 0` → positive |
| Acc-2 (pos vs neg) | `label > 0` → positive (neutral excluded) |
| Acc-7 | Round to nearest integer in [-3, 3], map to 7 classes |
| MAE | Mean absolute error of raw regression output |
| Pearson Corr | Pearson correlation between predictions and labels |

---

## Notes

- The pre-processed pkl files are sourced from https://github.com/declare-lab/MISA.
- Feature extraction pipeline follows the **ATCN** paper's preprocessing: word-level P2FA alignment, Facet 4.1 for visual, COVAREP for audio.
- GloVe word indices (`words` field) are included for legacy compatibility but are **not used** by this project's BERT-based model.
