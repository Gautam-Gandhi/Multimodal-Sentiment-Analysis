"""
MELD dataset loader for Phase 2.

Loads pre-extracted MM-Align pickle files (dict format) and raw CSV text for BERT.

Pickle format:
    key: "{dialogue_id}_{utterance_id}"
    value: {
        "token_ids":       list[float]  — GloVe word indices, length = seq_len
        "audio_features":  ndarray (seq_len, 32)  — Wave2Vec2.0, word-aligned
        "video_features":  ndarray (seq_len, 2048) — ResNet-101, word-aligned
        "label":           int (0–6) — emotion class
    }

CSV columns: Sr No., Utterance, Speaker, Emotion, Sentiment, Dialogue_ID, Utterance_ID, ...

Label mapping: 0=Neutral, 1=Surprise, 2=Fear, 3=Sadness, 4=Joy, 5=Disgust, 6=Anger
"""

import csv
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


EMOTION_LABELS = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
NUM_CLASSES = 7
PAD_TOKEN_ID = 0  # used for padding GloVe token_ids in collate

# CSV file names per split
CSV_FILES = {'train': 'train_sent_emo.csv', 'dev': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def _load_csv_text(csv_path):
    """Load raw utterance text from MELD CSV, keyed by '{Dialogue_ID}_{Utterance_ID}'."""
    lookup = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
            lookup[key] = row['Utterance']
    return lookup


class MELDDataset(Dataset):
    def __init__(self, pkl_path, csv_path):
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        text_lookup = _load_csv_text(csv_path)

        # Convert dict → list, attach raw text
        self.keys = list(raw.keys())
        self.data = []
        for k in self.keys:
            sample = raw[k]
            sample['text'] = text_lookup.get(k, '')
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):
    """
    Collate a list of MELD samples into padded tensors.

    Returns
    -------
    token_ids       : (batch, max_seq)         LongTensor — GloVe word indices, padded
    audio           : (batch, max_seq, 32)     FloatTensor — padded
    video           : (batch, max_seq, 2048)   FloatTensor — padded
    labels          : (batch,)                 LongTensor — emotion class [0-6]
    lengths         : (batch,)                 LongTensor — unpadded sequence lengths
    bert_ids        : (batch, 64)              LongTensor — BERT input_ids
    bert_mask       : (batch, 64)              LongTensor — BERT attention_mask
    bert_type_ids   : (batch, 64)              LongTensor — BERT token_type_ids
    """
    # Sort by descending sequence length
    batch = sorted(batch, key=lambda x: len(x['token_ids']), reverse=True)

    lengths = torch.LongTensor([len(s['token_ids']) for s in batch])
    labels = torch.LongTensor([s['label'] for s in batch])

    # Pad GloVe sequences (kept for backward compatibility / ablation)
    token_ids = pad_sequence(
        [torch.LongTensor([int(x) for x in s['token_ids']]) for s in batch],
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )
    audio = pad_sequence(
        [torch.FloatTensor(s['audio_features']) for s in batch],
        batch_first=True,
        padding_value=0.0,
    )
    video = pad_sequence(
        [torch.FloatTensor(np.asarray(s['video_features'], dtype=np.float32)) for s in batch],
        batch_first=True,
        padding_value=0.0,
    )

    # BERT tokenization from raw text
    bert_details = []
    for s in batch:
        encoded = bert_tokenizer(
            s['text'],
            max_length=64,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
        )
        bert_details.append(encoded)

    bert_ids = torch.LongTensor([d['input_ids'] for d in bert_details])
    bert_mask = torch.LongTensor([d['attention_mask'] for d in bert_details])
    bert_type_ids = torch.LongTensor([d['token_type_ids'] for d in bert_details])

    return token_ids, audio, video, labels, lengths, bert_ids, bert_mask, bert_type_ids


def get_loader(config, split, shuffle=True, generator=None, worker_init_fn=None):
    """
    Build a DataLoader for a MELD split.

    Parameters
    ----------
    config : SimpleNamespace with `data_dir` and `raw_csv_dir`
    split  : 'train', 'dev', or 'test'
    """
    pkl_path = f"{config.data_dir}/{split}.pkl"
    csv_path = f"{config.raw_csv_dir}/{CSV_FILES[split]}"
    dataset = MELDDataset(pkl_path, csv_path)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return loader
