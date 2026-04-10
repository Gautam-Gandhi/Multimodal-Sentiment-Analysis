"""
Training and evaluation script for Phase 2 — MELD 7-class emotion classification.

Usage:
    python phase2/train.py

Supports both GloVe (Exp 1-5) and BERT (Exp 6+) text encoders via config.use_bert.
Supports bimodal ablation via config.modalities (e.g. ['text', 'audio']).
"""

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from phase2.config import config, EMOTION_LABELS
from phase2.data_loader import get_loader, NUM_CLASSES
from phase2.model import MultiTaskModel


class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == 'mean' else focal.sum()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def make_padding_mask(lengths, max_len):
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def unpack_batch(batch, device, use_bert=False):
    """
    Unpack collated MELD batch, move to device, build padding mask.
    """
    token_ids, audio, video, labels, lengths = batch[:5]

    token_ids = token_ids.long().to(device)
    audio = audio.float().to(device)
    video = video.float().to(device)
    labels = labels.long().to(device)
    lengths = lengths.long().to(device)

    max_len = token_ids.size(1)
    av_mask = make_padding_mask(lengths, max_len)

    result = dict(token_ids=token_ids, audio=audio, video=video,
                  labels=labels, av_mask=av_mask)

    if use_bert:
        bert_ids, bert_mask, bert_type_ids = batch[5], batch[6], batch[7]
        result['bert_ids'] = bert_ids.long().to(device)
        result['bert_mask'] = bert_mask.long().to(device)
        result['bert_type_ids'] = bert_type_ids.long().to(device)

    return result


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    acc = accuracy_score(labels, preds) * 100
    f1_w = f1_score(labels, preds, average='weighted', zero_division=0) * 100
    f1_m = f1_score(labels, preds, average='macro', zero_division=0) * 100
    return {'Accuracy': acc, 'F1_weighted': f1_w, 'F1_macro': f1_m}


def print_full_report(preds, labels):
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=EMOTION_LABELS, zero_division=0))
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(labels, preds)
    header = "          " + " ".join(f"{EMOTION_LABELS[i][:4]:>5}" for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = " ".join(f"{cm[i, j]:5d}" for j in range(NUM_CLASSES))
        print(f"{EMOTION_LABELS[i]:<10}{row}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, use_bert=False):
    model.eval()
    use_amp = torch.cuda.is_available()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            b = unpack_batch(batch, device, use_bert)
            with autocast('cuda', enabled=use_amp):
                preds_dict, _ = model(
                    b['token_ids'], b['audio'], b['video'], b['av_mask'],
                    b.get('bert_ids'), b.get('bert_mask'), b.get('bert_type_ids'),
                )
            preds = preds_dict['recon'].argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(b['labels'].cpu().numpy())

    return compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels)), \
           np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Compute class weights from training set
# ---------------------------------------------------------------------------

def compute_class_weights(loader):
    counts = np.zeros(NUM_CLASSES)
    for batch in loader:
        for l in batch[3].numpy():
            counts[l] += 1
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.mean()
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_bert = getattr(config, 'use_bert', False)
    modalities = getattr(config, 'modalities', ['text', 'audio', 'video'])
    print(f"Modalities: {modalities}")

    # -----------------------------------------------------------------------
    # Load GloVe embeddings (only needed in GloVe mode)
    # -----------------------------------------------------------------------
    if not use_bert and 'text' in modalities:
        import pickle
        with open(config.embedding_path, 'rb') as f:
            config.pretrained_emb = pickle.load(f)
        print(f"GloVe embeddings: {config.pretrained_emb.shape}")
    elif use_bert:
        print("Using BERT for text encoding")

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = get_loader(config, 'train', shuffle=True, generator=g, worker_init_fn=_worker_init_fn)
    dev_loader = get_loader(config, 'dev', shuffle=False)
    test_loader = get_loader(config, 'test', shuffle=False)

    print(f"Train={len(train_loader.dataset)}, Dev={len(dev_loader.dataset)}, Test={len(test_loader.dataset)}")

    # -----------------------------------------------------------------------
    # Class weights
    # -----------------------------------------------------------------------
    if config.use_class_weights:
        class_weights = compute_class_weights(train_loader).to(device)
        print(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        class_weights = None

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = MultiTaskModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer — differentiated LR for BERT vs rest
    # -----------------------------------------------------------------------
    if use_bert and 'text' in modalities:
        bert_params = list(model.bert.parameters())
        bert_ids_set = set(id(p) for p in bert_params)
        other_params = [p for p in model.parameters() if id(p) not in bert_ids_set]

        optimizer = Adam([
            {'params': bert_params, 'lr': config.lr_bert},
            {'params': other_params, 'lr': config.lr},
        ])

        warmup = config.bert_warmup_epochs if config.use_bert_warmup else 0

        def bert_lr_lambda(t):
            if config.use_bert_warmup and t < warmup:
                return (t + 1) / warmup
            if config.use_lr_scheduler:
                progress = (t - warmup) / max(1, config.epochs - warmup)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0

        def other_lr_lambda(t):
            if config.use_lr_scheduler:
                progress = t / max(1, config.epochs)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=[bert_lr_lambda, other_lr_lambda])
        print(f"BERT lr={config.lr_bert}, other lr={config.lr}, warmup={warmup} epochs")
    else:
        optimizer = Adam(model.parameters(), lr=config.lr)
        if config.use_lr_scheduler:
            def lr_lambda(t):
                progress = t / max(1, config.epochs)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = None

    # -----------------------------------------------------------------------
    # Loss function
    # -----------------------------------------------------------------------
    if getattr(config, 'use_focal_loss', False):
        ce_loss = FocalLoss(
            weight=class_weights,
            gamma=getattr(config, 'focal_gamma', 2.0),
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )
        print(f"Using Focal Loss (gamma={config.focal_gamma})")
    else:
        ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=getattr(config, 'label_smoothing', 0.0),
        )
        print("Using CrossEntropyLoss")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_dev_f1 = 0.0
    best_epoch = 0
    os.makedirs('checkpoints', exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False,
                    unit='batch', dynamic_ncols=True)

        for batch in pbar:
            b = unpack_batch(batch, device, use_bert)

            with autocast('cuda', enabled=use_amp):
                preds_dict, _ = model(
                    b['token_ids'], b['audio'], b['video'], b['av_mask'],
                    b.get('bert_ids'), b.get('bert_mask'), b.get('bert_type_ids'),
                )

                # Sum CE losses for all active modalities + fusion + recon
                loss = sum(ce_loss(preds_dict[k], b['labels'])
                           for k in list(modalities) + ['fusion', 'recon'])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        dev_metrics, _, _ = evaluate(model, dev_loader, device, use_bert)

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
              f"Acc={dev_metrics['Accuracy']:.1f} | "
              f"F1w={dev_metrics['F1_weighted']:.1f} | "
              f"F1m={dev_metrics['F1_macro']:.1f}")

        if dev_metrics['F1_weighted'] > best_dev_f1:
            best_dev_f1 = dev_metrics['F1_weighted']
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/best_model_meld.pt')
            print(f"  -> New best Dev F1w={best_dev_f1:.2f} — model saved.")

        elif (epoch - best_epoch) >= config.early_stop:
            print(f"Early stopping: no improvement for {config.early_stop} epochs.")
            break

    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------
    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load('checkpoints/best_model_meld.pt', map_location=device))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, use_bert)

    print("\n========== Test Results ==========")
    print(f"Accuracy       : {test_metrics['Accuracy']:.1f}")
    print(f"F1 (weighted)  : {test_metrics['F1_weighted']:.1f}")
    print(f"F1 (macro)     : {test_metrics['F1_macro']:.1f}")
    print("==================================")

    print_full_report(test_preds, test_labels)


if __name__ == '__main__':
    train()
