"""
Enhanced training script for Phase 2 — Exp 8.

Adds three research-backed improvements over train.py (Exp 6):
  1. OGM-GE gradient modality balancing  (Peng et al. 2022)
  2. Cross-modal contrastive alignment loss (InfoNCE)
  3. Scheduled modality training (audio/video pre-train → full joint)

Usage:
    python phase2/train_enhanced.py
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
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from phase2.config import config, EMOTION_LABELS
from phase2.data_loader import get_loader, NUM_CLASSES
from phase2.model import MultiTaskModel, masked_mean


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
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def unpack_batch(batch, device, use_bert=False):
    token_ids, audio, video, labels, lengths = batch[:5]
    token_ids = token_ids.long().to(device)
    audio = audio.float().to(device)
    video = video.float().to(device)
    labels = labels.long().to(device)
    lengths = lengths.long().to(device)
    av_mask = make_padding_mask(lengths, token_ids.size(1))
    result = dict(token_ids=token_ids, audio=audio, video=video,
                  labels=labels, av_mask=av_mask)
    if use_bert:
        result['bert_ids'] = batch[5].long().to(device)
        result['bert_mask'] = batch[6].long().to(device)
        result['bert_type_ids'] = batch[7].long().to(device)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels):
    return {
        'Accuracy': accuracy_score(labels, preds) * 100,
        'F1_weighted': f1_score(labels, preds, average='weighted', zero_division=0) * 100,
        'F1_macro': f1_score(labels, preds, average='macro', zero_division=0) * 100,
    }


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
# Evaluation
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
            all_preds.append(preds_dict['recon'].argmax(dim=1).cpu().numpy())
            all_labels.append(b['labels'].cpu().numpy())
    return compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels)), \
           np.concatenate(all_preds), np.concatenate(all_labels)


def compute_class_weights(loader):
    counts = np.zeros(NUM_CLASSES)
    for batch in loader:
        for l in batch[3].numpy():
            counts[l] += 1
    w = 1.0 / np.clip(counts, 1, None)
    return torch.FloatTensor(w / w.mean())


# ---------------------------------------------------------------------------
# [CHANGE 2] Cross-modal contrastive alignment loss (InfoNCE)
# ---------------------------------------------------------------------------

def contrastive_loss(z1, z2, temperature=0.07):
    """
    InfoNCE loss between two sets of embeddings.
    Pulls same-sample pairs together, pushes different-sample pairs apart.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature                # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# [CHANGE 1] OGM-GE: On-the-fly Gradient Modulation (Peng et al. 2022)
# ---------------------------------------------------------------------------

def ogm_ge_modulate(model, modalities, max_scale=10.0):
    """
    Equalize gradient norms across modality UFEN branches.
    Scale weaker-modality gradients up to match the strongest modality.
    """
    norms = {}
    for m in modalities:
        ufen = getattr(model, f'ufen_{m[0]}', None)  # ufen_t, ufen_v, ufen_a
        if ufen is None:
            continue
        grads = [p.grad.flatten() for p in ufen.parameters() if p.grad is not None]
        if grads:
            norms[m] = torch.cat(grads).norm().item()

    if len(norms) < 2:
        return {}

    max_norm = max(norms.values())
    coeffs = {}
    for m, n in norms.items():
        coeff = min(max_norm / (n + 1e-8), max_scale)
        coeffs[m] = coeff
        if coeff > 1.05:   # only scale if meaningfully different
            ufen = getattr(model, f'ufen_{m[0]}')
            for p in ufen.parameters():
                if p.grad is not None:
                    p.grad.mul_(coeff)

    return coeffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_bert = getattr(config, 'use_bert', False)
    modalities = getattr(config, 'modalities', ['text', 'audio', 'video'])
    print(f"Modalities: {modalities}")

    # --- [CHANGE 3] Scheduled training config ---
    # Phase A: freeze BERT+text, train audio/video only (epochs 1-3)
    # Phase B: unfreeze all, train unimodal losses only (epochs 4-6)
    # Phase C: full end-to-end with fusion + contrastive (epoch 7+)
    PHASE_A_END = 3    # audio/video pre-training
    PHASE_B_END = 6    # all unimodal, no fusion yet
    # Phase C = PHASE_B_END+1 .. config.epochs

    # Contrastive loss weight
    LAMBDA_CONTRASTIVE = 0.1

    print(f"Scheduled training: Phase A (epochs 1-{PHASE_A_END}) audio/video only, "
          f"Phase B ({PHASE_A_END+1}-{PHASE_B_END}) all unimodal, "
          f"Phase C ({PHASE_B_END+1}+) full model + contrastive")
    print(f"Contrastive loss weight: {LAMBDA_CONTRASTIVE}")
    print(f"OGM-GE gradient balancing: enabled")

    # --- Load data ---
    if not use_bert and 'text' in modalities:
        import pickle
        with open(config.embedding_path, 'rb') as f:
            config.pretrained_emb = pickle.load(f)
    elif use_bert:
        print("Using BERT for text encoding")

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = get_loader(config, 'train', shuffle=True, generator=g, worker_init_fn=_worker_init_fn)
    dev_loader = get_loader(config, 'dev', shuffle=False)
    test_loader = get_loader(config, 'test', shuffle=False)
    print(f"Train={len(train_loader.dataset)}, Dev={len(dev_loader.dataset)}, Test={len(test_loader.dataset)}")

    class_weights = compute_class_weights(train_loader).to(device) if config.use_class_weights else None
    if class_weights is not None:
        print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # --- Model ---
    model = MultiTaskModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Optimizer: differentiated LR ---
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
            scheduler = LambdaLR(optimizer, lambda t: max(0.0, 0.5 * (1.0 + math.cos(math.pi * t / max(1, config.epochs)))))
        else:
            scheduler = None

    # --- Loss ---
    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=getattr(config, 'label_smoothing', 0.0),
    )
    print("Using CrossEntropyLoss")

    # --- Training loop ---
    best_dev_f1 = 0.0
    best_epoch = 0
    os.makedirs('checkpoints', exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        # ---- [CHANGE 3] Determine training phase & freeze/unfreeze ----
        if epoch <= PHASE_A_END:
            phase = 'A'
            # Freeze text (BERT + UFEN_t + norm_t), train audio/video only
            if use_bert and hasattr(model, 'bert'):
                for p in model.bert.parameters():
                    p.requires_grad = False
            if hasattr(model, 'ufen_t'):
                for p in model.ufen_t.parameters():
                    p.requires_grad = False
            if hasattr(model, 'norm_t'):
                for p in model.norm_t.parameters():
                    p.requires_grad = False
            # Freeze MTFN in phase A
            if model.mtfn is not None:
                for p in model.mtfn.parameters():
                    p.requires_grad = False

        elif epoch == PHASE_A_END + 1:
            phase = 'B'
            # Unfreeze text, keep MTFN frozen
            if use_bert and hasattr(model, 'bert'):
                for p in model.bert.parameters():
                    p.requires_grad = True
            if hasattr(model, 'ufen_t'):
                for p in model.ufen_t.parameters():
                    p.requires_grad = True
            if hasattr(model, 'norm_t'):
                for p in model.norm_t.parameters():
                    p.requires_grad = True
            # MTFN still frozen in phase B

        elif epoch == PHASE_B_END + 1:
            phase = 'C'
            # Unfreeze everything
            if model.mtfn is not None:
                for p in model.mtfn.parameters():
                    p.requires_grad = True

        else:
            phase = 'A' if epoch <= PHASE_A_END else ('B' if epoch <= PHASE_B_END else 'C')

        phase_desc = {'A': 'audio/video pre-train', 'B': 'all unimodal', 'C': 'full + contrastive'}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [{phase}]", leave=False,
                    unit='batch', dynamic_ncols=True)

        for batch in pbar:
            b = unpack_batch(batch, device, use_bert)

            with autocast('cuda', enabled=use_amp):
                preds_dict, pooled = model(
                    b['token_ids'], b['audio'], b['video'], b['av_mask'],
                    b.get('bert_ids'), b.get('bert_mask'), b.get('bert_type_ids'),
                )

                # --- Compute task losses based on phase ---
                if phase == 'A':
                    # Only audio + video unimodal losses
                    active_keys = [m for m in modalities if m != 'text']
                    loss = sum(ce_loss(preds_dict[k], b['labels']) for k in active_keys)
                elif phase == 'B':
                    # All unimodal losses, no fusion/recon
                    loss = sum(ce_loss(preds_dict[k], b['labels']) for k in modalities)
                else:
                    # Phase C: all losses (unimodal + fusion + recon)
                    loss = sum(ce_loss(preds_dict[k], b['labels'])
                               for k in list(modalities) + ['fusion', 'recon'])

                # --- [CHANGE 2] Contrastive alignment loss (Phase B onward) ---
                if phase in ('B', 'C') and len(pooled) >= 2:
                    L_align = 0.0
                    if 'text' in pooled and 'audio' in pooled:
                        L_align = L_align + contrastive_loss(pooled['text'], pooled['audio'])
                    if 'text' in pooled and 'video' in pooled:
                        L_align = L_align + contrastive_loss(pooled['text'], pooled['video'])
                    if 'audio' in pooled and 'video' in pooled:
                        L_align = L_align + contrastive_loss(pooled['audio'], pooled['video'])
                    loss = loss + LAMBDA_CONTRASTIVE * L_align

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # --- [CHANGE 1] OGM-GE gradient modulation ---
            if phase in ('B', 'C'):
                ogm_ge_modulate(model, modalities, max_scale=10.0)

            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        dev_metrics, _, _ = evaluate(model, dev_loader, device, use_bert)

        print(f"Epoch {epoch:02d} [{phase}: {phase_desc[phase]}] | loss={avg_loss:.4f} | "
              f"Acc={dev_metrics['Accuracy']:.1f} | "
              f"F1w={dev_metrics['F1_weighted']:.1f} | "
              f"F1m={dev_metrics['F1_macro']:.1f}")

        if dev_metrics['F1_weighted'] > best_dev_f1:
            best_dev_f1 = dev_metrics['F1_weighted']
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/best_model_enhanced.pt')
            print(f"  -> New best Dev F1w={best_dev_f1:.2f} — model saved.")

        elif (epoch - best_epoch) >= config.early_stop:
            print(f"Early stopping: no improvement for {config.early_stop} epochs.")
            break

    # --- Test ---
    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load('checkpoints/best_model_enhanced.pt', map_location=device))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, use_bert)

    print("\n========== Test Results ==========")
    print(f"Accuracy       : {test_metrics['Accuracy']:.1f}")
    print(f"F1 (weighted)  : {test_metrics['F1_weighted']:.1f}")
    print(f"F1 (macro)     : {test_metrics['F1_macro']:.1f}")
    print("==================================")
    print_full_report(test_preds, test_labels)


if __name__ == '__main__':
    train()
