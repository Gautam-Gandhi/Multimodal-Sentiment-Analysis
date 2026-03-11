"""
Training and evaluation script.

Usage:
    python train.py

Evaluation metrics (MOSI/MOSEI, following the paper):
    Acc-2  (two variants reported as "nn / np"):
        nn = negative / non-negative  (threshold = 0)
        np = negative / positive       (neutral samples excluded)
    Acc-7  = 7-class accuracy (round predictions to nearest int, clip to [-3, 3])
    F1     = weighted F1 for each Acc-2 variant
    MAE    = mean absolute error
    Corr   = Pearson correlation

Best model is selected by lowest Dev MAE.
Final report uses Y_m' (encoder-decoder output) as the sentiment prediction.
"""

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

# Make sure project root is on path when running from a subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_loader
from model import MultiTaskModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure cuDNN uses the same algorithm every run (slight speed cost, full determinism)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Force deterministic CUDA ops where available
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Disable the memory-efficient SDPA backend — it is non-deterministic on CUDA
    # and triggers a warning even when warn_only=True above.
    # Falls back to the math backend which is fully deterministic.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def _worker_init_fn(worker_id):
    """Give each DataLoader worker a unique but reproducible seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def make_padding_mask(lengths, max_len):
    """
    lengths : (batch,) LongTensor of valid sequence lengths
    Returns  : (batch, max_len) bool mask — True = padding position
    """
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    return idx >= lengths.unsqueeze(1)                                # (batch, max_len)


def unpack_batch(batch, device):
    """
    Unpack a DataLoader batch, move to device, build padding masks.

    DataLoader returns:
        sentences   – (T, batch)          word-id sequences  [unused by model]
        visual      – (T, batch, 47)      FACET features
        acoustic    – (T, batch, 74)      COVAREP features
        labels      – (batch, 1)          sentiment scores
        lengths     – (batch,)            actual word-level sequence lengths
        bert_ids    – (batch, 52)
        bert_types  – (batch, 52)
        bert_mask   – (batch, 52)

    Returns a dict with everything needed by the model.
    """
    sentences, visual, acoustic, labels, lengths, bert_ids, bert_types, bert_mask = batch

    # visual / acoustic: (T, batch, feat) → (batch, T, feat)
    visual   = visual.permute(1, 0, 2).float().to(device)
    acoustic = acoustic.permute(1, 0, 2).float().to(device)

    bert_ids   = bert_ids.long().to(device)
    bert_types = bert_types.long().to(device)
    bert_mask  = bert_mask.long().to(device)

    # labels after torch.cat in collate_fn: (batch, 1) → (batch,)
    labels  = labels.squeeze(-1).float().to(device)
    lengths = lengths.long().to(device)

    T_v = visual.size(1)
    T_a = acoustic.size(1)
    v_mask = make_padding_mask(lengths, T_v)   # (batch, T_v)
    a_mask = make_padding_mask(lengths, T_a)   # (batch, T_a)

    return bert_ids, bert_mask, bert_types, visual, v_mask, acoustic, a_mask, labels


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    preds, labels : 1-D numpy arrays of shape (N,)
    """
    mae  = float(np.mean(np.abs(preds - labels)))
    corr = float(pearsonr(preds, labels)[0])

    # Acc-7: round to nearest integer, clip to valid range [-3, 3]
    p7 = np.clip(np.round(preds),  -3, 3).astype(int)
    l7 = np.clip(np.round(labels), -3, 3).astype(int)
    acc7 = float(np.mean(p7 == l7) * 100)

    # Acc-2 variant 1: negative (< 0)  vs. non-negative (>= 0)
    p_nn = (preds  >= 0).astype(int)
    l_nn = (labels >= 0).astype(int)
    acc2_nn = float(np.mean(p_nn == l_nn) * 100)
    f1_nn   = float(f1_score(l_nn, p_nn, average='weighted') * 100)

    # Acc-2 variant 2: negative (< 0)  vs. positive (> 0)  [neutral=0 excluded]
    nz = labels != 0
    if nz.sum() > 0:
        p_np = (preds[nz]  > 0).astype(int)
        l_np = (labels[nz] > 0).astype(int)
        acc2_np = float(np.mean(p_np == l_np) * 100)
        f1_np   = float(f1_score(l_np, p_np, average='weighted') * 100)
    else:
        acc2_np, f1_np = acc2_nn, f1_nn

    return {
        'MAE':     mae,
        'Corr':    corr,
        'Acc7':    acc7,
        'Acc2_nn': acc2_nn,   # negative / non-negative
        'Acc2_np': acc2_np,   # negative / positive
        'F1_nn':   f1_nn,
        'F1_np':   f1_np,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            bert_ids, bert_mask, bert_types, visual, v_mask, acoustic, a_mask, labels = \
                unpack_batch(batch, device)
            # Y_m' is the final sentiment output (encoder-decoder reconstruction)
            *_, y_m_prime = model(bert_ids, bert_mask, bert_types, visual, v_mask, acoustic, a_mask)
            all_preds.append(y_m_prime.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels))


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Config  (hyperparameters from paper Table 2 + documented assumptions)
    # -----------------------------------------------------------------------
    config = SimpleNamespace(
        # --- data ---
        data_dir      = 'data/MOSI',
        dataset_dir   = 'data/MOSI',  # alias used by create_dataset.py
        sdk_dir       = None,         # mmsdk is pip-installed; 
        word_emb_path = None,         # not used when loading pre-computed pkl splits
        batch_size    = 32,           # Table 2

        # --- model dimensions ---
        d_m          = 128,
        conv_dim     = 64,           # restored from P2-3 lesson (32 hurts with lr_bert=2e-5)
        n_layers     = 2,            # Conv1D branches
        kernel_sizes = [1, 5],       # best from Phase 1
        d_ff         = 128,          # best from Phase 1

        # --- attention ---
        self_att_heads  = 1,         # Table 2
        cross_att_heads = 4,         # Table 2
        att_dropout     = 0.2,       # Table 2
        dropout         = 0.1,       # Table 2

        # --- training ---
        lr               = 5e-3,     # Table 2  (non-BERT params)
        lr_bert          = 2e-5,     # assumption: separate lower LR for BERT stability
        epochs           = 50,
        early_stop       = 15,
        grad_clip        = 1.0,      # assumption: standard gradient clipping
        use_lr_scheduler = True,     # cosine anneal both LRs to 0
        use_bert_warmup    = True,     # linearly warm up lr_bert from 0 → lr_bert over bert_warmup_epochs
        bert_warmup_epochs = 5,
        use_adamw          = False,
        weight_decay_bert  = 0.0,
        weight_decay       = 0.0,

        # set by DataLoader:
        visual_size   = None,
        acoustic_size = None,
    )



    # -----------------------------------------------------------------------
    # Data loaders  (seeded generator ensures reproducible shuffle order)
    # -----------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(42)
    config._generator       = g
    config._worker_init_fn  = _worker_init_fn
    config.mode = 'train';  train_loader = get_loader(config, shuffle=True)
    config.mode = 'dev';    dev_loader   = get_loader(config, shuffle=False)
    config.mode = 'test';   test_loader  = get_loader(config, shuffle=False)

    print(f"visual_size={config.visual_size}, acoustic_size={config.acoustic_size}")
    print(f"Train={len(train_loader.dataset)}, "
          f"Dev={len(dev_loader.dataset)}, "
          f"Test={len(test_loader.dataset)}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = MultiTaskModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer — differentiated learning rates
    # -----------------------------------------------------------------------
    bert_params  = list(model.bert.parameters())
    bert_ids_set = set(id(p) for p in bert_params)
    other_params = [p for p in model.parameters() if id(p) not in bert_ids_set]

    OptimizerClass = AdamW if config.use_adamw else Adam
    optimizer = OptimizerClass([
        {'params': bert_params,  'lr': config.lr_bert, 'weight_decay': config.weight_decay_bert},
        {'params': other_params, 'lr': config.lr,       'weight_decay': config.weight_decay},
    ])

    # LR schedule — handles three independent flags:
    #   use_bert_warmup:  linearly ramp lr_bert from 0 → lr_bert over bert_warmup_epochs
    #   use_lr_scheduler: cosine anneal both param groups from their base LR to 0
    # Old experiments (both False) → scheduler=None, LRs stay constant (no change).
    # LambdaLR calls step() on init, setting LR = base_lr * lambda(0) before epoch 1.
    if config.use_lr_scheduler or config.use_bert_warmup:
        warmup = config.bert_warmup_epochs if config.use_bert_warmup else 0

        def bert_lr_lambda(t):
            # t=0 before epoch 1, t=k after epoch k trains
            if config.use_bert_warmup and t < warmup:
                return (t + 1) / warmup          # linear: 1/w, 2/w, ..., w/w=1
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
    else:
        scheduler = None

    mse_loss = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_dev_mae = float('inf')
    best_epoch   = 0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            bert_ids, bert_mask, bert_types, visual, v_mask, acoustic, a_mask, labels = \
                unpack_batch(batch, device)

            y_t, y_v, y_a, y_m, y_m_prime = model(
                bert_ids, bert_mask, bert_types, visual, v_mask, acoustic, a_mask
            )

            # Total loss = sum of 5 MSE terms (Eq 17-18)
            loss = (mse_loss(y_t,       labels) +
                    mse_loss(y_v,       labels) +
                    mse_loss(y_a,       labels) +
                    mse_loss(y_m,       labels) +
                    mse_loss(y_m_prime, labels))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss    = total_loss / len(train_loader)
        dev_metrics = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
              f"MAE={dev_metrics['MAE']:.4f} | Corr={dev_metrics['Corr']:.4f} | "
              f"Acc2={dev_metrics['Acc2_nn']:.1f}/{dev_metrics['Acc2_np']:.1f} | "
              f"Acc7={dev_metrics['Acc7']:.1f}")

        if dev_metrics['MAE'] < best_dev_mae:
            best_dev_mae = dev_metrics['MAE']
            best_epoch   = epoch
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print(f"  -> New best Dev MAE={best_dev_mae:.4f} — model saved.")

        elif (epoch - best_epoch) >= config.early_stop:
            print(f"Early stopping: no improvement for {config.early_stop} epochs.")
            break

    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------
    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n========== Test Results ==========")
    print(f"Acc-2  (neg/non-neg / neg/pos) : {test_metrics['Acc2_nn']:.1f} / {test_metrics['Acc2_np']:.1f}")
    print(f"F1     (neg/non-neg / neg/pos) : {test_metrics['F1_nn']:.1f}  / {test_metrics['F1_np']:.1f}")
    print(f"MAE                            : {test_metrics['MAE']:.3f}")
    print(f"Corr                           : {test_metrics['Corr']:.3f}")
    print(f"Acc-7                          : {test_metrics['Acc7']:.1f}")
    print("==================================")


if __name__ == '__main__':
    train()
