"""
Phase 2 configuration — MELD 7-class emotion classification.
"""

from types import SimpleNamespace

EMOTION_LABELS = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']

config = SimpleNamespace(
    # --- data ---
    data_dir='data/MELD/meld',
    raw_csv_dir='data/MELD/meld/raw',
    embedding_path='data/MELD/meld/embedding.p',
    batch_size=16,         # reduced from 32 to fit BERT on RTX 3050 (3.68 GiB)

    # --- model dimensions (Exp 2 base — best so far) ---
    d_m=128,
    conv_dim=64,
    n_layers=2,
    kernel_sizes=[1, 5],
    d_ff=128,
    num_classes=7,

    # --- feature dimensions ---
    text_size=300,        # GloVe (unused when use_bert=True)
    visual_size=2048,     # ResNet-101
    visual_proj_dim=256,  # project 2048→256 before BiGRU to reduce VRAM
    acoustic_size=32,     # Wave2Vec2.0

    # --- modality selection (for ablation studies) ---
    # Full 3-modal: ['text', 'audio', 'video']
    # Bimodal ablations: ['text', 'audio'] | ['text', 'video'] | ['audio', 'video']
    modalities=['text'],            # Ablation Exp 7c: text only (BERT unimodal)

    # --- text encoder ---
    use_bert=True,        # BERT contextual embeddings (768-dim)

    # --- attention ---
    self_att_heads=1,
    cross_att_heads=4,
    att_dropout=0.2,
    dropout=0.1,

    # --- training ---
    lr=1e-3,              # non-BERT params
    lr_bert=2e-5,         # BERT fine-tuning (from Phase 1 best P3-2)
    epochs=50,
    early_stop=10,
    grad_clip=1.0,
    use_lr_scheduler=True,
    use_bert_warmup=True,
    bert_warmup_epochs=3, # fewer than Phase 1 (5) — MELD has 8x more data per epoch
    label_smoothing=0.1,

    # --- focal loss ---
    use_focal_loss=False,
    focal_gamma=2.0,

    # --- class imbalance ---
    use_class_weights=True,

    # --- set at runtime ---
    pretrained_emb=None,
)
