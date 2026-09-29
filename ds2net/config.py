"""
config.py — DS²Net hyper-parameters, one dict per stage, taken from the
notebooks. train.py saves the resolved config next to every run's outputs.
"""

PHASE_NAMES = ("nc", "art", "pvp", "delay")          # phase_0 .. phase_3
HU_WINDOWS  = {"nc": (-100, 200), "art": (-100, 300),
               "pvp": (-100, 250), "delay": (-100, 200)}

AUGMENT = {
    "aug_elastic_prob": 0.20, "aug_elastic_alpha": 100, "aug_elastic_sigma": 10,
    "aug_scale_prob": 0.20, "aug_scale_range": (0.75, 1.25),
    "aug_rotation_prob": 0.30, "aug_rotation_max_deg": 30,
    "aug_mirror_prob": 0.50,           # left-right
    "aug_ud_flip_prob": 0.05,          # up-down is anatomically implausible
    "aug_hu_jitter_prob": 0.15, "aug_hu_jitter": 0.10,
    "aug_noise_prob": 0.15, "aug_noise_std": 0.03,
    "aug_blur_prob": 0.20, "aug_blur_sigma": (0.5, 1.5),
    "aug_gamma_prob": 0.30, "aug_gamma_range": (0.7, 1.5),
}

# Stage 1 — liver (notebooks/liver_segmentation.ipynb)
LIVER = {
    "task": "liver",
    "n_context_slices": 5,             # 5 slices × 4 phases = 20 channels
    "img_size": 224,
    "target_spacing": 1.0,             # mm, all three axes (as in the notebook)
    "slices": "liver",                 # train on slices containing liver
    "batch_size": 48, "grad_accum_steps": 4,
    "lr": 1e-4, "weight_decay": 1e-4, "epochs": 60, "pct_start": 0.20,
    "early_stop_patience": 15, "val_every": 3,
    "pos_weight": 5.0, "boundary_weight": 0.3, "max_grad_norm": 1.0,
    "mha_heads": 8, "mha_dropout": 0.1,
    "seg_threshold": 0.5,
    "min_component_voxels": 1000,      # 3-D connected-component filter (liver)
    **AUGMENT,
}

# Stage 2 — tumor (notebooks/segmentation_classification.ipynb, cell 0)
TUMOR = {
    "task": "tumor",
    "n_context_slices": 3,             # 3 slices × 4 phases = 12 channels
    "img_size": 224,
    "target_spacing": 1.0,
    # Slices used for training/validation. "tumor" matches the notebook (its
    # mask was tumor-only, so it only ever saw tumor slices); "liver" also
    # includes tumor-free liver slices.
    "slices": "tumor",
    "batch_size": 64, "grad_accum_steps": 3,
    "lr": 2e-4, "weight_decay": 1e-4, "epochs": 50, "pct_start": 0.15,
    "early_stop_patience": 12, "val_every": 3,
    "pos_weight": 10.0, "pos_weight_warmup_extra": 5.0,
    "boundary_weight": 0.5, "stb_weight": 0.4, "stb_infer_weight": 0.3,
    "max_grad_norm": 1.0,
    "difficulty_multiplier": {"BCLM": 2.0, "CRLM": 2.0, "ICC": 1.5, "HCC": 1.0, "HH": 0.8},
    "seg_threshold": 0.5,
    "min_component_voxels": 0,
    **AUGMENT,
}

# Stage 3 — classification (notebooks/segmentation_classification.ipynb, cell 1)
CLS = {
    "task": "cls",
    "roi_margin_mm": 10, "roi_size": 224, "max_slices": 16,
    "min_tumor_voxels": 50, "bg_ring_width": 8,
    "batch_size": 128, "lr": 3e-4, "epochs": 50, "patience": 12,
    "unfreeze_epoch": 10, "label_smoothing": 0.1, "dropout": 0.3,
    "use_curve_features": True, "curve_embed_dim": 32,
}

STAGES = {1: LIVER, 2: TUMOR, 3: CLS}
