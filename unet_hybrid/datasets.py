"""
datasets.py — HERALD dataset utilities
  fetch_case / fetch_case_cached  — volume loading + LRU cache
  LiverDataset                    — Stage 1: 2.5D liver segmentation
  TumorDataset                    — Stage 2: 2.5D tumor segmentation (GT crop)
  ClassificationDataset           — Stage 3: dual-stream (slice + volume)
  CaseGroupedSampler              — cache-friendly sampler for Stages 1 & 2
"""

import os, random, time
from collections import OrderedDict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, Sampler

# ── Constants (overridden by train.py via module globals) ─────────────────────
LOCAL_DATA_DIR  = "/scratch/teh.c/mct_training/mct_local"
SPLITS_CSV      = "/scratch/teh.c/mct_training/mct_ltdiag_splits.csv"
IMG_SIZE        = 256
HU_MIN, HU_MAX  = -150, 250     # liver window
MAX_CACHED      = 30            # LRU cache capacity
# ── Constants — add phase HU windows ─────────────────────────────────────────
# Each phase has different optimal HU range
PHASE_HU = [
    (-150, 250),   # phase_0 — non-contrast (same as liver window)
    (-150, 250),   # phase_1 — arterial
    (-150, 250),   # phase_2 — portal venous (matches pvp)
    (-150, 250),   # phase_3 — delayed
]
# Names for phase_0..phase_3, in the channel order Stage 1 uses
PHASE_NAMES = ("nc", "art", "pvp", "del")
# ── LRU volume cache ──────────────────────────────────────────────────────────
_cache: "OrderedDict[str, dict]" = OrderedDict()
def _resample_depth(vol, target_d):
    """
    Resample (H, W, D) volume to (H, W, target_d) along depth axis
    using linear interpolation via torch.
    """
    if vol.shape[2] == target_d:
        return vol
    t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    # (1, 1, D, H, W) — interpolate along D
    t = torch.nn.functional.interpolate(
        t.float(), size=(target_d, vol.shape[0], vol.shape[1]),
        mode="trilinear", align_corners=False)
    return t.squeeze().permute(1, 2, 0).numpy().astype(np.float32)
def fetch_case(cid: str) -> dict:
    """
    Load one case from disk.
    Returns dict with keys:
      pvp        — HU-windowed float32 array (H, W, D)
      phases     — list of 4 HU-windowed float32 arrays (H, W, D)
                   resampled to match pvp depth
      phase_vols — the same 4 arrays keyed by PHASE_NAMES
      liver_mask — binary uint8 array (H, W, D)
      tumor_mask — binary uint8 array (H, W, D)
    """
    case_dir = os.path.join(LOCAL_DATA_DIR, cid)

    # load pvp and masks (reference depth)
    pvp   = nib.load(os.path.join(case_dir, "pvp.nii.gz")).get_fdata(dtype=np.float32)
    liver = nib.load(os.path.join(case_dir, "liver_mask.nii.gz")).get_fdata()
    tumor = nib.load(os.path.join(case_dir, "tumor_mask.nii.gz")).get_fdata()

    # HU windowing for pvp → [0, 1]
    pvp = np.clip(pvp, HU_MIN, HU_MAX)
    pvp = (pvp - HU_MIN) / (HU_MAX - HU_MIN)

    target_d = pvp.shape[2]

    # load and resample all 4 phases to match pvp depth
    phases = []
    for i, (hu_min, hu_max) in enumerate(PHASE_HU):
        phase_path = os.path.join(case_dir, f"phase_{i}.nii.gz")
        if os.path.exists(phase_path):
            ph = nib.load(phase_path).get_fdata(dtype=np.float32)
            ph = np.clip(ph, hu_min, hu_max)
            ph = (ph - hu_min) / (hu_max - hu_min)
            ph = _resample_depth(ph, target_d)
        else:
            # fallback to pvp if phase missing
            print(f"  WARNING: {cid} phase_{i} missing, using pvp")
            ph = pvp.copy()
        phases.append(ph)

    return {
        "pvp":        pvp.astype(np.float32),
        "phases":     phases,
        "phase_vols": dict(zip(PHASE_NAMES, phases)),
        "liver_mask": (liver > 0).astype(np.uint8),
        "tumor_mask": (tumor > 0).astype(np.uint8),
    }

def fetch_case_cached(cid: str) -> dict:
    if cid in _cache:
        _cache.move_to_end(cid)
        return _cache[cid]
    data = fetch_case(cid)
    _cache[cid] = data
    _cache.move_to_end(cid)
    if len(_cache) > MAX_CACHED:
        evicted, _ = _cache.popitem(last=False)
        del evicted  # just for clarity
    return data


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _augment_2d(img, mask):
    """img: (C, H, W) float32,  mask: (1, H, W) float32"""
    if random.random() < 0.5:
        img  = np.flip(img,  axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    if random.random() < 0.3:
        img  = np.flip(img,  axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if random.random() < 0.4:
        img = np.clip(img + np.random.uniform(-0.05, 0.05), 0, 1)
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)
    return img, mask

def _resize_slice(arr, size):
    """arr: (H, W) → resized to (size, size) via bilinear."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, (size, size),
                                        mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def _resize_mask(arr, size):
    """arr: (H, W) binary mask → (size, size) via nearest, so it stays binary."""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, (size, size), mode='nearest')
    return t.squeeze().numpy()


def _make_triplet(vol, sl_idx, size):
    """Extract 3-channel 2.5D slice triplet (prev, curr, next)."""
    D = vol.shape[2]
    lo = max(0, sl_idx - 1)
    hi = min(D - 1, sl_idx + 1)
    ch = np.stack([
        _resize_slice(vol[:, :, lo],     size),
        _resize_slice(vol[:, :, sl_idx], size),
        _resize_slice(vol[:, :, hi],     size),
    ])
    return ch  # (3, H, W)

def _make_triplet_4ph(data: dict, centre: int, size: int) -> np.ndarray:
    """Returns (12, H, W) — 3-slice triplet for each of 4 phases."""
    strips = []
    for ph in PHASE_NAMES:
        vol = data["phase_vols"][ph]
        strips.append(_make_triplet(vol, centre, size))  # (3, H, W)
    return np.concatenate(strips, axis=0)  # (12, H, W)


def _liver_bbox(liver_mask, pad=8, min_size=16):
    """Bounding box of liver in 2D (H, W) with padding and minimum size."""
    rows = np.any(liver_mask, axis=1)
    cols = np.any(liver_mask, axis=0)
    H, W = liver_mask.shape
    if not rows.any():
        return 0, H, 0, W
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - pad)
    r1 = min(H, r1 + pad + 1)
    c0 = max(0, c0 - pad)
    c1 = min(W, c1 + pad + 1)
    # enforce minimum crop size — expand symmetrically if too small
    if (r1 - r0) < min_size:
        centre = (r0 + r1) // 2
        r0 = max(0, centre - min_size // 2)
        r1 = min(H, r0 + min_size)
    if (c1 - c0) < min_size:
        centre = (c0 + c1) // 2
        c0 = max(0, centre - min_size // 2)
        c1 = min(W, c0 + min_size)
    return r0, r1, c0, c1


# ── Stage 1: LiverDataset ─────────────────────────────────────────────────────

class LiverDataset(Dataset):
    """
    Yields (img_triplet, liver_mask) pairs — 2.5D slices.
    Only slices with any liver voxel (or a random sample of empty slices)
    are included to avoid overwhelming the loader with pure background.
    """
    BG_RATIO = 0.15   # fraction of non-liver slices to keep

    def __init__(self, case_ids, augment=False):
        self.augment = augment
        self.samples: list[tuple[str, int]] = []  # (case_id, slice_idx)

        print(f"  LiverDataset: indexing {len(case_ids)} cases…")
        t0 = time.time()
        for cid in case_ids:
            data   = fetch_case_cached(cid)
            D      = data["pvp"].shape[2]
            pos_sl = [s for s in range(D)
                      if data["liver_mask"][:, :, s].any()]
            neg_sl = [s for s in range(D) if s not in set(pos_sl)]
            n_neg  = max(1, int(len(pos_sl) * self.BG_RATIO))
            neg_sl = random.sample(neg_sl, min(n_neg, len(neg_sl)))
            for s in pos_sl + neg_sl:
                self.samples.append((cid, s))

        # sort by case so consecutive batches hit same cached volume
        self.samples.sort(key=lambda x: x[0])
        print(f"  LiverDataset ready: {len(self.samples)} slices "
              f"from {len(case_ids)} cases ({time.time()-t0:.1f}s)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        cid, sl = self.samples[idx]
        data    = fetch_case_cached(cid)
        D       = data["pvp"].shape[2]

        def crop_resize(vol, sl_idx):
            s = np.clip(sl_idx, 0, D - 1)
            return _resize_slice(vol[:, :, s].astype(np.float32), IMG_SIZE)

        # build 12-channel input: 4 phases × 3 adjacent slices
        channels = []
        for phase_vol in data["phases"]:
            channels.append(crop_resize(phase_vol, sl - 1))
            channels.append(crop_resize(phase_vol, sl))
            channels.append(crop_resize(phase_vol, sl + 1))
        img = np.stack(channels)  # (12, IMG_SIZE, IMG_SIZE)

        msk = _resize_mask(
            data["liver_mask"][:, :, sl].astype(np.float32),
            IMG_SIZE)[None]  # (1, IMG_SIZE, IMG_SIZE)

        if self.augment:
            img, msk = _augment_2d(img, msk)

        return torch.from_numpy(img), torch.from_numpy(msk)

# ── Stage 2: TumorDataset ─────────────────────────────────────────────────────

class TumorDataset(Dataset):
    """
    Yields (img_crop, tumor_mask_crop) pairs — 2.5D liver-ROI crops.
    Uses ground-truth liver mask to crop, matching LiMT benchmark protocol.
    Only slices containing liver are included; when augment=True (training),
    tumor-positive slices are oversampled by OVERSAMPLE_FACTOR to address
    class imbalance. Val/test keep each slice once so Dice is unbiased.
    """
    OVERSAMPLE_FACTOR = 3

    def __init__(self, case_ids, augment=False):
        self.augment = augment
        self.samples: list[tuple[str, int]] = []

        print(f"  TumorDataset: indexing {len(case_ids)} cases…")
        t0 = time.time()
        for cid in case_ids:
            data = fetch_case_cached(cid)
            D    = data["pvp"].shape[2]
            for s in range(D):
                if not data["liver_mask"][:, :, s].any():
                    continue
                repeats = (self.OVERSAMPLE_FACTOR
                           if augment and data["tumor_mask"][:, :, s].any()
                           else 1)
                self.samples.extend([(cid, s)] * repeats)

        self.samples.sort(key=lambda x: x[0])
        print(f"  TumorDataset ready: {len(self.samples)} slices "
              f"from {len(case_ids)} cases ({time.time()-t0:.1f}s)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        cid, sl = self.samples[idx]
        data    = fetch_case_cached(cid)
        D       = data["pvp"].shape[2]

        # 1. compute bbox on full-res liver mask for this slice
        lmsk_2d        = data["liver_mask"][:, :, sl]
        r0, r1, c0, c1 = _liver_bbox(lmsk_2d)

        # 2. crop-then-resize each channel of the triplet (4 phases x 3 slices = 12ch)
        def crop_resize_ph(sl_idx, phase_key):
            s    = np.clip(sl_idx, 0, D - 1)
            vol  = data["phase_vols"][phase_key]
            crop = vol[r0:r1, c0:c1, s].astype(np.float32)
            if crop.shape[0] < 1 or crop.shape[1] < 1:
                crop = vol[:, :, s].astype(np.float32)
            return _resize_slice(crop, IMG_SIZE)

        strips = []
        for ph in PHASE_NAMES:
            strips += [
                crop_resize_ph(sl - 1, ph),
                crop_resize_ph(sl,     ph),
                crop_resize_ph(sl + 1, ph),
            ]

        img = np.stack(strips)  # (12, IMG_SIZE, IMG_SIZE)
        # 3. same crop for tumor mask
        tmsk_crop = data["tumor_mask"][r0:r1, c0:c1, sl].astype(np.float32)
        if tmsk_crop.shape[0] < 1 or tmsk_crop.shape[1] < 1:
            tmsk_crop = data["tumor_mask"][:, :, sl].astype(np.float32)
        msk = _resize_mask(tmsk_crop, IMG_SIZE)[None]  # (1, IMG_SIZE, IMG_SIZE)

        if self.augment:
            img, msk = _augment_2d(img, msk)

        return torch.from_numpy(img), torch.from_numpy(msk)


# ── Stage 3: ClassificationDataset ───────────────────────────────────────────


class ClassificationDataset(Dataset):
    """
    Dual-stream classification dataset:
      slice_x — centre slice 2.5D triplet (3, H, W)  — slice stream
      vol_x   — N_VOL_SLICES equally-spaced triplets  — volume stream
                shape (N_VOL_SLICES, 3, H, W)
      label   — int 0-4

    Samples one representative slice per __getitem__ call (random during
    training, centre during val/test). The volume stream always uses the
    same N_VOL_SLICES fixed samples for consistency.
    """
    N_VOL_SLICES = 12   # number of slices in the volume stream

    def __init__(self, case_ids, labels_df, augment=False):
        self.augment   = augment
        self.case_ids  = case_ids
        self.labels    = {}
        self.cls2idx   = {}
        idx = 0
        for _, row in labels_df.iterrows():
            cid = str(row["ID"]   if "ID"   in labels_df.columns else row.iloc[0])
            lbl = str(row["type"] if "type" in labels_df.columns else row.iloc[1])
            if lbl not in self.cls2idx:
                self.cls2idx[lbl] = idx; idx += 1
            self.labels[cid] = self.cls2idx[lbl]

        # Pre-index liver slices per case (lazy — no volume load)
        self._liver_slices: dict[str, list[int]] = {}
        print(f"  ClassificationDataset: {len(case_ids)} cases, "
              f"{len(self.cls2idx)} classes")
        print(f"    Class map: {self.cls2idx}")

    def _get_liver_slices(self, cid):
        if cid not in self._liver_slices:
            data = fetch_case_cached(cid)
            D    = data["liver_mask"].shape[2]
            sl   = [s for s in range(D)
                    if data["liver_mask"][:, :, s].any()]
            self._liver_slices[cid] = sl if sl else list(range(D))
        return self._liver_slices[cid]

    def __len__(self): return len(self.case_ids)

    def __getitem__(self, idx):
        cid   = self.case_ids[idx]
        label = self.labels[cid]
        data  = fetch_case_cached(cid)
        sl    = self._get_liver_slices(cid)

        # ── slice stream ──
        if self.augment:
            centre = random.choice(sl)
        else:
            centre = sl[len(sl) // 2]
        
        slice_x = torch.from_numpy(_make_triplet_4ph(data, centre, IMG_SIZE))

        # ── volume stream — N equally spaced slices ──
        N    = self.N_VOL_SLICES
        idxs = [sl[int(round(i * (len(sl) - 1) / (N - 1)))] for i in range(N)]
        vol  = np.stack([_make_triplet_4ph(data, s, IMG_SIZE)
                         for s in idxs])   # (N, 12, H, W)
        vol_x = torch.from_numpy(vol)

        return slice_x, vol_x, torch.tensor(label, dtype=torch.long)


# ── CaseGroupedSampler ────────────────────────────────────────────────────────

class CaseGroupedSampler(Sampler):
    """
    Shuffles cases, keeps all slices of each case contiguous.
    Maximises LRU cache hits: each volume loaded once per epoch.
    """
    def __init__(self, dataset):
        # dataset.samples must be list of (case_id, slice_idx)
        groups: dict[str, list[int]] = {}
        for i, (cid, _) in enumerate(dataset.samples):
            groups.setdefault(cid, []).append(i)
        self.groups = groups

    def __iter__(self):
        case_order = list(self.groups.keys())
        random.shuffle(case_order)
        indices = []
        for cid in case_order:
            sl_idxs = self.groups[cid][:]
            random.shuffle(sl_idxs)
            indices.extend(sl_idxs)
        return iter(indices)

    def __len__(self):
        return sum(len(v) for v in self.groups.values())