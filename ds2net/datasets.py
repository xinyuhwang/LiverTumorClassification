"""
datasets.py — DS²Net data pipeline (ported from the notebooks)
  load_case / build_volume  — NIfTI → HU-windowed, resampled, 2.5D stacks
  build_cache               — per-slice training cache (memory-mapped .npy)
  SliceDataset              — Stage 1 (liver) and Stage 2 (tumor) slices
  augment_nnunet_ct         — nnU-Net-style 2D augmentation
  to_original_grid          — map slice predictions back to the NIfTI grid
  build_roi_cache / TumorPatchDataset — Stage 3 4-phase tumor ROI patches

Case layout (produced by data_prep/prepare_mct_ltdiag.py):
  <data_dir>/<case>/phase_{0..3}.nii.gz, pvp.nii.gz,
                    liver_mask.nii.gz, tumor_mask.nii.gz
"""

import hashlib, json, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import (zoom, gaussian_filter, map_coordinates,
                           rotate as nd_rotate, binary_dilation)
from skimage.transform import resize as sk_resize

from config import PHASE_NAMES, HU_WINDOWS

CACHE_VERSION = "ds2net-v1"


# ── Volume loading ────────────────────────────────────────────────────────────

def load_case(case_dir, liver_includes_tumor=False):
    """Raw HU phases (on the PVP grid), binary masks, voxel spacing.
    liver_includes_tumor: liver label = liver_mask ∪ tumor_mask (the official
    liver mask leaves out part of the tumor in many cases; see E01)."""
    case_dir = Path(case_dir)
    pvp_img  = nib.load(case_dir / "pvp.nii.gz")
    shape    = pvp_img.shape
    phases   = []
    for i in range(len(PHASE_NAMES)):
        path = case_dir / f"phase_{i}.nii.gz"
        vol  = (nib.load(path).get_fdata(dtype=np.float32) if path.exists()
                else pvp_img.get_fdata(dtype=np.float32))
        if vol.shape != shape:
            vol = zoom(vol, tuple(s / v for s, v in zip(shape, vol.shape)), order=1)
        phases.append(vol)
    masks = {k: (np.asanyarray(nib.load(case_dir / f"{k}_mask.nii.gz").dataobj) > 0)
                .astype(np.uint8) for k in ("liver", "tumor")}
    if liver_includes_tumor:
        masks["liver"] = masks["liver"] | masks["tumor"]
    spacing = tuple(float(z) for z in pvp_img.header.get_zooms()[:3])
    return phases, masks, spacing


def _window(vol, name):
    lo, hi = HU_WINDOWS[name]
    return np.clip((np.clip(vol, lo, hi) - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def _resample_factors(spacing, target):
    f = tuple(s / target for s in spacing)
    return f if any(abs(x - 1) > 0.05 for x in f) else None


def build_volume(case_dir, cfg):
    """
    Returns
      phases_rs : list of 4 arrays (D_rs, S, S) float32 — windowed, resampled
                  to cfg target_spacing, each axial slice resized to S×S
      masks_rs  : {"liver","tumor"} → (H_rs, W_rs, D_rs) uint8
      rs_shape, orig_shape, spacing
    """
    phases, masks, spacing = load_case(case_dir, cfg.get("liver_includes_tumor", False))
    factors = _resample_factors(spacing, cfg["target_spacing"])
    S = cfg["img_size"]
    phases_rs = []
    for name, vol in zip(PHASE_NAMES, phases):
        if factors:
            vol = zoom(vol, factors, order=1)
        vol = _window(vol, name)                     # clip after zoom: no overshoot
        phases_rs.append(np.stack([
            np.clip(sk_resize(vol[:, :, z], (S, S), order=1, preserve_range=True), 0, 1)
            for z in range(vol.shape[2])]).astype(np.float32))
    masks_rs = {k: (zoom(m, factors, order=0) if factors else m).astype(np.uint8)
                for k, m in masks.items()}
    return phases_rs, masks_rs, masks_rs["liver"].shape, phases[0].shape, spacing


def stack_context(phases_rs, z, n_ctx):
    """(n_ctx × n_phases, S, S), context-major: [ph0_z-1, ph1_z-1, …, ph0_z, …]."""
    D, half = phases_rs[0].shape[0], n_ctx // 2
    return np.stack([ph[min(max(z + dz, 0), D - 1)]
                     for dz in range(-half, half + 1) for ph in phases_rs])


def to_original_grid(prob_slices, rs_shape, orig_shape):
    """(D_rs, S, S) probabilities → (H, W, D) on the original NIfTI grid."""
    H_rs, W_rs, D_rs = rs_shape
    vol = np.stack([sk_resize(prob_slices[d], (H_rs, W_rs), order=1, preserve_range=True)
                    for d in range(D_rs)], axis=-1).astype(np.float32)
    if vol.shape != tuple(orig_shape):
        vol = zoom(vol, tuple(o / r for o, r in zip(orig_shape, vol.shape)), order=1)
    return vol


# ── Stage 1/2 slice cache ─────────────────────────────────────────────────────

def cache_key(cfg):
    keys = ("task", "n_context_slices", "img_size", "target_spacing", "slices",
            "liver_includes_tumor")
    blob = json.dumps({k: cfg.get(k) for k in keys} | {"hu": HU_WINDOWS, "v": CACHE_VERSION},
                      sort_keys=True)
    return f"{cfg['task']}_{hashlib.md5(blob.encode()).hexdigest()[:10]}"


def _cache_one(args):
    case_id, data_dir, out_dir, cfg = args
    out = Path(out_dir)
    paths = {k: out / f"{case_id}_{k}.npy" for k in ("imgs", "masks", "w")}
    if all(p.exists() for p in paths.values()):
        return case_id, "cached"
    try:
        phases_rs, masks_rs, *_ = build_volume(Path(data_dir) / case_id, cfg)
        target = masks_rs[cfg["task"]]
        select = masks_rs[cfg["slices"]]
        S, n_ctx = cfg["img_size"], cfg["n_context_slices"]
        zs = [z for z in range(target.shape[2]) if select[:, :, z].any()]
        if not zs:
            return case_id, "empty"
        imgs  = np.lib.format.open_memmap(paths["imgs"].with_suffix(".tmp.npy"), "w+",
                                          np.float16, (len(zs), n_ctx * len(phases_rs), S, S))
        masks = np.zeros((len(zs), 1, S, S), np.uint8)
        w     = np.zeros(len(zs), np.float32)
        for i, z in enumerate(zs):
            imgs[i]  = stack_context(phases_rs, z, n_ctx)
            masks[i, 0] = sk_resize(target[:, :, z], (S, S), order=0,
                                    preserve_range=True, anti_aliasing=False)
            # sampling weight: liver area (Stage 1) or tumor voxels (Stage 2)
            w[i] = masks[i, 0].sum() if cfg["task"] == "liver" else target[:, :, z].sum()
        imgs.flush(); del imgs
        paths["imgs"].with_suffix(".tmp.npy").rename(paths["imgs"])
        np.save(paths["masks"], masks)
        np.save(paths["w"], w)
        return case_id, f"new ({len(zs)} slices)"
    except Exception as e:                                   # noqa: BLE001
        return case_id, f"ERROR: {type(e).__name__}: {e}"


def build_cache(case_ids, data_dir, cache_root, cfg, workers=8):
    out = Path(cache_root) / cache_key(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ok, errors = [], []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_cache_one, (c, str(data_dir), str(out), cfg)) for c in case_ids]
        for fut in as_completed(futs):
            cid, status = fut.result()
            (errors if status.startswith(("ERROR", "empty")) else ok).append(cid)
            if not status.startswith("cached"):
                print(f"  cache {cid}: {status}", flush=True)
    print(f"  Cache {out.name}: {len(ok)} ready, {len(errors)} skipped/failed")
    return out, sorted(ok)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_nnunet_ct(img, mask, cfg, n_phases):
    """img (C,H,W) float32 in [0,1], mask (1,H,W). Spatial transforms share one
    field across all channels; intensity transforms are per phase, shared
    across that phase's context slices."""
    _, H, W = img.shape
    n_ctx = img.shape[0] // n_phases
    if random.random() < cfg["aug_elastic_prob"]:
        dx = gaussian_filter(np.random.randn(H, W), cfg["aug_elastic_sigma"]) * cfg["aug_elastic_alpha"]
        dy = gaussian_filter(np.random.randn(H, W), cfg["aug_elastic_sigma"]) * cfg["aug_elastic_alpha"]
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        coords = [np.clip(gy + dy, 0, H - 1), np.clip(gx + dx, 0, W - 1)]
        img  = np.stack([map_coordinates(c, coords, order=1) for c in img]).astype(np.float32)
        mask = np.stack([map_coordinates(c, coords, order=0) for c in mask]).astype(np.float32)
    if random.random() < cfg["aug_scale_prob"]:
        sc = random.uniform(*cfg["aug_scale_range"])
        nh, nw = int(H * sc), int(W * sc)
        ir = np.stack([sk_resize(c, (nh, nw), order=1, preserve_range=True) for c in img])
        mr = np.stack([sk_resize(c, (nh, nw), order=0, preserve_range=True,
                                 anti_aliasing=False) for c in mask])
        if sc > 1:
            r0, c0 = (nh - H) // 2, (nw - W) // 2
            img, mask = ir[:, r0:r0 + H, c0:c0 + W], mr[:, r0:r0 + H, c0:c0 + W]
        else:
            ph, pw = (H - nh) // 2, (W - nw) // 2
            pad = ((0, 0), (ph, H - nh - ph), (pw, W - nw - pw))
            img, mask = np.pad(ir, pad), np.pad(mr, pad)
    if random.random() < cfg["aug_rotation_prob"]:
        ang = random.uniform(-cfg["aug_rotation_max_deg"], cfg["aug_rotation_max_deg"])
        img  = np.stack([nd_rotate(c, ang, reshape=False, order=1) for c in img])
        mask = np.stack([nd_rotate(c, ang, reshape=False, order=0) for c in mask])
    if random.random() < cfg["aug_mirror_prob"]:
        img, mask = img[:, :, ::-1], mask[:, :, ::-1]
    if random.random() < cfg["aug_ud_flip_prob"]:
        img, mask = img[:, ::-1, :], mask[:, ::-1, :]
    img = np.ascontiguousarray(img, dtype=np.float32)
    for ph in range(n_phases):
        ch = [ph + c * n_phases for c in range(n_ctx)]
        if random.random() < cfg["aug_hu_jitter_prob"]:
            j = cfg["aug_hu_jitter"]
            img[ch] = img[ch] * random.uniform(1 - j, 1 + j) + random.uniform(-j, j)
        if random.random() < cfg["aug_noise_prob"]:
            img[ch] += np.random.randn(H, W).astype(np.float32) * cfg["aug_noise_std"]
        if random.random() < cfg["aug_blur_prob"]:
            s = random.uniform(*cfg["aug_blur_sigma"])
            img[ch] = np.stack([gaussian_filter(c, s) for c in img[ch]])
        img[ch] = np.clip(img[ch], 0, 1)
        if random.random() < cfg["aug_gamma_prob"]:
            img[ch] = np.clip(img[ch], 1e-7, 1) ** random.uniform(*cfg["aug_gamma_range"])
    return np.clip(img, 0, 1), np.ascontiguousarray(mask, dtype=np.float32)


class SliceDataset(Dataset):
    """Memory-mapped 2.5D slices from build_cache. `slice_weights` feed a
    WeightedRandomSampler (liver area, or tumor voxels × type multiplier)."""
    def __init__(self, cache_dir, case_ids, cfg, augment=False, case_types=None):
        self.cache_dir, self.cfg, self.augment = Path(cache_dir), cfg, augment
        self.n_phases = len(PHASE_NAMES)
        self.index, self.slice_weights = [], []
        mult = cfg.get("difficulty_multiplier", {})
        floor = 100.0 if cfg["task"] == "liver" else 50.0
        for cid in case_ids:
            w = np.load(self.cache_dir / f"{cid}_w.npy")
            m = mult.get((case_types or {}).get(cid), 1.0)
            for i, wi in enumerate(w):
                self.index.append((cid, i))
                self.slice_weights.append(max(float(wi), floor) * m)
        print(f"  {'train' if augment else 'eval'} slices: {len(self.index)} "
              f"from {len(case_ids)} cases")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        cid, i = self.index[idx]
        img  = np.load(self.cache_dir / f"{cid}_imgs.npy", mmap_mode="r")[i].astype(np.float32)
        mask = np.load(self.cache_dir / f"{cid}_masks.npy", mmap_mode="r")[i].astype(np.float32)
        if self.augment:
            img, mask = augment_nnunet_ct(img, mask, self.cfg, self.n_phases)
        return torch.from_numpy(img), torch.from_numpy(mask)


# ── Stage 3: tumor ROI patches ────────────────────────────────────────────────

def extract_roi_patches(case_dir, cfg):
    """
    Up to max_slices 4-phase patches (roi_size², channels = nc/art/pvp/delay)
    cropped to the GT tumor bounding box + roi_margin_mm, and per-phase
    background means over a ring around the tumor. Slices are the ones with
    the largest tumor area.
    """
    phases, masks, spacing = load_case(case_dir)
    mask = masks["tumor"]
    if mask.sum() < cfg["min_tumor_voxels"]:
        return np.zeros((0, 4, cfg["roi_size"], cfg["roi_size"]), np.float32), \
               np.zeros((0, 4), np.float32)
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    margin = [int(cfg["roi_margin_mm"] / max(s, 1)) for s in spacing]
    lo = [max(0, l - m) for l, m in zip(lo, margin)]
    hi = [min(n - 1, h + m) for h, m, n in zip(hi, margin, mask.shape)]
    area = mask.sum(axis=(0, 1))
    zs = [int(z) for z in np.argsort(area)[::-1]
          if lo[2] <= z <= hi[2] and area[z] > 0][:cfg["max_slices"]]
    windowed = [_window(v, n) for v, n in zip(phases, PHASE_NAMES)]
    rs, ring = cfg["roi_size"], cfg["bg_ring_width"]
    patches, bgs = [], []
    for z in sorted(zs):
        crop = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), z)
        m_r = sk_resize(mask[crop].astype(np.float32), (rs, rs), order=0,
                        preserve_range=True, anti_aliasing=False) > 0.5
        ring_m = binary_dilation(m_r, iterations=ring) & ~m_r if m_r.any() else None
        chans, bg = [], []
        for v in windowed:
            sl = sk_resize(v[crop], (rs, rs), order=1, preserve_range=True).astype(np.float32)
            chans.append(sl)
            bg.append(sl[ring_m].mean() if ring_m is not None and ring_m.any() else sl.mean())
        patches.append(np.stack(chans)); bgs.append(bg)
    return np.stack(patches).astype(np.float32), np.asarray(bgs, np.float32)


def _roi_one(args):
    case_id, data_dir, out_dir, cfg = args
    path = Path(out_dir) / f"{case_id}.npz"
    if path.exists():
        return case_id, "cached"
    try:
        p, b = extract_roi_patches(Path(data_dir) / case_id, cfg)
        np.savez_compressed(path, patches=p.astype(np.float16), bg_means=b)
        return case_id, f"new ({len(p)} patches)"
    except Exception as e:                                   # noqa: BLE001
        return case_id, f"ERROR: {type(e).__name__}: {e}"


def build_roi_cache(case_ids, data_dir, cache_root, cfg, workers=8):
    keys = ("roi_margin_mm", "roi_size", "max_slices", "min_tumor_voxels", "bg_ring_width")
    blob = json.dumps({k: cfg[k] for k in keys} | {"hu": HU_WINDOWS, "v": CACHE_VERSION},
                      sort_keys=True)
    out = Path(cache_root) / f"cls_{hashlib.md5(blob.encode()).hexdigest()[:10]}"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_roi_one, (c, str(data_dir), str(out), cfg)) for c in case_ids]
        for fut in as_completed(futs):
            cid, status = fut.result()
            if status != "cached":
                print(f"  roi {cid}: {status}", flush=True)
    return out


def augment_patch(p):
    """p (4,H,W) in [0,1]: flips, 90° rotations, per-phase intensity jitter."""
    if random.random() < 0.5: p = p[:, :, ::-1]
    if random.random() < 0.5: p = p[:, ::-1, :]
    p = np.ascontiguousarray(np.rot90(p, random.randint(0, 3), axes=(1, 2)))
    for c in range(p.shape[0]):
        if random.random() < 0.15:
            p[c] = p[c] * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
        if random.random() < 0.15:
            p[c] = p[c] + np.random.randn(*p[c].shape).astype(np.float32) * 0.03
        if random.random() < 0.15:
            p[c] = gaussian_filter(p[c], random.uniform(0.5, 1.5))
        p[c] = np.clip(p[c], 0, 1)
        if random.random() < 0.20:
            p[c] = np.clip(p[c], 1e-7, 1) ** random.uniform(0.7, 1.5)
    return p


class TumorPatchDataset(Dataset):
    """One item per (case, ROI patch): (patch, bg_means, label, case_id)."""
    def __init__(self, roi_dir, case_ids, labels, augment=False):
        self.augment, self.items = augment, []
        for cid in case_ids:
            path = Path(roi_dir) / f"{cid}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            for p, b in zip(d["patches"], d["bg_means"]):
                self.items.append((p, b, labels[cid], cid))
        print(f"  {'train' if augment else 'eval'} patches: {len(self.items)} "
              f"from {len(case_ids)} cases")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, b, y, cid = self.items[idx]
        p = p.astype(np.float32)
        if self.augment:
            p = augment_patch(p)
        return torch.from_numpy(np.ascontiguousarray(p)), torch.from_numpy(b), y, cid
