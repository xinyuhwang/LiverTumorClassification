"""
models.py — DS²Net model architectures and losses (ported from the notebooks)
  Stage 1: DS2NetLiver  — Swin-Tiny + DEM/SEM + MHA self-attention, 1 channel
           (notebooks/liver_segmentation.ipynb)
  Stage 2: DS2NetUNet   — Swin-Tiny + DEM/SEM + PhaseNorm + LI-RADS phase
           attention + small-tumor branch, 1 channel (tumor)
           (notebooks/segmentation_classification.ipynb, cell 0)
  Stage 3: EfficientNet4Phase — EfficientNet-B0 + PhaseNorm + LI-RADS phase
           attention + CBAM + relative enhancement curve (cell 1)
  Losses:  DS² uncertainty-adaptive weighted IoU + weighted BCE (+ boundary)

See PORTING.md for what changed relative to the notebooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Shared building blocks ────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ic, oc, k, padding=p, bias=False),
                               nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

    def forward(self, x): return self.b(x)


class DEM(nn.Module):
    """
    Detail Enhancement Module (DS²Net §III-A).
      M_d = Sigmoid(Conv_k(Conv_k(channel_max(f_l))))
      out = M_d ⊙ Conv3(Conv1(cat(up(f_h), f_l))) + residual(f_l)
    """
    def __init__(self, c_lo, c_hi, c_out, k=7):
        super().__init__()
        self.dc1 = nn.Sequential(nn.Conv2d(1, c_out, k, padding=k // 2, bias=False),
                                 nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self.dc2 = nn.Sequential(nn.Conv2d(c_out, c_out, k, padding=k // 2, bias=False),
                                 nn.BatchNorm2d(c_out))
        self.sig = nn.Sigmoid()
        self.c1  = nn.Sequential(nn.Conv2d(c_hi + c_lo, c_out, 1, bias=False),
                                 nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self.c3  = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(c_out))
        self.res = nn.Conv2d(c_lo, c_out, 1, bias=False) if c_lo != c_out else nn.Identity()

    def forward(self, f_l, f_h):
        Md    = self.sig(self.dc2(self.dc1(f_l.max(1, keepdim=True).values)))
        f_h_u = F.interpolate(f_h, f_l.shape[2:], mode='bilinear', align_corners=True)
        return Md * self.c3(self.c1(torch.cat([f_h_u, f_l], 1))) + self.res(f_l)


class SEM(nn.Module):
    """
    Semantic Enhancement Module (DS²Net §III-B).
      M_s = SA(up(f_h)),  SA = Sigmoid(Conv7(cat(mean, max)))
      out = CA(M_s ⊙ proj(f_skip)) + proj_h(up(f_h))
    """
    def __init__(self, c_skip, c_prev, c_out):
        super().__init__()
        self.sa_p = nn.Conv2d(c_prev, c_out, 1, bias=False)
        self.sa_c = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.pl   = nn.Conv2d(c_skip, c_out, 1, bias=False) if c_skip != c_out else nn.Identity()
        mid = max(c_out // 16, 4)
        self.ca   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c_out, mid), nn.ReLU(inplace=True),
                                  nn.Linear(mid, c_out), nn.Sigmoid())
        self.ph   = nn.Conv2d(c_prev, c_out, 1, bias=False) if c_prev != c_out else nn.Identity()

    def forward(self, f_skip, f_prev):
        fu  = F.interpolate(f_prev, f_skip.shape[2:], mode='bilinear', align_corners=True)
        fp  = self.sa_p(fu)
        Ms  = self.sa_c(torch.cat([fp.mean(1, keepdim=True),
                                   fp.max(1, keepdim=True).values], 1))
        g   = Ms * self.pl(f_skip)
        caw = self.ca(g).view(g.shape[0], -1, 1, 1)
        return g * caw + self.ph(fu)


class MHASelfAttention(nn.Module):
    """Pre-LN multi-head self-attention + FFN over a (B, D, H, W) feature map."""
    def __init__(self, embed_dim=256, n_heads=8, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha   = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                           batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        B, D, H, W = x.shape
        t = x.flatten(2).permute(0, 2, 1)             # (B, HW, D)
        n = self.norm1(t)
        a, _ = self.mha(n, n, n)
        t = t + self.drop1(a)
        t = t + self.ffn(self.norm2(t))
        return t.permute(0, 2, 1).reshape(B, D, H, W)


def swin_backbone(n_input_channels, pretrained=True, grad_checkpointing=False):
    """Swin-Tiny feature extractor with the patch embedding widened to
    n_input_channels (RGB filters kept, extra channels = their mean)."""
    bb  = timm.create_model('swin_tiny_patch4_window7_224',
                            pretrained=pretrained, features_only=True)
    old = bb.patch_embed.proj
    new = nn.Conv2d(n_input_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = old.weight.mean(1, keepdim=True).expand(
            -1, n_input_channels - 3, -1, -1)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    bb.patch_embed.proj = new
    if grad_checkpointing and hasattr(bb, 'set_grad_checkpointing'):
        bb.set_grad_checkpointing(enable=True)
    return bb


def swin_features(backbone, x, stage0_ch=96):
    """Run the Swin backbone and return NCHW feature maps."""
    raw = backbone(x)
    # timm features_only returns NHWC for Swin; fail loudly if that changes
    assert raw[0].shape[-1] == stage0_ch, (
        f"Expected Swin NHWC output with last dim {stage0_ch}, got {raw[0].shape}")
    return [f.permute(0, 3, 1, 2).contiguous() for f in raw]


# ── Stage 1: DS2NetLiver ──────────────────────────────────────────────────────

class DS2NetLiver(nn.Module):
    """
    Encoder    : Swin-Tiny, patch embedding widened to n_input_channels
    Bottleneck : SEM@d4 → MHA self-attention (14×14 tokens) → conv4
    Decoder    : DEM@d3 → DEM@d2 → DEM@d1 → dec0 (2× upsample)
    Heads      : 5 × Conv(D→1) for DS² deep supervision
    forward() returns the 5 head logits, coarsest first, all at input size.
    """
    def __init__(self, n_classes=1, ch=(96, 192, 384, 768), n_input_channels=20,
                 D=256, mha_heads=8, mha_dropout=0.1, pretrained=True):
        super().__init__()
        self.backbone = swin_backbone(n_input_channels, pretrained,
                                      grad_checkpointing=True)
        self.stage0_ch = ch[0]
        self.mha_sa = MHASelfAttention(D, mha_heads, mha_dropout, ffn_ratio=4)
        self.sem4  = SEM(ch[2], ch[3], D)
        self.conv4 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem3  = DEM(ch[1], D, D, k=5)
        self.conv3 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem2  = DEM(ch[0], D, D, k=7)
        self.conv2 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem1  = DEM(ch[0], D, D, k=3)
        self.conv1 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dec0  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.heads = nn.ModuleList(nn.Conv2d(D, n_classes, 1) for _ in range(5))

    def forward(self, x):
        H, W = x.shape[2:]
        f1, f2, f3, f4 = swin_features(self.backbone, x, self.stage0_ch)
        up2 = lambda f: F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.conv4(self.mha_sa(self.sem4(f3, f4)))
        d3 = self.conv3(self.dem3(f2, up2(d4)))
        d2 = self.conv2(self.dem2(f1, up2(d3)))
        d1 = self.conv1(self.dem1(up2(f1), up2(d2)))
        d0 = self.dec0(d1)
        up = lambda f: F.interpolate(f, (H, W), mode='bilinear', align_corners=True)
        h4, h3, h2, h1, h0 = self.heads
        return [up(h4(d4)), up(h3(d3)), up(h2(d2)), up(h1(d1)), h0(d0)]


# ── Stage 2: DS2NetUNet ───────────────────────────────────────────────────────

class PhaseNorm(nn.Module):
    """Per-channel instance normalisation with learnable affine."""
    def __init__(self, n_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-5
        return self.gamma * (x - mean) / std + self.beta


class FixedPhaseScale(nn.Module):
    """Alternative to PhaseNorm (E12 v2): the same affine for every slice,
    gamma * x + beta per channel, initialised to (x − 0.5) / 0.25. Inputs are
    already HU-windowed to [0, 1] per phase, so intensities keep their meaning
    across slices, phases and patients (as nnU-Net's fixed CT normalisation)."""
    def __init__(self, n_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, n_channels, 1, 1), 4.0))
        self.beta  = nn.Parameter(torch.full((1, n_channels, 1, 1), -2.0))

    def forward(self, x):
        return self.gamma * x + self.beta


class LIRADSPhaseAttention(nn.Module):
    """
    LI-RADS-inspired phase gating: each non-arterial phase gets a weight
    sigmoid(gate(embed(mean_p) − embed(mean_art))); arterial keeps weight 1.
    Channel layout is context-major: [ph0_ctx0, ph1_ctx0, …, ph0_ctx1, …].

    Known issue (kept to match the notebook): after PhaseNorm every channel's
    spatial mean equals its learned beta, so these weights do not depend on
    the input image. With FixedPhaseScale (phase_norm="fixed") they do.
    """
    def __init__(self, n_phases=4, n_ctx=3, embed_dim=32, art_idx=1):
        super().__init__()
        self.n_phases, self.n_ctx, self.art_idx = n_phases, n_ctx, art_idx
        self.embed = nn.Linear(1, embed_dim)
        self.gate  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        mid = self.n_ctx // 2
        embs = [self.embed(x[:, ph + mid * self.n_phases].mean(dim=(1, 2)).unsqueeze(1))
                for ph in range(self.n_phases)]
        art = embs[self.art_idx]
        w = [torch.ones(B, 1, device=x.device, dtype=x.dtype) if ph == self.art_idx
             else self.gate(embs[ph] - art) for ph in range(self.n_phases)]
        w = torch.cat(w, dim=1).repeat(1, self.n_ctx)     # (B, C), context-major
        return x * w.view(B, C, 1, 1)


class SmallTumorBranch(nn.Module):
    """Dilated ASPP-style branch for small lesions at 56×56."""
    def __init__(self, in_channels, out_channels=128, n_classes=1):
        super().__init__()
        def block(ic, d):
            return nn.Sequential(
                nn.Conv2d(ic, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.d1, self.d2, self.d4 = (block(in_channels, 1), block(out_channels, 2),
                                     block(out_channels, 4))
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(out_channels, n_classes, 1)

    def forward(self, x):
        f1 = self.d1(x); f2 = self.d2(f1); f4 = self.d4(f2)
        return self.head(self.fuse(torch.cat([f1, f2, f4], dim=1)))


class DS2NetUNet(nn.Module):
    """
    PhaseNorm → LI-RADS phase attention → Swin-Tiny → SEM/DEM decoder (as in
    DS2NetLiver, without MHA) + small-tumor branch on cat(d2, f1) at 56×56.
    forward() returns (5 head logits coarsest first, stb_logits at 56×56).
    """
    def __init__(self, n_classes=1, ch=(96, 192, 384, 768), n_input_channels=12,
                 n_phases=4, D=256, pretrained=True, phase_norm="instance"):
        super().__init__()
        # "instance" = PhaseNorm (notebook); "fixed" = FixedPhaseScale (E12 v2)
        norm = {"instance": PhaseNorm, "fixed": FixedPhaseScale}[phase_norm]
        self.phase_norm  = norm(n_input_channels)
        self.lirads_attn = LIRADSPhaseAttention(n_phases, n_input_channels // n_phases)
        self.backbone    = swin_backbone(n_input_channels, pretrained)
        self.stage0_ch   = ch[0]
        self.sem4  = SEM(ch[2], ch[3], D)
        self.conv4 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem3  = DEM(ch[1], D, D, k=5)
        self.conv3 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem2  = DEM(ch[0], D, D, k=7)
        self.conv2 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dem1  = DEM(ch[0], D, D, k=3)
        self.conv1 = nn.Sequential(ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.dec0  = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnRelu(D, D), ConvBnRelu(D, D))
        self.heads = nn.ModuleList(nn.Conv2d(D, n_classes, 1) for _ in range(5))
        self.stb   = SmallTumorBranch(D + ch[0], 128, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.lirads_attn(self.phase_norm(x))
        f1, f2, f3, f4 = swin_features(self.backbone, x, self.stage0_ch)
        up2 = lambda f: F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.conv4(self.sem4(f3, f4))
        d3 = self.conv3(self.dem3(f2, up2(d4)))
        d2 = self.conv2(self.dem2(f1, up2(d3)))
        d1 = self.conv1(self.dem1(up2(f1), up2(d2)))
        d0 = self.dec0(d1)
        up = lambda f: F.interpolate(f, (H, W), mode='bilinear', align_corners=True)
        h4, h3, h2, h1, h0 = self.heads
        preds = [up(h4(d4)), up(h3(d3)), up(h2(d2)), up(h1(d1)), h0(d0)]
        return preds, self.stb(torch.cat([d2, f1], dim=1))


# ── Losses and probability fusion ─────────────────────────────────────────────

def boundary_loss(logits, gt, kernel_size=5):
    """BCE with weight 5 on the mask boundary (dilation − erosion), 1 elsewhere."""
    gt_f     = gt.float()
    pad      = kernel_size // 2
    dilated  = F.max_pool2d(gt_f, kernel_size, stride=1, padding=pad)
    eroded   = -F.max_pool2d(-gt_f, kernel_size, stride=1, padding=pad)
    weight   = 1.0 + 4.0 * ((dilated - eroded) > 0.5).float()
    return F.binary_cross_entropy_with_logits(logits.float(), gt_f, weight=weight)


def weighted_iou_bce(logit, gt, pos_weight):
    """Weighted IoU + weighted BCE (DS²Net eq. 9-10); pooling kernel ≤ 31,
    weight map clamped to [1, 5] for bf16 stability."""
    prob = torch.sigmoid(logit.float())
    gt_f = gt.float()
    H, W = gt_f.shape[-2:]
    if H >= 7 and W >= 7:
        k = min(31, H if H % 2 else H - 1, W if W % 2 else W - 1)
        w = (1 + torch.abs(gt_f - F.avg_pool2d(gt_f, k, stride=1, padding=k // 2))
             .clamp(0, 1)).clamp(1, 5)
    else:
        w = torch.ones_like(gt_f)
    inter = (prob * gt_f * w).sum(dim=[2, 3])
    union = ((prob + gt_f - prob * gt_f) * w).sum(dim=[2, 3])
    w_iou = 1 - (inter + 1) / (union + 1)
    w_bce = F.binary_cross_entropy_with_logits(
        logit.float(), gt_f, pos_weight=torch.tensor([pos_weight], device=gt.device))
    return w_iou.mean() + w_bce


def ds2_adaptive_loss(preds, gt, pos_weight, boundary_weight,
                      stb_logits=None, stb_weight=0.4):
    """
    DS² uncertainty-adaptive loss (paper §III-C):
      u_i = mean(1 − |σ(p_i) − 0.5| / 0.5),  λ_i = softmax(u) / max(softmax(u))
      L   = Σ λ_i (wIoU + wBCE)  + boundary loss on the finest head only
    Stage 2 adds the small-tumor branch: stb_weight × (wIoU + wBCE with
    1.5 × pos_weight + boundary).
    """
    with torch.no_grad():
        u    = torch.tensor([(1 - (torch.sigmoid(p.float()) - 0.5).abs() / 0.5).mean().item()
                             for p in preds]).clamp(min=1e-6)
        ub   = torch.softmax(u, dim=0)
        lams = (ub / ub.max().clamp(min=1e-6)).tolist()
    total = 0.
    for i, (p, lam) in enumerate(zip(preds, lams)):
        base = weighted_iou_bce(p, gt, pos_weight)
        if i == len(preds) - 1:                     # finest head (224×224)
            base = base + boundary_weight * boundary_loss(p, gt)
        total = total + lam * base
    if stb_logits is not None:
        stb_up = F.interpolate(stb_logits.float(), size=gt.shape[2:],
                               mode='bilinear', align_corners=True)
        total = total + stb_weight * (weighted_iou_bce(stb_up, gt, pos_weight * 1.5)
                                      + boundary_weight * boundary_loss(stb_up, gt))
    return total


def soft_dice_ce(logit, gt):
    """nnU-Net's loss for one binary output: soft Dice pooled over the batch
    (batch_dice, as nnU-Net's 3d_fullres) + unweighted BCE."""
    prob, gt_f = torch.sigmoid(logit.float()), gt.float()
    inter = (prob * gt_f).sum()
    dice = (2 * inter + 1e-5) / (prob.sum() + gt_f.sum() + 1e-5)
    return (1 - dice) + F.binary_cross_entropy_with_logits(logit.float(), gt_f)


def dice_ce_loss(preds, gt, stb_logits=None, stb_weight=0.4):
    """nnU-Net-style alternative to ds2_adaptive_loss (E12 v4): Dice + CE on
    every head, deep-supervision weights halving from the finest head
    (preds are coarsest first) and normalised to 1; no pos_weight, boundary
    or uncertainty weighting. The small-tumor branch gets the same loss."""
    w = [0.5 ** i for i in range(len(preds))][::-1]
    total = sum(wi * soft_dice_ce(p, gt) for wi, p in zip(w, preds)) / sum(w)
    if stb_logits is not None:
        stb_up = F.interpolate(stb_logits.float(), size=gt.shape[2:],
                               mode='bilinear', align_corners=True)
        total = total + stb_weight * soft_dice_ce(stb_up, gt)
    return total


def fuse_probs(preds, stb_logits=None, stb_weight=0.3):
    """Mean of the head sigmoids; Stage 2 blends in the small-tumor branch
    with weight 1.5 × stb_weight (the notebook's tumor-channel weighting)."""
    main = sum(torch.sigmoid(p.float()) for p in preds) / len(preds)
    if stb_logits is None:
        return main
    stb = torch.sigmoid(F.interpolate(stb_logits.float(), size=main.shape[2:],
                                      mode='bilinear', align_corners=True))
    w = min(max(1 - stb_weight * 1.5, 0.0), 1.0)
    return w * main + (1 - w) * stb


# ── Stage 3: EfficientNet4Phase classifier ───────────────────────────────────

class LIRADSPhaseAttnCls(nn.Module):
    """LI-RADS phase gating for single-context 4-channel classifier input."""
    def __init__(self, n_phases=4, embed_dim=32, art_idx=1):
        super().__init__()
        self.art_idx = art_idx
        self.embed = nn.Linear(1, embed_dim)
        self.gate  = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(embed_dim // 2, 1), nn.Sigmoid())

    def forward(self, x):
        B, C = x.shape[:2]
        embs = [self.embed(x[:, c].mean(dim=(1, 2)).unsqueeze(1)) for c in range(C)]
        w = [torch.ones(B, 1, device=x.device, dtype=x.dtype) if c == self.art_idx
             else self.gate(embs[c] - embs[self.art_idx]) for c in range(C)]
        return x * torch.cat(w, dim=1).view(B, C, 1, 1)


class CBAM(nn.Module):
    def __init__(self, channels, r=16, k=7):
        super().__init__()
        mid = max(channels // r, 4)
        self.avg_fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(), nn.Linear(mid, channels))
        self.max_fc = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(), nn.Linear(mid, channels))
        self.sa     = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.avg_fc(x.mean(dim=[2, 3])) + self.max_fc(x.amax(dim=[2, 3])))
        x  = x * ca[:, :, None, None]
        sa = torch.sigmoid(self.sa(torch.cat([x.mean(1, keepdim=True),
                                              x.amax(1, keepdim=True)], dim=1)))
        return x * sa


class RelativeEnhancementCurve(nn.Module):
    """
    LI-RADS-style enhancement features from per-phase patch means and
    background-liver-ring means: relative = patch − background, washout =
    rel_pvp − rel_art, progress = rel_delay − rel_art, amplitude.
    """
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(11, 32), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                nn.Linear(32, embed_dim), nn.ReLU(inplace=True))

    def forward(self, patch_means, bg_means):
        rel      = patch_means - bg_means
        washout  = rel[:, 2:3] - rel[:, 1:2]
        progress = rel[:, 3:4] - rel[:, 1:2]
        amp      = (rel.max(dim=1).values - rel.min(dim=1).values).unsqueeze(1)
        return self.fc(torch.cat([patch_means, rel, washout, progress, amp], dim=1))


class EfficientNet4Phase(nn.Module):
    """
    PhaseNorm → LI-RADS phase attention → EfficientNet-B0 → CBAM → GAP,
    concatenated with the relative-enhancement embedding → MLP head.

    Known issue (kept to match the notebook): patch_means are computed from
    the PhaseNorm-normalised input, where each channel's mean is ≈ its beta,
    so they carry almost no image information.
    """
    def __init__(self, n_classes=5, n_channels=4, dropout=0.3,
                 use_curve=True, curve_embed_dim=32, pretrained=True):
        super().__init__()
        self.use_curve   = use_curve
        self.phase_norm  = PhaseNorm(n_channels)
        self.lirads_attn = LIRADSPhaseAttnCls(n_channels)
        self.backbone    = timm.create_model("efficientnet_b0", pretrained=pretrained,
                                             num_classes=0)
        old = self.backbone.conv_stem
        new = nn.Conv2d(n_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(1, n_channels // 3 + 1, 1, 1)[:, :n_channels]
                             * (3.0 / n_channels))
        self.backbone.conv_stem = new
        n_feat    = self.backbone.num_features
        self.cbam = CBAM(n_feat)
        self.curve = RelativeEnhancementCurve(curve_embed_dim) if use_curve else None
        head_in   = n_feat + (curve_embed_dim if use_curve else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 256),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout / 2),
                                  nn.Linear(256, n_classes))
        self.freeze_backbone(True)

    def freeze_backbone(self, frozen):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)

    def forward(self, x, bg_means=None):
        x      = self.lirads_attn(self.phase_norm(x))
        pooled = self.cbam(self.backbone.forward_features(x)).mean(dim=(2, 3))
        if self.use_curve and bg_means is not None:
            pooled = torch.cat([pooled, self.curve(x.mean(dim=(2, 3)), bg_means)], dim=1)
        return self.head(pooled)
