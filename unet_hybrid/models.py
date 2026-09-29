"""
models.py — HERALD model architectures
  Stage 1 & 2: UNetTransformer (CNN encoder + Swin transformer bottleneck +
               DS2Net-style DEM/SEM dual supervision)
  Stage 3:     ResNetClassifier, EfficientNetClassifier, EnsembleClassifier
               with late fusion of 2.5D slice-level + volume-level streams
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import timm

# ── Helpers ───────────────────────────────────────────────────────────────────

def _gn(ch): return nn.GroupNorm(max(1, ch // 16), ch)

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBnRelu(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch))
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv2(self.conv1(x)) + x)


# ── Swin-style Window Attention (lightweight, no external dependency) ─────────

class WindowAttention(nn.Module):
    """Single-scale window multi-head self-attention for the bottleneck."""
    def __init__(self, dim, num_heads=8, win=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.win       = win
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        w = self.win
        # pad to multiple of window size
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nh, nw = Hp // w, Wp // w

        # partition into windows
        x = x.reshape(B, C, nh, w, nw, w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nh * nw, w * w, C)

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.num_heads,
                              C // self.num_heads).transpose(1, 2)
                   for t in qkv]
        attn = self.drop(F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B * nh * nw, w * w, C)
        out  = self.proj(out)

        # reverse window partition
        out = out.reshape(B, nh, nw, w, w, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        # remove padding
        return out[:, :, :H, :W].contiguous()


class TransformerBottleneck(nn.Module):
    """Two stacked window-attention blocks with LayerNorm and FFN."""
    def __init__(self, dim, num_heads=8, win=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn' : WindowAttention(dim, num_heads, win, dropout),
                'norm2': nn.LayerNorm(dim),
                'ffn'  : nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim), nn.Dropout(dropout))
            }) for _ in range(2)
        ])

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        for blk in self.layers:
            # attention with pre-norm (operate on channel dim via reshape)
            xf = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            xf = blk['norm1'](xf).reshape(B, H, W, C).permute(0, 3, 1, 2)
            x  = x + blk['attn'](xf)
            xf = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            xf = blk['norm2'](xf)
            xf = blk['ffn'](xf).reshape(B, H, W, C).permute(0, 3, 1, 2)
            x  = x + xf
        return x


# ── DEM / SEM — DS2Net-style dual supervision modules ────────────────────────

class DetailEnhanceModule(nn.Module):
    """Fuses low-level detail features with high-level semantics."""
    def __init__(self, lo_ch, hi_ch, out_ch):
        super().__init__()
        self.lo_conv = ConvBnRelu(lo_ch, out_ch)
        self.hi_conv = ConvBnRelu(hi_ch, out_ch)
        self.fuse    = ConvBnRelu(out_ch * 2, out_ch)
        self.pred    = nn.Conv2d(out_ch, 1, 1)   # auxiliary prediction head

    def forward(self, lo, hi, target_size=None):
        if target_size is None:
            target_size = lo.shape[-2:]
        lo_f = self.lo_conv(lo)
        hi_f = F.interpolate(self.hi_conv(hi), size=target_size,
                             mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([lo_f, hi_f], dim=1))
        aux   = self.pred(fused)
        return fused, aux


class SemanticEnhanceModule(nn.Module):
    """Refines high-level semantic features."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.pred  = nn.Conv2d(out_ch, 1, 1)    # auxiliary prediction head

    def forward(self, x):
        f   = self.conv2(self.conv1(x))
        aux = self.pred(f)
        return f, aux


# ── UNetTransformer ───────────────────────────────────────────────────────────

class UNetTransformer(nn.Module):
    """
    Standard UNet encoder-decoder with a Swin-style transformer bottleneck
    and DS2Net dual DEM/SEM supervision heads.

    Input:  (B, in_ch, H, W) — 2.5D input (12 = 4 phases x 3 slices)
    Output: (main_logit, dem_aux, sem_aux)
            main_logit: (B, 1, H, W)
            dem_aux:    (B, 1, H, W)       — auxiliary, training only
            sem_aux:    (B, 1, H/16, W/16) — auxiliary, training only
    DEM/SEM only provide auxiliary losses; their features do not feed the
    decoder.
    """
    BASE = 32

    def __init__(self, in_ch=3, out_ch=1, img_size=256,
                 tf_heads=8, tf_win=4, dropout=0.1, base=32):
        super().__init__()
        b = base

        # ── Encoder ──
        self.enc1 = nn.Sequential(ConvBnRelu(in_ch, b),   ResBlock(b))
        self.enc2 = nn.Sequential(ConvBnRelu(b,  b*2, s=2), ResBlock(b*2))
        self.enc3 = nn.Sequential(ConvBnRelu(b*2, b*4, s=2), ResBlock(b*4))
        self.enc4 = nn.Sequential(ConvBnRelu(b*4, b*8, s=2), ResBlock(b*8))

        # ── Transformer bottleneck ──
        self.bottleneck = nn.Sequential(
            nn.Conv2d(b*8, b*16, 3, stride=2, padding=1, bias=False),
            _gn(b*16), nn.ReLU(inplace=True),
            TransformerBottleneck(b*16, tf_heads, tf_win, dropout))

        # ── DEM / SEM auxiliary supervision ──
        # DEM: fuses enc1 (low-level) with enc4 (high-level)
        self.dem = DetailEnhanceModule(b, b*8, b*4)
        # SEM: refines bottleneck features
        self.sem = SemanticEnhanceModule(b*16, b*8)

        # ── Decoder ──
        self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = nn.Sequential(ConvBnRelu(b*16, b*8), ResBlock(b*8))
        self.up3  = nn.ConvTranspose2d(b*8,  b*4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBnRelu(b*8,  b*4), ResBlock(b*4))
        self.up2  = nn.ConvTranspose2d(b*4,  b*2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBnRelu(b*4,  b*2), ResBlock(b*2))
        self.up1  = nn.ConvTranspose2d(b*2,  b,   2, stride=2)
        self.dec1 = nn.Sequential(ConvBnRelu(b*2,  b),   ResBlock(b))

        # ── Final head ──
        self.head = nn.Conv2d(b, out_ch, 1)

        # Store feature dim for ClassificationDataset's feature extraction
        self.feature_dim = b * 4   # fused DEM output channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode(self, x):
        """Return encoder feature maps (used by classification head)."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)
        return e1, e2, e3, e4, bn

    def forward(self, x):
        e1, e2, e3, e4, bn = self.encode(x)

        # auxiliary supervision (features unused; only the aux logits are)
        _, dem_aux = self.dem(e1, e4)
        _, sem_aux = self.sem(bn)

        # decoder
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1), dem_aux, sem_aux


# ── Stage 3 Classification Models ────────────────────────────────────────────

class _StreamHead(nn.Module):
    """Shared projection MLP after global average pooling."""
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3))
        self.out_dim = proj_dim

    def forward(self, x): return self.net(x)


def _patch_first_conv(backbone, in_ch: int):
    """Replace the first Conv2d to accept in_ch channels, averaging pretrained weights."""
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            old = module
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            # average pretrained 3-ch weights across new channels
            with torch.no_grad():
                new.weight[:] = old.weight.mean(dim=1, keepdim=True).expand(
                    -1, in_ch, -1, -1) / in_ch * 3
            # set it on the parent module
            parts = name.split(".")
            parent = backbone
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new)
            return

class ResNetClassifier(nn.Module):
    """
    Dual-stream ResNet-50:
      slice_stream  — 2.5D triplet slices → GAP → MLP
      volume_stream — aggregates N slice embeddings → MLP
    Late-fused into a shared classifier.
    """
    def __init__(self, num_classes=5, proj_dim=256, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        _patch_first_conv(backbone, in_ch=12)        # ← patch BEFORE wrapping
        self.slice_backbone  = nn.Sequential(*list(backbone.children())[:-1])

        # separate (untied) backbone for the volume stream
        weights2 = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb2 = tvm.resnet50(weights=weights2)
        _patch_first_conv(bb2, in_ch=12)             # ← patch BEFORE wrapping
        self.volume_backbone = nn.Sequential(*list(bb2.children())[:-1])

        feat_dim = 2048  # ResNet-50 output before final FC

        self.slice_head  = _StreamHead(feat_dim, proj_dim)
        self.volume_head = _StreamHead(feat_dim, proj_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(proj_dim, num_classes))

    def forward_slice(self, x):
        """x: (B, 12, H, W)"""
        f = self.slice_backbone(x).flatten(1)   # (B, 2048)
        return self.slice_head(f)                # (B, proj_dim)

    def forward_volume(self, vx):
        """vx: (B, N, 12, H, W)  N = slices per volume"""
        B, N, C, H, W = vx.shape
        f = self.volume_backbone(vx.reshape(B * N, C, H, W)).flatten(1)
        f = f.reshape(B, N, -1).mean(dim=1)     # mean-pool over slices
        return self.volume_head(f)               # (B, proj_dim)

    def forward(self, slice_x, vol_x):
        s = self.forward_slice(slice_x)
        v = self.forward_volume(vol_x)
        return self.classifier(torch.cat([s, v], dim=1))


class EfficientNetClassifier(nn.Module):
    """
    Dual-stream EfficientNet-B3:
      slice_stream  — 2.5D triplet slices → GAP → MLP
      volume_stream — mean-pooled slice embeddings → MLP
    Late-fused into a shared classifier.
    """
    def __init__(self, num_classes=5, proj_dim=256, pretrained=True):
        super().__init__()
        weights  = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        bb1 = tvm.efficientnet_b3(weights=weights)
        bb2 = tvm.efficientnet_b3(weights=weights)

        _patch_first_conv(bb1, in_ch=12)
        _patch_first_conv(bb2, in_ch=12)

        self.slice_features  = bb1.features
        self.volume_features = bb2.features
        self.pool            = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1536  # EfficientNet-B3 output channels

        self.slice_head  = _StreamHead(feat_dim, proj_dim)
        self.volume_head = _StreamHead(feat_dim, proj_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(proj_dim, num_classes))

    def _extract(self, feats, x):
        return self.pool(feats(x)).flatten(1)   # (B, feat_dim)

    def forward_slice(self, x):
        return self.slice_head(self._extract(self.slice_features, x))

    def forward_volume(self, vx):
        B, N, C, H, W = vx.shape
        f = self._extract(self.volume_features,
                          vx.reshape(B * N, C, H, W))
        f = f.reshape(B, N, -1).mean(dim=1)
        return self.volume_head(f)

    def forward(self, slice_x, vol_x):
        s = self.forward_slice(slice_x)
        v = self.forward_volume(vol_x)
        return self.classifier(torch.cat([s, v], dim=1))

# ── Swin Classifier (4-phase, 12-channel input) ──────────────────────────

class SwinClassifier(nn.Module):
    """
    Swin-Tiny or Swin-Base classifier for 4-phase CT.
    Input: (B, 12, H, W)  — 3-slice triplet x 4 phases
    Patch embedding adapted from 3 → 12 channels.
    Last Swin stage unfrozen for fine-tuning.
    """
    CONFIGS = {
        "swin_tiny": ("swin_tiny_patch4_window7_224",  768),
        "swin_base": ("swin_base_patch4_window7_224", 1024),
    }

    def __init__(self, variant="swin_tiny", num_classes=5,
                 proj_dim=256, pretrained=True):
        super().__init__()
        model_name, feat_dim = self.CONFIGS[variant]
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg")

        # adapt patch embedding 3 → 12 channels
        pe = self.backbone.patch_embed.proj
        new_pe = nn.Conv2d(12, pe.out_channels,
                           kernel_size=pe.kernel_size,
                           stride=pe.stride,
                           padding=pe.padding,
                           bias=(pe.bias is not None))
        with torch.no_grad():
            # tile the 3-ch weights across 12 channels then scale
            new_pe.weight[:] = pe.weight.mean(dim=1, keepdim=True) \
                                         .expand(-1, 12, -1, -1) / 4.0
            if pe.bias is not None:
                new_pe.bias.copy_(pe.bias)
        self.backbone.patch_embed.proj = new_pe

        # freeze all, then unfreeze last stage
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.backbone.layers[-1].parameters():
            p.requires_grad_(True)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.BatchNorm1d(proj_dim // 2), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(proj_dim // 2, num_classes))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  SwinClassifier({variant}): "
              f"{trainable:,} trainable / {total:,} total params")

    def forward(self, slice_x, vol_x=None):
        """
        slice_x: (B, 12, H, W)
        vol_x ignored — Swin processes the centre-slice 12-ch input only.
        """
        x = slice_x
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224),
                              mode="bilinear", align_corners=False)
        return self.head(self.backbone(x))


class EnsembleClassifier(nn.Module):
    """
    Late-fuse ResNet and EfficientNet logits with learned weights.
    Call .resnet / .efficientnet attributes to access individual models.
    """
    def __init__(self, num_classes=5, proj_dim=256, pretrained=True):
        super().__init__()
        self.resnet      = ResNetClassifier(num_classes, proj_dim, pretrained)
        self.efficientnet= EfficientNetClassifier(num_classes, proj_dim, pretrained)
        # learnable scalar fusion weights (softmax-normalised at forward)
        self.logit_w     = nn.Parameter(torch.ones(2))
    def forward(self, slice_x, vol_x):
        r  = self.resnet(slice_x, vol_x)       # (B, num_classes)
        e  = self.efficientnet(slice_x, vol_x) # (B, num_classes)
        w  = F.softmax(self.logit_w, dim=0)    # (2,)
        return w[0] * r + w[1] * e
# ── Losses ────────────────────────────────────────────────────────────────────
class TverskyFocalLoss(nn.Module):
    """Tversky loss (alpha/beta trade-off) + binary focal loss."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0,
                 seg_w=0.6, focal_w=0.4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gamma = gamma
        self.seg_w, self.focal_w = seg_w, focal_w
    def forward(self, pred, target):
        p   = torch.sigmoid(pred)
        tp  = (p * target).sum()
        fp  = ((1 - target) * p).sum()
        fn  = (target * (1 - p)).sum()
        tv  = tp / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt  = torch.exp(-bce)
        fl  = ((1 - pt) ** self.gamma * bce).mean()
        return self.seg_w * (1 - tv) + self.focal_w * fl
class DeepSupervisionLoss(nn.Module):
    """
    Combines main seg loss + DEM auxiliary + SEM auxiliary.
    Weights follow the DS2Net uncertainty-adaptive scheme
    (fixed approximation: 0.6 / 0.2 / 0.2).
    """
    def __init__(self):
        super().__init__()
        self.seg_loss = TverskyFocalLoss()
    def forward(self, main, dem_aux, sem_aux, target):
        H, W = target.shape[-2:]
        # resize auxiliaries to target resolution
        dem_up = F.interpolate(dem_aux, (H, W),
                               mode='bilinear', align_corners=False)
        sem_up = F.interpolate(sem_aux, (H, W),
                               mode='bilinear', align_corners=False)
        l_main = self.seg_loss(main, target)
        l_dem  = self.seg_loss(dem_up, target)
        l_sem  = self.seg_loss(sem_up, target)
        return 0.6 * l_main + 0.2 * l_dem + 0.2 * l_sem