# LiverTumorClassification

## MCT-LTDiag — Multi-phase CT Liver-Tumor Diagnosis

A two-stage deep learning pipeline for liver segmentation and tumor detection in multi-phase abdominal CT scans, implemented in PyTorch and designed to run in Google Colab.


## Overview

MCT-LTDiag segments the liver from multi-phase contrast-enhanced CT volumes (Stage 1), then uses those liver masks as a spatial prior for tumor detection and classification (Stage 2). The current repository contains Stage 1 (v5).

The model uses a DS²Net architecture — combining Detail Enhancement Modules (DEM) and Semantic Enhancement Modules (SEM) in a U-Net-style encoder-decoder — augmented with a Multi-Head Self-Attention (MHA) bottleneck at the deepest feature scale.

## Pipeline Overview
 
```
Input CT (multi-phase)
        │
        ▼
1. Preprocessing
   └─ Data cleaning, HU windowing, resampling, data leakage check
        │
        ▼
2. Segmentation
   ├─ Stage 1 — Liver segmentation   (DS2Net | UNet-Hybrid)
   │             MHA / Cross-MTA attention
   └─ Stage 2 — Tumor segmentation  within liver ROI
        │
        ▼
3. Classification
   └─ Tumor subtype (EfficientNet | UNet | Swin-Tiny | Swin-Base | ViT)
        │
        ▼
Output: Liver mask + Tumor mask + Subtype label
        (BCLM | CRLM | HCC | HH | ICC)
```
 
---
 
## Dataset — MCT-LTDiag
 
517 contrast-enhanced multi-phase CT cases spanning five hepatic tumor subtypes:
 
| Subtype | Full name | N |
|---|---|---|
| BCLM | Breast cancer liver metastasis | 115 |
| CRLM | Colorectal liver metastasis | 103 |
| HCC | Hepatocellular carcinoma | 103 |
| HH | Hepatic hemangioma | 96 |
| ICC | Intrahepatic cholangiocarcinoma | 100 |
 
Each case includes four CT phases: **non-contrast (NC), arterial (art), portal-venous (pvp), and delayed**. Expert-annotated liver and tumor masks are provided.
 
> Wu et al., *MCT-LTDiag: Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors* (2025).
 
---
 
## Models
 
### Segmentation
 
| Model | Description |
|---|---|
| **DS2Net** | Detail-Semantic Dual-supervision Network. Swin-Tiny backbone (20-channel pseudo-3D input: 5 context slices × 4 phases), DS² deep supervision (5 heads), MHA self-attention bottleneck at d4 (14×14, 8 heads). |
| **UNet-Hybrid** | UNet-style encoder-decoder with transformer attention (Cross-MTA). Encoder initialized from Stage 1 weights for Stage 2 tumor segmentation. |
 
