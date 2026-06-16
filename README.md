# LiverTumorClassification

## MCT-LTDiag — Multi-phase CT Liver-Tumor Diagnosis

A two-stage deep learning pipeline for liver segmentation and tumor detection in multi-phase abdominal CT scans, implemented in PyTorch and designed to run in Google Colab.


## Overview

MCT-LTDiag segments the liver from multi-phase contrast-enhanced CT volumes (Stage 1), then uses those liver masks as a spatial prior for tumor detection and classification (Stage 2). The current repository contains Stage 1 (v5).

The model uses a DS²Net architecture — combining Detail Enhancement Modules (DEM) and Semantic Enhancement Modules (SEM) in a U-Net-style encoder-decoder — augmented with a Multi-Head Self-Attention (MHA) bottleneck at the deepest feature scale.
