# Technical Documentation

This directory contains detailed technical documentation for various components of the SmartCash system.

## Table of Contents

### Model Architecture
- [MODEL.md](MODEL.md) - Detailed model architecture with EfficientNet-B4 backbone
- [ARSITEKTUR.md](ARSITEKTUR.md) - Model architecture documentation in Indonesian

### Training System
- [two_phase_loss_calculation_analysis.md](two_phase_loss_calculation_analysis.md) - Comprehensive analysis of two-phase mode training loss calculation
- [uncertainty_weighting_analysis.md](uncertainty_weighting_analysis.md) - Detailed explanation of uncertainty-based weighting in multi-task learning
- [tensor_format_converter_fix.md](tensor_format_converter_fix.md) - Analysis and fix for tensor format converter warning

### Data and Evaluation
- [DATASET.md](DATASET.md) - Dataset documentation
- [EVALUASI.md](EVALUASI.md) - Evaluation methodology and metrics

## Overview

The SmartCash system uses a sophisticated multi-layer detection approach for Indonesian banknote recognition. The system implements a two-phase training approach with uncertainty-based weighting for optimal performance.

### Two-Phase Training

The training process is divided into two distinct phases:

1. **Phase 1**: Backbone freezing with single-layer training
2. **Phase 2**: Full model fine-tuning with multi-layer uncertainty-based weighting

This approach allows for stable initial learning followed by fine-tuned optimization across all detection layers.

### Multi-Layer Detection

The system uses three detection layers:
- **Layer 1**: Full banknote detection (classes 0-6)
- **Layer 2**: Denomination-specific features (classes 7-13)
- **Layer 3**: Common features (classes 14-16)

Each layer is optimized for detecting specific aspects of banknotes, allowing for more accurate and robust detection.