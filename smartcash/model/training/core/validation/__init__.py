#!/usr/bin/env python3
"""
Validation components for SmartCash training pipeline.

This package provides modular, SRP-compliant validation components:
- Configuration loading and class mapping
- Target processing and format conversion  
- Model inference with optimization
- Unified metrics processing (loss, mAP, classification)
- Batch processing orchestration

All components are designed to be under 400 lines and focused on single responsibilities.
"""

from .validation_config_loader import ValidationConfigLoader
from .validation_target_processor import ValidationTargetProcessor
from .validation_model_inference import ValidationModelInference
from .validation_metrics_processor import ValidationMetricsProcessor
from .batch_processor import (
    ValidationBatchProcessor,
    create_validation_batch_processor,
    BatchMetrics
)

__all__ = [
    'ValidationConfigLoader',
    'ValidationTargetProcessor', 
    'ValidationModelInference',
    'ValidationMetricsProcessor',
    'ValidationBatchProcessor',
    'create_validation_batch_processor',
    'BatchMetrics'
]