"""
File: smartcash/ui/model/evaluate/constants.py
Description: Constants and enums for evaluation module
"""

from enum import Enum
from typing import Dict, Any, List

class EvaluationOperation(Enum):
    """Evaluation operations enum."""
    TEST_SCENARIO = "test_scenario"
    TEST_ALL_SCENARIOS = "test_all_scenarios"
    LOAD_CHECKPOINT = "load_checkpoint"
    AUGMENT_DATASET = "augment_dataset"
    GENERATE_REPORT = "generate_report"

class EvaluationPhase(Enum):
    """Evaluation phases enum."""
    IDLE = "idle"
    AUGMENTING = "augmenting"
    EVALUATING = "evaluating"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    ERROR = "error"

class TestScenario(Enum):
    """Test scenarios enum."""
    POSITION_VARIATION = "position_variation"
    LIGHTING_VARIATION = "lighting_variation"

class BackboneModel(Enum):
    """Backbone models enum."""
    CSPDARKNET = "cspdarknet"
    EFFICIENTNET_B4 = "efficientnet_b4"

# Default configuration for evaluation
DEFAULT_CONFIG = {
    "evaluation": {
        "data": {
            "test_dir": "data/preprocessed/test",
            "evaluation_dir": "data/evaluation",
            "results_dir": "data/evaluation/results"
        },
        "scenarios": {
            "position_variation": {
                "name": "Position Variation",
                "enabled": True,
                "augmentation_config": {
                    "num_variations": 5,
                    "rotation_range": [-30, 30],
                    "translation_range": [-0.2, 0.2],
                    "scale_range": [0.8, 1.2]
                }
            },
            "lighting_variation": {
                "name": "Lighting Variation", 
                "enabled": True,
                "augmentation_config": {
                    "num_variations": 5,
                    "brightness_range": [-0.3, 0.3],
                    "contrast_range": [0.7, 1.3],
                    "gamma_range": [0.7, 1.3]
                }
            }
        },
        "metrics": {
            "map": {
                "enabled": True,
                "iou_thresholds": [0.5, 0.75]
            },
            "precision_recall": {
                "enabled": True,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.5,
                "per_class": True
            },
            "f1_score": {
                "enabled": True,
                "beta": 1.0,
                "per_class": True
            },
            "inference_time": {
                "enabled": True
            }
        },
        "checkpoints": {
            "auto_select_best": True,
            "sort_by": "val_map",
            "max_checkpoints": 10
        }
    },
    "inference": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 100,
        "img_size": 640,
        "half": True,
        "fuse": True,
        "nms": True
    }
}

# UI Configuration
UI_CONFIG = {
    "title": "Model Evaluation",
    "subtitle": "Test model performance with scenario-based evaluation",
    "icon": "🎯",
    "theme": {
        "primary_color": "#28a745",
        "secondary_color": "#6c757d",
        "success_color": "#28a745",
        "warning_color": "#ffc107",
        "error_color": "#dc3545"
    }
}

# Scenario configurations
SCENARIO_CONFIGS = {
    "position_variation": {
        "name": "Position Variation",
        "description": "Test model robustness against position changes",
        "icon": "📐",
        "color": "#007bff",
        "augmentations": [
            "rotation",
            "translation", 
            "scale"
        ]
    },
    "lighting_variation": {
        "name": "Lighting Variation",
        "description": "Test model robustness against lighting changes",
        "icon": "💡",
        "color": "#ffc107",
        "augmentations": [
            "brightness",
            "contrast",
            "gamma"
        ]
    }
}

# Model configurations
MODEL_CONFIGS = {
    "cspdarknet": {
        "name": "CSPDarknet",
        "description": "CSPDarknet backbone model",
        "icon": "🌙",
        "color": "#6f42c1"
    },
    "efficientnet_b4": {
        "name": "EfficientNet-B4",
        "description": "EfficientNet-B4 backbone model", 
        "icon": "⚡",
        "color": "#20c997"
    }
}

# Operation status messages
OPERATION_MESSAGES = {
    EvaluationOperation.TEST_SCENARIO: {
        "start": "🧪 Starting scenario evaluation...",
        "progress": "📊 Evaluating {scenario} scenario...",
        "success": "✅ Scenario evaluation completed successfully",
        "error": "❌ Scenario evaluation failed"
    },
    EvaluationOperation.TEST_ALL_SCENARIOS: {
        "start": "🚀 Starting comprehensive evaluation...",
        "progress": "🔄 Testing scenario {current}/{total}...",
        "success": "🎉 All scenarios evaluated successfully",
        "error": "❌ Evaluation failed for some scenarios"
    },
    EvaluationOperation.LOAD_CHECKPOINT: {
        "start": "📂 Loading checkpoint...",
        "progress": "⏳ Loading model from {checkpoint}...",
        "success": "✅ Checkpoint loaded successfully",
        "error": "❌ Failed to load checkpoint"
    },
    EvaluationOperation.AUGMENT_DATASET: {
        "start": "🔄 Starting dataset augmentation...",
        "progress": "🎨 Generating {scenario} variations...",
        "success": "✅ Dataset augmentation completed",
        "error": "❌ Dataset augmentation failed"
    },
    EvaluationOperation.GENERATE_REPORT: {
        "start": "📄 Generating evaluation report...",
        "progress": "📊 Compiling results...",
        "success": "📋 Report generated successfully",
        "error": "❌ Report generation failed"
    }
}

# Metric display configurations
METRIC_CONFIGS = {
    "map": {
        "name": "mAP",
        "description": "Mean Average Precision",
        "format": "{:.3f}",
        "color": "#007bff",
        "icon": "🎯",
        "default_enabled": True
    },
    "accuracy": {
        "name": "Accuracy",
        "description": "Detection Accuracy",
        "format": "{:.3f}",
        "color": "#17a2b8",
        "icon": "✅",
        "default_enabled": True
    },
    "precision": {
        "name": "Precision",
        "description": "Precision Score",
        "format": "{:.3f}",
        "color": "#28a745",
        "icon": "🔍",
        "default_enabled": True
    },
    "recall": {
        "name": "Recall", 
        "description": "Recall Score",
        "format": "{:.3f}",
        "color": "#ffc107",
        "icon": "📊",
        "default_enabled": True
    },
    "f1_score": {
        "name": "F1 Score",
        "description": "F1 Score",
        "format": "{:.3f}",
        "color": "#6f42c1",
        "icon": "⚖️",
        "default_enabled": True
    },
    "inference_time": {
        "name": "Inference Time",
        "description": "Average inference time per image",
        "format": "{:.2f}ms",
        "color": "#20c997",
        "icon": "⏱️",
        "default_enabled": False
    }
}

# Available metrics for selection
AVAILABLE_METRICS = list(METRIC_CONFIGS.keys())

# Default enabled metrics
DEFAULT_ENABLED_METRICS = [
    metric for metric, config in METRIC_CONFIGS.items() 
    if config.get("default_enabled", False)
]

# Available scenarios for the UI
AVAILABLE_SCENARIOS = list(SCENARIO_CONFIGS.keys())

# Available backbone models
AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())

# Default test matrix (2 scenarios x 2 models = 4 total tests)
DEFAULT_TEST_MATRIX = [
    {"scenario": "position_variation", "model": "cspdarknet"},
    {"scenario": "position_variation", "model": "efficientnet_b4"},
    {"scenario": "lighting_variation", "model": "cspdarknet"},
    {"scenario": "lighting_variation", "model": "efficientnet_b4"}
]