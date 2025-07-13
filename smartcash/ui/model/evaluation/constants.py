"""
Constants for evaluation module with 2×4 model evaluation matrix.
Total tests: 2 scenarios × 2 backbones × 2 layer modes = 8 evaluations
"""

from enum import Enum
from typing import Dict, Any, List

class EvaluationOperation(Enum):
    """Evaluation operations."""
    POSITION_VARIATION = "position_variation"
    LIGHTING_VARIATION = "lighting_variation"
    ALL_SCENARIOS = "all_scenarios"
    LOAD_CHECKPOINT = "load_checkpoint"
    GENERATE_REPORT = "generate_report"

class EvaluationPhase(Enum):
    """Evaluation phases."""
    IDLE = "idle"
    LOADING_CHECKPOINT = "loading_checkpoint"
    RUNNING_SCENARIO = "running_scenario"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    ERROR = "error"

class TestScenario(Enum):
    """2 Research test scenarios from evaluation_config.yaml."""
    POSITION_VARIATION = "position_variation"
    LIGHTING_VARIATION = "lighting_variation"

class BackboneType(Enum):
    """2 Backbone model types."""
    CSPDARKNET = "cspdarknet"
    EFFICIENTNET_B4 = "efficientnet_b4"

class LayerMode(Enum):
    """2 Layer training modes.""" 
    SINGLE = "single"
    MULTILAYER = "multilayer"

# UI Configuration
UI_CONFIG = {
    'module_name': 'evaluation',
    'parent_module': 'model',
    'title': '🎯 Model Evaluation',
    'description': 'Evaluate model performance across 4 research scenarios',
    'icon': '🎯',
    'version': '2.0.0'
}

# 2 Research Scenarios from evaluation_config.yaml
RESEARCH_SCENARIOS = {
    "position_variation": {
        "name": "Position Variation",
        "description": "Test model robustness against position and rotation changes",
        "icon": "📐",
        "color": "#007bff",
        "enabled": True,
        "augmentation_config": {
            "num_variations": 5
        }
    },
    "lighting_variation": {
        "name": "Lighting Variation", 
        "description": "Test model robustness against lighting condition changes",
        "icon": "💡",
        "color": "#ffc107",
        "enabled": True,
        "augmentation_config": {
            "num_variations": 5
        }
    }
}

# 4 Model combinations: 2 backbones × 2 layer modes
MODEL_COMBINATIONS = [
    {"backbone": "cspdarknet", "layer_mode": "single", "name": "CSPDarknet Single-Layer"},
    {"backbone": "cspdarknet", "layer_mode": "multilayer", "name": "CSPDarknet Multi-Layer"},
    {"backbone": "efficientnet_b4", "layer_mode": "single", "name": "EfficientNet-B4 Single-Layer"},
    {"backbone": "efficientnet_b4", "layer_mode": "multilayer", "name": "EfficientNet-B4 Multi-Layer"}
]

# Total evaluation matrix: 2 scenarios × 4 models = 8 tests
EVALUATION_MATRIX = []
for scenario in RESEARCH_SCENARIOS.keys():
    for model in MODEL_COMBINATIONS:
        EVALUATION_MATRIX.append({
            "scenario": scenario,
            "backbone": model["backbone"], 
            "layer_mode": model["layer_mode"],
            "test_name": f"{RESEARCH_SCENARIOS[scenario]['name']} - {model['name']}"
        })

# Evaluation metrics configuration
EVALUATION_METRICS = {
    "map": {
        "name": "mAP@0.5", 
        "description": "Mean Average Precision at IoU=0.5",
        "format": "{:.3f}",
        "icon": "🎯",
        "color": "#007bff"
    },
    "map_75": {
        "name": "mAP@0.75",
        "description": "Mean Average Precision at IoU=0.75", 
        "format": "{:.3f}",
        "icon": "🎯",
        "color": "#0056b3"
    },
    "precision": {
        "name": "Precision",
        "description": "Detection Precision",
        "format": "{:.3f}",
        "icon": "🔍",
        "color": "#28a745"
    },
    "recall": {
        "name": "Recall",
        "description": "Detection Recall", 
        "format": "{:.3f}",
        "icon": "📊",
        "color": "#ffc107"
    },
    "f1_score": {
        "name": "F1 Score",
        "description": "F1 Score (2*precision*recall)/(precision+recall)",
        "format": "{:.3f}", 
        "icon": "⚖️",
        "color": "#6f42c1"
    },
    "accuracy": {
        "name": "Accuracy",
        "description": "Overall classification accuracy",
        "format": "{:.3f}",
        "icon": "🎯",
        "color": "#e83e8c"
    },
    "inference_time": {
        "name": "Inference Time",
        "description": "Average inference time per image",
        "format": "{:.2f}ms",
        "icon": "⏱️", 
        "color": "#20c997"
    }
}

# Button configurations
BUTTON_CONFIG = {
    'run_all_scenarios': {
        'text': '🚀 Run All Scenarios',
        'style': 'primary',
        'tooltip': 'Run evaluation across all scenarios and models (8 total tests)'
    },
    'run_position_scenario': {
        'text': '📐 Position Variation',
        'style': 'info',
        'tooltip': 'Run position variation scenario only (4 model tests)'
    },
    'run_lighting_scenario': {
        'text': '💡 Lighting Variation', 
        'style': 'info',
        'tooltip': 'Run lighting variation scenario only (4 model tests)'
    },
    'load_checkpoint': {
        'text': '📂 Load Models',
        'style': 'secondary',
        'tooltip': 'Load best trained model checkpoints for evaluation'
    },
    'export_results': {
        'text': '📊 Export Results',
        'style': 'success',
        'tooltip': 'Export evaluation results and metrics'
    },
    'clear_results': {
        'text': '🗑️ Clear Results',
        'style': 'warning',
        'tooltip': 'Clear previous evaluation results'
    }
}

# Operation messages for logging
OPERATION_MESSAGES = {
    EvaluationOperation.POSITION_VARIATION: {
        "start": "📐 Starting position variation scenario...",
        "progress": "📐 Testing position variations... {progress}%",
        "success": "✅ Position variation scenario completed",
        "error": "❌ Position variation scenario failed"
    },
    EvaluationOperation.LIGHTING_VARIATION: {
        "start": "💡 Starting lighting variation scenario...",
        "progress": "💡 Testing lighting variations... {progress}%", 
        "success": "✅ Lighting variation scenario completed",
        "error": "❌ Lighting variation scenario failed"
    },
    EvaluationOperation.ALL_SCENARIOS: {
        "start": "🚀 Starting comprehensive evaluation...",
        "progress": "🔄 Testing {current}/{total} model combinations...",
        "success": "🎉 All scenarios evaluation completed successfully",
        "error": "❌ Comprehensive evaluation failed"
    },
    EvaluationOperation.LOAD_CHECKPOINT: {
        "start": "📂 Loading model checkpoint...",
        "progress": "⏳ Loading model from {checkpoint}...",
        "success": "✅ Model checkpoint loaded successfully",
        "error": "❌ Failed to load model checkpoint"
    },
    EvaluationOperation.GENERATE_REPORT: {
        "start": "📄 Generating evaluation report...",
        "progress": "📊 Compiling results from {scenarios} scenarios...",
        "success": "📋 Evaluation report generated successfully", 
        "error": "❌ Report generation failed"
    }
}

# Default configuration
DEFAULT_EVALUATION_CONFIG = {
    "evaluation": {
        "scenarios": RESEARCH_SCENARIOS,
        "metrics": list(EVALUATION_METRICS.keys()),
        "checkpoint": {
            "path": "data/checkpoints",
            "auto_select_best": True,
            "sort_by": "val_map"
        },
        "output": {
            "results_dir": "data/evaluation/results",
            "export_format": ["json", "csv"],
            "save_images": True
        },
        "inference": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "img_size": 640
        }
    }
}

# Available scenarios for UI selection
AVAILABLE_SCENARIOS = list(RESEARCH_SCENARIOS.keys())

# Available metrics for UI selection  
AVAILABLE_METRICS = list(EVALUATION_METRICS.keys())

# Module metadata
MODULE_METADATA = {
    'name': 'evaluation',
    'display_name': 'Model Evaluation',
    'description': 'Comprehensive model evaluation with 4 research scenarios',
    'version': '2.0.0',
    'author': 'SmartCash Team',
    'scenarios_count': len(RESEARCH_SCENARIOS),
    'metrics_count': len(EVALUATION_METRICS)
}