"""
Default configuration for evaluation module.
Loads from smartcash/configs/evaluation_config.yaml
"""

from typing import Dict, Any
from smartcash.ui.model.evaluation.constants import (
    RESEARCH_SCENARIOS,
    EVALUATION_METRICS,
    MODEL_COMBINATIONS,
    EVALUATION_MATRIX
)

def get_default_evaluation_config() -> Dict[str, Any]:
    """
    Get default evaluation configuration matching evaluation_config.yaml structure.
    
    Returns:
        Default evaluation configuration dictionary
    """
    return {
        "evaluation": {
            # Data paths (matches evaluation_config.yaml)
            "data": {
                "test_dir": "data/preprocessed/test",
                "evaluation_dir": "data/evaluation", 
                "results_dir": "data/evaluation/results"
            },
            
            # 2 Scenarios from evaluation_config.yaml
            "scenarios": {
                "position_variation": {
                    "name": "Position Variation",
                    "enabled": True,
                    "augmentation_config": {
                        "num_variations": 5
                    }
                },
                "lighting_variation": {
                    "name": "Lighting Variation",
                    "enabled": True,
                    "augmentation_config": {
                        "num_variations": 5
                    }
                }
            },
            
            # Metrics configuration (matches evaluation_config.yaml)
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
                "accuracy": {
                    "enabled": True,
                    "per_class": True
                },
                "inference_time": {
                    "enabled": True
                }
            },
            
            # Checkpoint management (matches evaluation_config.yaml)
            "checkpoints": {
                "auto_select_best": True,
                "sort_by": "val_map",
                "max_checkpoints": 10
            },
            
            # Analysis modules (matches evaluation_config.yaml)
            "analysis": {
                "currency_analysis": {
                    "enabled": True,
                    "primary_layer": "banknote",
                    "confidence_threshold": 0.3
                },
                "class_analysis": {
                    "enabled": True,
                    "compute_confusion_matrix": True
                }
            },
            
            # Model selection (4 combinations: 2 backbones × 2 layer modes)
            "models": {
                "selected_models": [
                    "cspdarknet_single",
                    "cspdarknet_multilayer", 
                    "efficientnet_b4_single",
                    "efficientnet_b4_multilayer"
                ],
                "auto_select_best": True
            },
            
            # Execution options
            "execution": {
                "run_mode": "all_scenarios",  # "all_scenarios", "position_only", "lighting_only"
                "parallel_execution": False,
                "save_intermediate_results": True
            }
        },
        
        # Inference settings
        "inference": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "img_size": 640,
            "half": True,
            "fuse": True,
            "nms": True
        },
        
        # UI configuration
        "ui": {
            "show_progress": True,
            "show_live_metrics": True,
            "auto_refresh_results": True,
            "results_per_page": 10
        }
    }

def get_scenario_configs() -> Dict[str, Any]:
    """Get scenario configuration details."""
    return RESEARCH_SCENARIOS

def get_model_combinations() -> list:
    """Get all 4 model combinations (2 backbones × 2 layer modes)."""
    return MODEL_COMBINATIONS

def get_evaluation_matrix() -> list:
    """Get complete evaluation matrix (2 scenarios × 4 models = 8 tests)."""
    return EVALUATION_MATRIX

def get_available_metrics() -> Dict[str, Any]:
    """Get available evaluation metrics."""
    return EVALUATION_METRICS

def validate_evaluation_config(config: Dict[str, Any]) -> bool:
    """
    Validate evaluation configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not config:
            return False
            
        # Check if config has the expected top-level structure
        if not isinstance(config, dict):
            return False
            
        # Check for required top-level sections with flexible validation
        if 'evaluation' not in config:
            return False
            
        # Get evaluation config with defaults
        eval_config = config.get('evaluation', {})
        
        # Ensure we have basic required sections with empty defaults
        required_sections = ['data', 'scenarios', 'models', 'metrics', 'settings']
        for section in required_sections:
            if section not in eval_config:
                eval_config[section] = {}
        
        # Ensure we have basic model configurations
        if 'models' in eval_config and not isinstance(eval_config['models'], dict):
            eval_config['models'] = {}
            
        # Ensure we have basic scenario configurations
        if 'scenarios' in eval_config and not isinstance(eval_config['scenarios'], dict):
            eval_config['scenarios'] = {}
            
        # Ensure we have basic metrics
        if 'metrics' in eval_config and not isinstance(eval_config['metrics'], list):
            eval_config['metrics'] = ['accuracy']  # Default metric
            
        # Update the config with any defaults we've set
        config['evaluation'] = eval_config
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Validation error: {e}")
        traceback.print_exc()
        return False