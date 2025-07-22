"""
File: smartcash/ui/model/evaluation/configs/evaluation_defaults.py
Default configuration for evaluation module using BaseUIModule pattern.
"""

from typing import Dict, Any
from enum import Enum


class EvaluationPhase(Enum):
    """Evaluation phase enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"

def get_default_evaluation_config() -> Dict[str, Any]:
    """
    Get default evaluation configuration.
    
    Returns:
        Default evaluation configuration dictionary
    """
    return {
        'evaluation': {
            'scenarios': {
                'position_variation': {
                    'enabled': True,
                    'description': 'Test model performance across different object positions',
                    'test_cases': ['center', 'left', 'right', 'top', 'bottom', 'corner']
                },
                'lighting_variation': {
                    'enabled': True,
                    'description': 'Test model performance under different lighting conditions',
                    'test_cases': ['bright', 'dim', 'artificial', 'natural', 'mixed']
                }
            },
            'models': {
                'source': 'trained',  # 'trained', 'checkpoint', 'best'
                'auto_discover': True,
                'selection_criteria': 'best_map',  # 'best_map', 'latest', 'manual'
                'max_models': 4
            },
            'metrics': {
                'primary': ['mAP@0.5', 'mAP@0.75', 'precision', 'recall'],
                'secondary': ['accuracy', 'f1_score', 'inference_time'],
                'thresholds': {
                    'mAP@0.5': 0.5,
                    'mAP@0.75': 0.3,
                    'precision': 0.7,
                    'recall': 0.7
                }
            },
            'execution': {
                'run_mode': 'all_scenarios',  # 'all_scenarios', 'position_only', 'lighting_only', 'single_model'
                'parallel_execution': False,
                'timeout_per_test': 300,  # seconds
                'save_intermediate_results': True,
                'compare_with_baseline': True
            },
            'output': {
                'save_dir': 'runs/evaluation',
                'name': 'smartcash_evaluation',
                'exist_ok': True,
                'save_detailed_results': True,
                'generate_report': True
            }
        },
        'model_selection': {
            'auto_detect': True,
            'source': 'trained',  # 'trained', 'checkpoint', 'pretrained'
            'filter_criteria': {
                'min_map': 0.5,
                'min_epochs': 10,
                'status': 'completed'
            }
        },
        'ui': {
            'show_advanced_options': False,
            'auto_refresh_models': True,
            'real_time_updates': True,
            'compact_view': False
        },
        'reporting': {
            'format': 'html',  # 'html', 'json', 'csv'
            'include_charts': True,
            'include_comparisons': True,
            'detailed_metrics': True
        }
    }

def get_evaluation_scenarios() -> Dict[str, Dict[str, Any]]:
    """Get available evaluation scenarios."""
    return {
        'position_variation': {
            'name': 'Position Variation',
            'description': 'Test model robustness across different object positions',
            'icon': '📐',
            'test_count': 6,
            'estimated_time': '5-10 minutes'
        },
        'lighting_variation': {
            'name': 'Lighting Variation', 
            'description': 'Test model performance under different lighting conditions',
            'icon': '💡',
            'test_count': 5,
            'estimated_time': '5-8 minutes'
        },
        'all_scenarios': {
            'name': 'Comprehensive Evaluation',
            'description': 'Run all evaluation scenarios for complete assessment',
            'icon': '🎯',
            'test_count': 8,
            'estimated_time': '10-20 minutes'
        }
    }


def get_evaluation_metrics() -> Dict[str, Dict[str, Any]]:
    """Get available evaluation metrics."""
    return {
        'mAP@0.5': {
            'name': 'mAP@0.5',
            'description': 'Mean Average Precision at IoU threshold 0.5',
            'format': '.3f',
            'higher_better': True,
            'critical': True
        },
        'mAP@0.75': {
            'name': 'mAP@0.75',
            'description': 'Mean Average Precision at IoU threshold 0.75',
            'format': '.3f',
            'higher_better': True,
            'critical': True
        },
        'precision': {
            'name': 'Precision',
            'description': 'Precision score for all classes',
            'format': '.3f',
            'higher_better': True,
            'critical': True
        },
        'recall': {
            'name': 'Recall',
            'description': 'Recall score for all classes',
            'format': '.3f',
            'higher_better': True,
            'critical': True
        },
        'f1_score': {
            'name': 'F1-Score',
            'description': 'F1 score combining precision and recall',
            'format': '.3f',
            'higher_better': True,
            'critical': False
        },
        'accuracy': {
            'name': 'Accuracy',
            'description': 'Overall accuracy across all classes',
            'format': '.3f',
            'higher_better': True,
            'critical': False
        },
        'inference_time': {
            'name': 'Inference Time',
            'description': 'Average inference time per image (ms)',
            'format': '.1f',
            'higher_better': False,
            'critical': False
        }
    }


def get_model_selection_criteria() -> Dict[str, str]:
    """Get model selection criteria options."""
    return {
        'best_map': 'Best mAP@0.5',
        'best_f1': 'Best F1-Score',
        'latest': 'Latest Trained',
        'fastest': 'Fastest Inference',
        'manual': 'Manual Selection'
    }

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


# Constants for evaluation validation
EVALUATION_VALIDATION_CONFIG = {
    'min_test_timeout': 60,  # seconds
    'max_test_timeout': 1800,  # 30 minutes
    'min_models': 1,
    'max_models': 10,
    'required_metrics': ['mAP@0.5', 'precision', 'recall'],
    'min_threshold_map': 0.1,
    'max_threshold_map': 1.0
}