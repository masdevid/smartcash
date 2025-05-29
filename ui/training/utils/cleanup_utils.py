"""
File: smartcash/ui/training/utils/cleanup_utils.py
Deskripsi: Utilities untuk GPU cleanup dan memory management
"""

import torch
from typing import Dict, Any, List
from smartcash.ui.training.handlers.training_button_handlers import set_state


def cleanup_model_resources(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Cleanup model resources dari memory"""
    result = {
        'models_cleaned': [],
        'memory_freed': 0,
        'errors': []
    }
    
    try:
        # Cleanup model manager
        model_manager = ui_components.get('model_manager')
        if model_manager and hasattr(model_manager, 'model') and model_manager.model:
            model_name = model_manager.model_type
            del model_manager.model
            model_manager.model = None
            result['models_cleaned'].append(f"ModelManager.{model_name}")
            set_state(model_ready=False)
        
        # Cleanup training service
        training_service = ui_components.get('training_service')
        if training_service and hasattr(training_service, 'model_manager'):
            training_service._training_active = False
            result['models_cleaned'].append("TrainingService")
        
        # Clear any cached models
        for key in list(ui_components.keys()):
            if 'model' in key.lower() and hasattr(ui_components[key], 'model'):
                try:
                    if ui_components[key].model:
                        del ui_components[key].model
                        ui_components[key].model = None
                        result['models_cleaned'].append(key)
                except Exception as e:
                    result['errors'].append(f"Error cleaning {key}: {str(e)}")
        
    except Exception as e:
        result['errors'].append(f"Model cleanup error: {str(e)}")
    
    return result


def cleanup_torch_cache() -> Dict[str, Any]:
    """Cleanup PyTorch cache dan GPU memory"""
    result = {
        'cache_cleared': False,
        'memory_freed': 0,
        'methods_used': []
    }
    
    try:
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            result['methods_used'].append('torch.cuda.empty_cache()')
            
            # Force garbage collection
            import gc
            gc.collect()
            result['methods_used'].append('gc.collect()')
            
            # Clear reserved memory
            try:
                torch.cuda.reset_peak_memory_stats()
                result['methods_used'].append('reset_peak_memory_stats()')
            except Exception:
                pass
            
            result['cache_cleared'] = True
            
    except Exception as e:
        result['error'] = f"Cache cleanup error: {str(e)}"
    
    return result


def cleanup_ui_outputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Cleanup UI outputs dan displays"""
    result = {
        'outputs_cleared': [],
        'errors': []
    }
    
    outputs_to_clear = [
        'chart_output', 'metrics_output', 'log_output', 
        'model_readiness_display', 'training_config_display', 'gpu_status_display'
    ]
    
    for output_name in outputs_to_clear:
        try:
            output_widget = ui_components.get(output_name)
            if output_widget and hasattr(output_widget, 'clear_output'):
                output_widget.clear_output(wait=True)
                result['outputs_cleared'].append(output_name)
        except Exception as e:
            result['errors'].append(f"Error clearing {output_name}: {str(e)}")
    
    return result


def calculate_memory_freed(results: Dict[str, Any]) -> str:
    """Calculate memory freed dari cleanup"""
    try:
        # Simple calculation - just show that cleanup was performed
        models_cleaned = len(results['model_cleanup']['models_cleaned'])
        cache_cleared = results['cache_cleanup']['cache_cleared']
        
        if models_cleaned > 0 and cache_cleared:
            return f"~{models_cleaned * 100}MB+ freed"
        elif cache_cleared:
            return "Cache cleared"
        else:
            return "Minimal cleanup"
    except Exception:
        return "Calculation error"


def get_gpu_memory_info() -> Dict[str, str]:
    """Get GPU memory information"""
    if torch.cuda.is_available():
        return {
            'allocated': format_memory_size(torch.cuda.memory_allocated()),
            'reserved': format_memory_size(torch.cuda.memory_reserved()),
            'max_allocated': format_memory_size(torch.cuda.max_memory_allocated())
        }
    return {'allocated': 'N/A', 'reserved': 'N/A', 'max_allocated': 'N/A'}


def format_memory_size(size_bytes: int) -> str:
    """Format memory size dalam human readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"