"""
File: smartcash/ui/training/handlers/cleanup_handler.py
Deskripsi: Handler untuk GPU cleanup dan memory management
"""

import torch
from typing import Dict, Any
from smartcash.ui.training.handlers.training_button_handlers import get_state, set_state
from smartcash.ui.training.utils.training_status_utils import update_training_status
from smartcash.ui.training.utils.cleanup_utils import (
    cleanup_model_resources, cleanup_torch_cache, cleanup_ui_outputs, 
    get_gpu_memory_info, calculate_memory_freed
)
from smartcash.ui.training.utils.training_display_utils import display_cleanup_results


def handle_cleanup_gpu(ui_components: Dict[str, Any]):
    """Handle GPU cleanup dan memory management"""
    if get_state()['active']:
        return
    
    logger = ui_components.get('logger')
    gpu_status_display = ui_components.get('gpu_status_display')
    
    logger and logger.info("🧹 Memulai GPU cleanup...")
    
    # Perform cleanup operations
    cleanup_results = _perform_cleanup_operations(ui_components)
    
    # Display results
    if gpu_status_display:
        display_cleanup_results(gpu_status_display, cleanup_results)
    
    # Update status
    memory_freed = calculate_memory_freed(cleanup_results)
    status_msg = f"🧹 GPU cleanup selesai - {memory_freed} memory dibebaskan"
    update_training_status(ui_components, status_msg, 'success')
    
    logger and logger.success(status_msg)


def _perform_cleanup_operations(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Perform semua cleanup operations"""
    cleanup_results = {
        'gpu_memory_before': get_gpu_memory_info(),
        'model_cleanup': cleanup_model_resources(ui_components),
        'cache_cleanup': cleanup_torch_cache(),
        'ui_cleanup': cleanup_ui_outputs(ui_components),
        'gpu_memory_after': None
    }
    
    # Update model readiness state
    set_state(model_ready=False)
    
    # Get memory info after cleanup
    cleanup_results['gpu_memory_after'] = get_gpu_memory_info()
    
    return cleanup_results


def cleanup_training_outputs(ui_components: Dict[str, Any]):
    """Quick cleanup untuk training outputs saja"""
    outputs_to_clear = ['chart_output', 'metrics_output']
    
    for output_name in outputs_to_clear:
        try:
            output_widget = ui_components.get(output_name)
            if output_widget and hasattr(output_widget, 'clear_output'):
                output_widget.clear_output(wait=True)
        except Exception:
            pass  # Silent fail


def emergency_gpu_cleanup():
    """Emergency GPU cleanup tanpa UI dependencies"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            import gc
            gc.collect()
            
            return True
    except Exception:
        return False
    
    return False