"""
File: smartcash/ui/training_config/training_strategy/handlers/form_handlers.py
Deskripsi: Handler untuk form UI pada komponen strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    update_config_from_ui,
    update_training_strategy_info
)

logger = get_logger(__name__)

def setup_training_strategy_form_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk form pada komponen UI strategi pelatihan.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Handler untuk perubahan komponen
        def on_component_change(change):
            if change['name'] == 'value':
                # Update config dari UI
                if 'update_config_from_ui' in ui_components and callable(ui_components['update_config_from_ui']):
                    ui_components['update_config_from_ui'](ui_components)
                
                # Update info strategi pelatihan
                update_training_strategy_info(ui_components)
        
        # Register observers untuk semua komponen
        # Tab 1: Parameter Utilitas Training
        ui_components['experiment_name'].observe(on_component_change)
        ui_components['checkpoint_dir'].observe(on_component_change)
        ui_components['tensorboard'].observe(on_component_change)
        ui_components['log_metrics_every'].observe(on_component_change)
        ui_components['visualize_batch_every'].observe(on_component_change)
        ui_components['gradient_clipping'].observe(on_component_change)
        ui_components['mixed_precision'].observe(on_component_change)
        ui_components['layer_mode'].observe(on_component_change)
        
        # Tab 2: Validasi dan Evaluasi
        ui_components['validation_frequency'].observe(on_component_change)
        ui_components['iou_threshold'].observe(on_component_change)
        ui_components['conf_threshold'].observe(on_component_change)
        
        # Tab 3: Multi-scale Training
        ui_components['multi_scale'].observe(on_component_change)
        
        # Cleanup function
        def cleanup():
            try:
                # Hapus semua observer
                # Tab 1: Parameter Utilitas Training
                ui_components['experiment_name'].unobserve(on_component_change)
                ui_components['checkpoint_dir'].unobserve(on_component_change)
                ui_components['tensorboard'].unobserve(on_component_change)
                ui_components['log_metrics_every'].unobserve(on_component_change)
                ui_components['visualize_batch_every'].unobserve(on_component_change)
                ui_components['gradient_clipping'].unobserve(on_component_change)
                ui_components['mixed_precision'].unobserve(on_component_change)
                ui_components['layer_mode'].unobserve(on_component_change)
                
                # Tab 2: Validasi dan Evaluasi
                ui_components['validation_frequency'].unobserve(on_component_change)
                ui_components['iou_threshold'].unobserve(on_component_change)
                ui_components['conf_threshold'].unobserve(on_component_change)
                
                # Tab 3: Multi-scale Training
                ui_components['multi_scale'].unobserve(on_component_change)
                
                logger.info(f"{ICONS.get('success', '✅')} Training strategy form handlers cleaned up")
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup'] = cleanup
        
        # Tambahkan update_training_strategy_info ke ui_components
        ui_components['update_training_strategy_info'] = lambda: update_training_strategy_info(ui_components)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup form handlers: {str(e)}")
        return ui_components
