"""
File: smartcash/ui/training_config/training_strategy/handlers/form_handlers.py
Deskripsi: Handler untuk form UI pada komponen strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

# Import langsung dari modul terpisah
from smartcash.ui.training_config.training_strategy.handlers.config_extractor import update_config_from_ui
from smartcash.ui.training_config.training_strategy.handlers.info_updater import update_training_strategy_info
from smartcash.ui.training_config.training_strategy.handlers.status_handlers import update_status_panel

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
        logger.info(f"{ICONS.get('info', 'ℹ️')} Setting up form handlers untuk strategi pelatihan")
        
        # List semua komponen yang perlu diobservasi
        components_to_observe = [
            # Parameter Utilitas Training
            'experiment_name', 'checkpoint_dir', 'tensorboard', 
            'log_metrics_every', 'visualize_batch_every', 'gradient_clipping', 
            'mixed_precision', 'layer_mode',
            
            # Parameter utama
            'enabled_checkbox', 'batch_size_slider', 'epochs_slider', 'learning_rate_slider',
            
            # Optimizer
            'optimizer_dropdown', 'weight_decay_slider', 'momentum_slider',
            
            # Scheduler
            'scheduler_checkbox', 'scheduler_dropdown', 'warmup_epochs_slider', 'min_lr_slider',
            
            # Early stopping
            'early_stopping_checkbox', 'patience_slider', 'min_delta_slider',
            
            # Checkpoint
            'checkpoint_checkbox', 'save_best_only_checkbox', 'save_freq_slider',
            
            # Validasi dan Evaluasi
            'validation_frequency', 'iou_threshold', 'conf_threshold',
            
            # Multi-scale Training
            'multi_scale'
        ]
        
        # Daftar komponen yang berhasil diobservasi
        observed_components = []
        
        # Handler untuk perubahan komponen
        def on_component_change(change):
            if change['name'] == 'value':
                try:
                    # Update info strategi pelatihan
                    update_training_strategy_info(ui_components)
                    
                    # Jika perubahan pada komponen utama, tampilkan pesan di status panel
                    widget_name = None
                    for name, widget in ui_components.items():
                        if widget is change['owner']:
                            widget_name = name
                            break
                    
                    if widget_name:
                        update_status_panel(
                            ui_components, 
                            f"Parameter '{widget_name}' diubah. Klik 'Simpan' untuk menyimpan perubahan.", 
                            "info"
                        )
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memproses perubahan: {str(e)}")
        
        # Register observers untuk semua komponen yang ada
        for component_name in components_to_observe:
            if component_name in ui_components and hasattr(ui_components[component_name], 'observe'):
                try:
                    ui_components[component_name].observe(on_component_change, names='value')
                    observed_components.append(component_name)
                except Exception as e:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak dapat register observer untuk '{component_name}': {str(e)}")
        
        logger.info(f"{ICONS.get('success', '✅')} Berhasil register observer untuk {len(observed_components)} komponen")
        
        # Cleanup function untuk unregister semua observer
        def cleanup():
            try:
                for component_name in observed_components:
                    if component_name in ui_components and hasattr(ui_components[component_name], 'unobserve'):
                        ui_components[component_name].unobserve(on_component_change, names='value')
                
                logger.info(f"{ICONS.get('success', '✅')} Training strategy form handlers cleaned up")
            except Exception as e:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup_form_handlers'] = cleanup
        
        # Tambahkan update_training_strategy_info ke ui_components jika belum ada
        if 'update_training_strategy_info' not in ui_components:
            ui_components['update_training_strategy_info'] = lambda: update_training_strategy_info(ui_components)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat setup form handlers: {str(e)}")
        return ui_components