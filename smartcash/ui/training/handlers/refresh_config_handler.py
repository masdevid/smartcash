"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/training/handlers/refresh_config_handler.py
Deskripsi: Handler untuk tombol refresh konfigurasi pada modul training
"""

from typing import Dict, Any
from smartcash.common.config.manager import ConfigManager
from smartcash.common.logger import get_logger

logger = get_logger('smartcash.ui.training')

def handle_refresh_config(ui_components: Dict[str, Any]) -> None:
    """Handle refresh config button dengan mengambil konfigurasi terbaru dari modul lain"""
    try:
        # Update status panel untuk memberikan feedback
        status_panel = ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            status_panel.value = "⏳ Memperbarui konfigurasi dari modul lain..."
        
        # Langsung refresh config tanpa threading
        _refresh_config(ui_components)
        
    except Exception as e:
        logger.error(f"❌ Error refreshing config: {str(e)}")
        if ui_components.get('status_panel') and hasattr(ui_components.get('status_panel'), 'value'):
            ui_components['status_panel'].value = f"❌ Error refreshing config: {str(e)}"

def _refresh_config(ui_components: Dict[str, Any]) -> None:
    """Async config refresh dengan proper error handling"""
    try:
        # Mendapatkan config manager
        config_manager = ConfigManager.get_instance()
        
        # Mendapatkan konfigurasi dari semua modul terkait
        backbone_config = config_manager.get_module_config('backbone')
        hyperparameters_config = config_manager.get_module_config('hyperparameters')
        training_strategy_config = config_manager.get_module_config('training_strategy')
        
        # Gabungkan konfigurasi
        combined_config = {
            'model': backbone_config.get('model', {}),
            'hyperparameters': hyperparameters_config.get('hyperparameters', {}),
            'training_strategy': training_strategy_config.get('training_strategy', {}),
            'paths': {
                'data_dir': '/data/preprocessed',
                'checkpoint_dir': config_manager.get_config_value(['training_strategy', 'training_strategy', 'utils', 'checkpoint_dir'], 'runs/train/checkpoints'),
                'tensorboard_dir': 'runs/tensorboard'
            }
        }
        
        # Update config tabs jika tersedia
        _update_config_ui(ui_components, combined_config)
        
        # Jika training initializer tersedia di globals, trigger config update
        _notify_config_updates(combined_config)
        
        # Update status panel
        if ui_components.get('status_panel') and hasattr(ui_components.get('status_panel'), 'value'):
            ui_components['status_panel'].value = "✅ Konfigurasi berhasil diperbarui dari modul lain"
        
        logger.success("✅ Konfigurasi training berhasil diperbarui")
        
    except Exception as e:
        logger.error(f"❌ Error melakukan refresh config: {str(e)}")
        if ui_components.get('status_panel') and hasattr(ui_components.get('status_panel'), 'value'):
            ui_components['status_panel'].value = f"❌ Error melakukan refresh config: {str(e)}"

def _update_config_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI config tabs dengan konfigurasi baru"""
    try:
        from smartcash.ui.training.components.config_tabs import update_config_tabs
        
        # Update config tabs jika tersedia
        if ui_components.get('config_tabs') and hasattr(ui_components['config_tabs'], 'children'):
            config_tabs = ui_components['config_tabs']
            new_tabs = update_config_tabs(config_tabs, config)
            ui_components['config_tabs'] = new_tabs
            
    except Exception as e:
        logger.error(f"❌ Error updating config tabs: {str(e)}")

def _notify_config_updates(config: Dict[str, Any]) -> None:
    """Notifikasi perubahan config ke handlers lain jika tersedia"""
    try:
        # Get training initializer jika tersedia
        from smartcash.ui.training.training_init import get_training_initializer
        initializer = get_training_initializer()
        
        # Trigger config update callbacks
        if initializer and hasattr(initializer, 'trigger_config_update'):
            initializer.trigger_config_update(config)
            
    except Exception as e:
        # Silent fail - tidak perlu menginterupsi proses
        pass
