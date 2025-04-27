"""
File: smartcash/ui/training_config/hyperparameters/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_hyperparameters_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI hyperparameter.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Import dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Validasi config
        if config is None: config = {}
        
        # Default config
        default_config = {
            'hyperparameters': {
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 0.001,
                    'weight_decay': 0.0005,
                    'momentum': 0.9
                },
                'scheduler': {
                    'type': 'cosine',
                    'warmup_epochs': 3,
                    'step_size': 30,
                    'gamma': 0.1
                },
                'augmentation': {
                    'use_augmentation': True,
                    'mosaic': True,
                    'mixup': False,
                    'flip': True,
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4
                }
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Update config dari nilai UI
            if 'hyperparameters' not in current_config:
                current_config['hyperparameters'] = {}
                
            # Optimizer
            if 'optimizer' not in current_config['hyperparameters']:
                current_config['hyperparameters']['optimizer'] = {}
                
            current_config['hyperparameters']['optimizer']['type'] = ui_components['optimizer_type'].value
            current_config['hyperparameters']['optimizer']['lr'] = ui_components['learning_rate'].value
            current_config['hyperparameters']['optimizer']['weight_decay'] = ui_components['weight_decay'].value
            current_config['hyperparameters']['optimizer']['momentum'] = ui_components['momentum'].value
            
            # Scheduler
            if 'scheduler' not in current_config['hyperparameters']:
                current_config['hyperparameters']['scheduler'] = {}
                
            current_config['hyperparameters']['scheduler']['type'] = ui_components['scheduler_type'].value
            current_config['hyperparameters']['scheduler']['warmup_epochs'] = ui_components['warmup_epochs'].value
            current_config['hyperparameters']['scheduler']['step_size'] = ui_components['step_size'].value
            current_config['hyperparameters']['scheduler']['gamma'] = ui_components['gamma'].value
            
            # Augmentation
            if 'augmentation' not in current_config['hyperparameters']:
                current_config['hyperparameters']['augmentation'] = {}
                
            current_config['hyperparameters']['augmentation']['use_augmentation'] = ui_components['use_augmentation'].value
            current_config['hyperparameters']['augmentation']['mosaic'] = ui_components['mosaic'].value
            current_config['hyperparameters']['augmentation']['mixup'] = ui_components['mixup'].value
            current_config['hyperparameters']['augmentation']['flip'] = ui_components['flip'].value
            current_config['hyperparameters']['augmentation']['hsv_h'] = ui_components['hsv_h'].value
            current_config['hyperparameters']['augmentation']['hsv_s'] = ui_components['hsv_s'].value
            current_config['hyperparameters']['augmentation']['hsv_v'] = ui_components['hsv_v'].value
            
            # Update info hyperparameter
            update_hyperparameters_info()
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config():
            if not config or 'hyperparameters' not in config: return
            
            try:
                # Optimizer
                if 'optimizer' in config['hyperparameters']:
                    if 'type' in config['hyperparameters']['optimizer']:
                        ui_components['optimizer_type'].value = config['hyperparameters']['optimizer']['type']
                    if 'lr' in config['hyperparameters']['optimizer']:
                        ui_components['learning_rate'].value = config['hyperparameters']['optimizer']['lr']
                    if 'weight_decay' in config['hyperparameters']['optimizer']:
                        ui_components['weight_decay'].value = config['hyperparameters']['optimizer']['weight_decay']
                    if 'momentum' in config['hyperparameters']['optimizer']:
                        ui_components['momentum'].value = config['hyperparameters']['optimizer']['momentum']
                
                # Scheduler
                if 'scheduler' in config['hyperparameters']:
                    if 'type' in config['hyperparameters']['scheduler']:
                        ui_components['scheduler_type'].value = config['hyperparameters']['scheduler']['type']
                    if 'warmup_epochs' in config['hyperparameters']['scheduler']:
                        ui_components['warmup_epochs'].value = config['hyperparameters']['scheduler']['warmup_epochs']
                    if 'step_size' in config['hyperparameters']['scheduler']:
                        ui_components['step_size'].value = config['hyperparameters']['scheduler']['step_size']
                    if 'gamma' in config['hyperparameters']['scheduler']:
                        ui_components['gamma'].value = config['hyperparameters']['scheduler']['gamma']
                
                # Augmentation
                if 'augmentation' in config['hyperparameters']:
                    if 'use_augmentation' in config['hyperparameters']['augmentation']:
                        ui_components['use_augmentation'].value = config['hyperparameters']['augmentation']['use_augmentation']
                    if 'mosaic' in config['hyperparameters']['augmentation']:
                        ui_components['mosaic'].value = config['hyperparameters']['augmentation']['mosaic']
                    if 'mixup' in config['hyperparameters']['augmentation']:
                        ui_components['mixup'].value = config['hyperparameters']['augmentation']['mixup']
                    if 'flip' in config['hyperparameters']['augmentation']:
                        ui_components['flip'].value = config['hyperparameters']['augmentation']['flip']
                    if 'hsv_h' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_h'].value = config['hyperparameters']['augmentation']['hsv_h']
                    if 'hsv_s' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_s'].value = config['hyperparameters']['augmentation']['hsv_s']
                    if 'hsv_v' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_v'].value = config['hyperparameters']['augmentation']['hsv_v']
                
                # Update info hyperparameter
                update_hyperparameters_info()
                
                if logger: logger.info("✅ UI hyperparameter diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi hyperparameter
        def update_hyperparameters_info():
            try:
                # Dapatkan nilai dari UI
                optimizer_type = ui_components['optimizer_type'].value
                lr = ui_components['learning_rate'].value
                weight_decay = ui_components['weight_decay'].value
                scheduler_type = ui_components['scheduler_type'].value
                use_augmentation = ui_components['use_augmentation'].value
                
                # Buat informasi HTML
                info_html = f"""
                <h4>Ringkasan Hyperparameter</h4>
                <ul>
                    <li><b>Optimizer:</b> {optimizer_type} (LR: {lr:.6f}, Weight Decay: {weight_decay:.6f})</li>
                    <li><b>Scheduler:</b> {scheduler_type.capitalize()}</li>
                    <li><b>Augmentasi:</b> {'Aktif' if use_augmentation else 'Nonaktif'}</li>
                </ul>
                <p><i>Catatan: Hyperparameter yang tepat sangat penting untuk performa model yang optimal.</i></p>
                """
                
                ui_components['hyperparameters_info'].value = info_html
            except Exception as e:
                ui_components['hyperparameters_info'].value = f"<p style='color:red'>❌ Error: {str(e)}</p>"
        
        # Handler buttons
        def on_save_click(b): 
            save_config(ui_components, config, "configs/hyperparameters_config.yaml", update_config_from_ui, "Hyperparameter")
        
        def on_reset_click(b): 
            reset_config(ui_components, config, default_config, update_ui_from_config, "Hyperparameter")
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_hyperparameters_info'] = update_hyperparameters_info
        
        # Inisialisasi UI dari config
        update_ui_from_config()
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup hyperparameter button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup hyperparameter button handler: {str(e)}")
    
    return ui_components
