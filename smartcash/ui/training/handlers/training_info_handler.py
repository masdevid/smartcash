"""
File: smartcash/ui/training/handlers/training_info_handler.py
Deskripsi: Handler untuk pembaruan informasi training
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.training.handlers.training_handler_utils import display_status_panel

def update_training_info(ui_components: Dict[str, Any], logger=None):
    """
    Memperbarui informasi training dari konfigurasi.
    
    Args:
        ui_components: Komponen UI
        logger: Logger untuk mencatat aktivitas
    """
    try:
        # Dapatkan logger jika tidak disediakan
        logger = logger or get_logger("training_ui")
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager()
        
        # Ambil konfigurasi dari ConfigManager
        hyperparams_config = config_manager.get_module_config('hyperparameters')
        training_config = config_manager.get_module_config('training')
        model_config = config_manager.get_module_config('model')
        
        # Tampilkan informasi
        with ui_components['info_box']:
            ui_components['info_box'].clear_output()
            
            # Informasi model
            display(HTML(f"""
            <h3 style="margin-bottom:10px">📊 Konfigurasi Training</h3>
            <div style="display:flex;flex-wrap:wrap">
                <div style="flex:1;min-width:300px;margin-right:10px">
                    <h4>Model</h4>
                    <ul>
                        <li><b>Backbone:</b> {model_config.get('backbone', 'efficientnet_b4')}</li>
                        <li><b>Layer Mode:</b> {training_config.get('training_utils', {}).get('layer_mode', 'single')}</li>
                    </ul>
                </div>
                <div style="flex:1;min-width:300px;margin-right:10px">
                    <h4>Hyperparameter</h4>
                    <ul>
                        <li><b>Batch Size:</b> {hyperparams_config.get('hyperparameters', {}).get('batch_size', 16)}</li>
                        <li><b>Epochs:</b> {hyperparams_config.get('hyperparameters', {}).get('epochs', 100)}</li>
                        <li><b>Image Size:</b> {hyperparams_config.get('hyperparameters', {}).get('image_size', 640)}</li>
                        <li><b>Optimizer:</b> {hyperparams_config.get('hyperparameters', {}).get('optimizer', 'Adam')}</li>
                        <li><b>Learning Rate:</b> {hyperparams_config.get('hyperparameters', {}).get('learning_rate', 0.001)}</li>
                    </ul>
                </div>
                <div style="flex:1;min-width:300px">
                    <h4>Training Strategy</h4>
                    <ul>
                        <li><b>Multi-scale:</b> {str(training_config.get('multi_scale', True))}</li>
                        <li><b>Validasi:</b> {str(training_config.get('validation', {}).get('enabled', True))}</li>
                        <li><b>Augmentasi:</b> {str(hyperparams_config.get('hyperparameters', {}).get('augment', True))}</li>
                    </ul>
                </div>
            </div>
            """))
        
        return True
    except Exception as e:
        logger.error(f"❌ Error saat memperbarui informasi training: {str(e)}")
        
        # Tampilkan pesan error
        display_status_panel(
            ui_components, 
            f"Error saat memperbarui informasi training: {str(e)}", 
            is_error=True
        )
        
        return False
