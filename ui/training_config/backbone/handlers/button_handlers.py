"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen backbone
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_backbone_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI backbone.
    
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
        
        # Import ModelManager untuk mendapatkan model yang dioptimalkan
        from smartcash.model.manager import ModelManager
        
        # Default config berdasarkan model yang dioptimalkan
        default_model_type = 'efficient_optimized'
        default_model_config = ModelManager.OPTIMIZED_MODELS[default_model_type]
        
        default_config = {
            'model': {
                'model_type': default_model_type,
                'backbone': default_model_config['backbone'],
                'pretrained': True,
                'freeze_backbone': True,
                'freeze_layers': 3,
                'use_attention': default_model_config.get('use_attention', False),
                'use_residual': default_model_config.get('use_residual', False),
                'use_ciou': default_model_config.get('use_ciou', False)
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Update config dari nilai UI
            if 'model' not in current_config:
                current_config['model'] = {}
            
            # Simpan model_type yang dipilih
            current_config['model']['model_type'] = ui_components['model_type'].value
                
            # Simpan backbone dan pengaturan dasar
            current_config['model']['backbone'] = ui_components['backbone_type'].value
            current_config['model']['pretrained'] = ui_components['pretrained'].value
            current_config['model']['freeze_backbone'] = ui_components['freeze_backbone'].value
            current_config['model']['freeze_layers'] = ui_components['freeze_layers'].value
            
            # Simpan fitur optimasi
            current_config['model']['use_attention'] = ui_components['use_attention'].value
            current_config['model']['use_residual'] = ui_components['use_residual'].value
            current_config['model']['use_ciou'] = ui_components['use_ciou'].value
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config():
            if not config or 'model' not in config: return
            
            try:
                # Import display di dalam fungsi untuk menghindari masalah scope
                from IPython.display import display, HTML
                
                # Update model_type jika tersedia dengan penanganan error
                if 'model_type' in config['model']:
                    try:
                        model_type = config['model']['model_type']
                        if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
                            ui_components['model_type'].value = model_type
                            if logger: logger.debug(f"✅ Model type diperbarui ke: {model_type}")
                        else:
                            # Jika model_type tidak valid, gunakan default
                            ui_components['model_type'].value = default_model_type
                            if logger: logger.info(f"ℹ️ Model type tidak valid, menggunakan default: {default_model_type}")
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update model_type: {str(e)}")
                
                # Update nilai UI dari config dengan penanganan error
                if 'backbone' in config['model']:
                    try:
                        backbone = config['model']['backbone']
                        # Periksa apakah dropdown dinonaktifkan
                        is_disabled = getattr(ui_components['backbone_type'], 'disabled', False)
                        
                        if is_disabled:
                            # Aktifkan sementara untuk mengubah nilai
                            ui_components['backbone_type'].disabled = False
                            ui_components['backbone_type'].value = backbone
                            ui_components['backbone_type'].disabled = True
                        else:
                            ui_components['backbone_type'].value = backbone
                            
                        if logger: logger.debug(f"✅ Backbone diperbarui ke: {backbone}")
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update backbone: {str(e)}")
                    
                # Update checkbox pretrained dengan penanganan error
                if 'pretrained' in config['model']:
                    try:
                        ui_components['pretrained'].value = config['model']['pretrained']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update pretrained: {str(e)}")
                    
                # Update checkbox freeze_backbone dengan penanganan error
                if 'freeze_backbone' in config['model']:
                    try:
                        ui_components['freeze_backbone'].value = config['model']['freeze_backbone']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update freeze_backbone: {str(e)}")
                    
                # Update slider freeze_layers dengan penanganan error
                if 'freeze_layers' in config['model']:
                    try:
                        ui_components['freeze_layers'].value = config['model']['freeze_layers']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update freeze_layers: {str(e)}")
                
                # Update fitur optimasi dengan penanganan error
                if 'use_attention' in config['model']:
                    try:
                        ui_components['use_attention'].value = config['model']['use_attention']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_attention: {str(e)}")
                    
                if 'use_residual' in config['model']:
                    try:
                        ui_components['use_residual'].value = config['model']['use_residual']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_residual: {str(e)}")
                    
                if 'use_ciou' in config['model']:
                    try:
                        ui_components['use_ciou'].value = config['model']['use_ciou']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_ciou: {str(e)}")
                
                if logger: logger.info("✅ UI backbone diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi backbone sudah tidak diperlukan karena sudah dihandle oleh on_model_change
        def update_backbone_info():
            # Tidak melakukan apa-apa karena sudah dihandle oleh on_model_change
            pass
        
        # Handler buttons
        def on_save_click(b): 
            save_config(ui_components, config, "configs/model_config.yaml", update_config_from_ui, "Model Backbone")
        
        def on_reset_click(b): 
            reset_config(ui_components, config, default_config, update_ui_from_config, "Model Backbone")
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_backbone_info'] = update_backbone_info
        
        # Inisialisasi UI dari config
        update_ui_from_config()
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup backbone button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup backbone button handler: {str(e)}")
    
    return ui_components
