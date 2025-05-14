"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen backbone
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_backbone_button_handlers(ui_components: Dict[str, Any], env: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handlers untuk tombol-tombol di UI backbone.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dictionary berisi komponen UI yang diupdate
    """
    # Import LogLevel untuk logging
    from smartcash.common.logger import LogLevel
    
    try:
        # Import dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Dapatkan logger jika tersedia atau buat UI logger baru
        logger = ui_components.get('logger', None)
        if logger is None and 'status' in ui_components:
            from smartcash.ui.utils.ui_logger import create_direct_ui_logger
            logger = create_direct_ui_logger(ui_components, 'backbone_buttons')
            ui_components['logger'] = logger
        
        # Validasi config
        if config is None: config = {}
        
        # Import ModelManager untuk mendapatkan model yang dioptimalkan
        from smartcash.model.manager import ModelManager
        
        # Default config berdasarkan model yang dioptimalkan
        default_model_type = 'efficient_basic'
        
        # Pastikan model type ada dalam OPTIMIZED_MODELS
        if hasattr(ModelManager, 'OPTIMIZED_MODELS') and default_model_type in ModelManager.OPTIMIZED_MODELS:
            default_model_config = ModelManager.OPTIMIZED_MODELS[default_model_type]
        else:
            # Fallback jika model tidak ditemukan
            default_model_config = {
                'backbone': 'efficientnet_b4',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            }
        
        default_config = {
            'model': {
                'type': default_model_type,
                'backbone': default_model_config['backbone'],
                'backbone_pretrained': True,
                'backbone_freeze': False,
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
            if 'model_type' in ui_components:
                current_config['model']['type'] = ui_components['model_type'].value
                
            # Simpan backbone dan pengaturan dasar
            if 'backbone_type' in ui_components:
                current_config['model']['backbone'] = ui_components['backbone_type'].value
            
            if 'pretrained' in ui_components:
                current_config['model']['backbone_pretrained'] = ui_components['pretrained'].value
            
            if 'freeze_backbone' in ui_components:
                current_config['model']['backbone_freeze'] = ui_components['freeze_backbone'].value
            
            if 'freeze_layers' in ui_components:
                current_config['model']['freeze_layers'] = ui_components['freeze_layers'].value
            
            # Sesuaikan fitur optimasi berdasarkan model type
            model_type = current_config['model'].get('type', 'efficient_basic')
            
            # Default semua fitur optimasi ke False
            current_config['model']['use_attention'] = False
            current_config['model']['use_residual'] = False
            current_config['model']['use_ciou'] = False
            
            # Jika model_type adalah efficient_advanced, sesuaikan properti
            if model_type == 'efficient_advanced':
                current_config['model']['use_attention'] = True
                current_config['model']['use_residual'] = True
                current_config['model']['use_ciou'] = True
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config(current_ui_components=None, current_config=None):
            # Gunakan parameter jika disediakan, jika tidak gunakan variabel dari scope luar
            nonlocal ui_components, config
            _ui_components = current_ui_components or ui_components
            _config = current_config or config
            
            if not _config or 'model' not in _config: return
            
            try:
                # Import display di dalam fungsi untuk menghindari masalah scope
                from IPython.display import display, HTML
                
                # Update model_type jika tersedia dengan penanganan error
                if 'type' in _config['model']:
                    try:
                        model_type = _config['model']['type']
                        if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
                            _ui_components['model_type'].value = model_type
                            if logger:
                                if hasattr(logger, 'log'):
                                    logger.log(LogLevel.DEBUG, f"Model type diperbarui ke: {model_type}")
                                else:
                                    logger.debug(f"✅ Model type diperbarui ke: {model_type}")
                        else:
                            # Jika model_type tidak valid, gunakan default
                            _ui_components['model_type'].value = default_model_type
                            if logger: 
                                if hasattr(logger, 'log'):
                                    logger.log(LogLevel.INFO, f"Model type tidak valid, menggunakan default: {default_model_type}")
                                else:
                                    logger.info(f"ℹ️ Model type tidak valid, menggunakan default: {default_model_type}")
                    except Exception as e:
                        if logger:
                            if hasattr(logger, 'log'):
                                logger.log(LogLevel.WARNING, f"Error saat update model_type: {str(e)}")
                            else:
                                logger.warning(f"⚠️ Error saat update model_type: {str(e)}")
                
                # Update nilai UI dari config dengan penanganan error yang lebih kuat
                if 'backbone' in _config['model']:
                    try:
                        backbone = _config['model']['backbone']
                        
                        # Periksa apakah backbone ada dalam opsi dropdown
                        available_options = getattr(_ui_components['backbone_type'], 'options', [])
                        if not backbone in available_options:
                            if logger:
                                if hasattr(logger, 'log'):
                                    logger.log(LogLevel.WARNING, f"Backbone '{backbone}' tidak ditemukan dalam opsi dropdown")
                                else:
                                    logger.warning(f"⚠️ Backbone '{backbone}' tidak ditemukan dalam opsi dropdown")
                            
                            # Cari backbone alternatif yang valid
                            if 'efficientnet_b4' in available_options:
                                backbone = 'efficientnet_b4'
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(LogLevel.INFO, f"Menggunakan backbone alternatif: efficientnet_b4")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: efficientnet_b4")
                            elif 'cspdarknet_s' in available_options:
                                backbone = 'cspdarknet_s'
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(LogLevel.INFO, f"Menggunakan backbone alternatif: cspdarknet_s")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: cspdarknet_s")
                            elif available_options:
                                backbone = available_options[0]
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(LogLevel.INFO, f"Menggunakan backbone alternatif: {backbone}")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: {backbone}")
                            else:
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(LogLevel.INFO, f"Menggunakan backbone alternatif: {backbone}")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: {backbone}")
                        
                        # Update nilai backbone di UI
                        if backbone in available_options:
                            _ui_components['backbone_type'].value = backbone
                        else:
                            if logger:
                                logger.warning(f"⚠️ Backbone '{backbone}' tidak tersedia dalam opsi")
                    except Exception as e2:
                        logger.warning(f"⚠️ Error saat update backbone_type: {str(e2)}")
                
                # Update pretrained
                if 'backbone_pretrained' in _config['model'] and 'pretrained' in _ui_components:
                    try:
                        _ui_components['pretrained'].value = _config['model']['backbone_pretrained']
                    except Exception as e3:
                        logger.warning(f"⚠️ Error saat update pretrained: {str(e3)}")
                
                # Update freeze_backbone
                if 'backbone_freeze' in _config['model'] and 'freeze_backbone' in _ui_components:
                    try:
                        _ui_components['freeze_backbone'].value = _config['model']['backbone_freeze']
                    except Exception as e4:
                        logger.warning(f"⚠️ Error saat update freeze_backbone: {str(e4)}")
                
                # Update freeze_layers
                if 'freeze_layers' in _config['model'] and 'freeze_layers' in _ui_components:
                    try:
                        _ui_components['freeze_layers'].value = _config['model']['freeze_layers']
                    except Exception as e5:
                        logger.warning(f"⚠️ Error saat update freeze_layers: {str(e5)}")
                
                # Fitur optimasi tidak perlu diupdate karena sudah dihapus dari UI
                
                if logger:
                    if hasattr(logger, 'log'):
                        logger.log(LogLevel.INFO, "UI backbone diperbarui dari config")
                    else:
                        logger.info("✅ UI backbone diperbarui dari config")
            except Exception as e:
                if logger:
                    if hasattr(logger, 'log'):
                        logger.log(LogLevel.ERROR, f"Error update UI: {e}")
                    else:
                        logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi backbone sudah tidak diperlukan karena sudah dihandle oleh on_model_change
        def update_backbone_info():
            # Tidak melakukan apa-apa karena sudah dihandle oleh on_model_change
            pass
        
        # Handler buttons
        def on_save_click(b): 
            save_config(ui_components, config, "configs/model_config.yaml", update_config_from_ui, "Model Backbone")
        
        def on_reset_click(b): 
            # Panggil reset_config dengan fungsi update_ui_from_config yang sudah menerima parameter
            reset_config(ui_components, config, default_config, lambda: update_ui_from_config(ui_components, config), "Model Backbone")
        
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
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator("error", f"Error setup backbone button handler: {str(e)}"))
        else: print(f"❌ Error setup backbone button handler: {str(e)}")
    
    return ui_components
