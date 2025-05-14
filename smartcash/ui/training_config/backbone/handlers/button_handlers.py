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
                if 'model_type' in _config['model']:
                    try:
                        model_type = _config['model']['model_type']
                        if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
                            _ui_components['model_type'].value = model_type
                            if logger:
                                if hasattr(logger, 'log'):
                                    logger.log(logger.DEBUG, f"Model type diperbarui ke: {model_type}")
                                else:
                                    logger.debug(f"✅ Model type diperbarui ke: {model_type}")
                        else:
                            # Jika model_type tidak valid, gunakan default
                            _ui_components['model_type'].value = default_model_type
                            if logger: 
                                if hasattr(logger, 'log'):
                                    logger.log(logger.INFO, f"Model type tidak valid, menggunakan default: {default_model_type}")
                                else:
                                    logger.info(f"ℹ️ Model type tidak valid, menggunakan default: {default_model_type}")
                    except Exception as e:
                        if logger:
                            if hasattr(logger, 'log'):
                                logger.log(logger.WARNING, f"Error saat update model_type: {str(e)}")
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
                                    logger.log(logger.WARNING, f"Backbone '{backbone}' tidak ditemukan dalam opsi dropdown")
                                else:
                                    logger.warning(f"⚠️ Backbone '{backbone}' tidak ditemukan dalam opsi dropdown")
                            
                            # Cari backbone alternatif yang valid
                            if 'efficientnet_b4' in available_options:
                                backbone = 'efficientnet_b4'
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.INFO, f"Menggunakan backbone alternatif: efficientnet_b4")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: efficientnet_b4")
                            elif 'cspdarknet_s' in available_options:
                                backbone = 'cspdarknet_s'
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.INFO, f"Menggunakan backbone alternatif: cspdarknet_s")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: cspdarknet_s")
                            elif available_options:
                                backbone = available_options[0]
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.INFO, f"Menggunakan backbone alternatif: {backbone}")
                                    else:
                                        logger.info(f"ℹ️ Menggunakan backbone alternatif: {backbone}")
                            else:
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.ERROR, f"Tidak ada opsi backbone yang tersedia")
                                    else:
                                        logger.error(f"❌ Tidak ada opsi backbone yang tersedia")
                                return  # Keluar dari fungsi jika tidak ada opsi yang tersedia
                            
                            # Update config dengan backbone yang valid
                            _config['model']['backbone'] = backbone
                        
                        # Periksa apakah dropdown dinonaktifkan
                        is_disabled = getattr(_ui_components['backbone_type'], 'disabled', False)
                        
                        try:
                            # Aktifkan sementara untuk mengubah nilai jika dinonaktifkan
                            if is_disabled:
                                _ui_components['backbone_type'].disabled = False
                            
                            # Update nilai dengan penanganan error
                            _ui_components['backbone_type'].value = backbone
                            
                            # Kembalikan status disabled
                            if is_disabled:
                                _ui_components['backbone_type'].disabled = True
                                
                            if logger:
                                if hasattr(logger, 'log'):
                                    logger.log(logger.DEBUG, f"Backbone diperbarui ke: {backbone}")
                                else:
                                    logger.debug(f"✅ Backbone diperbarui ke: {backbone}")
                        except Exception as set_error:
                            if logger: logger.warning(f"⚠️ Error saat mengatur nilai backbone: {str(set_error)}")
                            
                            # Jika gagal, coba metode alternatif dengan membuat ulang dropdown
                            try:
                                import ipywidgets as widgets
                                # Buat dropdown baru dengan nilai yang benar
                                new_dropdown = widgets.Dropdown(
                                    options=available_options,
                                    value=backbone,
                                    description='Backbone:',
                                    style={'description_width': 'initial'},
                                    layout=widgets.Layout(width='100%'),
                                    disabled=is_disabled
                                )
                                
                                # Ganti dropdown lama dengan yang baru
                                _ui_components['backbone_type'] = new_dropdown
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.INFO, f"Berhasil mengatur backbone dengan metode alternatif")
                                    else:
                                        logger.info(f"✅ Berhasil mengatur backbone dengan metode alternatif")
                            except Exception as alt_error:
                                if logger:
                                    if hasattr(logger, 'log'):
                                        logger.log(logger.ERROR, f"Gagal mengatur backbone dengan metode alternatif: {str(alt_error)}")
                                    else:
                                        logger.error(f"❌ Gagal mengatur backbone dengan metode alternatif: {str(alt_error)}")
                    except Exception as e:
                        if logger:
                            if hasattr(logger, 'log'):
                                logger.log(logger.WARNING, f"Error saat update backbone: {str(e)}")
                            else:
                                logger.warning(f"⚠️ Error saat update backbone: {str(e)}")
                    
                # Update checkbox pretrained dengan penanganan error
                if 'pretrained' in _config['model']:
                    try:
                        _ui_components['pretrained'].value = _config['model']['pretrained']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update pretrained: {str(e)}")
                    
                # Update checkbox freeze_backbone dengan penanganan error
                if 'freeze_backbone' in _config['model']:
                    try:
                        _ui_components['freeze_backbone'].value = _config['model']['freeze_backbone']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update freeze_backbone: {str(e)}")
                    
                # Update slider freeze_layers dengan penanganan error
                if 'freeze_layers' in _config['model']:
                    try:
                        _ui_components['freeze_layers'].value = _config['model']['freeze_layers']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update freeze_layers: {str(e)}")
                
                # Update fitur optimasi dengan penanganan error
                if 'use_attention' in _config['model']:
                    try:
                        _ui_components['use_attention'].value = _config['model']['use_attention']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_attention: {str(e)}")
                    
                if 'use_residual' in _config['model']:
                    try:
                        _ui_components['use_residual'].value = _config['model']['use_residual']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_residual: {str(e)}")
                    
                if 'use_ciou' in _config['model']:
                    try:
                        _ui_components['use_ciou'].value = _config['model']['use_ciou']
                    except Exception as e:
                        if logger: logger.warning(f"⚠️ Error saat update use_ciou: {str(e)}")
                
                if logger:
                    if hasattr(logger, 'log'):
                        logger.log(logger.INFO, "UI backbone diperbarui dari config")
                    else:
                        logger.info("✅ UI backbone diperbarui dari config")
            except Exception as e:
                if logger:
                    if hasattr(logger, 'log'):
                        logger.log(logger.ERROR, f"Error update UI: {e}")
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
