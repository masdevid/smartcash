"""
File: smartcash/ui/training_config/backbone/backbone_initializer.py
Deskripsi: Initializer untuk UI pemilihan backbone model
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML
import os
import yaml
import copy

def initialize_backbone_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk pemilihan backbone model.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': 'backbone'}
    
    # Setup logging
    import logging
    logger = logging.getLogger('backbone_ui')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    ui_components['logger'] = logger
    
    logger.info("🚀 Memulai inisialisasi UI backbone model")
    
    try:
        # Import dependency
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config.manager import get_config_manager
        from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
        
        # Dapatkan environment jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Load konfigurasi dari config manager
        if config is None:
            # Dapatkan konfigurasi dari config manager
            from smartcash.common.default_config import generate_default_config
            default_config = generate_default_config()
            config = config_manager.get_module_config('model', default_config)
        
        # Buat komponen UI dengan penanganan error yang lebih baik
        try:
            from smartcash.ui.training_config.backbone.components.backbone_components import create_backbone_ui
            # Pastikan config memiliki nilai default yang valid untuk backbone
            if 'model' not in config:
                config['model'] = {}
            if 'backbone' not in config['model']:
                config['model']['backbone'] = 'cspdarknet_s'  # Default ke YOLOv5s backbone
                logger.info(f"🔧 Menggunakan default backbone: cspdarknet_s")
            
            # Buat UI components
            ui_components.update(create_backbone_ui(config))
        except Exception as ui_error:
            logger.error(f"❌ Error saat membuat komponen UI backbone: {str(ui_error)}")
            # Fallback ke config default
            from smartcash.common.default_config import generate_default_config
            default_config = generate_default_config()
            if 'model' not in default_config:
                default_config['model'] = {}
            if 'backbone' not in default_config['model']:
                default_config['model']['backbone'] = 'cspdarknet_s'
            
            # Coba lagi dengan config default
            try:
                ui_components.update(create_backbone_ui(default_config))
                logger.info(f"✅ Berhasil membuat UI dengan config default")
            except Exception as fallback_error:
                logger.error(f"❌ Error fallback UI backbone: {str(fallback_error)}")
                # Jika masih gagal, gunakan fallback minimal
        
        # Tambahkan tombol konfigurasi dari komponen standar
        from smartcash.ui.components.config_buttons import create_config_buttons
        config_buttons = create_config_buttons()
        ui_components.update({
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            'config_buttons': config_buttons['container']
        })
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(ui_components, "backbone", "backbone_step")
        
        # Setup handlers untuk tombol save dan reset
        def on_save_config(b):
            try:
                # Import ConfigManager
                from smartcash.common.config.manager import get_config_manager
                from smartcash.ui.utils.persistence_utils import extract_config_from_ui
                
                # Dapatkan config manager
                config_manager = get_config_manager()
                
                # Ekstrak konfigurasi dari UI
                current_config = config_manager.get_module_config('model')
                updated_config = extract_config_from_ui(ui_components, current_config, 'model', logger)
                
                # Simpan konfigurasi menggunakan config manager
                success = config_manager.save_module_config('model', updated_config)
                
                # Tampilkan pesan sukses atau error
                with ui_components['status']:
                    if success:
                        display(HTML(f"<div style='color:green'>✅ Konfigurasi backbone berhasil disimpan</div>"))
                    else:
                        display(HTML(f"<div style='color:orange'>⚠️ Konfigurasi backbone mungkin tidak tersimpan dengan benar</div>"))
                    
                if logger:
                    if success:
                        logger.info(f"✅ Konfigurasi backbone berhasil disimpan")
                    else:
                        logger.warning(f"⚠️ Konfigurasi backbone mungkin tidak tersimpan dengan benar")
                    
            except Exception as e:
                # Tampilkan pesan error
                with ui_components['status']:
                    display(HTML(f"<div style='color:red'>❌ Error: {str(e)}</div>"))
                    
                if logger:
                    logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
                    
                # Pastikan UI components tetap terdaftar untuk persistensi
                try:
                    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                    ensure_ui_persistence(ui_components, 'backbone', logger)
                except Exception:
                    pass
        
        def on_reset_config(b):
            # Reset ke default config
            from smartcash.common.default_config import generate_default_config
            default_config = generate_default_config().get('model', {})
            ui_components = update_ui_from_config(ui_components, default_config)
            
            # Tampilkan status
            if 'status' in ui_components:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                with ui_components['status']:
                    display(create_status_indicator('info', "🔄 Konfigurasi direset ke default"))
        
        # Register handler untuk tombol
        ui_components['save_button'].on_click(on_save_config)
        ui_components['reset_button'].on_click(on_reset_config)
        
        # Tambahkan fungsi helper ke ui_components
        ui_components.update({
            'update_config_from_ui': update_config_from_ui,
            'update_ui_from_config': update_ui_from_config,
            'config': config
        })
        
        # Setup handlers lainnya
        from smartcash.ui.training_config.backbone.handlers.button_handlers import setup_backbone_button_handlers
        from smartcash.ui.training_config.backbone.handlers.form_handlers import setup_backbone_form_handlers
        
        ui_components = setup_backbone_button_handlers(ui_components, env, config)
        ui_components = setup_backbone_form_handlers(ui_components, env, config)
        
        # Update UI dari config yang tersimpan
        ui_components = update_ui_from_config(ui_components, config)
        
        # Tampilkan container utama
        if 'main_container' in ui_components:
            # Ganti placeholder tombol dengan config_buttons
            if 'buttons_placeholder' in ui_components and hasattr(ui_components['main_container'], 'children'):
                # Dapatkan indeks buttons_placeholder dalam children
                children_list = list(ui_components['main_container'].children)
                placeholder_idx = next((i for i, child in enumerate(children_list) 
                                    if child is ui_components['buttons_placeholder']), -1)
                
                if placeholder_idx >= 0:
                    # Ganti placeholder dengan config_buttons
                    children_list[placeholder_idx] = config_buttons['container']
                    ui_components['main_container'].children = tuple(children_list)
                else:
                    # Jika placeholder tidak ditemukan, tambahkan sebelum status
                    if 'status' in ui_components:
                        status_idx = next((i for i, child in enumerate(children_list) 
                                        if child is ui_components['status']), len(children_list))
                        children_list.insert(status_idx, config_buttons['container'])
                        ui_components['main_container'].children = tuple(children_list)
            
            display(ui_components['main_container'])
        else:
            # Fallback jika main_container tidak ada
            from smartcash.ui.utils.alert_utils import create_alert_html
            display(HTML(create_alert_html(
                "Container utama tidak ditemukan. Mencoba menampilkan komponen yang tersedia.",
                "warning"
            )))
            # Coba tampilkan komponen yang tersedia
            if 'form' in ui_components:
                display(ui_components['form'])
            display(config_buttons['container'])
            if 'status' in ui_components:
                display(ui_components['status'])
        
    except Exception as e:
        # Log error
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error inisialisasi UI backbone: {str(e)}")
        
        # Coba tampilkan UI meskipun ada error
        try:
            # Import display di awal blok try untuk menghindari error
            from IPython.display import display, HTML
            
            # Coba buat komponen UI dengan fallback
            from smartcash.ui.training_config.backbone.components.backbone_components import create_backbone_ui
            ui_components.update(create_backbone_ui(config))
            
            # Tambahkan pesan error ke status
            if 'status' in ui_components:
                from smartcash.ui.utils.alert_utils import create_alert_html
                with ui_components['status']:
                    display(HTML(create_alert_html(
                        f"Error inisialisasi UI backbone: {str(e)}. Menggunakan konfigurasi fallback.",
                        "warning"
                    )))
            
            # Tampilkan komponen yang tersedia
            if 'main_container' in ui_components:
                display(ui_components['main_container'])
            elif 'form' in ui_components:
                display(ui_components['form'])
                if 'status' in ui_components:
                    display(ui_components['status'])
        except Exception as inner_e:
            # Jika masih gagal, gunakan utilitas fallback yang ada
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Error fallback UI backbone: {str(inner_e)}")
            from smartcash.ui.utils.fallback_utils import create_fallback_ui
            ui_components = create_fallback_ui(ui_components, f"{str(e)}\n\nError fallback: {str(inner_e)}", "error")
    
    return ui_components

# Fungsi untuk mengupdate konfigurasi dari UI
def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi model
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Import utilitas persistensi
    from smartcash.ui.utils.persistence_utils import extract_config_from_ui, validate_ui_param
    
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    
    # Gunakan utilitas extract_config_from_ui untuk mengekstrak nilai dari UI
    updated_config = extract_config_from_ui(ui_components, config, 'model', logger)
    
    # Validasi nilai-nilai penting
    if 'model' in updated_config:
        # Validasi backbone
        if 'backbone' in updated_config['model']:
            backbone = updated_config['model']['backbone']
            valid_backbones = ['cspdarknet_s', 'efficientnet_b4']
            updated_config['model']['backbone'] = validate_ui_param(backbone, 'efficientnet_b4', str, valid_backbones, logger)
        
        # Validasi pretrained
        if 'pretrained' in updated_config['model']:
            pretrained = updated_config['model']['pretrained']
            updated_config['model']['pretrained'] = validate_ui_param(pretrained, True, bool, None, logger)
        
        # Validasi freeze_backbone
        if 'freeze_backbone' in updated_config['model']:
            freeze_backbone = updated_config['model']['freeze_backbone']
            updated_config['model']['freeze_backbone'] = validate_ui_param(freeze_backbone, True, bool, None, logger)
    
    # Pastikan UI components terdaftar untuk persistensi
    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
    ensure_ui_persistence(ui_components, 'backbone', logger)
    
    return updated_config

# Fungsi untuk mengupdate UI dari konfigurasi
def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi model
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import utilitas persistensi
    from smartcash.ui.utils.persistence_utils import validate_ui_param, update_ui_from_config as update_ui_util
    
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    
    # Dapatkan konfigurasi model dengan validasi
    if not isinstance(config, dict):
        if logger: logger.warning(f"⚠️ Config bukan dictionary, menggunakan empty dict")
        config = {}
    
    model_config = config.get('model', {})
    if not isinstance(model_config, dict):
        if logger: logger.warning(f"⚠️ Model config bukan dictionary, menggunakan empty dict")
        model_config = {}
    
    # Update dropdown backbone dengan validasi yang lebih kuat
    if 'backbone_dropdown' in ui_components:
        try:
            # Validasi nilai backbone
            backbone = validate_ui_param(
                model_config.get('backbone'), 
                'cspdarknet_s',  # Default ke cspdarknet_s (yolov5s)
                str,
                None,  # Validasi terhadap options akan dilakukan di bawah
                logger
            )
            
            # Pastikan backbone ada dalam opsi dropdown
            if hasattr(ui_components['backbone_dropdown'], 'options'):
                options = ui_components['backbone_dropdown'].options
                if backbone in options:
                    ui_components['backbone_dropdown'].value = backbone
                else:
                    # Jika tidak ada, gunakan opsi default yang aman
                    if logger: 
                        logger.warning(f"⚠️ Backbone '{backbone}' tidak ditemukan dalam opsi dropdown, menggunakan default")
                    
                    # Gunakan opsi default yang aman (cspdarknet_s untuk yolov5s)
                    if 'cspdarknet_s' in options:
                        ui_components['backbone_dropdown'].value = 'cspdarknet_s'
                    elif options and len(options) > 0:
                        # Fallback ke opsi pertama jika cspdarknet_s tidak tersedia
                        ui_components['backbone_dropdown'].value = options[0]
                        if logger: logger.info(f"ℹ️ Menggunakan backbone: {options[0]}")
            else:
                if logger: logger.warning(f"⚠️ Dropdown backbone tidak memiliki opsi")
        except Exception as e:
            if logger: logger.error(f"❌ Error saat update dropdown backbone: {str(e)}")
    
    # Update checkbox pretrained dengan validasi
    if 'pretrained_checkbox' in ui_components:
        try:
            pretrained = validate_ui_param(model_config.get('pretrained'), True, bool, None, logger)
            ui_components['pretrained_checkbox'].value = pretrained
        except Exception as e:
            if logger: logger.warning(f"⚠️ Error saat update checkbox pretrained: {str(e)}")
    
    # Update checkbox freeze backbone dengan validasi
    if 'freeze_backbone_checkbox' in ui_components:
        try:
            freeze_backbone = validate_ui_param(model_config.get('freeze_backbone'), True, bool, None, logger)
            ui_components['freeze_backbone_checkbox'].value = freeze_backbone
        except Exception as e:
            if logger: logger.warning(f"⚠️ Error saat update checkbox freeze_backbone: {str(e)}")
    
    # Pastikan UI components terdaftar untuk persistensi
    from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
    ensure_ui_persistence(ui_components, 'backbone', logger)
    
    if logger: logger.debug(f"✅ UI berhasil diupdate dari konfigurasi")
    
    return ui_components
