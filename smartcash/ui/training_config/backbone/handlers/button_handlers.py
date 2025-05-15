"""
File: smartcash/ui/training_config/backbone/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen backbone
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.utils.constants import ICONS

def setup_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol-tombol pada UI konfigurasi backbone model.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Inisialisasi ui_components jika None
    if ui_components is None:
        ui_components = {}
    
    # Dapatkan ui_logger untuk logging yang lebih baik di UI
    logger = ui_components.get('logger')
    
    # Pastikan konfigurasi model ada
    if not config:
        # Import fungsi yang diperlukan
        from smartcash.ui.training_config.backbone.handlers.config_handlers import load_config, get_config_manager_instance
        
        # Coba dapatkan konfigurasi dari ConfigManager
        config_manager = get_config_manager_instance()
        if config_manager:
            try:
                config = config_manager.get_module_config('model')
                if logger: logger.debug(f"{ICONS['info']} Konfigurasi berhasil dimuat dari ConfigManager")
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari ConfigManager: {str(e)}")
                # Fallback ke load dari file
                config = load_config()
        else:
            # Fallback ke load dari file
            config = load_config()
    
    # Pastikan struktur konfigurasi benar
    if not isinstance(config, dict):
        config = {}
    if 'model' not in config:
        config['model'] = {}
    
    # Pastikan UI components terdaftar untuk persistensi
    from smartcash.ui.training_config.backbone.handlers.ui_handlers import ensure_ui_persistence
    ensure_ui_persistence(ui_components, config, logger)
    
    # Inisialisasi UI dari konfigurasi
    from smartcash.ui.training_config.backbone.handlers.ui_handlers import initialize_ui_from_config
    initialize_ui_from_config(ui_components, config)
        
    # Register handler untuk save button
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button(b, ui_components, config, env, logger)
        )
        if logger: logger.debug(f"{ICONS['link']} Handler untuk save button terdaftar")
    
    # Register handler untuk reset button
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button(b, ui_components, config, env, logger)
        )
        if logger: logger.debug(f"{ICONS['link']} Handler untuk reset button terdaftar")
    
    return ui_components


def handle_save_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Handler untuk tombol save konfigurasi.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    # Import fungsi yang diperlukan
    from smartcash.ui.training_config.backbone.handlers.config_handlers import update_config_from_ui, save_config_with_manager
    from smartcash.ui.utils.constants import COLORS
    
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        print(f"{ICONS['warning']} Status panel tidak tersedia")
        return
    
    # Tampilkan status processing
    status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS['processing']} Menyimpan konfigurasi backbone...</p>
        </div>"""
    
    try:
        # Update config dari UI
        updated_config = update_config_from_ui(config, ui_components)
        
        # Simpan config dengan ConfigManager atau fallback
        success = save_config_with_manager(updated_config, ui_components, logger)
        
        # Tampilkan hasil
        if success:
            status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                     color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                     border-left:4px solid {COLORS['alert_success_text']}">
                <p style="margin:5px 0">{ICONS['success']} Konfigurasi backbone berhasil disimpan</p>
            </div>"""
            if logger: logger.debug(f"{ICONS['success']} Konfigurasi backbone berhasil disimpan")
        else:
            status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                     color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                     border-left:4px solid {COLORS['alert_danger_text']}">
                <p style="margin:5px 0">{ICONS['error']} Gagal menyimpan konfigurasi</p>
            </div>"""
            if logger: logger.error(f"{ICONS['error']} Gagal menyimpan konfigurasi backbone")
    except Exception as e:
        # Tampilkan error
        status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                 color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_danger_text']}">
            <p style="margin:5px 0">{ICONS['error']} Error: {str(e)}</p>
        </div>"""
        if logger: logger.error(f"{ICONS['error']} Error saat menyimpan konfigurasi backbone: {str(e)}")

def handle_reset_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Handler untuk tombol reset konfigurasi.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    # Import fungsi yang diperlukan
    from smartcash.ui.training_config.backbone.handlers.config_handlers import load_default_config, save_config_with_manager
    from smartcash.ui.training_config.backbone.handlers.ui_handlers import update_ui_from_config
    from smartcash.ui.utils.constants import COLORS
    
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        print(f"{ICONS['warning']} Status panel tidak tersedia")
        return
    
    # Tampilkan status processing
    status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS['processing']} Mereset konfigurasi...</p>
        </div>"""
    
    try:
        # Load konfigurasi default
        default_config = load_default_config()
        
        # Update UI dari konfigurasi default
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default dengan ConfigManager atau fallback
        success = save_config_with_manager(default_config, ui_components, logger)
        
        # Tampilkan hasil
        if success:
            status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                     color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                     border-left:4px solid {COLORS['alert_success_text']}">
                <p style="margin:5px 0">{ICONS['success']} Konfigurasi berhasil direset ke default</p>
            </div>"""
            if logger: logger.debug(f"{ICONS['success']} Konfigurasi backbone berhasil direset ke default")
        else:
            status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                     color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                     border-left:4px solid {COLORS['alert_danger_text']}">
                <p style="margin:5px 0">{ICONS['error']} Gagal mereset konfigurasi</p>
            </div>"""
            if logger: logger.error(f"{ICONS['error']} Gagal mereset konfigurasi backbone")
    except Exception as e:
        # Tampilkan error
        status_panel.value = f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                 color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_danger_text']}">
            <p style="margin:5px 0">{ICONS['error']} Error: {str(e)}</p>
        </div>"""
        if logger: logger.error(f"{ICONS['error']} Error saat mereset konfigurasi backbone: {str(e)}")
