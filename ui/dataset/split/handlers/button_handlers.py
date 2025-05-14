"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI konfigurasi split dataset
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import yaml
from pathlib import Path

def setup_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol-tombol pada UI konfigurasi split dataset.
    
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
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger', None)
    
    # Register handler untuk save button
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button(b, ui_components, config, env, logger)
        )
        if logger: logger.info("🔗 Handler untuk save button terdaftar")
    
    # Register handler untuk reset button
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button(b, ui_components, config, env, logger)
        )
        if logger: logger.info("🔗 Handler untuk reset button terdaftar")
    
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
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Validasi komponen yang diperlukan
    if 'output_box' not in ui_components:
        if logger: logger.error("❌ Output box tidak tersedia untuk aksi save")
        return
    
    output = ui_components['output_box']
    
    with output:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['save']} Menyimpan konfigurasi split..."))
        
        try:
            # Update config dari UI
            config = update_config_from_ui(config, ui_components)
            
            # Simpan konfigurasi
            success = save_config(config)
            
            if success:
                display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan"))
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = (
                        f'<div style="padding:10px; background-color:{COLORS["alert_success_bg"]}; '
                        f'color:{COLORS["alert_success_text"]}; margin:10px 0; border-radius:4px; '
                        f'border-left:4px solid {COLORS["alert_success_text"]};">'
                        f'<p style="margin:5px 0">{ICONS["success"]} Konfigurasi split berhasil disimpan</p>'
                        '</div>'
                    )
                if logger: logger.info("✅ Konfigurasi split berhasil disimpan")
            else:
                display(create_status_indicator("error", f"{ICONS['error']} Gagal menyimpan konfigurasi"))
                if logger: logger.error("❌ Gagal menyimpan konfigurasi")
        
        except Exception as e:
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")

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
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Validasi ui_components untuk mencegah KeyError
    if ui_components is None:
        if logger: logger.error("❌ ui_components adalah None saat reset split_config")
        return
    
    # Validasi komponen yang diperlukan
    if 'output_box' not in ui_components:
        if logger: logger.error("❌ Output box tidak tersedia untuk aksi reset")
        return
    
    output = ui_components['output_box']
    
    with output:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['refresh']} Mereset konfigurasi split..."))
        
        try:
            # Load default config
            config = load_default_config()
            
            # Initialize UI dari config
            initialize_ui_from_config(ui_components, config)
            
            # Tampilkan pesan sukses dengan cara yang lebih aman
            success_message = f"{ICONS['success']} Konfigurasi split berhasil direset"
            display(create_status_indicator("success", success_message))
            
            # Update status panel jika ada
            if 'status_panel' in ui_components and ui_components['status_panel'] is not None:
                try:
                    ui_components['status_panel'].value = (
                        f'<div style="padding:10px; background-color:{COLORS["alert_success_bg"]}; '
                        f'color:{COLORS["alert_success_text"]}; margin:10px 0; border-radius:4px; '
                        f'border-left:4px solid {COLORS["alert_success_text"]};">' 
                        f'<p style="margin:5px 0">{ICONS["success"]} Konfigurasi split berhasil direset</p>'
                        '</div>'
                    )
                except Exception as panel_error:
                    if logger: logger.warning(f"⚠️ Error saat update status panel: {str(panel_error)}")
            
            # Update status jika ada
            if 'status' in ui_components and ui_components['status'] is not None:
                try:
                    with ui_components['status']:
                        display(create_info_alert(success_message, alert_type="success"))
                except Exception as status_error:
                    if logger: logger.warning(f"⚠️ Error saat update status: {str(status_error)}")
                    
            if logger: logger.info("✅ Konfigurasi split berhasil direset")
        
        except Exception as e:
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            if logger: logger.error(f"❌ Error saat mereset konfigurasi: {str(e)}")

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari komponen UI dengan validasi.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang telah diupdate
    """
    # Initialize config if needed
    if config is None:
        config = {'data': {}}
    elif 'data' not in config:
        config['data'] = {}
        
    data = config['data']

    # Update split ratios
    if ui_components and 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
        train, val, test = [s.value for s in ui_components['split_sliders'][:3]]
        total = train + val + test
        factor = 100.0 / total if total else 1.0
        data['split_ratios'] = {
            'train': train * factor / 100.0,
            'valid': val * factor / 100.0,
            'test': test * factor / 100.0
        }

    # Update stratified split
    if ui_components and 'stratified' in ui_components:
        stratified = ui_components['stratified']
        if hasattr(stratified, 'value'):
            data['stratified_split'] = stratified.value
    
    # Update advanced options
    if ui_components and 'advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children'):
        advanced_options = ui_components['advanced_options'].children
        if len(advanced_options) > 0 and hasattr(advanced_options[0], 'value'):
            data['random_seed'] = advanced_options[0].value
        if len(advanced_options) > 1 and hasattr(advanced_options[1], 'value'):
            data['backup_before_split'] = advanced_options[1].value
        if len(advanced_options) > 2 and hasattr(advanced_options[2], 'value'):
            data['backup_dir'] = advanced_options[2].value
    
    # Update data paths
    if ui_components and 'data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children'):
        data_paths = ui_components['data_paths'].children
        if len(data_paths) > 0 and hasattr(data_paths[0], 'value'):
            data['dataset_path'] = data_paths[0].value
        if len(data_paths) > 1 and hasattr(data_paths[1], 'value'):
            data['preprocessed_path'] = data_paths[1].value
    
    return config

def save_config(config: Dict[str, Any], config_path: str = 'configs/dataset_config.yaml') -> bool:
    """
    Simpan konfigurasi dataset ke file.
    
    Args:
        config: Konfigurasi aplikasi
        config_path: Path ke file konfigurasi
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump({'data': config['data']}, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"❌ Error menyimpan konfigurasi: {str(e)}")
        return False

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk dataset.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLIT_RATIOS, DRIVE_BACKUP_PATH, DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH
    
    return {
        'data': {
            'split_ratios': DEFAULT_SPLIT_RATIOS,
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': DRIVE_BACKUP_PATH,
            'dataset_path': DRIVE_DATASET_PATH,
            'preprocessed_path': DRIVE_PREPROCESSED_PATH
        }
    }

def initialize_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update komponen UI dari konfigurasi dengan validasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    if logger: logger.debug(f"🔄 Menginisialisasi UI dari konfigurasi: {config}")
    
    # Validasi komponen yang diperlukan
    if 'split_sliders' not in ui_components or not ui_components['split_sliders'] or len(ui_components['split_sliders']) < 3:
        if logger: logger.warning("⚠️ Komponen split_sliders tidak tersedia")
        return
    
    # Split sliders dengan validasi
    split_ratios = config.get('data', {}).get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
    ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
    ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
    ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
    
    # Stratified checkbox
    if 'stratified' in ui_components:
        ui_components['stratified'].value = config.get('data', {}).get('stratified_split', True)
    
    # Advanced options dengan validasi
    if ('advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') 
            and len(ui_components['advanced_options'].children) >= 3):
        ui_components['advanced_options'].children[0].value = config.get('data', {}).get('random_seed', 42)
        ui_components['advanced_options'].children[1].value = config.get('data', {}).get('backup_before_split', True)
        ui_components['advanced_options'].children[2].value = config.get('data', {}).get('backup_dir', 'data/splits_backup')
    
    # Data paths dengan validasi
    if ('data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children') 
            and len(ui_components['data_paths'].children) >= 2):
        ui_components['data_paths'].children[0].value = config.get('data', {}).get('dataset_path', 'data')
        ui_components['data_paths'].children[1].value = config.get('data', {}).get('preprocessed_path', 'data/preprocessed')
    
    # Simpan referensi konfigurasi di ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    if logger: logger.debug("✅ UI berhasil diinisialisasi dari konfigurasi")
