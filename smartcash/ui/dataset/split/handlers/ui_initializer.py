"""
File: smartcash/ui/dataset/split/handlers/ui_initializer.py
Deskripsi: Handler untuk inisialisasi UI split dataset dari konfigurasi
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.logger import get_logger

def initialize_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Inisialisasi UI components dari konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    if logger: logger.debug(f"{ICONS['refresh']} Menginisialisasi UI dari konfigurasi")
    
    try:
        # Dapatkan konfigurasi split
        split_config = config.get('data', {}).get('split', {})
        
        # Update slider values - menggunakan akses langsung ke slider
        if 'train_slider' in ui_components and 'val_slider' in ui_components and 'test_slider' in ui_components:
            # Set nilai slider berdasarkan konfigurasi
            train_value = split_config.get('train', 0.7)
            val_value = split_config.get('valid', 0.15)  # Konfigurasi menggunakan 'valid' tapi UI menggunakan 'val'
            test_value = split_config.get('test', 0.15)
            
            # Pastikan nilai valid (antara 0 dan 1)
            train_value = max(0, min(1, train_value))
            val_value = max(0, min(1, val_value))
            test_value = max(0, min(1, test_value))
            
            # Update nilai slider
            ui_components['train_slider'].value = train_value
            ui_components['val_slider'].value = val_value
            ui_components['test_slider'].value = test_value
            
            if logger: logger.debug(f"{ICONS['info']} Slider diupdate: Train={train_value:.1f}%, Val={val_value:.1f}%, Test={test_value:.1f}%")
        
        # Update checkbox stratified
        if 'stratified' in ui_components and ui_components['stratified']:
            stratified_value = split_config.get('stratified', True)
            ui_components['stratified'].value = stratified_value
            if logger: logger.debug(f"{ICONS['info']} Stratified={stratified_value}")
        
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
        
        # Update info panel
        if 'config_info_html' in ui_components and ui_components['config_info_html']:
            ui_components['config_info_html'].value = f"""
            <div style="text-align:center; padding:15px;">
                <p style="color:{COLORS['success']};">{ICONS['success']} Konfigurasi split dataset berhasil dimuat</p>
                <p>Train: {split_config.get('train', 0.7):.2f}, Val: {split_config.get('val', 0.15):.2f}, Test: {split_config.get('test', 0.15):.2f}</p>
                <p>Stratified: {"Ya" if split_config.get('stratified', True) else "Tidak"}</p>
            </div>"""
        
        # Simpan referensi konfigurasi di ui_components untuk persistensi
        ui_components['config'] = config
        
        if logger: logger.debug(f"{ICONS['success']} UI berhasil diinisialisasi dari konfigurasi")
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat menginisialisasi UI: {str(e)}")
        print(f"{ICONS['error']} Error saat menginisialisasi UI dari konfigurasi: {str(e)}")

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Perbarui UI berdasarkan konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
    """
    # Dapatkan nilai dari konfigurasi dengan fallback ke default
    split_config = config.get('data', {}).get('split', {})
    train_pct = split_config.get('train', 0.7)
    valid_pct = split_config.get('valid', 0.15)
    test_pct = split_config.get('test', 0.15)
    stratified = split_config.get('stratified', True)
    
    # Perbarui UI
    if 'train_slider' in ui_components:
        ui_components['train_slider'].value = train_pct
    if 'val_slider' in ui_components:
        ui_components['val_slider'].value = valid_pct  # Menggunakan val_slider untuk valid_pct
    if 'test_slider' in ui_components:
        ui_components['test_slider'].value = test_pct
    if 'stratified_checkbox' in ui_components:
        ui_components['stratified_checkbox'].value = stratified

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None) -> None:
    """
    Memastikan UI components terdaftar untuk persistensi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
    """
    # Import get_config_manager_instance dari config_handlers
    from smartcash.ui.dataset.split.handlers.config_handlers import get_config_manager_instance
    
    try:
        # Dapatkan instance ConfigManager
        config_manager = get_config_manager_instance()
        
        if config_manager:
            # Register UI components
            config_manager.register_ui_components('dataset_split', ui_components)
            
            # Simpan referensi konfigurasi di ui_components
            ui_components['config'] = config
            
            if logger: logger.debug(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
        else:
            if logger: logger.warning(f"{ICONS['warning']} ConfigManager tidak tersedia")
            # Fallback: simpan referensi konfigurasi di ui_components
            ui_components['config'] = config
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan UI components: {str(e)}")
        
        # Fallback: simpan referensi konfigurasi di ui_components
        ui_components['config'] = config
