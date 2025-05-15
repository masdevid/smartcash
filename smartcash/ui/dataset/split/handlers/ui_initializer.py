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
        
        # Update slider values
        if 'split_sliders' in ui_components and ui_components['split_sliders'] and len(ui_components['split_sliders']) >= 3:
            train_slider, valid_slider, test_slider = ui_components['split_sliders']
            
            # Set nilai slider berdasarkan konfigurasi
            train_value = split_config.get('train', 0.8) * 100  # Konversi ke persentase
            val_value = split_config.get('val', 0.1) * 100
            test_value = split_config.get('test', 0.1) * 100
            
            # Pastikan nilai valid (antara 0 dan 100)
            train_value = max(0, min(100, train_value))
            val_value = max(0, min(100, val_value))
            test_value = max(0, min(100, test_value))
            
            # Update nilai slider
            train_slider.value = train_value
            valid_slider.value = val_value
            test_slider.value = test_value
            
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
                <p>Train: {split_config.get('train', 0.8):.2f}, Val: {split_config.get('val', 0.1):.2f}, Test: {split_config.get('test', 0.1):.2f}</p>
                <p>Stratified: {"Ya" if split_config.get('stratified', True) else "Tidak"}</p>
            </div>"""
        
        # Simpan referensi konfigurasi di ui_components untuk persistensi
        ui_components['config'] = config
        
        if logger: logger.debug(f"{ICONS['success']} UI berhasil diinisialisasi dari konfigurasi")
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat menginisialisasi UI: {str(e)}")
        print(f"{ICONS['error']} Error saat menginisialisasi UI dari konfigurasi: {str(e)}")

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None) -> None:
    """
    Memastikan UI components terdaftar untuk persistensi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
    """
    # Import ConfigManager untuk persistensi
    try:
        from smartcash.common.config.manager import ConfigManager
        
        # Dapatkan instance ConfigManager
        config_manager = ConfigManager.get_instance()
        
        # Register UI components
        config_manager.register_ui_components('dataset_split', ui_components)
        
        # Simpan referensi konfigurasi di ui_components
        ui_components['config'] = config
        
        if logger: logger.debug(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Gagal mendaftarkan UI components: {str(e)}")
        
        # Fallback: simpan referensi konfigurasi di ui_components
        ui_components['config'] = config
