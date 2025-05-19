"""
File: smartcash/ui/dataset/split/handlers/ui_handlers.py
Deskripsi: Handler untuk interaksi UI split dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.dataset.split.handlers.config_handlers import get_default_split_config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Pastikan struktur konfigurasi benar
    if not config:
        if logger: logger.debug(f"{ICONS['warning']} Konfigurasi kosong, menggunakan default")
        config = get_default_split_config()
    
    if 'data' not in config:
        config['data'] = {}
    if 'split' not in config['data']:
        config['data']['split'] = {'train': 0.7, 'val': 0.15, 'test': 0.15, 'stratified': True}
    
    split_config = config['data']['split']
    
    # Update nilai train slider
    if 'train_slider' in ui_components and 'train' in split_config:
        try:
            ui_components['train_slider'].value = split_config['train']
            if logger: logger.debug(f"{ICONS['success']} Train slider diupdate ke {split_config['train']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update train slider: {str(e)}")
    
    # Update nilai val slider
    if 'val_slider' in ui_components and 'val' in split_config:
        try:
            ui_components['val_slider'].value = split_config['val']
            if logger: logger.debug(f"{ICONS['success']} Val slider diupdate ke {split_config['val']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update val slider: {str(e)}")
    
    # Update nilai test slider
    if 'test_slider' in ui_components and 'test' in split_config:
        try:
            ui_components['test_slider'].value = split_config['test']
            if logger: logger.debug(f"{ICONS['success']} Test slider diupdate ke {split_config['test']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update test slider: {str(e)}")
    
    # Update nilai stratified checkbox
    if 'stratified_checkbox' in ui_components and 'stratified' in split_config:
        try:
            ui_components['stratified_checkbox'].value = split_config['stratified']
            if logger: logger.debug(f"{ICONS['success']} Stratified checkbox diupdate ke {split_config['stratified']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update stratified checkbox: {str(e)}")
    
    # Update nilai random seed
    if 'random_seed' in ui_components and 'random_seed' in config['data']:
        try:
            ui_components['random_seed'].value = config['data']['random_seed']
            if logger: logger.debug(f"{ICONS['success']} Random seed diupdate ke {config['data']['random_seed']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update random seed: {str(e)}")
    
    # Update nilai backup checkbox
    if 'backup_checkbox' in ui_components and 'backup_before_split' in config['data']:
        try:
            ui_components['backup_checkbox'].value = config['data']['backup_before_split']
            if logger: logger.debug(f"{ICONS['success']} Backup checkbox diupdate ke {config['data']['backup_before_split']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update backup checkbox: {str(e)}")
    
    # Update nilai backup dir
    if 'backup_dir' in ui_components and 'backup_dir' in config['data']:
        try:
            ui_components['backup_dir'].value = config['data']['backup_dir']
            if logger: logger.debug(f"{ICONS['success']} Backup dir diupdate ke {config['data']['backup_dir']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update backup dir: {str(e)}")
    
    # Update nilai dataset path
    if 'dataset_path' in ui_components and 'dataset_path' in config['data']:
        try:
            ui_components['dataset_path'].value = config['data']['dataset_path']
            if logger: logger.debug(f"{ICONS['success']} Dataset path diupdate ke {config['data']['dataset_path']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update dataset path: {str(e)}")
    
    # Update nilai preprocessed path
    if 'preprocessed_path' in ui_components and 'preprocessed_path' in config['data']:
        try:
            ui_components['preprocessed_path'].value = config['data']['preprocessed_path']
            if logger: logger.debug(f"{ICONS['success']} Preprocessed path diupdate ke {config['data']['preprocessed_path']}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update preprocessed path: {str(e)}")
    
    # Update total label
    if 'total_label' in ui_components and 'train' in split_config and 'val' in split_config and 'test' in split_config:
        try:
            from smartcash.ui.utils.constants import COLORS
            total = round(split_config['train'] + split_config['val'] + split_config['test'], 2)
            color = COLORS['success'] if total == 1.0 else COLORS['danger']
            ui_components['total_label'].value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"
            if logger: logger.debug(f"{ICONS['success']} Total label diupdate ke {total}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat update total label: {str(e)}")
    
    return ui_components

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None) -> None:
    """
    Pastikan UI components terdaftar untuk persistensi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
    """
    try:
        from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
        ensure_ui_persistence(ui_components, 'dataset_split', logger)
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error saat mendaftarkan UI components untuk persistensi: {str(e)}")

def initialize_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inisialisasi UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Update UI dari konfigurasi
    ui_components = update_ui_from_config(ui_components, config)
    
    # Pastikan UI components terdaftar untuk persistensi
    ensure_ui_persistence(ui_components, config, logger)
    
    return ui_components

def on_slider_change(train_slider, val_slider, test_slider, total_label) -> None:
    """
    Handler untuk perubahan nilai slider.
    
    Args:
        train_slider: Slider untuk proporsi data training
        val_slider: Slider untuk proporsi data validasi
        test_slider: Slider untuk proporsi data testing
        total_label: Label untuk menampilkan total proporsi
    """
    from smartcash.ui.utils.constants import COLORS
    
    # Hitung total proporsi
    total = round(train_slider.value + val_slider.value + test_slider.value, 2)
    
    # Tentukan warna berdasarkan total
    color = COLORS['success'] if total == 1.0 else COLORS['danger']
    
    # Update label total
    total_label.value = f"<div style='padding: 10px; color: {color}; font-weight: bold;'>Total: {total:.2f}</div>"

def validate_sliders(train_slider, val_slider, test_slider) -> bool:
    """
    Validasi nilai slider untuk memastikan total = 1.
    
    Args:
        train_slider: Slider untuk proporsi data training
        val_slider: Slider untuk proporsi data validasi
        test_slider: Slider untuk proporsi data testing
        
    Returns:
        Boolean yang menunjukkan validitas nilai slider
    """
    # Hitung total proporsi
    total = round(train_slider.value + val_slider.value + test_slider.value, 2)
    
    # Kembalikan hasil validasi
    return total == 1.0
