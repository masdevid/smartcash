"""
File: smartcash/ui/dataset/augmentation/handlers/status_handler.py
Deskripsi: Handler untuk mengelola status panel pada UI augmentasi
"""

import ipywidgets as widgets
from typing import Dict, Any

# Import status handler utama untuk menjaga konsistensi
from smartcash.ui.handlers.status_handler import update_status_panel as global_update_status_panel
from smartcash.ui.handlers.status_handler import create_status_panel as global_create_status_panel
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS, COLORS
from smartcash.ui.dataset.augmentation.utils.notification_manager import get_notification_manager

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel UI dengan pesan dan tipe yang ditentukan.

    Args:
        ui_components: Dictionary komponen UI dengan kunci 'status_panel'
        status_type: Tipe pesan ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    # Delegasikan ke fungsi global untuk konsistensi
    global_update_status_panel(ui_components, status_type, message)

def create_status_panel(message: str = "", status_type: str = "info") -> widgets.HTML:
    """
    Buat komponen status panel untuk modul augmentasi dengan styling yang konsisten.

    Args:
        message: Pesan awal untuk status
        status_type: Tipe status awal ('info', 'success', 'warning', 'error')

    Returns:
        Widget HTML berisi status panel
    """
    # Delegasikan ke fungsi global untuk konsistensi
    return global_create_status_panel(message, status_type)

def setup_status_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk status panel augmentasi.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan status panel tersedia
    if 'status_panel' not in ui_components:
        ui_components['status_panel'] = create_status_panel("Siap untuk augmentasi dataset", "info")

    # Inisialisasi notification manager
    notification_manager = get_notification_manager(ui_components)

    # Tambahkan fungsi ke ui_components
    ui_components['update_status_panel'] = update_status_panel
    ui_components['create_status_panel'] = create_status_panel

    return ui_components

def update_augmentation_info(ui_components: Dict[str, Any]) -> None:
    """
    Update informasi augmentasi di panel status.

    Args:
        ui_components: Dictionary komponen UI
    """
    # Dapatkan konfigurasi dari UI
    from smartcash.ui.dataset.augmentation.utils.config_utils import get_module_config
    config = get_module_config(ui_components)

    # Dapatkan konfigurasi augmentasi
    aug_config = config.get('augmentation', {})

    # Dapatkan nilai-nilai penting
    enabled = aug_config.get('enabled', True)
    rotation_range = aug_config.get('rotation_range', 20)
    width_shift = aug_config.get('width_shift_range', 0.2)
    height_shift = aug_config.get('height_shift_range', 0.2)
    zoom_range = aug_config.get('zoom_range', 0.2)
    horizontal_flip = aug_config.get('horizontal_flip', True)

    # Dapatkan split dari UI
    split_selector = ui_components.get('split_selector')
    split = 'Train Only'  # Default
    if split_selector and hasattr(split_selector, 'value'):
        split = split_selector.value

    # Buat pesan status
    if enabled:
        message = f"Augmentasi aktif: rotasi {rotation_range}°, shift {width_shift}/{height_shift}, zoom {zoom_range} pada split {split}"
    else:
        message = "Augmentasi tidak aktif"

    # Update panel status
    update_status_panel(ui_components, 'info', message)
