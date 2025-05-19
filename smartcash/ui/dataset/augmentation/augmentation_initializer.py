"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi dataset dengan pendekatan DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.setup_handlers import setup_augmentation_handlers
from smartcash.common.logger import get_logger

logger = get_logger("augmentation_initializer")

def initialize_augmentation_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk augmentasi dataset.
    
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    # Konfigurasi multi-progress tracking
    multi_progress_config = {
        "module_name": "augmentation",
        "step_key": "augmentation_step",
        "progress_bar_key": "progress_bar",
        "current_progress_key": "current_progress",
        "overall_label_key": "overall_label",
        "step_label_key": "step_label"
    }
    
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['augment_button', 'augmentation_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button']
    
    # Gunakan base initializer dengan konfigurasi minimal
    ui_components = initialize_module_ui(
        module_name='augmentation',
        create_ui_func=create_augmentation_ui,
        setup_specific_handlers_func=setup_augmentation_handlers,
        button_keys=button_keys,
        multi_progress_config=multi_progress_config,
        observer_group="augmentation_observers"
    )
    
    return ui_components


def initialize_augmentation(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inisialisasi modul augmentasi dataset (fungsi lama untuk kompatibilitas).
    Sebaiknya gunakan initialize_augmentation_ui() untuk pendekatan DRY.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger.warning("⚠️ Menggunakan initialize_augmentation() yang sudah usang. Sebaiknya gunakan initialize_augmentation_ui()")
    
    # Tambahkan logger ke ui_components jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = logger

    # Setup handlers
    try:
        ui_components = setup_augmentation_handlers(ui_components)
        logger.info("✅ Modul augmentasi berhasil diinisialisasi")
    except Exception as e:
        logger.error(f"❌ Error saat setup handlers: {str(e)}")
    
    return ui_components
