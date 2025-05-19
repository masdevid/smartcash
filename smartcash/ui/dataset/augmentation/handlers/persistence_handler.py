"""
File: smartcash/ui/dataset/augmentation/handlers/persistence_handler.py
Deskripsi: Handler untuk persistensi konfigurasi augmentasi
"""

from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager

logger = get_logger("augmentation_persistence")

def validate_param(value: Any, default_value: Any, 
                  valid_types: Optional[Union[type, List[type]]] = None, 
                  valid_values: Optional[List[Any]] = None) -> Any:
    """Validasi parameter dengan fallback ke nilai default.

    Args:
        value: Nilai yang akan divalidasi
        default_value: Nilai default jika validasi gagal
        valid_types: Tipe data yang valid
        valid_values: Nilai-nilai yang valid

    Returns:
        Nilai yang sudah divalidasi atau default_value jika validasi gagal
    """
    # Validasi None
    if value is None:
        return default_value

    # Validasi tipe data
    if valid_types:
        # Konversi ke list jika bukan list
        if not isinstance(valid_types, (list, tuple)):
            valid_types = [valid_types]

        # Cek apakah value memiliki tipe yang valid
        if not any(isinstance(value, t) for t in valid_types):
            return default_value

    # Validasi nilai
    if valid_values and value not in valid_values:
        return default_value

    return value

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pastikan persistensi UI components dengan ConfigManager.

    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)

    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Dapatkan instance ConfigManager
        config_manager = get_config_manager()

        # Register UI components untuk persistensi
        config_manager.register_ui_components('augmentation', ui_components)

        # Dapatkan konfigurasi augmentasi jika tidak disediakan
        if not config:
            config = config_manager.get_module_config('augmentation')

            # Jika masih None, coba load dari file
            if not config:
                try:
                    from smartcash.ui.dataset.augmentation.utils.config_utils import load_augmentation_config
                    config = load_augmentation_config(ui_components=ui_components)
                except Exception as e:
                    logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi: {str(e)}")

        # Update UI dari konfigurasi jika ada
        if config:
            from smartcash.ui.dataset.augmentation.utils.config_utils import update_ui_from_config
            ui_components = update_ui_from_config(ui_components, config)

        return ui_components
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat memastikan persistensi UI: {str(e)}")
        return ui_components

def get_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari UI components.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Dictionary konfigurasi augmentasi
    """
    # Coba dapatkan dari ConfigManager terlebih dahulu
    try:
        config_manager = get_config_manager()
        config = config_manager.get_module_config('augmentation')
        if config and isinstance(config, dict) and 'augmentation' in config:
            logger.debug(f"{ICONS['info']} Menggunakan konfigurasi dari ConfigManager")
            return config
    except Exception as e:
        logger.debug(f"{ICONS['info']} Tidak dapat memuat dari ConfigManager: {str(e)}")

    # Jika tidak ada di ConfigManager, ekstrak dari UI
    try:
        # Import fungsi dari utils
        from smartcash.ui.dataset.augmentation.utils.config_utils import get_config_from_ui

        # Dapatkan konfigurasi dari UI
        config = get_config_from_ui(ui_components)

        # Pastikan format konfigurasi benar
        if not isinstance(config, dict):
            config = {'augmentation': {}}
        elif 'augmentation' not in config:
            config = {'augmentation': config}

        return config
    except Exception as e:
        logger.warning(f"{ICONS['warning']} Error saat mendapatkan konfigurasi dari UI: {str(e)}")

        # Kembalikan konfigurasi default jika gagal
        return {'augmentation': {
            'enabled': True,
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True
        }}

def sync_config_with_drive(ui_components: Dict[str, Any]) -> bool:
    """Sinkronisasi konfigurasi dengan file di drive.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan konfigurasi dari UI
        config = get_augmentation_config(ui_components)

        # Simpan ke ConfigManager
        config_manager = get_config_manager()
        success = config_manager.save_module_config('augmentation', config)

        if not success:
            logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi ke ConfigManager")
            return False

        # Simpan ke file lokal
        try:
            config_path = "configs/augmentation_config.yaml"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"{ICONS['success']} Konfigurasi berhasil disimpan ke {config_path}")
        except Exception as e:
            logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi ke file: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat sinkronisasi konfigurasi: {str(e)}")
        return False

def get_persisted_ui_components() -> Optional[Dict[str, Any]]:
    """Dapatkan UI components yang tersimpan dari ConfigManager.

    Returns:
        Dictionary UI components atau None jika tidak ditemukan
    """
    try:
        config_manager = get_config_manager()
        return config_manager.get_ui_components('augmentation')
    except Exception as e:
        logger.debug(f"{ICONS['info']} Tidak dapat memuat UI components dari ConfigManager: {str(e)}")
        return None

def reset_config_to_default(ui_components: Dict[str, Any]) -> bool:
    """Reset konfigurasi ke default dan perbarui UI.

    Args:
        ui_components: Dictionary komponen UI

    Returns:
        Boolean status keberhasilan
    """
    try:
        # Buat konfigurasi default
        default_config = {
            "augmentation": {
                "enabled": True,
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "output_dir": "data/augmented"
            },
            "data": {
                "dir": "data"
            }
        }

        # Simpan ke ConfigManager
        config_manager = get_config_manager()
        success = config_manager.save_module_config('augmentation', default_config)

        # Update UI dari konfigurasi default
        if success and ui_components:
            from smartcash.ui.dataset.augmentation.utils.config_utils import update_ui_from_config
            ui_components = update_ui_from_config(ui_components, default_config)
            ui_components['config'] = default_config

            # Simpan juga ke file lokal
            try:
                config_path = "configs/augmentation_config.yaml"
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            except Exception as e:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan default ke file: {str(e)}")

        return success
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}")
        return False

def setup_persistence_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup persistence handler untuk augmentasi dataset.

    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)

    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Pastikan persistensi UI components
    ui_components = ensure_ui_persistence(ui_components, config)

    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'get_augmentation_config': get_augmentation_config,
        'sync_config_with_drive': sync_config_with_drive,
        'reset_config_to_default': reset_config_to_default
    })

    # Setup handler untuk tombol save
    if 'save_button' in ui_components:
        def on_save_click(b):
            # Dapatkan konfigurasi dari UI
            config = get_augmentation_config(ui_components)

            # Simpan ke file dan drive
            success = sync_config_with_drive(ui_components)

            # Update status
            if success and 'update_status_panel' in ui_components:
                ui_components['update_status_panel'](
                    ui_components, 
                    "success", 
                    f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan"
                )

        # Register handler
        ui_components['save_button'].on_click(on_save_click)
        ui_components['on_save_click'] = on_save_click

    return ui_components
