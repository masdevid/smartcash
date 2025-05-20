"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk split dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_base_dir():
    """Dapatkan direktori base default."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_default_split_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "split": {
            "enabled": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "stratify": True
        }
    }

def is_colab_environment() -> bool:
    """
    Menentukan apakah kode dijalankan di Google Colab.
    
    Returns:
        bool: True jika di Google Colab, False jika tidak
    """
    try:
        from smartcash.common.config.force_sync import detect_colab_environment
        return detect_colab_environment()
    except ImportError:
        # Fallback ke metode lama jika force_sync tidak tersedia
        return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ

def load_config() -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari config manager.
    
    Returns:
        Dictionary konfigurasi split
    """
    try:
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        config = config_manager.get_module_config('split', {})
        
        # Pastikan config memiliki struktur yang benar
        if not config or 'split' not in config:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan konfigurasi default untuk split dataset")
            default_config = get_default_split_config()
            # Simpan default config ke file
            config_manager.save_module_config('split', default_config)
            return default_config
            
        return config
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat load konfigurasi split: {str(e)}")
        return get_default_split_config()

def sync_with_drive(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Sinkronisasi konfigurasi dengan Google Drive.
    
    Args:
        config: Konfigurasi yang akan disinkronkan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disinkronkan
    """
    try:
        # Gunakan fungsi sync_with_drive dari force_sync jika tersedia
        try:
            from smartcash.common.config.force_sync import sync_with_drive as force_sync
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Menyinkronkan konfigurasi dengan Google Drive...", 'info')
            
            synced_config = force_sync(config, 'split', ui_components)
            
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Konfigurasi berhasil disinkronkan dengan Google Drive", 'success')
            
            return synced_config
        except ImportError:
            # Jika force_sync tidak tersedia, gunakan metode lama
            pass
        
        if not is_colab_environment():
            # Tidak perlu sinkronisasi jika bukan di Colab
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
            return config
            
        # Dapatkan config manager
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        
        # Log info
        if ui_components and 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Menyinkronkan konfigurasi dengan Google Drive...", 'info')
        
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'split' not in config:
            config = {'split': config}
        
        # Simpan konfigurasi terlebih dahulu
        config_save_success = config_manager.save_module_config('split', config)
        if not config_save_success:
            # Log error jika gagal menyimpan
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi lokal sebelum sinkronisasi", 'error')
            return config
        
        # Sinkronisasi dengan Google Drive
        success, message = config_manager.sync_to_drive('split')
        
        # Verifikasi sinkronisasi berhasil dengan membaca ulang konfigurasi
        if success:
            # Muat ulang konfigurasi untuk verifikasi
            synced_config = config_manager.get_module_config('split', {})
            
            # Verifikasi konsistensi konfigurasi
            if 'split' in synced_config and 'split' in config:
                is_consistent = True
                for key, value in config['split'].items():
                    if key not in synced_config['split'] or synced_config['split'][key] != value:
                        is_consistent = False
                        logger.warning(f"⚠️ Inkonsistensi data pada key '{key}': {value} vs {synced_config['split'].get(key, 'tidak ada')}")
                        break
                
                # Log hasil verifikasi
                if is_consistent:
                    # Berhasil dan konsisten
                    if ui_components and 'status_panel' in ui_components:
                        from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                        update_sync_status_only(ui_components, "Konfigurasi berhasil disinkronkan dengan Google Drive", 'success')
                    return synced_config
                else:
                    # Berhasil tapi tidak konsisten
                    if ui_components and 'status_panel' in ui_components:
                        from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                        update_sync_status_only(ui_components, "Sinkronisasi berhasil tapi data tidak konsisten", 'warning')
                    # Kembalikan config asli untuk keamanan
                    return config
            
            # Log hasil sinkronisasi jika tidak bisa memverifikasi konsistensi
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Konfigurasi berhasil disinkronkan dengan Google Drive", 'success')
            
            return synced_config
        else:
            # Gagal sinkronisasi
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, f"Gagal menyinkronkan konfigurasi: {message}", 'error')
            return config
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat sinkronisasi dengan Google Drive: {str(e)}")
        
        # Log error ke UI
        if ui_components and 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, f"Error saat sinkronisasi: {str(e)}", 'error')
            
        return config

def save_config(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Simpan konfigurasi split dataset ke config manager dan sinkronkan dengan Google Drive.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disimpan dan disinkronkan
    """
    try:
        # Update status panel
        if ui_components and 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Menyimpan konfigurasi...", 'info')
        
        # Simpan konfigurasi asli untuk verifikasi
        original_config = config.copy() if config else get_default_split_config()
        
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'split' not in original_config:
            original_config = {'split': original_config}
        
        # Simpan konfigurasi
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        save_success = config_manager.save_module_config('split', original_config)
        
        if not save_success:
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi split", 'error')
            return original_config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi split berhasil disimpan")
        
        # Log ke UI jika ui_components tersedia
        if ui_components and 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Konfigurasi split berhasil disimpan", 'success')
        
        # Verifikasi konfigurasi tersimpan dengan benar
        saved_config = config_manager.get_module_config('split', {})
        
        # Verifikasi konsistensi
        if 'split' in saved_config and 'split' in original_config:
            is_consistent = True
            for key, value in original_config['split'].items():
                if key not in saved_config['split'] or saved_config['split'][key] != value:
                    is_consistent = False
                    logger.warning(f"⚠️ Inkonsistensi data pada key '{key}': {value} vs {saved_config['split'].get(key, 'tidak ada')}")
                    if ui_components and 'status_panel' in ui_components:
                        from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                        update_sync_status_only(ui_components, f"Inkonsistensi data pada key '{key}'", 'warning')
                    break
            
            if not is_consistent:
                # Coba simpan ulang jika tidak konsisten
                config_manager.save_module_config('split', original_config)
                # Log warning
                logger.warning(f"⚠️ Data tidak konsisten setelah penyimpanan, mencoba kembali")
                if ui_components and 'status_panel' in ui_components:
                    from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Data tidak konsisten, mencoba kembali", 'warning')
                
                # Verifikasi ulang setelah simpan ulang
                saved_config = config_manager.get_module_config('split', {})
        
        # Sinkronisasi dengan Google Drive jika di Colab
        if is_colab_environment():
            if ui_components and 'status_panel' in ui_components:
                from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Menyinkronkan dengan Google Drive...", 'info')
            
            # Pastikan nilai yang disinkronkan menggunakan nilai original_config
            # untuk menghindari inkonsistensi
            synced_config = sync_with_drive(original_config, ui_components)
            
            # Verifikasi konfigurasi yang disinkronkan dengan membandingkan dengan nilai asli
            is_synced_consistent = True
            if 'split' in synced_config and 'split' in original_config:
                for key, value in original_config['split'].items():
                    if key not in synced_config['split'] or synced_config['split'][key] != value:
                        is_synced_consistent = False
                        logger.warning(f"⚠️ Inkonsistensi data setelah sinkronisasi pada key '{key}': {value} vs {synced_config['split'].get(key, 'tidak ada')}")
                        if ui_components and 'status_panel' in ui_components:
                            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                            update_sync_status_only(ui_components, f"Inkonsistensi data setelah sinkronisasi", 'warning')
                        break
            
            if is_synced_consistent:
                if ui_components and 'status_panel' in ui_components:
                    from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Konfigurasi berhasil disimpan dan disinkronkan", 'success')
            
            return synced_config
        
        return saved_config
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi split: {str(e)}")
        
        # Log ke UI jika ui_components tersedia
        if ui_components and 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", 'error')
        
        return config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Import status handler
        from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel
        
        # Update status panel
        update_status_panel(ui_components, "Memperbarui UI dari konfigurasi...", 'info')
        
        # Get config if not provided
        if config is None:
            config = load_config()
            
        # Pastikan config memiliki struktur yang benar
        if not config or 'split' not in config:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan konfigurasi default untuk split dataset")
            config = get_default_split_config()
            
        split_config = config['split']
        
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = split_config.get('enabled', True)
            
        if 'train_ratio_slider' in ui_components:
            ui_components['train_ratio_slider'].value = split_config.get('train_ratio', 0.7)
        elif 'train_slider' in ui_components:
            ui_components['train_slider'].value = split_config.get('train_ratio', 0.7)
            
        if 'val_ratio_slider' in ui_components:
            ui_components['val_ratio_slider'].value = split_config.get('val_ratio', 0.15)
        elif 'val_slider' in ui_components:
            ui_components['val_slider'].value = split_config.get('val_ratio', 0.15)
            
        if 'test_ratio_slider' in ui_components:
            ui_components['test_ratio_slider'].value = split_config.get('test_ratio', 0.15)
        elif 'test_slider' in ui_components:
            ui_components['test_slider'].value = split_config.get('test_ratio', 0.15)
            
        if 'random_seed_input' in ui_components:
            ui_components['random_seed_input'].value = split_config.get('random_seed', 42)
        elif 'random_seed' in ui_components:
            ui_components['random_seed'].value = split_config.get('random_seed', 42)
            
        if 'stratify_checkbox' in ui_components:
            ui_components['stratify_checkbox'].value = split_config.get('stratify', True)
        elif 'stratified_checkbox' in ui_components:
            ui_components['stratified_checkbox'].value = split_config.get('stratify', True)
            
        update_status_panel(ui_components, "UI berhasil diupdate dari konfigurasi split", 'success')
        logger.info(f"{ICONS.get('success', '✅')} UI berhasil diupdate dari konfigurasi split")
        
    except Exception as e:
        # Log error
        if 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel
            update_status_panel(ui_components, f"Error saat mengupdate UI dari konfigurasi: {str(e)}", 'error')
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengupdate UI dari konfigurasi: {str(e)}")
        
        # Jika terjadi error, gunakan konfigurasi default
        default_config = get_default_split_config()
        update_ui_from_config(ui_components, default_config)

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Import status handler
        from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel
        from smartcash.ui.dataset.split.handlers.ui_value_handlers import get_ui_values
        
        # Update status panel
        update_status_panel(ui_components, "Memperbarui konfigurasi dari UI...", 'info')
        
        # Get current config
        config = load_config()
        
        # Simpan config sebelum diupdate untuk verifikasi nantinya
        pre_update_config = config.copy() if config else get_default_split_config()
        
        # Dapatkan nilai dari UI
        ui_values = get_ui_values(ui_components)
        
        # Update config from UI values
        config['split'].update(ui_values)
        
        # Save config dan sinkronkan dengan Google Drive
        saved_config = save_config(config, ui_components)
        
        # Verifikasi perubahan telah disimpan dengan benar dengan membaca ulang config
        post_update_config = load_config()
        
        # Periksa apakah data yang disimpan sesuai
        is_consistent = True
        inconsistent_keys = []
        
        for key, value in ui_values.items():
            if key in post_update_config['split'] and post_update_config['split'][key] != value:
                is_consistent = False
                inconsistent_keys.append(key)
        
        if not is_consistent:
            update_status_panel(ui_components, f"Konfigurasi tidak disimpan dengan benar pada: {', '.join(inconsistent_keys)}", 'warning')
            logger.warning(f"⚠️ Konfigurasi tidak disimpan dengan benar pada: {', '.join(inconsistent_keys)}")
        else:
            update_status_panel(ui_components, "Konfigurasi berhasil diupdate dari UI", 'success')
            logger.info(f"{ICONS.get('success', '✅')} Konfigurasi berhasil diupdate dari UI")
        
        return saved_config
        
    except Exception as e:
        # Log error
        if 'status_panel' in ui_components:
            from smartcash.ui.dataset.split.handlers.status_handlers import update_status_panel
            update_status_panel(ui_components, f"Error saat update config dari UI: {str(e)}", 'error')
        logger.error(f"{ICONS.get('error', '❌')} Error saat update config dari UI: {str(e)}")
        return load_config()
