"""
File: smartcash/ui/dataset/augmentation/handlers/symlink_handler.py
Deskripsi: Handler untuk setup symlink sebelum augmentasi (SRP)
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple

def setup_augmentation_symlinks(ui_components: Dict[str, Any], params: Dict[str, Any], ui_logger) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Setup symlink yang diperlukan untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        ui_logger: UI Logger bridge
        
    Returns:
        Tuple (success, message, symlink_info)
    """
    try:
        # Deteksi environment
        is_colab = _detect_colab_environment()
        
        if not is_colab:
            ui_logger.info("ℹ️ Environment lokal, symlink tidak diperlukan")
            return True, "Local environment", {
                'uses_symlink': False,
                'storage_type': 'Local',
                'data_path': params.get('data_dir', 'data')
            }
        
        ui_logger.info("🔗 Setup symlink untuk Google Drive")
        
        # Cek drive mount
        if not _is_drive_mounted():
            return False, "Google Drive belum di-mount", {}
        
        # Setup data symlink
        success, message, symlink_info = _setup_data_symlink(ui_logger)
        
        if success:
            # Verify paths untuk augmentasi
            verify_success, verify_message, path_info = _verify_augmentation_paths(params.get('split', 'train'), ui_logger)
            if verify_success:
                symlink_info.update(path_info)
                return True, f"{message} | {verify_message}", symlink_info
            else:
                return False, verify_message, {}
        
        return success, message, symlink_info
        
    except Exception as e:
        error_msg = f"Error setup symlink: {str(e)}"
        ui_logger.error(f"❌ {error_msg}")
        return False, error_msg, {}

def _detect_colab_environment() -> bool:
    """Deteksi apakah berjalan di Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _is_drive_mounted() -> bool:
    """Cek apakah Google Drive sudah di-mount."""
    drive_path = Path("/content/drive/MyDrive")
    return drive_path.exists() and drive_path.is_dir()

def _setup_data_symlink(ui_logger) -> Tuple[bool, str, Dict[str, Any]]:
    """Setup symlink untuk direktori data."""
    colab_data_path = Path("/content/data")
    drive_data_path = Path("/content/drive/MyDrive/SmartCash/data")
    
    try:
        # Pastikan direktori SmartCash ada di Drive
        drive_data_path.mkdir(parents=True, exist_ok=True)
        
        # Cek apakah symlink sudah ada dan valid
        if colab_data_path.is_symlink():
            try:
                actual_target = colab_data_path.resolve()
                if actual_target == drive_data_path:
                    ui_logger.info(f"🔗 Symlink sudah aktif: {colab_data_path} → {actual_target}")
                    return True, "Symlink data sudah aktif", {
                        'uses_symlink': True,
                        'storage_type': 'Google Drive (via symlink)',
                        'data_path': str(colab_data_path)
                    }
                else:
                    colab_data_path.unlink()
                    ui_logger.info("🔄 Symlink lama dihapus")
            except OSError:
                colab_data_path.unlink()
                ui_logger.info("🔄 Broken symlink dihapus")
        
        # Backup direktori lokal jika ada
        elif colab_data_path.exists():
            import time
            backup_path = Path(f"/content/data_backup_{int(time.time())}")
            colab_data_path.rename(backup_path)
            ui_logger.info(f"📦 Data di-backup ke {backup_path}")
        
        # Buat symlink baru
        os.symlink(str(drive_data_path), str(colab_data_path))
        
        # Verifikasi symlink
        if colab_data_path.is_symlink() and colab_data_path.resolve() == drive_data_path:
            return True, f"Symlink berhasil dibuat: {colab_data_path} → {drive_data_path}", {
                'uses_symlink': True,
                'storage_type': 'Google Drive (via symlink)',
                'data_path': str(colab_data_path)
            }
        else:
            return False, "Gagal membuat symlink yang valid", {}
            
    except Exception as e:
        return False, f"Error setup symlink: {str(e)}", {}

def _verify_augmentation_paths(split: str, ui_logger) -> Tuple[bool, str, Dict[str, str]]:
    """Verifikasi path yang diperlukan untuk augmentasi."""
    base_path = Path("/content/data")
    
    # Path yang diperlukan
    paths = {
        'preprocessed_split': str(base_path / "preprocessed" / split),
        'preprocessed_images': str(base_path / "preprocessed" / split / "images"),
        'preprocessed_labels': str(base_path / "preprocessed" / split / "labels"),
        'augmented': str(base_path / "augmented"),
        'augmented_images': str(base_path / "augmented" / "images"),
        'augmented_labels': str(base_path / "augmented" / "labels")
    }
    
    # Verifikasi path input
    required_paths = ['preprocessed_images', 'preprocessed_labels']
    missing_paths = [k for k in required_paths if not Path(paths[k]).exists()]
    
    if missing_paths:
        return False, f"Path input tidak ditemukan: {', '.join(missing_paths)}", paths
    
    # Buat path output
    output_paths = ['augmented', 'augmented_images', 'augmented_labels']
    for path_key in output_paths:
        Path(paths[path_key]).mkdir(parents=True, exist_ok=True)
    