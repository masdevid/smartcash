"""
File: smartcash/ui/dataset/augmentation/utils/symlink_setup_manager.py
Deskripsi: Manager untuk setup symlink sebelum augmentasi dimulai
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class SymlinkSetupManager:
    """Manager untuk setup dan verifikasi symlink sebelum augmentasi."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi symlink setup manager.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.is_colab = self._detect_colab_environment()
        
        # Path constants
        self.colab_data_path = Path("/content/data")
        self.drive_smartcash_path = Path("/content/drive/MyDrive/SmartCash")
        self.drive_data_path = self.drive_smartcash_path / "data"
    
    def _detect_colab_environment(self) -> bool:
        """Deteksi apakah berjalan di Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def setup_symlinks_for_augmentation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Setup symlink yang diperlukan untuk augmentasi.
        
        Returns:
            Tuple (success, message, symlink_info)
        """
        if not self.is_colab:
            log_message(self.ui_components, "ℹ️ Environment lokal, symlink tidak diperlukan", "info")
            return True, "Local environment - symlink tidak diperlukan", {
                'uses_symlink': False,
                'storage_type': 'Local',
                'data_path': 'data'
            }
        
        log_message(self.ui_components, "🔗 Memeriksa dan setup symlink untuk Google Drive", "info")
        
        # Cek apakah Google Drive sudah di-mount
        if not self._is_drive_mounted():
            return False, "Google Drive belum di-mount. Jalankan drive.mount() terlebih dahulu.", {}
        
        # Setup symlink data directory
        success, message, symlink_info = self._setup_data_symlink()
        
        if success:
            log_message(self.ui_components, f"✅ {message}", "success")
        else:
            log_message(self.ui_components, f"❌ {message}", "error")
        
        return success, message, symlink_info
    
    def _is_drive_mounted(self) -> bool:
        """Cek apakah Google Drive sudah di-mount."""
        drive_path = Path("/content/drive/MyDrive")
        return drive_path.exists() and drive_path.is_dir()
    
    def _setup_data_symlink(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Setup symlink untuk direktori data.
        
        Returns:
            Tuple (success, message, symlink_info)
        """
        try:
            # Pastikan direktori SmartCash ada di Drive
            self.drive_smartcash_path.mkdir(parents=True, exist_ok=True)
            self.drive_data_path.mkdir(parents=True, exist_ok=True)
            
            # Cek apakah symlink sudah ada dan valid
            if self.colab_data_path.is_symlink():
                # Symlink sudah ada, cek apakah valid
                try:
                    actual_target = self.colab_data_path.resolve()
                    if actual_target == self.drive_data_path:
                        log_message(self.ui_components, f"🔗 Symlink sudah aktif: {self.colab_data_path} → {actual_target}", "info")
                        return True, "Symlink data sudah aktif dan valid", {
                            'uses_symlink': True,
                            'storage_type': 'Google Drive (via symlink)',
                            'data_path': str(self.colab_data_path),
                            'target_path': str(actual_target)
                        }
                    else:
                        # Symlink mengarah ke path yang salah, hapus
                        self.colab_data_path.unlink()
                        log_message(self.ui_components, "🔄 Symlink lama dihapus, membuat yang baru", "info")
                except OSError:
                    # Broken symlink, hapus
                    self.colab_data_path.unlink()
                    log_message(self.ui_components, "🔄 Broken symlink dihapus, membuat yang baru", "info")
            
            # Jika ada direktori biasa (bukan symlink), backup dan hapus
            elif self.colab_data_path.exists():
                backup_path = Path(f"/content/data_backup_{int(time.time())}")
                self.colab_data_path.rename(backup_path)
                log_message(self.ui_components, f"📦 Direktori data lama di-backup ke {backup_path}", "info")
            
            # Buat symlink baru
            os.symlink(str(self.drive_data_path), str(self.colab_data_path))
            
            # Verifikasi symlink
            if self.colab_data_path.is_symlink() and self.colab_data_path.resolve() == self.drive_data_path:
                return True, f"Symlink berhasil dibuat: {self.colab_data_path} → {self.drive_data_path}", {
                    'uses_symlink': True,
                    'storage_type': 'Google Drive (via symlink)',
                    'data_path': str(self.colab_data_path),
                    'target_path': str(self.drive_data_path)
                }
            else:
                return False, "Gagal membuat symlink yang valid", {}
                
        except Exception as e:
            return False, f"Error saat setup symlink: {str(e)}", {}
    
    def verify_augmentation_paths(self, split: str = "train") -> Tuple[bool, str, Dict[str, str]]:
        """
        Verifikasi path yang diperlukan untuk augmentasi.
        
        Args:
            split: Split dataset yang akan diaugmentasi
            
        Returns:
            Tuple (success, message, path_info)
        """
        if self.is_colab:
            base_path = self.colab_data_path
        else:
            base_path = Path("data")
        
        # Path yang diperlukan
        paths = {
            'base_data': str(base_path),
            'preprocessed': str(base_path / "preprocessed"),
            'preprocessed_split': str(base_path / "preprocessed" / split),
            'preprocessed_images': str(base_path / "preprocessed" / split / "images"),
            'preprocessed_labels': str(base_path / "preprocessed" / split / "labels"),
            'augmented': str(base_path / "augmented"),
            'augmented_split': str(base_path / "augmented" / split),
            'augmented_images': str(base_path / "augmented" / split / "images"),
            'augmented_labels': str(base_path / "augmented" / split / "labels")
        }
        
        # Verifikasi path input (harus ada)
        required_input_paths = ['preprocessed_split', 'preprocessed_images', 'preprocessed_labels']
        missing_paths = []
        
        for path_key in required_input_paths:
            if not Path(paths[path_key]).exists():
                missing_paths.append(path_key)
        
        if missing_paths:
            return False, f"Path input tidak ditemukan: {', '.join(missing_paths)}", paths
        
        # Buat path output (jika belum ada)
        output_paths = ['augmented', 'augmented_split', 'augmented_images', 'augmented_labels']
        created_paths = []
        
        for path_key in output_paths:
            path = Path(paths[path_key])
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    created_paths.append(path_key)
                except Exception as e:
                    return False, f"Gagal membuat direktori {path_key}: {str(e)}", paths
        
        if created_paths:
            log_message(self.ui_components, f"📁 Dibuat direktori: {', '.join(created_paths)}", "info")
        
        # Count file input
        images_path = Path(paths['preprocessed_images'])
        labels_path = Path(paths['preprocessed_labels'])
        
        image_count = len([f for f in images_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        label_count = len([f for f in labels_path.glob('*.txt')])
        
        log_message(self.ui_components, f"📊 Dataset {split}: {image_count} gambar, {label_count} label", "info")
        
        return True, f"Path verifikasi berhasil untuk split {split}", paths
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi storage yang sedang digunakan.
        
        Returns:
            Dictionary informasi storage
        """
        if not self.is_colab:
            return {
                'environment': 'Local',
                'uses_symlink': False,
                'storage_type': 'Local',
                'data_path': 'data'
            }
        
        # Cek status symlink
        if self.colab_data_path.is_symlink():
            try:
                target = self.colab_data_path.resolve()
                return {
                    'environment': 'Google Colab',
                    'uses_symlink': True,
                    'storage_type': 'Google Drive (via symlink)',
                    'data_path': str(self.colab_data_path),
                    'target_path': str(target),
                    'symlink_active': True
                }
            except OSError:
                return {
                    'environment': 'Google Colab',
                    'uses_symlink': False,
                    'storage_type': 'Local (broken symlink)',
                    'data_path': str(self.colab_data_path),
                    'symlink_active': False
                }
        else:
            return {
                'environment': 'Google Colab',
                'uses_symlink': False,
                'storage_type': 'Local (no symlink)',
                'data_path': str(self.colab_data_path),
                'symlink_active': False
            }

def get_symlink_setup_manager(ui_components: Dict[str, Any]) -> SymlinkSetupManager:
    """
    Factory function untuk mendapatkan symlink setup manager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance SymlinkSetupManager
    """
    return SymlinkSetupManager(ui_components)