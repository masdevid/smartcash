"""
File: smartcash/dataset/downloader/cleanup_service.py
Deskripsi: Backend service untuk cleanup dataset files dengan preservasi direktori
"""

import shutil
from pathlib import Path
from typing import Dict, Any, List
from smartcash.dataset.downloader.base import BaseDownloaderComponent
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage

class CleanupService(BaseDownloaderComponent):
    """Backend service untuk cleanup dataset files"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.progress_tracker = None
    
    def set_progress_callback(self, callback) -> None:
        """Set progress callback dan create tracker"""
        super().set_progress_callback(callback)
        self.progress_tracker = DownloadProgressTracker(callback)
    
    def cleanup_dataset_files(self, targets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cleanup dataset files dengan preservasi direktori
        
        Args:
            targets: Dictionary target cleanup dari scanner
            
        Returns:
            Cleanup result dictionary
        """
        try:
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.INIT, "Mempersiapkan cleanup...")
            
            cleaned_targets = []
            errors = []
            total_targets = len(targets)
            current_target = 0
            
            for target_name, target_info in targets.items():
                current_target += 1
                
                if self.progress_tracker:
                    progress = int((current_target / total_targets) * 80)
                    self.progress_tracker.update_stage(progress, f"🧹 Cleaning {target_name}...")
                
                try:
                    target_path = Path(target_info['path'])
                    
                    if target_path.exists():
                        # Count files before cleanup
                        file_count_before = target_info.get('file_count', 0)
                        
                        # Clean files tapi preserve directory
                        cleaned_count = self._clean_directory_preserve_structure(target_path)
                        
                        cleaned_targets.append({
                            'name': target_name,
                            'path': str(target_path),
                            'files_cleaned': cleaned_count,
                            'files_before': file_count_before
                        })
                        
                        self.logger.info(f"✅ Cleaned {target_name}: {cleaned_count} files dari {target_path}")
                    else:
                        self.logger.warning(f"⚠️ Target {target_name} tidak ditemukan: {target_path}")
                        
                except Exception as e:
                    error_msg = f"Error cleaning {target_name}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.COMPLETE, "Finalisasi cleanup...")
                self.progress_tracker.complete_all(f"Cleanup selesai: {len(cleaned_targets)} targets")
            
            return self._create_success_result(
                cleaned_targets=cleaned_targets,
                errors=errors,
                total_cleaned=len(cleaned_targets),
                total_errors=len(errors)
            )
            
        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.error(f"Cleanup failed: {str(e)}")
            return self._create_error_result(f"Cleanup failed: {str(e)}")
    
    def _clean_directory_preserve_structure(self, directory: Path) -> int:
        """
        Clean directory contents tapi preserve struktur direktori
        
        Args:
            directory: Directory yang akan dibersihkan
            
        Returns:
            Jumlah files yang dibersihkan
        """
        if not directory.exists() or not directory.is_dir():
            return 0
        
        cleaned_count = 0
        
        try:
            # Iterate semua items dalam directory
            for item in directory.iterdir():
                if item.is_file():
                    # Hapus file
                    item.unlink()
                    cleaned_count += 1
                elif item.is_dir():
                    # Recursively clean subdirectory
                    subdir_count = self._clean_directory_preserve_structure(item)
                    cleaned_count += subdir_count
                    
                    # Keep subdirectory structure (don't remove empty dirs)
                    # Tapi bisa remove jika benar-benar kosong dan bukan struktur penting
                    if not any(item.iterdir()) and item.name not in ['images', 'labels']:
                        try:
                            item.rmdir()
                        except OSError:
                            pass  # Directory not empty or permission issue
            
            return cleaned_count
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleaning {directory}: {str(e)}")
            return cleaned_count
    
    def cleanup_downloads_only(self, downloads_path: Path) -> Dict[str, Any]:
        """Cleanup hanya downloads directory"""
        try:
            if not downloads_path.exists():
                return self._create_success_result(message="Downloads directory tidak ditemukan")
            
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.CLEANUP, "Cleaning downloads...")
            
            # Count files before
            files_before = sum(1 for _ in downloads_path.rglob('*') if _.is_file())
            
            # Clean directory
            cleaned_count = self._clean_directory_preserve_structure(downloads_path)
            
            if self.progress_tracker:
                self.progress_tracker.complete_all(f"Downloads cleaned: {cleaned_count} files")
            
            return self._create_success_result(
                files_cleaned=cleaned_count,
                files_before=files_before,
                target_path=str(downloads_path)
            )
            
        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.error(f"Downloads cleanup failed: {str(e)}")
            return self._create_error_result(f"Downloads cleanup failed: {str(e)}")
    
    def cleanup_splits_only(self, splits_paths: List[Path]) -> Dict[str, Any]:
        """Cleanup hanya split directories (train/valid/test)"""
        try:
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.CLEANUP, "Cleaning splits...")
            
            cleaned_splits = []
            total_cleaned = 0
            
            for split_path in splits_paths:
                if split_path.exists():
                    # Clean images dan labels directories
                    images_dir = split_path / 'images'
                    labels_dir = split_path / 'labels'
                    
                    split_cleaned = 0
                    
                    if images_dir.exists():
                        split_cleaned += self._clean_directory_preserve_structure(images_dir)
                    
                    if labels_dir.exists():
                        split_cleaned += self._clean_directory_preserve_structure(labels_dir)
                    
                    if split_cleaned > 0:
                        cleaned_splits.append({
                            'split': split_path.name,
                            'path': str(split_path),
                            'files_cleaned': split_cleaned
                        })
                        total_cleaned += split_cleaned
            
            if self.progress_tracker:
                self.progress_tracker.complete_all(f"Splits cleaned: {total_cleaned} files")
            
            return self._create_success_result(
                cleaned_splits=cleaned_splits,
                total_cleaned=total_cleaned
            )
            
        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.error(f"Splits cleanup failed: {str(e)}")
            return self._create_error_result(f"Splits cleanup failed: {str(e)}")
    
    def verify_directory_structure(self, base_path: Path) -> Dict[str, Any]:
        """Verify dan create directory structure setelah cleanup"""
        try:
            required_dirs = [
                base_path / 'downloads',
                base_path / 'train' / 'images',
                base_path / 'train' / 'labels',
                base_path / 'valid' / 'images', 
                base_path / 'valid' / 'labels',
                base_path / 'test' / 'images',
                base_path / 'test' / 'labels'
            ]
            
            created_dirs = []
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
            
            return self._create_success_result(
                created_directories=created_dirs,
                message=f"Directory structure verified, created {len(created_dirs)} directories"
            )
            
        except Exception as e:
            return self._create_error_result(f"Directory structure verification failed: {str(e)}")

def create_cleanup_service(logger=None) -> CleanupService:
    """Factory untuk CleanupService"""
    return CleanupService(logger)