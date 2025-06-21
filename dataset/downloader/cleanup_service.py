"""
File: smartcash/dataset/downloader/cleanup_service.py
Deskripsi: Fixed cleanup service yang preserve direktori struktur dataset
"""

from pathlib import Path
from typing import Dict, Any, List
from smartcash.dataset.downloader.base import BaseDownloaderComponent, DirectoryManager
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage

class CleanupService(BaseDownloaderComponent):
    """Enhanced cleanup service dengan directory preservation dan structure validation"""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.progress_tracker = None
        self.directory_manager = DirectoryManager()
    
    def set_progress_callback(self, callback) -> None:
        """Set progress callback dan create tracker"""
        super().set_progress_callback(callback)
        self.progress_tracker = DownloadProgressTracker(callback)
    
    def cleanup_dataset_files(self, targets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced cleanup dengan directory structure preservation
        
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
                        
                        # Clean files tapi preserve directory structure
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
            
            # Ensure directory structure after cleanup
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.COMPLETE, "Memastikan struktur direktori...")
            
            self._ensure_dataset_structure_after_cleanup(targets)
            
            if self.progress_tracker:
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
        Clean directory contents tapi preserve essential dataset structure
        
        Args:
            directory: Directory yang akan dibersihkan
            
        Returns:
            Jumlah files yang dibersihkan
        """
        if not directory.exists() or not directory.is_dir():
            return 0
        
        cleaned_count = 0
        essential_dirs = {'images', 'labels', 'train', 'valid', 'test', 'downloads'}
        
        try:
            # Iterate semua items dalam directory
            for item in directory.iterdir():
                if item.is_file():
                    # Hapus file kecuali config files penting
                    if not self._is_essential_file(item):
                        item.unlink()
                        cleaned_count += 1
                elif item.is_dir():
                    # Recursively clean subdirectory
                    if item.name in essential_dirs:
                        # Clean contents tapi preserve directory
                        subdir_count = self._clean_directory_preserve_structure(item)
                        cleaned_count += subdir_count
                    else:
                        # Remove non-essential directories completely
                        try:
                            import shutil
                            shutil.rmtree(item)
                            cleaned_count += self._count_files_recursive(item)
                        except Exception:
                            pass
            
            return cleaned_count
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleaning {directory}: {str(e)}")
            return cleaned_count
    
    def _is_essential_file(self, file_path: Path) -> bool:
        """Check apakah file essential yang harus dipertahankan"""
        essential_files = {
            'dataset.yaml', 'data.yaml', '.gitkeep', 
            'README.md', 'README.txt', 'requirements.txt'
        }
        
        return (
            file_path.name in essential_files or
            file_path.name.startswith('.git') or
            file_path.suffix in {'.gitignore', '.gitkeep'}
        )
    
    def _count_files_recursive(self, directory: Path) -> int:
        """Count files recursively untuk logging purposes"""
        if not directory.exists():
            return 0
            
        count = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    count += 1
        except Exception:
            pass
        return count
    
    def _ensure_dataset_structure_after_cleanup(self, targets: Dict[str, Dict[str, Any]]) -> None:
        """Ensure dataset structure exists after cleanup"""
        try:
            # Determine base path dari targets
            base_paths = set()
            for target_info in targets.values():
                target_path = Path(target_info['path'])
                
                # Find potential base dataset directory
                if target_path.name in {'downloads', 'train', 'valid', 'test'}:
                    base_paths.add(target_path.parent)
                elif target_path.name in {'images', 'labels'}:
                    base_paths.add(target_path.parent.parent)
            
            # Ensure structure for each base path
            for base_path in base_paths:
                structure_result = self.directory_manager.ensure_dataset_structure(base_path)
                if structure_result['status'] == 'success':
                    created_count = structure_result['total_created']
                    if created_count > 0:
                        self.logger.info(f"📁 Recreated {created_count} directories in {base_path}")
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Error ensuring structure: {str(e)}")
    
    def cleanup_downloads_only(self, downloads_path: Path) -> Dict[str, Any]:
        """Enhanced cleanup hanya downloads directory dengan structure preservation"""
        try:
            if not downloads_path.exists():
                # Create downloads directory if not exists
                self.directory_manager.ensure_directory(downloads_path)
                return self._create_success_result(
                    message="Downloads directory created",
                    files_cleaned=0,
                    target_path=str(downloads_path)
                )
            
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.CLEANUP, "Cleaning downloads...")
            
            # Count files before
            files_before = sum(1 for _ in downloads_path.rglob('*') if _.is_file())
            
            # Clean directory but preserve structure
            cleaned_count = self._clean_directory_preserve_structure(downloads_path)
            
            # Ensure downloads directory still exists
            self.directory_manager.ensure_directory(downloads_path)
            
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
        """Enhanced cleanup splits dengan structure preservation"""
        try:
            if self.progress_tracker:
                self.progress_tracker.start_stage(DownloadStage.CLEANUP, "Cleaning splits...")
            
            cleaned_splits = []
            total_cleaned = 0
            
            for split_path in splits_paths:
                split_cleaned = 0
                
                if split_path.exists():
                    # Clean images dan labels directories
                    images_dir = split_path / 'images'
                    labels_dir = split_path / 'labels'
                    
                    if images_dir.exists():
                        split_cleaned += self._clean_directory_preserve_structure(images_dir)
                    
                    if labels_dir.exists():
                        split_cleaned += self._clean_directory_preserve_structure(labels_dir)
                
                # Ensure split structure exists
                self.directory_manager.ensure_directory(split_path / 'images')
                self.directory_manager.ensure_directory(split_path / 'labels')
                
                if split_cleaned > 0 or not split_path.exists():
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
    
    def verify_and_create_structure(self, base_path: Path) -> Dict[str, Any]:
        """Enhanced verify dan create directory structure dengan validation"""
        try:
            # Use DirectoryManager untuk comprehensive structure creation
            result = self.directory_manager.ensure_dataset_structure(base_path)
            
            if result['status'] == 'success':
                created_dirs = result.get('created_directories', [])
                self.logger.info(f"📁 Structure verified: {len(created_dirs)} directories created")
                
                return self._create_success_result(
                    created_directories=created_dirs,
                    total_created=len(created_dirs),
                    message=f"Dataset structure verified and created in {base_path}"
                )
            else:
                return self._create_error_result(result.get('message', 'Failed to create structure'))
                
        except Exception as e:
            return self._create_error_result(f"Structure verification failed: {str(e)}")


def create_cleanup_service(logger=None) -> CleanupService:
    """Factory untuk CleanupService dengan enhanced directory management"""
    return CleanupService(logger)