"""
File: smartcash/dataset/services/organizer/dataset_organizer.py
Deskripsi: Fixed dataset organizer dengan path validator dan val->valid mapping
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.utils.path_validator import get_path_validator

class DatasetOrganizer:
    """Service untuk organize dataset dengan fixed path validation."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.env_manager = get_environment_manager()
        self.path_validator = get_path_validator(logger)
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress dengan gradual updates."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def organize_dataset(self, source_dir: str, remove_source: bool = True) -> Dict[str, Any]:
        """Organize dataset dengan fixed path validation."""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            return {'status': 'error', 'message': f'Source tidak ditemukan: {source_dir}'}
        
        self.logger.info(f"📁 Mengorganisir dataset: {source_dir}")
        self._notify_progress("organize", 0, 100, "Memulai organisasi dataset")
        
        try:
            # Get target paths dari path validator
            target_paths = self.path_validator.get_dataset_paths()
            
            # Detect splits dengan path validator
            splits_found = self.path_validator.detect_available_splits(str(source_path))
            if not splits_found:
                return {'status': 'error', 'message': 'Tidak ada split dataset yang valid ditemukan'}
            
            self.logger.info(f"📊 Splits ditemukan: {', '.join(splits_found)}")
            self.logger.info(f"🎯 Target base: {target_paths['data_root']}")
            
            self._prepare_target_directories(target_paths)
            self._notify_progress("organize", 20, 100, "Menyiapkan direktori target")
            
            moved_stats = {}
            total_splits = len(splits_found)
            
            for i, split in enumerate(splits_found):
                start_progress = 20 + (i * 60 // total_splits)
                end_progress = 20 + ((i + 1) * 60 // total_splits)
                
                self._notify_progress("organize", start_progress, 100, f"Memindahkan split {split}")
                
                split_stats = self._move_split_with_progress(
                    source_path, split, target_paths[split], 
                    start_progress, end_progress
                )
                moved_stats[split] = split_stats
                
                if split_stats['status'] != 'success':
                    self.logger.warning(f"⚠️ Error memindahkan split {split}: {split_stats.get('message')}")
            
            if remove_source:
                self._notify_progress("organize", 90, 100, "Membersihkan source directory")
                self._cleanup_source(source_path)
            
            total_images = sum(stats.get('images', 0) for stats in moved_stats.values())
            total_labels = sum(stats.get('labels', 0) for stats in moved_stats.values())
            
            self._notify_progress("organize", 100, 100, f"Organisasi selesai: {total_images} gambar")
            
            self.logger.success(
                f"✅ Dataset berhasil diorganisir ke {target_paths['data_root']}\n"
                f"   • Total gambar: {total_images}\n"
                f"   • Total label: {total_labels}\n"
                f"   • Splits: {', '.join(splits_found)}"
            )
            
            return {
                'status': 'success',
                'message': f'Dataset berhasil diorganisir: {total_images} gambar',
                'total_images': total_images,
                'total_labels': total_labels,
                'splits': moved_stats,
                'target_paths': target_paths
            }
            
        except Exception as e:
            error_msg = f"Error organizasi dataset: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'message': error_msg}
    
    def _move_split_with_progress(self, source_path: Path, split_name: str, 
                                 target_path_str: str, start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Move split dengan fixed path detection."""
        target_path = Path(target_path_str)
        
        try:
            # Get actual source path menggunakan path validator
            source_split_path = self.path_validator.get_split_path(str(source_path), split_name)
            
            if not source_split_path.exists():
                return {'status': 'error', 'message': f'Split directory tidak ditemukan: {source_split_path}'}
            
            source_images = source_split_path / 'images'
            source_labels = source_split_path / 'labels'
            
            image_count = len(list(source_images.glob('*.*'))) if source_images.exists() else 0
            label_count = len(list(source_labels.glob('*.txt'))) if source_labels.exists() else 0
            
            total_files = image_count + label_count
            current_file = 0
            
            # Move images dengan progress
            if source_images.exists() and image_count > 0:
                target_images = target_path / 'images'
                target_images.mkdir(parents=True, exist_ok=True)
                
                for img_file in source_images.glob('*.*'):
                    shutil.copy2(img_file, target_images / img_file.name)
                    current_file += 1
                    
                    # Update progress secara gradual
                    if current_file % max(1, total_files // 10) == 0:
                        file_progress = int((current_file / total_files) * (end_progress - start_progress))
                        current_progress = start_progress + file_progress
                        self._notify_progress("organize", current_progress, 100, 
                                            f"Menyalin {split_name}: {current_file}/{total_files}")
            
            # Move labels dengan progress
            if source_labels.exists() and label_count > 0:
                target_labels = target_path / 'labels'
                target_labels.mkdir(parents=True, exist_ok=True)
                
                for label_file in source_labels.glob('*.txt'):
                    shutil.copy2(label_file, target_labels / label_file.name)
                    current_file += 1
                    
                    # Update progress secara gradual
                    if current_file % max(1, total_files // 10) == 0:
                        file_progress = int((current_file / total_files) * (end_progress - start_progress))
                        current_progress = start_progress + file_progress
                        self._notify_progress("organize", current_progress, 100, 
                                            f"Menyalin {split_name}: {current_file}/{total_files}")
            
            self._copy_additional_files(source_split_path, target_path)
            
            self.logger.info(f"📁 Split {split_name}: {image_count} gambar, {label_count} label → {target_path}")
            
            return {
                'status': 'success',
                'images': image_count,
                'labels': label_count,
                'path': str(target_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _prepare_target_directories(self, target_paths: Dict[str, str]) -> None:
        """Siapkan direktori target."""
        for split in ['train', 'valid', 'test']:
            if split in target_paths:
                target_path = Path(target_paths[split])
                target_path.mkdir(parents=True, exist_ok=True)
                (target_path / 'images').mkdir(exist_ok=True)
                (target_path / 'labels').mkdir(exist_ok=True)
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy file tambahan."""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        for filename in additional_files:
            source_file = source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, target_dir / filename)
    
    def _cleanup_source(self, source_path: Path) -> None:
        """Cleanup source directory."""
        try:
            if source_path.exists():
                shutil.rmtree(source_path, ignore_errors=True)
                self.logger.info(f"🗑️ Source directory dihapus: {source_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal hapus source: {str(e)}")
    
    def check_organized_dataset(self) -> Dict[str, Any]:
        """Check status dataset yang sudah diorganisir."""
        target_paths = self.path_validator.get_dataset_paths()
        validation_result = self.path_validator.validate_dataset_structure(target_paths['data_root'])
        
        return {
            'is_organized': validation_result['valid'] and validation_result['total_images'] > 0,
            'total_images': validation_result['total_images'],
            'total_labels': validation_result['total_labels'],
            'splits': validation_result['splits'],
            'issues': validation_result['issues']
        }
    
    def cleanup_all_dataset_folders(self) -> Dict[str, Any]:
        """Cleanup dengan gradual progress tracking."""
        target_paths = self.path_validator.get_dataset_paths()
        
        cleanup_stats = {
            'total_files_removed': 0,
            'folders_cleaned': [],
            'errors': []
        }
        
        # Count total files untuk progress calculation
        total_files_to_remove = 0
        folders_to_clean = []
        
        self._notify_progress("cleanup", 5, 100, "Menghitung file yang akan dihapus")
        
        # Count files in splits
        for split in ['train', 'valid', 'test']:
            split_path = Path(target_paths[split])
            if split_path.exists():
                try:
                    file_count = sum(1 for f in split_path.rglob('*') if f.is_file())
                    if file_count > 0:
                        total_files_to_remove += file_count
                        folders_to_clean.append((split, split_path, file_count))
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error counting {split}: {str(e)}")
        
        # Count downloads folder
        downloads_path = Path(target_paths.get('downloads', f"{target_paths['data_root']}/downloads"))
        if downloads_path.exists():
            try:
                file_count = sum(1 for f in downloads_path.rglob('*') if f.is_file())
                if file_count > 0:
                    total_files_to_remove += file_count
                    folders_to_clean.append(('downloads', downloads_path, file_count))
            except Exception as e:
                cleanup_stats['errors'].append(f"Error counting downloads: {str(e)}")
        
        if total_files_to_remove == 0:
            self._notify_progress("cleanup", 100, 100, "Tidak ada file untuk dihapus")
            return {
                'status': 'empty',
                'message': 'Tidak ada file untuk dihapus di folder dataset',
                'stats': cleanup_stats
            }
        
        # Cleanup dengan gradual progress
        files_removed = 0
        
        for i, (folder_name, folder_path, file_count) in enumerate(folders_to_clean):
            start_progress = 10 + (i * 80 // len(folders_to_clean))
            end_progress = 10 + ((i + 1) * 80 // len(folders_to_clean))
            
            self._notify_progress("cleanup", start_progress, 100, f"Menghapus folder {folder_name}")
            
            try:
                # Delete files dengan progress tracking
                current_file = 0
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            current_file += 1
                            files_removed += 1
                            
                            # Update progress setiap 10% dari folder ini
                            if current_file % max(1, file_count // 10) == 0:
                                folder_progress = int((current_file / file_count) * (end_progress - start_progress))
                                current_progress = start_progress + folder_progress
                                self._notify_progress("cleanup", current_progress, 100, 
                                                    f"Menghapus {folder_name}: {current_file}/{file_count}")
                        except Exception:
                            pass
                
                # Remove empty directories
                try:
                    if folder_path.exists():
                        shutil.rmtree(folder_path, ignore_errors=True)
                        cleanup_stats['folders_cleaned'].append(f"{folder_name} ({file_count} files)")
                        self.logger.info(f"🗑️ Cleaned {folder_name}: {file_count} files")
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error cleaning {folder_name}: {str(e)}")
                    
            except Exception as e:
                cleanup_stats['errors'].append(f"Error processing {folder_name}: {str(e)}")
        
        cleanup_stats['total_files_removed'] = files_removed
        
        self._notify_progress("cleanup", 100, 100, f"Cleanup selesai: {files_removed} file dihapus")
        
        return {
            'status': 'success', 
            'message': f"Berhasil menghapus {files_removed} file",
            'stats': cleanup_stats
        }