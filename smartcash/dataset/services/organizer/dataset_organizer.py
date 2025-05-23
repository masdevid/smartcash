"""
File: smartcash/dataset/services/organizer/dataset_organizer.py
Deskripsi: Fixed dataset organizer dengan path handling yang benar dan move ke /data di Drive
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager

class DatasetOrganizer:
    """Service untuk memindahkan dataset dari downloads ke struktur final di /data."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.env_manager = get_environment_manager()
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def _get_target_paths(self) -> Dict[str, str]:
        """Get target paths berdasarkan environment - selalu ke /data untuk consistency."""
        if self.env_manager.is_colab and self.env_manager.is_drive_mounted:
            # Target ke Drive /data bukan Colab /content/data
            base_path = self.env_manager.drive_path / 'data'
        elif self.env_manager.is_colab:
            # Target ke Colab /content/data
            base_path = Path('/content/data')
        else:
            # Local environment
            base_path = Path('data')
        
        return {
            'train': str(base_path / 'train'),
            'valid': str(base_path / 'valid'), 
            'test': str(base_path / 'test'),
            'base': str(base_path)
        }
    
    def organize_dataset(self, source_dir: str, remove_source: bool = True) -> Dict[str, Any]:
        """Pindahkan dataset dari downloads ke /data structure."""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            return {'status': 'error', 'message': f'Source tidak ditemukan: {source_dir}'}
        
        self.logger.info(f"📁 Mengorganisir dataset: {source_dir}")
        self._notify_progress("organize", 0, 100, "Memulai organisasi dataset")
        
        try:
            # Get target paths
            target_paths = self._get_target_paths()
            
            # Check structure di source
            splits_found = self._detect_splits(source_path)
            if not splits_found:
                return {'status': 'error', 'message': 'Tidak ada split dataset yang valid ditemukan'}
            
            self.logger.info(f"📊 Splits ditemukan: {', '.join(splits_found)}")
            self.logger.info(f"🎯 Target base: {target_paths['base']}")
            
            # Prepare target directories
            self._prepare_target_directories(target_paths)
            self._notify_progress("organize", 20, 100, "Menyiapkan direktori target")
            
            # Move each split
            moved_stats = {}
            total_splits = len(splits_found)
            
            for i, split in enumerate(splits_found):
                split_progress = 30 + (i * 60 // total_splits)
                self._notify_progress("organize", split_progress, 100, f"Memindahkan split {split}")
                
                split_stats = self._move_split(source_path / split, target_paths[split], split)
                moved_stats[split] = split_stats
                
                if split_stats['status'] != 'success':
                    self.logger.warning(f"⚠️ Error memindahkan split {split}: {split_stats.get('message')}")
            
            # Cleanup source jika diminta
            if remove_source:
                self._notify_progress("organize", 90, 100, "Membersihkan source directory")
                self._cleanup_source(source_path)
            
            # Calculate final stats
            total_images = sum(stats.get('images', 0) for stats in moved_stats.values())
            total_labels = sum(stats.get('labels', 0) for stats in moved_stats.values())
            
            self._notify_progress("organize", 100, 100, f"Organisasi selesai: {total_images} gambar")
            
            self.logger.success(
                f"✅ Dataset berhasil diorganisir ke {target_paths['base']}\n"
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
    
    def _detect_splits(self, source_path: Path) -> list:
        """Deteksi splits yang tersedia di source."""
        splits = []
        for split_name in ['train', 'valid', 'test', 'val']:
            split_dir = source_path / split_name
            if split_dir.exists() and (split_dir / 'images').exists():
                # Normalize val -> valid
                normalized_name = 'valid' if split_name == 'val' else split_name
                splits.append(normalized_name)
        return splits
    
    def _prepare_target_directories(self, target_paths: Dict[str, str]) -> None:
        """Siapkan direktori target."""
        for split in ['train', 'valid', 'test']:
            if split in target_paths:
                target_path = Path(target_paths[split])
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Buat subdirectories
                (target_path / 'images').mkdir(exist_ok=True)
                (target_path / 'labels').mkdir(exist_ok=True)
    
    def _move_split(self, source_split_path: Path, target_path_str: str, split_name: str) -> Dict[str, Any]:
        """Pindahkan satu split ke target directory."""
        target_path = Path(target_path_str)
        
        try:
            # Handle val -> valid mapping
            actual_source = source_split_path
            if not actual_source.exists() and split_name == 'valid':
                val_path = source_split_path.parent / 'val'
                if val_path.exists():
                    actual_source = val_path
            
            if not actual_source.exists():
                return {'status': 'error', 'message': f'Split directory tidak ditemukan: {actual_source}'}
            
            # Count files before moving
            source_images = actual_source / 'images'
            source_labels = actual_source / 'labels'
            
            image_count = len(list(source_images.glob('*.*'))) if source_images.exists() else 0
            label_count = len(list(source_labels.glob('*.txt'))) if source_labels.exists() else 0
            
            # Move images
            if source_images.exists() and image_count > 0:
                target_images = target_path / 'images'
                self._copy_files(source_images, target_images, ['*.jpg', '*.jpeg', '*.png', '*.bmp'])
            
            # Move labels
            if source_labels.exists() and label_count > 0:
                target_labels = target_path / 'labels'
                self._copy_files(source_labels, target_labels, ['*.txt'])
            
            # Copy additional files (data.yaml, classes.txt, etc)
            self._copy_additional_files(actual_source, target_path)
            
            self.logger.info(f"📁 Split {split_name}: {image_count} gambar, {label_count} label → {target_path}")
            
            return {
                'status': 'success',
                'images': image_count,
                'labels': label_count,
                'path': str(target_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _copy_files(self, source_dir: Path, target_dir: Path, patterns: list) -> None:
        """Copy files dengan multiple patterns."""
        if not source_dir.exists():
            return
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern in patterns:
            for file_path in source_dir.glob(pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, target_dir / file_path.name)
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy file tambahan seperti data.yaml, classes.txt."""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        for filename in additional_files:
            source_file = source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, target_dir / filename)
    
    def _cleanup_source(self, source_path: Path) -> None:
        """Cleanup source directory setelah move."""
        try:
            if source_path.exists():
                shutil.rmtree(source_path, ignore_errors=True)
                self.logger.info(f"🗑️ Source directory dihapus: {source_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal hapus source: {str(e)}")
    
    def check_organized_dataset(self) -> Dict[str, Any]:
        """Check status dataset yang sudah diorganisir."""
        target_paths = self._get_target_paths()
        stats = {'total_images': 0, 'total_labels': 0, 'splits': {}}
        
        for split in ['train', 'valid', 'test']:
            split_path = Path(target_paths[split])
            
            if split_path.exists():
                images_dir = split_path / 'images'
                labels_dir = split_path / 'labels'
                
                image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                
                stats['splits'][split] = {
                    'images': image_count,
                    'labels': label_count,
                    'path': str(split_path)
                }
                
                stats['total_images'] += image_count
                stats['total_labels'] += label_count
        
        stats['is_organized'] = stats['total_images'] > 0
        return stats
    
    def cleanup_all_dataset_folders(self) -> Dict[str, Any]:
        """Cleanup dataset folders (/data/train, /data/valid, /data/test) dan downloads."""
        target_paths = self._get_target_paths()
        
        cleanup_stats = {
            'total_files_removed': 0,
            'folders_cleaned': [],
            'errors': []
        }
        
        # Cleanup splits folders
        for split in ['train', 'valid', 'test']:
            split_path = Path(target_paths[split])
            if split_path.exists():
                try:
                    file_count = sum(1 for f in split_path.rglob('*') if f.is_file())
                    if file_count > 0:
                        shutil.rmtree(split_path, ignore_errors=True)
                        cleanup_stats['total_files_removed'] += file_count
                        cleanup_stats['folders_cleaned'].append(f"{split} ({file_count} files)")
                        self.logger.info(f"🗑️ Cleaned {split}: {file_count} files")
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error cleaning {split}: {str(e)}")
        
        # Cleanup downloads folder
        downloads_paths = [
            Path(target_paths['base']) / 'downloads',  # data/downloads
            Path(target_paths['base']).parent / 'downloads'  # downloads at same level
        ]
        
        for downloads_path in downloads_paths:
            if downloads_path.exists():
                try:
                    file_count = sum(1 for f in downloads_path.rglob('*') if f.is_file())
                    if file_count > 0:
                        shutil.rmtree(downloads_path, ignore_errors=True)
                        cleanup_stats['total_files_removed'] += file_count
                        cleanup_stats['folders_cleaned'].append(f"downloads ({file_count} files)")
                        self.logger.info(f"🗑️ Cleaned downloads: {file_count} files")
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error cleaning downloads: {str(e)}")
        
        if cleanup_stats['total_files_removed'] == 0:
            return {
                'status': 'empty',
                'message': 'Tidak ada file untuk dihapus di folder dataset',
                'stats': cleanup_stats
            }
        
        return {
            'status': 'success', 
            'message': f"Berhasil menghapus {cleanup_stats['total_files_removed']} file",
            'stats': cleanup_stats
        }