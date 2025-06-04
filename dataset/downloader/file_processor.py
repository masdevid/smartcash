"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Complete file processor dengan proper path handling dan progress tracking
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class FileProcessor:
    """Processor untuk operasi file dataset dengan progress callback."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress dengan error handling."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract ZIP file dengan progress tracking."""
        try:
            if not zipfile.is_zipfile(zip_path):
                return {'status': 'error', 'message': 'File bukan ZIP yang valid'}
            
            self._notify_progress("extract", 0, 100, "Memulai ekstraksi ZIP")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                total_files = len(file_list)
                
                if total_files == 0:
                    return {'status': 'error', 'message': 'ZIP file kosong'}
                
                extract_to.mkdir(parents=True, exist_ok=True)
                extracted_count = 0
                
                for i, file_info in enumerate(file_list):
                    try:
                        zip_ref.extract(file_info, extract_to)
                        extracted_count += 1
                        
                        # Update progress setiap 10%
                        if i % max(1, total_files // 10) == 0:
                            progress = int((i / total_files) * 100)
                            self._notify_progress("extract", progress, 100, f"Ekstrak: {extracted_count}/{total_files}")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal ekstrak {file_info.filename}: {str(e)}")
                        continue
                
                self._notify_progress("extract", 100, 100, f"Ekstraksi selesai: {extracted_count} file")
                
                return {
                    'status': 'success',
                    'extracted_files': extracted_count,
                    'total_files': total_files,
                    'extract_path': str(extract_to)
                }
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error ekstraksi: {str(e)}'}
    
    def organize_dataset(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset struktur dengan progress tracking."""
        try:
            if not source_dir.exists():
                return {'status': 'error', 'message': f'Source tidak ditemukan: {source_dir}'}
            
            self._notify_progress("organize", 0, 100, "Memulai organisasi dataset")
            
            # Detect dataset structure
            structure = self._detect_structure(source_dir)
            if not structure['valid']:
                return {'status': 'error', 'message': 'Struktur dataset tidak valid'}
            
            # Create target structure
            target_dir.mkdir(parents=True, exist_ok=True)
            organized_stats = {}
            
            splits_found = structure['splits']
            total_splits = len(splits_found)
            
            for i, split in enumerate(splits_found):
                start_progress = (i * 80) // total_splits
                end_progress = ((i + 1) * 80) // total_splits
                
                self._notify_progress("organize", start_progress, 100, f"Mengorganisir split {split}")
                
                split_result = self._organize_split(source_dir, target_dir, split, start_progress, end_progress)
                organized_stats[split] = split_result
            
            # Copy additional files
            self._copy_additional_files(source_dir, target_dir)
            
            total_images = sum(stats.get('images', 0) for stats in organized_stats.values())
            total_labels = sum(stats.get('labels', 0) for stats in organized_stats.values())
            
            self._notify_progress("organize", 100, 100, f"Organisasi selesai: {total_images} gambar")
            
            return {
                'status': 'success',
                'total_images': total_images,
                'total_labels': total_labels,
                'splits': organized_stats,
                'target_dir': str(target_dir)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error organisasi: {str(e)}'}
    
    def _detect_structure(self, source_dir: Path) -> Dict[str, Any]:
        """Detect struktur dataset."""
        splits = []
        
        # Check standard splits
        for split in ['train', 'valid', 'test', 'val']:
            split_path = source_dir / split
            if split_path.exists() and (split_path / 'images').exists():
                splits.append(split)
        
        # Check flat structure
        if not splits and (source_dir / 'images').exists():
            splits = ['train']  # Treat as single training set
        
        return {'valid': len(splits) > 0, 'splits': splits}
    
    def _organize_split(self, source_dir: Path, target_dir: Path, split: str, 
                       start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Organize single split dengan progress tracking."""
        try:
            # Handle val -> valid mapping
            normalized_split = 'valid' if split == 'val' else split
            
            source_split = source_dir / split
            target_split = target_dir / normalized_split
            
            # Create target directories
            (target_split / 'images').mkdir(parents=True, exist_ok=True)
            (target_split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Copy images
            source_images = source_split / 'images'
            source_labels = source_split / 'labels'
            
            image_files = list(source_images.glob('*.*')) if source_images.exists() else []
            label_files = list(source_labels.glob('*.txt')) if source_labels.exists() else []
            
            total_files = len(image_files) + len(label_files)
            copied_files = 0
            
            # Copy images
            for img_file in image_files:
                shutil.copy2(img_file, target_split / 'images' / img_file.name)
                copied_files += 1
                
                # Update progress
                if copied_files % max(1, total_files // 10) == 0:
                    progress = start_progress + int((copied_files / total_files) * (end_progress - start_progress))
                    self._notify_progress("organize", progress, 100, f"Copy {split}: {copied_files}/{total_files}")
            
            # Copy labels
            for label_file in label_files:
                shutil.copy2(label_file, target_split / 'labels' / label_file.name)
                copied_files += 1
                
                # Update progress
                if copied_files % max(1, total_files // 10) == 0:
                    progress = start_progress + int((copied_files / total_files) * (end_progress - start_progress))
                    self._notify_progress("organize", progress, 100, f"Copy {split}: {copied_files}/{total_files}")
            
            return {
                'status': 'success',
                'images': len(image_files),
                'labels': len(label_files),
                'path': str(target_split)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error split {split}: {str(e)}'}
    
    def cleanup_temp_files(self, temp_dir: Path) -> Dict[str, Any]:
        """Cleanup temporary files dan directories."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                return {'status': 'success', 'message': f'Temp directory cleaned: {temp_dir}'}
            return {'status': 'success', 'message': 'Temp directory tidak ada'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error cleanup: {str(e)}'}
    
    def validate_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate struktur dataset yang sudah diorganisir."""
        try:
            if not dataset_dir.exists():
                return {'valid': False, 'message': 'Dataset directory tidak ditemukan'}
            
            validation_result = {
                'valid': True,
                'splits': {},
                'total_images': 0,
                'total_labels': 0,
                'issues': []
            }
            
            # Check each split
            for split in ['train', 'valid', 'test']:
                split_path = dataset_dir / split
                images_dir = split_path / 'images'
                labels_dir = split_path / 'labels'
                
                if split_path.exists():
                    image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                    label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                    
                    validation_result['splits'][split] = {
                        'exists': True,
                        'images': image_count,
                        'labels': label_count,
                        'path': str(split_path)
                    }
                    
                    validation_result['total_images'] += image_count
                    validation_result['total_labels'] += label_count
                    
                    # Check for issues
                    if image_count == 0:
                        validation_result['issues'].append(f"Split {split}: Tidak ada gambar")
                    if label_count == 0:
                        validation_result['issues'].append(f"Split {split}: Tidak ada label")
                    if abs(image_count - label_count) > image_count * 0.1:  # 10% tolerance
                        validation_result['issues'].append(f"Split {split}: Mismatch gambar vs label")
                else:
                    validation_result['splits'][split] = {
                        'exists': False,
                        'images': 0,
                        'labels': 0,
                        'path': str(split_path)
                    }
            
            # Overall validation
            if validation_result['total_images'] == 0:
                validation_result['valid'] = False
                validation_result['issues'].append("Tidak ada gambar ditemukan")
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'Error validasi: {str(e)}'}
    
    def get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get statistik file."""
        try:
            stat = file_path.stat()
            return {
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'exists': True,
                'is_zip': zipfile.is_zipfile(file_path) if file_path.suffix.lower() == '.zip' else False
            }
        except Exception:
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False, 'is_zip': False}

# Factory function
def create_file_processor(logger=None) -> FileProcessor:
    """Factory untuk create FileProcessor."""
    return FileProcessor(logger)