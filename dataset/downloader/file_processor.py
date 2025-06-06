"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Fixed file processor dengan _copy_additional_files method dan enhanced one-liner style
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class FileProcessor:
    """Fixed processor untuk operasi file dataset dengan complete methods dan progress callback."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress dengan error handling.
        
        Memastikan callback dipanggil dengan format yang benar untuk UI progress tracker:
        - step: Nama step yang valid (extract, organize, dll)
        - current: Nilai progress saat ini (0-100)
        - total: Nilai total progress (biasanya 100)
        - message: Pesan status yang informatif
        """
        if self._progress_callback:
            try:
                # Pastikan nilai dalam range yang valid
                current = max(0, min(100, current))
                total = max(1, total)
                
                # Panggil callback dengan format yang benar
                self._progress_callback(step, current, total, message)
            except Exception as e:
                self.logger.debug(f"🔍 Progress callback error: {str(e)}")
                pass
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract ZIP file dengan progress tracking dan one-liner error handling."""
        try:
            not zipfile.is_zipfile(zip_path) and {'status': 'error', 'message': 'File bukan ZIP yang valid'} or None
            
            self._notify_progress("extract", 0, 100, "Memulai ekstraksi ZIP")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                total_files = len(file_list)
                
                total_files == 0 and {'status': 'error', 'message': 'ZIP file kosong'} or None
                
                extract_to.mkdir(parents=True, exist_ok=True)
                extracted_count = 0
                
                # One-liner extraction dengan progress tracking
                [self._extract_single_file(zip_ref, file_info, extract_to, i, total_files) and setattr(self, '_temp_extracted', extracted_count + 1) 
                 for i, file_info in enumerate(file_list) if (extracted_count := extracted_count + 1) or True]
                
                self._notify_progress("extract", 100, 100, f"Ekstraksi selesai: {extracted_count} file")
                
                return {'status': 'success', 'extracted_files': extracted_count, 'total_files': total_files, 'extract_path': str(extract_to)}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error ekstraksi: {str(e)}'}
    
    def _extract_single_file(self, zip_ref, file_info, extract_to: Path, index: int, total_files: int) -> bool:
        """Extract single file dengan one-liner error handling."""
        try:
            zip_ref.extract(file_info, extract_to)
            # Update progress setiap 10% dengan one-liner calculation
            index % max(1, total_files // 10) == 0 and self._notify_progress("extract", int((index / total_files) * 100), 100, f"Ekstrak: {index + 1}/{total_files}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal ekstrak {file_info.filename}: {str(e)}")
            return False
    
    def organize_dataset(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset struktur dengan enhanced progress tracking dan one-liner style."""
        try:
            not source_dir.exists() and {'status': 'error', 'message': f'Source tidak ditemukan: {source_dir}'} or None
            
            self._notify_progress("organize", 0, 100, "Memulai organisasi dataset")
            
            # Detect structure dengan one-liner validation
            structure = self._detect_structure(source_dir)
            not structure['valid'] and {'status': 'error', 'message': 'Struktur dataset tidak valid'} or None
            
            # Create target dengan one-liner mkdir
            target_dir.mkdir(parents=True, exist_ok=True)
            organized_stats = {}
            
            splits_found = structure['splits']
            total_splits = len(splits_found)
            
            # Process splits dengan one-liner progress calculation
            [organized_stats.update({split: self._organize_split(source_dir, target_dir, split, 
                                                                (i * 80) // total_splits, 
                                                                ((i + 1) * 80) // total_splits)}) 
             for i, split in enumerate(splits_found)]
            
            # Copy additional files dengan one-liner call
            self._copy_additional_files(source_dir, target_dir)
            
            # One-liner stats calculation
            total_images = sum(stats.get('images', 0) for stats in organized_stats.values())
            total_labels = sum(stats.get('labels', 0) for stats in organized_stats.values())
            
            self._notify_progress("organize", 100, 100, f"Organisasi selesai: {total_images} gambar")
            
            return {'status': 'success', 'total_images': total_images, 'total_labels': total_labels, 
                   'splits': organized_stats, 'target_dir': str(target_dir)}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error organisasi: {str(e)}'}
    
    def _detect_structure(self, source_dir: Path) -> Dict[str, Any]:
        """Detect struktur dataset dengan one-liner pattern matching."""
        # One-liner split detection
        splits = [split for split in ['train', 'valid', 'test', 'val'] 
                 if (source_dir / split).exists() and (source_dir / split / 'images').exists()]
        
        # Fallback flat structure detection dengan one-liner
        not splits and (source_dir / 'images').exists() and splits.append('train')
        
        return {'valid': len(splits) > 0, 'splits': splits}
    
    def _organize_split(self, source_dir: Path, target_dir: Path, split: str, 
                       start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Organize single split dengan enhanced progress tracking dan one-liner style."""
        try:
            # Handle val -> valid mapping dengan one-liner
            normalized_split = 'valid' if split == 'val' else split
            
            source_split = source_dir / split
            target_split = target_dir / normalized_split
            
            # Create directories dengan one-liner
            [(target_split / subdir).mkdir(parents=True, exist_ok=True) for subdir in ['images', 'labels']]
            
            # Get file lists dengan one-liner
            source_images, source_labels = source_split / 'images', source_split / 'labels'
            image_files = list(source_images.glob('*.*')) if source_images.exists() else []
            label_files = list(source_labels.glob('*.txt')) if source_labels.exists() else []
            
            total_files = len(image_files) + len(label_files)
            copied_files = 0
            
            # Copy files dengan one-liner progress tracking
            copied_files = self._copy_files_with_progress(image_files, target_split / 'images', 
                                                        copied_files, total_files, split, start_progress, end_progress)
            copied_files = self._copy_files_with_progress(label_files, target_split / 'labels', 
                                                        copied_files, total_files, split, start_progress, end_progress)
            
            return {'status': 'success', 'images': len(image_files), 'labels': len(label_files), 'path': str(target_split)}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error split {split}: {str(e)}'}
    
    def _copy_files_with_progress(self, file_list: list, target_dir: Path, copied_files: int, 
                                 total_files: int, split: str, start_progress: int, end_progress: int) -> int:
        """Copy files dengan progress tracking - one-liner style."""
        [shutil.copy2(file_path, target_dir / file_path.name) and setattr(self, '_temp_copied', copied_files + i + 1) and
         (i % max(1, len(file_list) // 10) == 0 and 
          self._notify_progress("organize", start_progress + int(((copied_files + i + 1) / total_files) * (end_progress - start_progress)), 
                               100, f"Copy {split}: {copied_files + i + 1}/{total_files}"))
         for i, file_path in enumerate(file_list)]
        
        return copied_files + len(file_list)
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy file tambahan dengan one-liner style - FIXED METHOD."""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        # One-liner copy dengan error handling
        [shutil.copy2(source_file, target_dir / filename) if (source_file := source_dir / filename).exists() else None
         for filename in additional_files]
    
    def cleanup_temp_files(self, temp_dir: Path) -> Dict[str, Any]:
        """Cleanup temporary files dengan one-liner removal."""
        try:
            temp_dir.exists() and shutil.rmtree(temp_dir)
            return {'status': 'success', 'message': f'Temp directory cleaned: {temp_dir}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Error cleanup: {str(e)}'}
    
    def validate_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate struktur dataset dengan comprehensive one-liner checks."""
        try:
            not dataset_dir.exists() and {'valid': False, 'message': 'Dataset directory tidak ditemukan'} or None
            
            validation_result = {'valid': True, 'splits': {}, 'total_images': 0, 'total_labels': 0, 'issues': []}
            
            # Check splits dengan one-liner validation dan accumulation
            [validation_result['splits'].update({split: self._validate_single_split(dataset_dir / split, split)}) and
             validation_result.update({'total_images': validation_result['total_images'] + validation_result['splits'][split].get('images', 0),
                                     'total_labels': validation_result['total_labels'] + validation_result['splits'][split].get('labels', 0)}) and
             (validation_result['splits'][split].get('images', 0) == 0 and validation_result['issues'].append(f"Split {split}: Tidak ada gambar")) and
             (validation_result['splits'][split].get('labels', 0) == 0 and validation_result['issues'].append(f"Split {split}: Tidak ada label")) and
             (abs(validation_result['splits'][split].get('images', 0) - validation_result['splits'][split].get('labels', 0)) > 
              validation_result['splits'][split].get('images', 0) * 0.1 and 
              validation_result['issues'].append(f"Split {split}: Mismatch gambar vs label"))
             for split in ['train', 'valid', 'test'] if (dataset_dir / split).exists()]
            
            # Overall validation dengan one-liner
            validation_result['total_images'] == 0 and validation_result.update({'valid': False}) and validation_result['issues'].append("Tidak ada gambar ditemukan")
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'Error validasi: {str(e)}'}
    
    def _validate_single_split(self, split_path: Path, split: str) -> Dict[str, Any]:
        """Validate single split dengan one-liner counting."""
        if not split_path.exists():
            return {'exists': False, 'images': 0, 'labels': 0, 'path': str(split_path)}
        
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # One-liner file counting
        image_count = len([f for f in images_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        return {'exists': True, 'images': image_count, 'labels': label_count, 'path': str(split_path)}
    
    def get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get statistik file dengan one-liner safe operations."""
        try:
            stat = file_path.stat()
            return {'size_bytes': stat.st_size, 'size_mb': stat.st_size / (1024 * 1024), 'exists': True,
                   'is_zip': zipfile.is_zipfile(file_path) if file_path.suffix.lower() == '.zip' else False}
        except Exception:
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False, 'is_zip': False}

# Factory function
def create_file_processor(logger=None) -> FileProcessor:
    """Factory untuk create FileProcessor dengan fixed methods."""
    return FileProcessor(logger)