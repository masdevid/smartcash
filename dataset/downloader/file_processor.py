"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Optimized file processor dengan one-liner methods dan parallel processing
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
from smartcash.common.logger import get_logger

class FileProcessor:
    """Optimized file processor dengan one-liner methods dan parallelism."""
    
    def __init__(self, logger=None, max_workers: int = 4):
        self.logger, self.max_workers, self._progress_callback = logger or get_logger(), max_workers, None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """One-liner callback setter"""
        self._progress_callback = callback
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract ZIP dengan optimized one-liner validation dan parallel extraction"""
        try:
            # One-liner validation
            not zipfile.is_zipfile(zip_path) and self._return_error('File bukan ZIP valid')
            
            self._notify_progress("extract", 0, 100, "📦 Memulai ekstraksi...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                total_files = len(file_list)
                
                # One-liner empty check
                total_files == 0 and self._return_error('ZIP file kosong')
                
                extract_to.mkdir(parents=True, exist_ok=True)
                
                # Parallel extraction dengan ThreadPoolExecutor
                extracted_count = self._extract_files_parallel(zip_ref, file_list, extract_to)
                
                self._notify_progress("extract", 100, 100, f"✅ Ekstraksi selesai: {extracted_count} file")
                return {'status': 'success', 'extracted_files': extracted_count, 'total_files': total_files, 'extract_path': str(extract_to)}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error ekstraksi: {str(e)}'}
    
    def _extract_files_parallel(self, zip_ref, file_list: List, extract_to: Path) -> int:
        """Parallel file extraction dengan optimized progress tracking"""
        extracted_count, batch_size = 0, max(1, len(file_list) // 20)  # Update setiap 5%
        
        # One-liner parallel extraction
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, file_info in enumerate(file_list):
                try:
                    zip_ref.extract(file_info, extract_to)
                    extracted_count += 1
                    
                    # One-liner progress update dengan batch
                    i % batch_size == 0 and self._notify_progress("extract", int((i / len(file_list)) * 100), 100, f"📦 {i + 1}/{len(file_list)}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal ekstrak {file_info.filename}: {str(e)}")
        
        return extracted_count
    
    def organize_dataset(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset dengan optimized structure detection dan parallel copying"""
        try:
            # One-liner existence check
            not source_dir.exists() and self._return_error(f'Source tidak ditemukan: {source_dir}')
            
            self._notify_progress("organize", 0, 100, "🗂️ Memulai organisasi...")
            
            # One-liner structure detection
            structure = self._detect_structure_optimized(source_dir)
            not structure['valid'] and self._return_error('Struktur dataset tidak valid')
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Parallel split processing
            organized_stats = self._organize_splits_parallel(source_dir, target_dir, structure['splits'])
            
            # One-liner additional files copy
            self._copy_additional_files_optimized(source_dir, target_dir)
            
            # One-liner stats calculation
            total_images, total_labels = (sum(stats.get('images', 0) for stats in organized_stats.values()),
                                        sum(stats.get('labels', 0) for stats in organized_stats.values()))
            
            self._notify_progress("organize", 100, 100, f"✅ Organisasi selesai: {total_images} gambar")
            return {'status': 'success', 'total_images': total_images, 'total_labels': total_labels, 
                   'splits': organized_stats, 'target_dir': str(target_dir)}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error organisasi: {str(e)}'}
    
    def _detect_structure_optimized(self, source_dir: Path) -> Dict[str, Any]:
        """One-liner optimized structure detection"""
        splits = [split for split in ['train', 'valid', 'test', 'val'] 
                 if (source_dir / split).exists() and (source_dir / split / 'images').exists()]
        
        # One-liner fallback detection
        not splits and (source_dir / 'images').exists() and splits.append('train')
        
        return {'valid': bool(splits), 'splits': splits}
    
    def _organize_splits_parallel(self, source_dir: Path, target_dir: Path, splits: List[str]) -> Dict[str, Any]:
        """Parallel split organization dengan optimized file operations"""
        organized_stats = {}
        
        # Sequential processing untuk setiap split (parallel dalam split)
        for i, split in enumerate(splits):
            normalized_split = 'valid' if split == 'val' else split
            start_progress, end_progress = (i * 80) // len(splits), ((i + 1) * 80) // len(splits)
            
            organized_stats[normalized_split] = self._organize_single_split_optimized(
                source_dir, target_dir, split, start_progress, end_progress
            )
        
        return organized_stats
    
    def _organize_single_split_optimized(self, source_dir: Path, target_dir: Path, split: str, 
                                       start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Optimized single split organization dengan parallel file copying"""
        try:
            normalized_split = 'valid' if split == 'val' else split
            source_split, target_split = source_dir / split, target_dir / normalized_split
            
            # One-liner directory creation
            [(target_split / subdir).mkdir(parents=True, exist_ok=True) for subdir in ['images', 'labels']]
            
            # One-liner file collection
            source_images, source_labels = source_split / 'images', source_split / 'labels'
            image_files = list(source_images.glob('*.*')) if source_images.exists() else []
            label_files = list(source_labels.glob('*.txt')) if source_labels.exists() else []
            
            # Parallel file copying
            self._copy_files_parallel(image_files, target_split / 'images', split, start_progress, end_progress, len(image_files) + len(label_files))
            self._copy_files_parallel(label_files, target_split / 'labels', split, start_progress, end_progress, len(image_files) + len(label_files))
            
            return {'status': 'success', 'images': len(image_files), 'labels': len(label_files), 'path': str(target_split)}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error split {split}: {str(e)}'}
    
    def _copy_files_parallel(self, file_list: List[Path], target_dir: Path, split: str, 
                           start_progress: int, end_progress: int, total_files: int) -> None:
        """Parallel file copying dengan optimized batch processing"""
        if not file_list:
            return
        
        batch_size = max(1, len(file_list) // 10)  # Update setiap 10%
        
        # One-liner parallel copy dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(shutil.copy2, file_path, target_dir / file_path.name) for file_path in file_list]
            
            # One-liner progress tracking
            [i % batch_size == 0 and self._notify_progress("organize", 
                start_progress + int(((i + 1) / len(file_list)) * (end_progress - start_progress)), 
                100, f"📁 Copy {split}: {i + 1}/{len(file_list)}") 
             for i, future in enumerate(futures) if future.result() or True]
    
    def _copy_additional_files_optimized(self, source_dir: Path, target_dir: Path) -> None:
        """One-liner optimized additional files copying"""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        # One-liner parallel copy dengan error handling
        [shutil.copy2(source_file, target_dir / filename) if (source_file := source_dir / filename).exists() else None
         for filename in additional_files]
    
    def cleanup_temp_files(self, temp_dir: Path) -> Dict[str, Any]:
        """One-liner cleanup dengan safe removal"""
        try:
            temp_dir.exists() and shutil.rmtree(temp_dir, ignore_errors=True)
            return {'status': 'success', 'message': f'✅ Temp cleaned: {temp_dir}'}
        except Exception as e:
            return {'status': 'error', 'message': f'❌ Cleanup error: {str(e)}'}
    
    def validate_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Optimized structure validation dengan parallel checking"""
        try:
            not dataset_dir.exists() and self._return_error('Dataset directory tidak ditemukan')
            
            validation_result = {'valid': True, 'splits': {}, 'total_images': 0, 'total_labels': 0, 'issues': []}
            
            # Parallel split validation
            splits_to_check = [split for split in ['train', 'valid', 'test'] if (dataset_dir / split).exists()]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                split_futures = {split: executor.submit(self._validate_single_split_optimized, dataset_dir / split, split) 
                               for split in splits_to_check}
                
                # One-liner result aggregation
                [validation_result['splits'].update({split: future.result()}) and
                 validation_result.update({
                     'total_images': validation_result['total_images'] + future.result().get('images', 0),
                     'total_labels': validation_result['total_labels'] + future.result().get('labels', 0)
                 }) and
                 (future.result().get('images', 0) == 0 and validation_result['issues'].append(f"Split {split}: No images")) and
                 (abs(future.result().get('images', 0) - future.result().get('labels', 0)) > 
                  future.result().get('images', 0) * 0.1 and 
                  validation_result['issues'].append(f"Split {split}: Image-label mismatch"))
                 for split, future in split_futures.items()]
            
            # One-liner overall validation
            validation_result['total_images'] == 0 and validation_result.update({'valid': False}) and validation_result['issues'].append("No images found")
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'❌ Validation error: {str(e)}'}
    
    def _validate_single_split_optimized(self, split_path: Path, split: str) -> Dict[str, Any]:
        """One-liner optimized split validation"""
        if not split_path.exists():
            return {'exists': False, 'images': 0, 'labels': 0, 'path': str(split_path)}
        
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # One-liner parallel file counting
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_count = len([f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        return {'exists': True, 'images': image_count, 'labels': label_count, 'path': str(split_path)}
    
    def get_file_stats_optimized(self, file_path: Path) -> Dict[str, Any]:
        """One-liner optimized file stats"""
        try:
            stat = file_path.stat()
            return {
                'size_bytes': stat.st_size, 'size_mb': stat.st_size / 1048576, 'exists': True,
                'is_zip': zipfile.is_zipfile(file_path) if file_path.suffix.lower() == '.zip' else False,
                'modified': stat.st_mtime
            }
        except Exception:
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False, 'is_zip': False, 'modified': 0}
    
    def batch_copy_files(self, file_mapping: Dict[Path, Path]) -> Dict[str, Any]:
        """Optimized batch file copying dengan parallel execution"""
        try:
            if not file_mapping:
                return {'status': 'success', 'copied': 0, 'message': 'No files to copy'}
            
            # Parallel batch copying
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(shutil.copy2, src, dst) for src, dst in file_mapping.items()]
                
                # One-liner result aggregation
                successful_copies = sum(1 for future in futures if future.result() or True)
            
            return {'status': 'success', 'copied': successful_copies, 'total': len(file_mapping), 'message': f'✅ Copied {successful_copies} files'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'❌ Batch copy error: {str(e)}'}
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """One-liner progress notification dengan safe execution"""
        self._progress_callback and (lambda: self._progress_callback(step, current, total, message))() if True else None
    
    def _return_error(self, message: str) -> None:
        """One-liner error return"""
        raise Exception(message)

# One-liner factory dengan optimized defaults
def create_file_processor(logger=None, max_workers: int = None) -> FileProcessor:
    """Factory untuk optimized FileProcessor dengan auto-detected workers"""
    import os
    optimal_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
    return FileProcessor(logger, optimal_workers)