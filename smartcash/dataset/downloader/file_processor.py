"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Enhanced file processor dengan UUID file renaming dan one-liner methods
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import FileNamingManager

class FileProcessor:
    """Enhanced file processor dengan UUID renaming dan optimized operations."""
    
    def __init__(self, logger=None, max_workers: int = 4):
        self.logger, self.max_workers, self._progress_callback = logger or get_logger(), max_workers, None
        self.naming_manager = FileNamingManager(logger=logger)
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """One-liner callback setter"""
        self._progress_callback = callback
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract ZIP dengan optimized one-liner validation dan parallel extraction"""
        try:
            # One-liner validation
            if not zipfile.is_zipfile(zip_path):
                return self._return_error('File bukan ZIP valid')
            
            self._notify_progress("extract", 0, 100, "📦 Memulai ekstraksi...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                total_files = len(file_list)
                
                # One-liner empty check
                if total_files == 0:
                    return self._return_error('ZIP file kosong')
                
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
                    if i % batch_size == 0:
                        self._notify_progress("extract", int((i / len(file_list)) * 100), 100, f"📦 {i + 1}/{len(file_list)}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal ekstrak {file_info.filename}: {str(e)}")
        
        return extracted_count
    
    def organize_dataset_with_renaming(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset dengan UUID renaming dan optimized structure detection"""
        try:
            # One-liner existence check
            if not source_dir.exists():
                return self._return_error(f'Source tidak ditemukan: {source_dir}')
            
            self._notify_progress("organize", 0, 100, "🗂️ Memulai organisasi...")
            
            # One-liner structure detection
            structure = self._detect_structure_optimized(source_dir)
            if not structure['valid']:
                return self._return_error('Struktur dataset tidak valid')
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Process splits dengan renaming
            organized_stats = self._organize_splits_with_renaming(source_dir, target_dir, structure['splits'])
            
            # One-liner additional files copy
            self._copy_additional_files_optimized(source_dir, target_dir)
            
            # One-liner stats calculation
            total_images, total_labels = (sum(stats.get('images', 0) for stats in organized_stats.values()),
                                        sum(stats.get('labels', 0) for stats in organized_stats.values()))
            
            self._notify_progress("organize", 100, 100, f"✅ Organisasi selesai: {total_images} gambar dengan UUID format")
            return {'status': 'success', 'total_images': total_images, 'total_labels': total_labels, 
                   'splits': organized_stats, 'target_dir': str(target_dir), 'uuid_renamed': True}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error organisasi: {str(e)}'}
    
    def _organize_splits_with_renaming(self, source_dir: Path, target_dir: Path, splits: List[str]) -> Dict[str, Any]:
        """Parallel split organization dengan UUID renaming"""
        organized_stats = {}
        
        # Sequential processing untuk setiap split (parallel dalam split)
        for i, split in enumerate(splits):
            normalized_split = 'valid' if split == 'val' else split
            start_progress, end_progress = (i * 80) // len(splits), ((i + 1) * 80) // len(splits)
            
            organized_stats[normalized_split] = self._organize_single_split_with_renaming(
                source_dir, target_dir, split, start_progress, end_progress
            )
        
        return organized_stats
    
    def _organize_single_split_with_renaming(self, source_dir: Path, target_dir: Path, split: str, 
                                           start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Optimized single split organization dengan UUID file renaming"""
        try:
            normalized_split = 'valid' if split == 'val' else split
            source_split, target_split = source_dir / split, target_dir / normalized_split
            
            # One-liner directory creation
            [(target_split / subdir).mkdir(parents=True, exist_ok=True) for subdir in ['images', 'labels']]
            
            # One-liner file collection
            source_images, source_labels = source_split / 'images', source_split / 'labels'
            image_files = list(source_images.glob('*.*')) if source_images.exists() else []
            label_files = list(source_labels.glob('*.txt')) if source_labels.exists() else []
            
            # Process dengan UUID renaming
            renamed_images = self._copy_and_rename_images(image_files, target_split / 'images', source_labels, split, start_progress, end_progress)
            renamed_labels = self._copy_and_rename_labels(label_files, target_split / 'labels', split)
            
            return {'status': 'success', 'images': renamed_images, 'labels': renamed_labels, 'path': str(target_split), 'uuid_format': True}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error split {split}: {str(e)}'}
    
    def _copy_and_rename_images(self, image_files: List[Path], target_dir: Path, labels_dir: Path,
                               split: str, start_progress: int, end_progress: int) -> int:
        """Copy dan rename images dengan UUID format"""
        if not image_files:
            return 0
        
        renamed_count, batch_size = 0, max(1, len(image_files) // 10)
        
        for i, img_file in enumerate(image_files):
            try:
                # Extract primary class dari corresponding label
                label_path = labels_dir / f"{img_file.stem}.txt"
                primary_class = self.naming_manager.extract_primary_class_from_label(label_path)
                
                # Generate new filename dengan UUID
                file_info = self.naming_manager.generate_file_info(img_file.name, primary_class, 'raw')
                new_filename = file_info.get_filename()
                
                # Copy dengan nama baru
                target_path = target_dir / new_filename
                shutil.copy2(img_file, target_path)
                renamed_count += 1
                
                # Progress update dengan batch
                if i % batch_size == 0:
                    progress = start_progress + int(((i + 1) / len(image_files)) * (end_progress - start_progress))
                    self._notify_progress("organize", progress, 100, f"📸 Rename {split}: {i + 1}/{len(image_files)}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error rename image {img_file.name}: {str(e)}")
        
        return renamed_count
    
    def _copy_and_rename_labels(self, label_files: List[Path], target_dir: Path, split: str) -> int:
        """Copy dan rename labels dengan UUID consistency"""
        if not label_files:
            return 0
        
        renamed_count = 0
        
        for label_file in label_files:
            try:
                # Extract primary class dari label content
                primary_class = self.naming_manager.extract_primary_class_from_label(label_file)
                
                # Generate new filename dengan same UUID as corresponding image
                original_stem = label_file.stem
                if original_stem in self.naming_manager.uuid_registry:
                    # Use existing UUID dari image
                    file_info = self.naming_manager.generate_file_info(label_file.name, primary_class, 'raw')
                    new_filename = f"{Path(file_info.get_filename()).stem}.txt"
                    
                    # Copy dengan nama baru
                    target_path = target_dir / new_filename
                    shutil.copy2(label_file, target_path)
                    renamed_count += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error rename label {label_file.name}: {str(e)}")
        
        return renamed_count
    
    def _detect_structure_optimized(self, source_dir: Path) -> Dict[str, Any]:
        """One-liner optimized structure detection"""
        splits = [split for split in ['train', 'valid', 'test', 'val'] 
                 if (source_dir / split).exists() and (source_dir / split / 'images').exists()]
        
        # Fallback detection untuk dataset tanpa split
        if not splits and (source_dir / 'images').exists():
            splits.append('train')
        
        return {'valid': bool(splits), 'splits': splits}
    
    def _copy_additional_files_optimized(self, source_dir: Path, target_dir: Path) -> None:
        """One-liner optimized additional files copying"""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        # One-liner parallel copy dengan error handling
        [shutil.copy2(source_file, target_dir / filename) if (source_file := source_dir / filename).exists() else None
         for filename in additional_files]
    
    def cleanup_temp_files(self, temp_dir: Path) -> Dict[str, Any]:
        """One-liner cleanup dengan safe removal"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return {'status': 'success', 'message': f'✅ Temp cleaned: {temp_dir}'}
        except Exception as e:
            return {'status': 'error', 'message': f'❌ Cleanup error: {str(e)}'}
    
    def validate_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Optimized structure validation dengan UUID format checking"""
        try:
            if not dataset_dir.exists():
                return self._return_error('Dataset directory tidak ditemukan')
            
            validation_result = {'valid': True, 'splits': {}, 'total_images': 0, 'total_labels': 0, 'issues': [], 'uuid_format': True}
            
            # Parallel split validation
            splits_to_check = [split for split in ['train', 'valid', 'test'] if (dataset_dir / split).exists()]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                split_futures = {split: executor.submit(self._validate_single_split_with_uuid, dataset_dir / split, split) 
                               for split in splits_to_check}
                
                # Result aggregation dengan validasi
                for split, future in split_futures.items():
                    result = future.result()
                    validation_result['splits'][split] = result
                    
                    # Update totals
                    validation_result['total_images'] += result.get('images', 0)
                    validation_result['total_labels'] += result.get('labels', 0)
                    
                    # Validasi UUID format
                    if not result.get('uuid_consistent', True):
                        validation_result['uuid_format'] = False
                        validation_result['issues'].append(f"Split {split}: Inconsistent UUID format")
                    
                    # Validasi images
                    if result.get('images', 0) == 0:
                        validation_result['issues'].append(f"Split {split}: No images")
                    
                    # Validasi image-label mismatch
                    if abs(result.get('images', 0) - result.get('labels', 0)) > result.get('images', 0) * 0.1:
                        validation_result['issues'].append(f"Split {split}: Image-label mismatch")
            
            # Overall validation
            if validation_result['total_images'] == 0:
                validation_result.update({'valid': False})
                validation_result['issues'].append("No images found")
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'message': f'❌ Validation error: {str(e)}'}
    
    def _validate_single_split_with_uuid(self, split_path: Path, split: str) -> Dict[str, Any]:
        """One-liner optimized split validation dengan UUID format checking"""
        if not split_path.exists():
            return {'exists': False, 'images': 0, 'labels': 0, 'path': str(split_path), 'uuid_consistent': False}
        
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # One-liner file counting dan UUID validation
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions] if images_dir.exists() else []
        label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
        
        # Check UUID consistency
        uuid_consistent = self._check_uuid_consistency(image_files, label_files)
        
        return {
            'exists': True, 'images': len(image_files), 'labels': len(label_files), 
            'path': str(split_path), 'uuid_consistent': uuid_consistent
        }
    
    def _check_uuid_consistency(self, image_files: List[Path], label_files: List[Path]) -> bool:
        """Check UUID consistency antara images dan labels"""
        try:
            # Sample check untuk performance (check 10 files atau semua jika < 10)
            sample_size = min(10, len(image_files))
            sample_images = image_files[:sample_size]
            
            uuid_consistent_count = 0
            for img_file in sample_images:
                parsed_img = self.naming_manager.parse_existing_filename(img_file.name)
                if parsed_img:
                    # Check corresponding label
                    label_stem = f"{Path(img_file.name).stem}"
                    corresponding_label = next((lf for lf in label_files if lf.stem == label_stem), None)
                    
                    if corresponding_label:
                        parsed_label = self.naming_manager.parse_existing_filename(corresponding_label.name)
                        if parsed_label and parsed_img['uuid'] == parsed_label['uuid']:
                            uuid_consistent_count += 1
            
            # Consider consistent jika > 80% sample konsisten
            return (uuid_consistent_count / sample_size) > 0.8 if sample_size > 0 else True
            
        except Exception:
            return False
    
    def get_file_stats_optimized(self, file_path: Path) -> Dict[str, Any]:
        """One-liner optimized file stats dengan UUID info"""
        try:
            stat = file_path.stat()
            parsed_filename = self.naming_manager.parse_existing_filename(file_path.name)
            
            return {
                'size_bytes': stat.st_size, 'size_mb': stat.st_size / 1048576, 'exists': True,
                'is_zip': zipfile.is_zipfile(file_path) if file_path.suffix.lower() == '.zip' else False,
                'modified': stat.st_mtime, 'uuid_format': parsed_filename is not None,
                'nominal': parsed_filename['nominal'] if parsed_filename else None
            }
        except Exception:
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False, 'is_zip': False, 'modified': 0, 'uuid_format': False}
    
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
    
    def get_naming_statistics(self) -> Dict[str, Any]:
        """Get statistics dari naming manager"""
        return self.naming_manager.get_nominal_statistics()
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Progress notification dengan safe execution"""
        if self._progress_callback:
            self._progress_callback(step, current, total, message)
    
    def _return_error(self, message: str) -> None:
        """One-liner error return"""
        raise Exception(message)

# One-liner factory dengan optimized defaults
def create_file_processor(logger=None, max_workers: int = None) -> FileProcessor:
    """Factory untuk optimized FileProcessor dengan auto-detected workers dan UUID support"""
    import os
    optimal_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
    return FileProcessor(logger, optimal_workers)