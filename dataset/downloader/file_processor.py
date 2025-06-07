"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Simplified file processor menggunakan base components
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from smartcash.dataset.downloader.base import BaseDownloaderComponent, FileHelper
from smartcash.common.utils.file_naming_manager import FileNamingManager


class FileProcessor(BaseDownloaderComponent):
    """Simplified file processor dengan shared components"""
    
    def __init__(self, logger=None, max_workers: int = 4):
        super().__init__(logger)
        self.max_workers = max_workers
        self.naming_manager = FileNamingManager(logger=logger)
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract ZIP dengan validation"""
        try:
            if not zipfile.is_zipfile(zip_path):
                return self._create_error_result('File bukan ZIP valid')
            
            self._notify_progress("extract", 0, 100, "📦 Memulai ekstraksi...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                
                if not file_list:
                    return self._create_error_result('ZIP file kosong')
                
                FileHelper.ensure_directory(extract_to)
                zip_ref.extractall(extract_to)
            
            self._notify_progress("extract", 100, 100, "✅ Ekstraksi selesai")
            return self._create_success_result(
                extracted_files=len(file_list),
                extract_path=str(extract_to)
            )
            
        except Exception as e:
            return self._create_error_result(f'Error ekstraksi: {str(e)}')
    
    def organize_dataset_with_renaming(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset dengan UUID renaming"""
        try:
            if not source_dir.exists():
                return self._create_error_result(f'Source tidak ditemukan: {source_dir}')
            
            self._notify_progress("organize", 0, 100, "🗂️ Memulai organisasi...")
            
            structure = self._detect_structure(source_dir)
            if not structure['valid']:
                return self._create_error_result('Struktur dataset tidak valid')
            
            FileHelper.ensure_directory(target_dir)
            organized_stats = self._organize_splits_with_renaming(
                source_dir, target_dir, structure['splits']
            )
            
            self._copy_additional_files(source_dir, target_dir)
            
            total_images = sum(stats.get('images', 0) for stats in organized_stats.values())
            total_labels = sum(stats.get('labels', 0) for stats in organized_stats.values())
            
            self._notify_progress("organize", 100, 100, f"✅ Organisasi selesai: {total_images} gambar")
            
            return self._create_success_result(
                total_images=total_images,
                total_labels=total_labels,
                splits=organized_stats,
                target_dir=str(target_dir),
                uuid_renamed=True
            )
            
        except Exception as e:
            return self._create_error_result(f'Error organisasi: {str(e)}')
    
    def organize_dataset(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset tanpa renaming"""
        try:
            if not source_dir.exists():
                return self._create_error_result(f'Source tidak ditemukan: {source_dir}')
            
            # Simple copy operation
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(source_dir, target_dir)
            
            # Count files
            stats = self._count_dataset_files(target_dir)
            
            return self._create_success_result(
                total_images=stats['total_images'],
                total_labels=stats['total_labels'],
                splits=stats['splits'],
                target_dir=str(target_dir),
                uuid_renamed=False
            )
            
        except Exception as e:
            return self._create_error_result(f'Error copy: {str(e)}')
    
    def validate_dataset_structure(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate dataset structure"""
        from smartcash.dataset.downloader.base import ValidationHelper
        return ValidationHelper.validate_dataset_structure(dataset_dir)
    
    def _detect_structure(self, source_dir: Path) -> Dict[str, Any]:
        """Detect dataset structure"""
        splits = [split for split in ['train', 'valid', 'test', 'val'] 
                 if (source_dir / split).exists() and (source_dir / split / 'images').exists()]
        
        if not splits and (source_dir / 'images').exists():
            splits.append('train')
        
        return {'valid': bool(splits), 'splits': splits}
    
    def _organize_splits_with_renaming(self, source_dir: Path, target_dir: Path, 
                                     splits: List[str]) -> Dict[str, Any]:
        """Organize splits dengan UUID renaming"""
        organized_stats = {}
        
        for i, split in enumerate(splits):
            normalized_split = 'valid' if split == 'val' else split
            progress = int((i / len(splits)) * 80)
            
            self._notify_progress("organize", progress, 100, f"📁 Processing {split}...")
            
            organized_stats[normalized_split] = self._organize_single_split(
                source_dir, target_dir, split, with_renaming=True
            )
        
        return organized_stats
    
    def _organize_single_split(self, source_dir: Path, target_dir: Path, 
                             split: str, with_renaming: bool = False) -> Dict[str, Any]:
        """Organize single split"""
        try:
            normalized_split = 'valid' if split == 'val' else split
            source_split = source_dir / split
            target_split = target_dir / normalized_split
            
            # Create directories
            for subdir in ['images', 'labels']:
                FileHelper.ensure_directory(target_split / subdir)
            
            # Process files
            if with_renaming:
                images, labels = self._copy_and_rename_files(source_split, target_split)
            else:
                images, labels = self._copy_files_simple(source_split, target_split)
            
            return {
                'status': 'success',
                'images': images,
                'labels': labels,
                'path': str(target_split)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error split {split}: {str(e)}'}
    
    def _copy_and_rename_files(self, source_split: Path, target_split: Path) -> tuple:
        """Copy dan rename files dengan UUID"""
        source_images = source_split / 'images'
        source_labels = source_split / 'labels'
        target_images = target_split / 'images'
        target_labels = target_split / 'labels'
        
        image_files = list(source_images.glob('*.*')) if source_images.exists() else []
        renamed_images = 0
        renamed_labels = 0
        
        for img_file in image_files:
            try:
                # Extract class dari label
                label_path = source_labels / f"{img_file.stem}.txt"
                primary_class = self.naming_manager.extract_primary_class_from_label(label_path)
                
                # Generate UUID filename
                file_info = self.naming_manager.generate_file_info(img_file.name, primary_class)
                new_filename = file_info.get_filename()
                
                # Copy image
                shutil.copy2(img_file, target_images / new_filename)
                renamed_images += 1
                
                # Copy corresponding label
                if label_path.exists():
                    label_filename = f"{Path(new_filename).stem}.txt"
                    shutil.copy2(label_path, target_labels / label_filename)
                    renamed_labels += 1
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error rename {img_file.name}: {str(e)}")
        
        return renamed_images, renamed_labels
    
    def _copy_files_simple(self, source_split: Path, target_split: Path) -> tuple:
        """Simple file copy tanpa renaming"""
        source_images = source_split / 'images'
        source_labels = source_split / 'labels'
        
        images_count = 0
        labels_count = 0
        
        if source_images.exists():
            for img_file in source_images.glob('*.*'):
                shutil.copy2(img_file, target_split / 'images' / img_file.name)
                images_count += 1
        
        if source_labels.exists():
            for label_file in source_labels.glob('*.txt'):
                shutil.copy2(label_file, target_split / 'labels' / label_file.name)
                labels_count += 1
        
        return images_count, labels_count
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy additional files"""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        for filename in additional_files:
            source_file = source_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, target_dir / filename)
    
    def _count_dataset_files(self, dataset_dir: Path) -> Dict[str, Any]:
        """Count files dalam dataset"""
        stats = {'total_images': 0, 'total_labels': 0, 'splits': {}}
        
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                
                stats['splits'][split] = {'images': image_count, 'labels': label_count}
                stats['total_images'] += image_count
                stats['total_labels'] += label_count
        
        return stats
    
    def get_naming_statistics(self) -> Dict[str, Any]:
        """Get naming statistics"""
        return self.naming_manager.get_nominal_statistics()


def create_file_processor(logger=None, max_workers: int = None) -> FileProcessor:
    """Factory untuk FileProcessor"""
    import os
    optimal_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
    return FileProcessor(logger, optimal_workers)