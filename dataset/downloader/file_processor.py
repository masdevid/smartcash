"""
File: smartcash/dataset/downloader/file_processor.py
Deskripsi: Fixed file processor dengan proper imports dan Path handling
"""

import shutil
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from smartcash.dataset.downloader.base import BaseDownloaderComponent, FileHelper


class FileProcessor(BaseDownloaderComponent):
    """File processor dengan config-aware workers dan fixed imports"""
    
    def __init__(self, logger=None, max_workers: int = None):
        super().__init__(logger)
        # Use provided max_workers or get optimal count
        if max_workers is None:
            from smartcash.common.worker_utils import get_optimal_worker_count
            max_workers = get_optimal_worker_count('io')
        
        self.max_workers = max_workers
        self._setup_naming_manager()
        self.logger.info(f"🔧 FileProcessor initialized with {self.max_workers} workers")
    
    def _setup_naming_manager(self):
        """Setup naming manager dengan fallback"""
        try:
            from smartcash.common.utils.file_naming_manager import FileNamingManager
            self.naming_manager = FileNamingManager(logger=self.logger)
        except ImportError:
            self.logger.warning("⚠️ FileNamingManager not available, using basic naming")
            self.naming_manager = None
    
    def organize_dataset_with_renaming(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset dengan UUID renaming menggunakan optimal workers"""
        try:
            if not source_dir.exists():
                return self._create_error_result(f'Source tidak ditemukan: {source_dir}')
            
            self._notify_progress("organize", 0, 100, f"🗂️ Memulai organisasi dengan {self.max_workers} workers...")
            
            structure = self._detect_structure(source_dir)
            if not structure['valid']:
                return self._create_error_result('Struktur dataset tidak valid')
            
            FileHelper.ensure_directory(target_dir)
            organized_stats = self._organize_splits_with_renaming_parallel(
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
                uuid_renamed=bool(self.naming_manager)
            )
            
        except Exception as e:
            return self._create_error_result(f'Error organisasi: {str(e)}')
    
    def organize_dataset(self, source_dir: Path, target_dir: Path) -> Dict[str, Any]:
        """Organize dataset tanpa renaming"""
        try:
            if not source_dir.exists():
                return self._create_error_result(f'Source tidak ditemukan: {source_dir}')
            
            self._notify_progress("organize", 0, 100, "🗂️ Memulai organisasi dataset...")
            
            structure = self._detect_structure(source_dir)
            if not structure['valid']:
                return self._create_error_result('Struktur dataset tidak valid')
            
            FileHelper.ensure_directory(target_dir)
            organized_stats = self._organize_splits_without_renaming(
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
                uuid_renamed=False
            )
            
        except Exception as e:
            return self._create_error_result(f'Error organisasi: {str(e)}')
    
    def _detect_structure(self, source_dir: Path) -> Dict[str, Any]:
        """Detect dataset structure"""
        structure = {'valid': False, 'splits': [], 'type': 'unknown'}
        
        # Check for YOLOv5 structure
        possible_splits = ['train', 'valid', 'test', 'val']
        found_splits = []
        
        for split in possible_splits:
            split_dir = source_dir / split
            if split_dir.exists():
                images_dir = split_dir / 'images'
                if images_dir.exists() and any(images_dir.iterdir()):
                    # Normalize 'val' to 'valid'
                    normalized_split = 'valid' if split == 'val' else split
                    found_splits.append(normalized_split)
        
        if found_splits:
            structure.update({
                'valid': True,
                'splits': found_splits,
                'type': 'yolov5'
            })
        
        return structure
    
    def _organize_splits_with_renaming_parallel(self, source_dir: Path, target_dir: Path, 
                                              splits: List[str]) -> Dict[str, Any]:
        """Organize splits dengan parallel processing dan UUID renaming"""
        organized_stats = {}
        
        # Use ThreadPoolExecutor dengan optimal workers
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(splits))) as executor:
            future_to_split = {
                executor.submit(self._organize_single_split, source_dir, target_dir, split, True): split 
                for split in splits
            }
            
            for i, future in enumerate(future_to_split):
                split = future_to_split[future]
                progress = int((i / len(splits)) * 80)
                
                self._notify_progress("organize", progress, 100, f"📁 Processing {split}...")
                
                try:
                    organized_stats[split] = future.result()
                except Exception as e:
                    self.logger.error(f"❌ Error processing {split}: {str(e)}")
                    organized_stats[split] = {'status': 'error', 'message': str(e)}
        
        return organized_stats
    
    def _organize_splits_without_renaming(self, source_dir: Path, target_dir: Path, 
                                        splits: List[str]) -> Dict[str, Any]:
        """Organize splits tanpa UUID renaming"""
        organized_stats = {}
        
        for i, split in enumerate(splits):
            progress = int((i / len(splits)) * 80)
            self._notify_progress("organize", progress, 100, f"📁 Processing {split}...")
            
            try:
                organized_stats[split] = self._organize_single_split(source_dir, target_dir, split, False)
            except Exception as e:
                self.logger.error(f"❌ Error processing {split}: {str(e)}")
                organized_stats[split] = {'status': 'error', 'message': str(e)}
        
        return organized_stats
    
    def _organize_single_split(self, source_dir: Path, target_dir: Path, split: str, use_renaming: bool) -> Dict[str, Any]:
        """Organize single split dengan atau tanpa renaming"""
        # Handle val -> valid mapping
        source_split_name = 'val' if split == 'valid' and (source_dir / 'val').exists() else split
        source_split = source_dir / source_split_name
        target_split = target_dir / split
        
        if not source_split.exists():
            return {'status': 'not_found', 'images': 0, 'labels': 0}
        
        # Create target directories
        target_images = target_split / 'images'
        target_labels = target_split / 'labels'
        FileHelper.ensure_directory(target_images)
        FileHelper.ensure_directory(target_labels)
        
        if use_renaming and self.naming_manager:
            return self._copy_and_rename_files_batch(source_split, target_split)
        else:
            return self._copy_files_without_renaming(source_split, target_split)
    
    def _copy_files_without_renaming(self, source_split: Path, target_split: Path) -> Dict[str, int]:
        """Copy files tanpa renaming"""
        source_images = source_split / 'images'
        source_labels = source_split / 'labels'
        target_images = target_split / 'images'
        target_labels = target_split / 'labels'
        
        if not source_images.exists():
            return {'images': 0, 'labels': 0}
        
        copied_images = 0
        copied_labels = 0
        
        # Copy images
        for img_file in source_images.glob('*.*'):
            if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                try:
                    shutil.copy2(img_file, target_images / img_file.name)
                    copied_images += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Error copying image {img_file.name}: {str(e)}")
        
        # Copy labels
        if source_labels.exists():
            for label_file in source_labels.glob('*.txt'):
                try:
                    shutil.copy2(label_file, target_labels / label_file.name)
                    copied_labels += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Error copying label {label_file.name}: {str(e)}")
        
        return {'images': copied_images, 'labels': copied_labels}
    
    def _copy_and_rename_files_batch(self, source_split: Path, target_split: Path) -> Dict[str, int]:
        """Copy dan rename files dengan batch processing untuk performance"""
        if not self.naming_manager:
            return self._copy_files_without_renaming(source_split, target_split)
        
        source_images = source_split / 'images'
        source_labels = source_split / 'labels'
        target_images = target_split / 'images'
        target_labels = target_split / 'labels'
        
        if not source_images.exists():
            return {'images': 0, 'labels': 0}
        
        image_files = [f for f in source_images.glob('*.*') 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        renamed_images = 0
        renamed_labels = 0
        
        # Process in batches untuk memory efficiency
        batch_size = 100
        batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_file_batch, batch, source_labels, target_images, target_labels)
                for batch in batches
            ]
            
            for future in futures:
                try:
                    batch_images, batch_labels = future.result()
                    renamed_images += batch_images
                    renamed_labels += batch_labels
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing batch: {str(e)}")
        
        return {'images': renamed_images, 'labels': renamed_labels}
    
    def _process_file_batch(self, image_batch: List[Path], source_labels: Path, 
                           target_images: Path, target_labels: Path) -> tuple:
        """Process batch of files dengan UUID renaming"""
        batch_images = 0
        batch_labels = 0
        
        for img_file in image_batch:
            try:
                # Extract class dari label jika naming manager tersedia
                if self.naming_manager:
                    label_path = source_labels / f"{img_file.stem}.txt"
                    primary_class = self.naming_manager.extract_primary_class_from_label(label_path)
                    
                    # Generate UUID filename
                    file_info = self.naming_manager.generate_file_info(img_file.name, primary_class)
                    new_filename = file_info.get_filename()
                    
                    # Skip if already in UUID format
                    if self.naming_manager.parse_existing_filename(img_file.name):
                        new_filename = img_file.name
                else:
                    # Fallback tanpa renaming
                    new_filename = img_file.name
                
                # Copy image
                shutil.copy2(img_file, target_images / new_filename)
                batch_images += 1
                
                # Copy corresponding label
                label_path = source_labels / f"{img_file.stem}.txt"
                if label_path.exists():
                    label_filename = f"{Path(new_filename).stem}.txt"
                    shutil.copy2(label_path, target_labels / label_filename)
                    batch_labels += 1
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error processing {img_file.name}: {str(e)}")
        
        return batch_images, batch_labels
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path):
        """Copy additional files seperti dataset.yaml"""
        additional_files = ['dataset.yaml', 'data.yaml', 'README.md', 'README.txt']
        
        for filename in additional_files:
            source_file = source_dir / filename
            if source_file.exists():
                try:
                    shutil.copy2(source_file, target_dir / filename)
                    self.logger.info(f"📄 Copied {filename}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error copying {filename}: {str(e)}")


def create_file_processor(logger=None, max_workers: int = None) -> FileProcessor:
    """Factory dengan optimal workers support"""
    return FileProcessor(logger, max_workers)