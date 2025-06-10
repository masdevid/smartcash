"""
File: smartcash/dataset/preprocessor/core/engine.py
Deskripsi: Enhanced preprocessing engine dengan dual progress tracker compatibility dan improved normalization
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils.file_processor import FileProcessor
from smartcash.dataset.preprocessor.utils.file_scanner import FileScanner
from smartcash.dataset.preprocessor.utils.path_resolver import PathResolver
from smartcash.dataset.preprocessor.utils.cleanup_manager import CleanupManager
from smartcash.dataset.preprocessor.utils.filename_manager import FilenameManager
from smartcash.dataset.preprocessor.validators import (
    create_image_validator,
    create_label_validator,
    create_pair_validator
)

class PreprocessingValidator:
    """🔍 Enhanced validator dengan dual progress support"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config.get('preprocessing', {})
        self.logger = logger or get_logger()
        
        # Initialize validators dengan enhanced config
        self.image_validator = create_image_validator(self.config.get('validation', {}))
        self.label_validator = create_label_validator(self.config.get('validation', {}))
        self.pair_validator = create_pair_validator(self.config.get('validation', {}))
    
    def validate_split(self, split: str) -> Dict[str, Any]:
        """🎯 Enhanced validation dengan detailed stats"""
        try:
            from smartcash.dataset.preprocessor.utils.path_resolver import PathResolver
            resolver = PathResolver({'data': {'dir': 'data'}})
            
            # Get source directories
            img_dir = Path('data') / split / 'images'
            label_dir = Path('data') / split / 'labels'
            
            if not img_dir.exists() or not label_dir.exists():
                return {
                    'is_valid': False,
                    'message': f"❌ Direktori tidak ditemukan: {img_dir} atau {label_dir}",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # Scan image files
            scanner = FileScanner()
            img_files = scanner.scan_directory(img_dir, {'.jpg', '.jpeg', '.png'})
            
            if not img_files:
                return {
                    'is_valid': False,
                    'message': f"❌ Tidak ada gambar ditemukan di {img_dir}",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # Enhanced validation stats
            stats = {
                'total_images': len(img_files),
                'valid_images': 0,
                'invalid_images': 0,
                'missing_labels': 0,
                'class_distribution': {},
                'avg_image_size': None,
                'validation_errors': []
            }
            
            valid_count = 0
            for img_file in img_files:
                # Validate image
                img_valid, img_errors, img_stats = self.image_validator.validate(img_file)
                
                # Find corresponding label
                label_file = label_dir / f"{img_file.stem}.txt"
                label_valid, label_errors, label_stats = self.label_validator.validate(
                    label_file, (img_stats.get('width', 640), img_stats.get('height', 640))
                )
                
                if img_valid and label_valid:
                    valid_count += 1
                    # Update class distribution
                    for cls_id in label_stats.get('class_distribution', {}):
                        stats['class_distribution'][cls_id] = stats['class_distribution'].get(cls_id, 0) + 1
                else:
                    stats['validation_errors'].extend(img_errors + label_errors)
                    if not label_file.exists():
                        stats['missing_labels'] += 1
            
            stats['valid_images'] = valid_count
            stats['invalid_images'] = len(img_files) - valid_count
            
            success = valid_count == len(img_files)
            return {
                'is_valid': success,
                'message': f"✅ Validasi berhasil: {valid_count}/{len(img_files)} gambar valid" if success else f"⚠️ Validasi partial: {valid_count}/{len(img_files)} gambar valid",
                'summary': stats
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'message': f"❌ Error validasi: {str(e)}",
                'summary': {'total_images': 0, 'valid_images': 0}
            }

class PreprocessingEngine:
    """🚀 Enhanced preprocessing engine dengan dual progress tracker compatibility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.logger = get_logger()
        
        # Initialize components
        self.validator = PreprocessingValidator(config, self.logger)
        self.file_processor = FileProcessor(config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(config)
        self.cleanup_manager = CleanupManager(config)
        
        # Enhanced normalization config (consistent dengan augmentor)
        norm_config = self.preprocessing_config.get('normalization', {})
        self.target_size = tuple(norm_config.get('target_size', [640, 640]))
        self.normalization_method = norm_config.get('method', 'minmax')
        self.preserve_aspect_ratio = norm_config.get('preserve_aspect_ratio', True)
        self.normalize_enabled = norm_config.get('enabled', True)
        
        # Multi-split support
        self.target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(self.target_splits, str):
            self.target_splits = [self.target_splits] if self.target_splits != 'all' else ['train', 'valid', 'test']
        
        # Performance config
        self.batch_size = config.get('performance', {}).get('batch_size', 32)
        self.max_workers = min(4, self.batch_size // 8) if self.batch_size > 8 else 2
        
        # Progress tracking
        self.progress_bridge = None
    
    def register_progress_callback(self, callback: Callable[[str, int, int, str], None]):
        """📊 Register dual progress callback untuk UI integration"""
        self.progress_callback = callback
    
    def preprocess_dataset(self) -> Dict[str, Any]:
        """🎯 Enhanced preprocessing untuk multi-split dengan dual progress tracking"""
        try:
            self._report_progress("overall", 0, 100, "🚀 Memulai preprocessing dataset")
            
            # Phase 1: Validation (0-20%)
            validation_results = self._validate_all_splits()
            if not validation_results['valid']:
                return {
                    'success': False,
                    'message': validation_results['message'],
                    'stats': validation_results.get('stats', {})
                }
            
            self._report_progress("overall", 20, 100, "✅ Validasi selesai, mulai preprocessing")
            
            # Phase 2: Setup directories (20-30%)
            self._setup_output_directories()
            self._report_progress("overall", 30, 100, "📁 Direktori output siap")
            
            # Phase 3: Process each split (30-90%)
            all_stats = {}
            split_progress_step = 60 / len(self.target_splits)
            
            for i, split in enumerate(self.target_splits):
                split_start = 30 + (i * split_progress_step)
                split_end = 30 + ((i + 1) * split_progress_step)
                
                self._report_progress("overall", int(split_start), 100, f"🔄 Processing split: {split}")
                
                split_result = self._process_single_split(
                    split, 
                    lambda current, total, msg: self._report_split_progress(split, current, total, msg, split_start, split_end)
                )
                
                all_stats[split] = split_result.get('stats', {})
                
                if split_result.get('status') != 'success':
                    self.logger.warning(f"⚠️ Split {split} processing had issues: {split_result.get('message', 'Unknown error')}")
            
            # Phase 4: Finalization (90-100%)
            self._report_progress("overall", 90, 100, "🏁 Finalizing preprocessing")
            
            final_stats = self._compile_final_stats(all_stats)
            self._report_progress("overall", 100, 100, "✅ Preprocessing selesai")
            
            return {
                'success': True,
                'message': f"✅ Preprocessing berhasil untuk {len(self.target_splits)} splits",
                'stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"❌ Error preprocessing: {str(e)}"
            self.logger.error(error_msg)
            self._report_progress("overall", 0, 100, error_msg)
            return {'success': False, 'message': error_msg, 'stats': {}}
    
    def _validate_all_splits(self) -> Dict[str, Any]:
        """🔍 Validate semua target splits"""
        all_valid = True
        total_images = 0
        validation_messages = []
        
        for split in self.target_splits:
            result = self.validator.validate_split(split)
            if not result['is_valid']:
                all_valid = False
                validation_messages.append(f"❌ {split}: {result['message']}")
            else:
                total_images += result['summary'].get('total_images', 0)
                validation_messages.append(f"✅ {split}: {result['summary'].get('valid_images', 0)} gambar valid")
        
        return {
            'valid': all_valid,
            'message': '; '.join(validation_messages),
            'stats': {'total_images': total_images, 'splits_validated': len(self.target_splits)}
        }
    
    def _setup_output_directories(self):
        """📁 Setup direktori output untuk semua splits"""
        output_dir = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
        
        for split in self.target_splits:
            for subdir in ['images', 'labels']:
                dir_path = output_dir / split / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Created output directories untuk {len(self.target_splits)} splits")
    
    def _process_single_split(self, split: str, progress_callback: Callable) -> Dict[str, Any]:
        """🔄 Process single split dengan enhanced normalization"""
        try:
            # Get source and destination paths
            src_img_dir = Path('data') / split / 'images'
            src_label_dir = Path('data') / split / 'labels'
            dst_base = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed')) / split
            
            # Scan image files
            img_files = self.file_scanner.scan_directory(src_img_dir, {'.jpg', '.jpeg', '.png'})
            
            if not img_files:
                return {'status': 'skipped', 'message': f'Tidak ada gambar di {split}', 'stats': {}}
            
            # Process with threading
            stats = {'total': len(img_files), 'processed': 0, 'normalized': 0, 'errors': 0}
            
            def process_file(img_file: Path) -> Dict[str, Any]:
                return self._process_single_file(img_file, src_label_dir, dst_base)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {executor.submit(process_file, img_file): img_file for img_file in img_files}
                
                for i, future in enumerate(as_completed(future_to_file)):
                    result = future.result()
                    
                    if result['status'] == 'success':
                        stats['processed'] += 1
                        if result.get('normalized'):
                            stats['normalized'] += 1
                    else:
                        stats['errors'] += 1
                    
                    # Report progress dengan detail
                    progress_pct = ((i + 1) / len(img_files)) * 100
                    progress_callback(i + 1, len(img_files), f"Processed {stats['processed']} files ({progress_pct:.1f}%)")
            
            return {
                'status': 'success',
                'message': f"✅ {split}: {stats['processed']}/{stats['total']} files processed",
                'stats': stats
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"❌ Error processing {split}: {str(e)}", 'stats': {}}
    
    def _process_single_file(self, img_file: Path, label_dir: Path, dst_base: Path) -> Dict[str, Any]:
        """🔧 Process single file dengan updated filename pattern - convert dari raw ke preprocessed"""
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize dengan aspect ratio handling
            if self.preserve_aspect_ratio:
                resized_image = self._resize_with_aspect_ratio(image)
            else:
                resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply normalization (consistent dengan augmentor)
            normalized = False
            if self.normalize_enabled:
                processed_image = self._apply_normalization_method(resized_image)
                normalized = True
            else:
                processed_image = resized_image
            
            # Parse existing filename untuk extract metadata
            filename_manager = FilenameManager()
            parsed = filename_manager.parse_filename(img_file.name)
            
            if parsed and parsed['pattern'] == 'raw':
                # Source sudah dalam format raw: rp_nominal_uuid_increment.ext
                # Convert ke preprocessed: pre_rp_nominal_uuid_increment_variance.ext
                preprocessed_name = filename_manager.generate_preprocessed_filename(
                    img_file.name, 
                    variance=1  # Default variance untuk preprocessed
                )
            else:
                # Fallback untuk file yang belum sesuai format
                self.logger.warning(f"⚠️ File {img_file.name} tidak sesuai format raw, menggunakan fallback naming")
                nominal = self._extract_nominal_from_source(img_file)
                preprocessed_name = f"pre_rp_{nominal}_{filename_manager._generate_uuid()}_{1:03d}_{1:02d}"
            
            # Save processed image
            if normalized:
                # Save sebagai .npy dengan preprocessed pattern
                output_path = dst_base / 'images' / f"{preprocessed_name}.npy"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, processed_image.astype(np.float32))
            else:
                # Save sebagai image dengan preprocessed pattern
                output_path = dst_base / 'images' / f"{preprocessed_name}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                processed_bgr = cv2.cvtColor(processed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), processed_bgr)
            
            # Copy corresponding label dengan matching filename
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                label_output_path = dst_base / 'labels' / f"{preprocessed_name}.txt"
                label_output_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(label_file, label_output_path)
            
            return {
                'status': 'success', 
                'normalized': normalized, 
                'output': str(output_path),
                'source_format': parsed['pattern'] if parsed else 'unknown',
                'preprocessed_name': preprocessed_name
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _extract_nominal_from_source(self, img_file: Path) -> str:
        """💰 Extract nominal dari source filename atau directory structure (fallback only)"""
        # Ini hanya fallback untuk file yang tidak sesuai format
        filename_lower = img_file.name.lower()
        nominal_patterns = [
            (r'1000|rp1000', '001000'),
            (r'2000|rp2000', '002000'), 
            (r'5000|rp5000', '005000'),
            (r'10000|rp10000', '010000'),
            (r'20000|rp20000', '020000'),
            (r'50000|rp50000', '050000'),
            (r'100000|rp100000', '100000')
        ]
        
        for pattern, nominal in nominal_patterns:
            if re.search(pattern, filename_lower):
                return nominal
        
        # Try dari directory path
        path_str = str(img_file.parent).lower()
        for pattern, nominal in nominal_patterns:
            if re.search(pattern, path_str):
                return nominal
        
        # Default fallback
        return '000000'
    
    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """🖼️ Resize dengan preserve aspect ratio (consistent dengan augmentor)"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to target size
        result = np.full((target_h, target_w, 3), 128, dtype=np.uint8)  # Gray padding
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    def _apply_normalization_method(self, image: np.ndarray) -> np.ndarray:
        """🎨 Apply normalization method (consistent dengan augmentor)"""
        normalized = image.astype(np.float32)
        
        if self.normalization_method == 'minmax':
            normalized = normalized / 255.0
        elif self.normalization_method == 'standard':
            mean = normalized.mean()
            std = normalized.std()
            if std > 0:
                normalized = (normalized - mean) / std
        elif self.normalization_method == 'imagenet':
            normalized = normalized / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
        else:  # 'none' or default
            normalized = normalized / 255.0
        
        return normalized
    
    def _compile_final_stats(self, all_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """📊 Compile final statistics"""
        total_images = sum(stats.get('total', 0) for stats in all_stats.values())
        total_processed = sum(stats.get('processed', 0) for stats in all_stats.values())
        total_normalized = sum(stats.get('normalized', 0) for stats in all_stats.values())
        total_errors = sum(stats.get('errors', 0) for stats in all_stats.values())
        
        return {
            'total_images': total_images,
            'total_processed': total_processed,
            'total_normalized': total_normalized,
            'total_errors': total_errors,
            'success_rate': (total_processed / total_images * 100) if total_images > 0 else 0,
            'normalization_rate': (total_normalized / total_processed * 100) if total_processed > 0 else 0,
            'splits': all_stats,
            'target_splits': self.target_splits,
            'configuration': {
                'method': self.normalization_method,
                'target_size': f"{self.target_size[0]}x{self.target_size[1]}",
                'preserve_aspect_ratio': self.preserve_aspect_ratio,
                'normalize_enabled': self.normalize_enabled
            }
        }
    
    def _report_progress(self, level: str, current: int, total: int, message: str):
        """📈 Report progress ke dual tracker"""
        if hasattr(self, 'progress_callback') and callable(self.progress_callback):
            try:
                self.progress_callback(level, current, total, message)
            except Exception:
                pass
        
        # Log milestone progress
        if current % max(1, total // 10) == 0 or current == total:
            self.logger.info(f"🔄 {message} ({current}/{total})")
    
    def _report_split_progress(self, split: str, current: int, total: int, message: str, start_pct: float, end_pct: float):
        """📊 Report progress untuk individual split"""
        split_progress = (current / total) if total > 0 else 0
        overall_progress = int(start_pct + (split_progress * (end_pct - start_pct)))
        
        self._report_progress("current", current, total, f"{split}: {message}")
        self._report_progress("overall", overall_progress, 100, f"Processing {split}: {message}")

def create_preprocessing_engine(config: Dict[str, Any]) -> PreprocessingEngine:
    """🏭 Factory untuk create preprocessing engine"""
    return PreprocessingEngine(config)