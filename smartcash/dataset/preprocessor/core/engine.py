"""
File: smartcash/dataset/preprocessor/core/engine.py
Deskripsi: Enhanced preprocessing engine dengan proper file format output (.npy dan .txt)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import re
import time

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
    """🔍 Enhanced validator dengan progress callback support"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config.get('preprocessing', {})
        self.logger = logger or get_logger()
        self.data_config = config.get('data', {})
        
        # Initialize validators dengan enhanced config
        self.image_validator = create_image_validator(self.config.get('validation', {}))
        self.label_validator = create_label_validator(self.config.get('validation', {}))
        self.pair_validator = create_pair_validator(self.config.get('validation', {}))
    
    def validate_split(self, split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Enhanced validation dengan progress callback"""
        try:
            # Periksa apakah validasi dinonaktifkan
            validation_config = self.config.get('validation', {})
            if not validation_config.get('enabled', True):
                if progress_callback:
                    progress_callback("current", 100, 100, "Validasi dinonaktifkan")
                return {
                    'is_valid': True,
                    'message': "✅ Validasi dinonaktifkan dalam konfigurasi",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # Get source directories dengan proper path resolution
            local_paths = self.data_config.get('local', {})
            if split in local_paths:
                base_path = Path(local_paths[split])
            else:
                base_dir = self.data_config.get('dir', 'data')
                base_path = Path(base_dir) / split
            
            img_dir = base_path / 'images'
            label_dir = base_path / 'labels'
            
            if not img_dir.exists() or not label_dir.exists():
                if progress_callback:
                    progress_callback("current", 0, 100, f"Direktori {split} tidak ditemukan")
                return {
                    'is_valid': False,
                    'message': f"❌ Direktori tidak ditemukan: {img_dir} atau {label_dir}",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # Scan image files dengan progress
            if progress_callback:
                progress_callback("current", 10, 100, f"Scanning {split} images...")
            
            scanner = FileScanner()
            img_files = scanner.scan_directory(img_dir, {'.jpg', '.jpeg', '.png'})
            
            if not img_files:
                if progress_callback:
                    progress_callback("current", 0, 100, f"Tidak ada gambar di {split}")
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
            
            # Process files dengan progress updates (NO THREADING)
            total_files = len(img_files)
            valid_count = 0
            
            # Process files dengan progress updates
            for i, img_file in enumerate(img_files):
                # Update progress
                progress = 10 + int((i / total_files) * 80)  # 10-90%
                if progress_callback:
                    progress_callback("current", progress, 100, f"Validating {img_file.name}")
                
                # Validate image
                img_valid = True
                img_errors = []
                img_stats = {}
                
                # Hanya jalankan validasi gambar jika diaktifkan
                if validation_config.get('check_image_quality', True):
                    img_valid, img_errors, img_stats = self.image_validator.validate(img_file)
                
                # Find corresponding label
                label_file = label_dir / f"{img_file.stem}.txt"
                label_valid = True
                label_errors = []
                label_stats = {}
                
                # Hanya jalankan validasi label jika diaktifkan
                if validation_config.get('check_labels', True) or validation_config.get('check_coordinates', True):
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
            
            # Completion
            if progress_callback:
                progress_callback("current", 100, 100, f"Validation {split} completed")
            
            stats['valid_images'] = valid_count
            stats['invalid_images'] = len(img_files) - valid_count
            
            # Jika validasi dinonaktifkan, anggap semua gambar valid
            success = valid_count == len(img_files) if validation_config.get('enabled', True) else True
            return {
                'is_valid': success,
                'message': f"✅ Validasi berhasil: {valid_count}/{len(img_files)} gambar valid" if success else f"⚠️ Validasi partial: {valid_count}/{len(img_files)} gambar valid",
                'summary': stats
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback("current", 0, 100, f"Error validasi: {str(e)}")
            return {
                'is_valid': False,
                'message': f"❌ Error validasi: {str(e)}",
                'summary': {'total_images': 0, 'valid_images': 0}
            }

class PreprocessingEngine:
    """🚀 Enhanced preprocessing engine dengan proper file format output"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.data_config = config.get('data', {})
        self.file_naming_config = config.get('file_naming', {})
        self.logger = get_logger()
        
        # Initialize components
        self.validator = PreprocessingValidator(config, self.logger)
        self.file_processor = FileProcessor(config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(config)
        self.cleanup_manager = CleanupManager(config)
        self.filename_manager = FilenameManager(self.file_naming_config)
        
        # Enhanced normalization config
        norm_config = self.preprocessing_config.get('normalization', {})
        self.target_size = tuple(norm_config.get('target_size', [640, 640]))
        self.normalization_method = norm_config.get('method', 'minmax')
        self.preserve_aspect_ratio = norm_config.get('preserve_aspect_ratio', True)
        self.normalize_enabled = norm_config.get('enabled', True)
        
        # Multi-split support
        self.target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(self.target_splits, str):
            self.target_splits = [self.target_splits] if self.target_splits != 'all' else ['train', 'valid', 'test']
        
        # Performance config - NO THREADING
        self.batch_size = config.get('performance', {}).get('batch_size', 32)
        
        # Progress tracking
        self.progress_callback = None
    
    def register_progress_callback(self, callback: Callable[[str, int, int, str], None]):
        """📊 Register progress callback untuk UI integration"""
        self.progress_callback = callback
    
    def preprocess_dataset(self, progress_callback: Optional[Callable[[str, int, int, str], None]] = None) -> Dict[str, Any]:
        """🎯 Main preprocessing dengan proper .npy dan .txt output"""
        # Use provided callback atau yang sudah terdaftar
        current_callback = progress_callback or self.progress_callback
        
        try:
            if current_callback:
                current_callback("overall", 0, 100, "🚀 Memulai preprocessing dataset")
            
            # Phase 1: Validation (0-20%)
            validation_results = self._validate_all_splits_with_progress(current_callback)
            if not validation_results['valid']:
                return {
                    'success': False,
                    'message': validation_results['message'],
                    'stats': validation_results.get('stats', {})
                }
            
            self._report_progress("overall", 20, 100, "✅ Validasi selesai, mulai preprocessing", current_callback)
            
            # Phase 2: Setup directories (20-30%)
            self._setup_output_directories()
            self._report_progress("overall", 30, 100, "📁 Direktori output siap", current_callback)
            
            # Phase 3: Process each split (30-90%) - TANPA THREADING
            all_stats = {}
            split_progress_step = 60 / len(self.target_splits)
            
            for i, split in enumerate(self.target_splits):
                split_start = 30 + (i * split_progress_step)
                split_end = 30 + ((i + 1) * split_progress_step)
                
                self._report_progress("overall", int(split_start), 100, f"🔄 Processing split: {split}", current_callback)
                
                split_result = self._process_single_split_sequential(
                    split, 
                    lambda current, total, msg: self._report_split_progress(split, current, total, msg, split_start, split_end, current_callback)
                )
                
                all_stats[split] = split_result.get('stats', {})
                
                if split_result.get('status') != 'success':
                    self.logger.warning(f"⚠️ Split {split} processing had issues: {split_result.get('message', 'Unknown error')}")
            
            # Phase 4: Finalization (90-100%)
            self._report_progress("overall", 90, 100, "🏁 Finalizing preprocessing", current_callback)
            
            final_stats = self._compile_final_stats(all_stats)
            self._report_progress("overall", 100, 100, "✅ Preprocessing selesai", current_callback)
            
            return {
                'success': True,
                'message': f"✅ Preprocessing berhasil untuk {len(self.target_splits)} splits",
                'stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"❌ Error preprocessing: {str(e)}"
            self.logger.error(error_msg)
            self._report_progress("overall", 0, 100, error_msg, current_callback)
            return {'success': False, 'message': error_msg, 'stats': {}}
    
    def _process_single_split_sequential(self, split: str, progress_callback: Callable) -> Dict[str, Any]:
        """🔄 Process single split dengan proper file format output"""
        try:
            # Get source and destination paths dari config
            local_paths = self.data_config.get('local', {})
            if split in local_paths:
                src_base = Path(local_paths[split])
            else:
                base_dir = self.data_config.get('dir', 'data')
                src_base = Path(base_dir) / split
            
            src_img_dir = src_base / 'images'
            src_label_dir = src_base / 'labels'
            dst_base = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed')) / split
            
            # Scan image files
            img_files = self.file_scanner.scan_directory(src_img_dir, {'.jpg', '.jpeg', '.png'})
            
            if not img_files:
                return {'status': 'skipped', 'message': f'Tidak ada gambar di {split}', 'stats': {}}
            
            # Process SEQUENTIAL - NO THREADING
            stats = {'total': len(img_files), 'processed': 0, 'normalized': 0, 'errors': 0}
            
            # Process files satu per satu dengan progress updates
            for i, img_file in enumerate(img_files):
                try:
                    # Update progress
                    progress_pct = ((i + 1) / len(img_files)) * 100
                    progress_callback(i + 1, len(img_files), f"Processing {img_file.name} ({progress_pct:.1f}%)")
                    
                    # Process single file dengan proper output format
                    result = self._process_single_file_enhanced(img_file, src_label_dir, dst_base)
                    
                    if result['status'] == 'success':
                        stats['processed'] += 1
                        if result.get('normalized'):
                            stats['normalized'] += 1
                    else:
                        stats['errors'] += 1
                        self.logger.warning(f"⚠️ Error processing {img_file.name}: {result.get('error', 'Unknown')}")
                    
                    # Small delay untuk UI responsiveness
                    time.sleep(0.001)  # 1ms delay
                    
                except Exception as e:
                    stats['errors'] += 1
                    self.logger.error(f"❌ Error processing {img_file.name}: {str(e)}")
            
            return {
                'status': 'success',
                'message': f"✅ {split}: {stats['processed']}/{stats['total']} files processed",
                'stats': stats
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"❌ Error processing {split}: {str(e)}", 'stats': {}}
    
    def _process_single_file_enhanced(self, img_file: Path, label_dir: Path, dst_base: Path) -> Dict[str, Any]:
        """🔧 Process single file dengan enhanced output format (.npy untuk images, .txt untuk labels)"""
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
                processed_image = resized_image.astype(np.float32) / 255.0  # Basic normalization
                normalized = True
            
            # Generate preprocessed filename
            preprocessed_name = self._generate_preprocessed_filename(img_file)
            
            # 🔑 KEY: Save sebagai .npy file (normalized arrays)
            output_img_path = dst_base / 'images' / f"{preprocessed_name}.npy"
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save normalized image sebagai .npy
            np.save(output_img_path, processed_image.astype(np.float32))
            
            # 🔑 KEY: Copy corresponding label sebagai .txt dengan matching filename
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                output_label_path = dst_base / 'labels' / f"{preprocessed_name}.txt"
                output_label_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy label file
                import shutil
                shutil.copy2(label_file, output_label_path)
            
            return {
                'status': 'success', 
                'normalized': normalized, 
                'output_image': str(output_img_path),
                'output_label': str(output_label_path) if label_file.exists() else None,
                'preprocessed_name': preprocessed_name,
                'format': 'npy'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_preprocessed_filename(self, img_file: Path) -> str:
        """🔧 Generate preprocessed filename dengan proper pattern"""
        try:
            # Parse existing filename untuk extract metadata
            parsed = self.filename_manager.parse_filename(img_file.name)
            
            if parsed and parsed['pattern'] in ['raw', 'preprocessed']:
                # Source sudah dalam format yang benar
                nominal = parsed.get('nominal', '000000')
                file_uuid = parsed.get('uuid', self._generate_uuid())
                sequence = parsed.get('increment', 1)
                variance = 1  # Default variance untuk preprocessed
                
                return f"pre_rp_{nominal}_{file_uuid}_{sequence:03d}_{variance:02d}"
            else:
                # Fallback untuk file yang belum sesuai format
                nominal = self._extract_nominal_from_source(img_file)
                file_uuid = self._generate_uuid()
                sequence = 1
                variance = 1
                
                return f"pre_rp_{nominal}_{file_uuid}_{sequence:03d}_{variance:02d}"
                
        except Exception:
            # Emergency fallback
            import uuid
            return f"pre_rp_000000_{str(uuid.uuid4())}_{1:03d}_{1:02d}"
    
    def _generate_uuid(self) -> str:
        """🆔 Generate UUID untuk filename"""
        import uuid
        return str(uuid.uuid4())
    
    def _extract_nominal_from_source(self, img_file: Path) -> str:
        """💰 Extract nominal dari source filename atau directory structure"""
        # Try dari filename
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
        
        import re
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
        """🖼️ Resize dengan preserve aspect ratio"""
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
        """🎨 Apply normalization method"""
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
    
    def _validate_all_splits_with_progress(self, callback: Optional[Callable]) -> Dict[str, Any]:
        """🔍 Validate semua target splits dengan progress"""
        all_valid = True
        total_images = 0
        validation_messages = []
        
        for i, split in enumerate(self.target_splits):
            # Update progress untuk validation phase
            progress = int((i / len(self.target_splits)) * 20)  # 0-20%
            self._report_progress("overall", progress, 100, f"Validating {split}...", callback)
            
            # Validation dengan progress callback
            def validation_progress(level, current, total, message):
                # Map validation progress ke current level
                if callback:
                    callback("current", current, total, f"{split}: {message}")
            
            result = self.validator.validate_split(split, validation_progress)
            
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
        
        # Pastikan output directory root ada
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buat direktori untuk setiap split
        for split in self.target_splits:
            split_dir = output_dir / split
            for subdir in ['images', 'labels']:
                dir_path = split_dir / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 Created output directories untuk {len(self.target_splits)} splits di {output_dir}")
    
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
                'output_formats': {
                    'images': '.npy (normalized arrays)',
                    'labels': '.txt (YOLO format)'
                }
            }
        }
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """📈 Report progress dengan proper logging"""
        # Log milestone progress (every 10% or completion)
        if current % max(1, total // 10) == 0 or current == total:
            self.logger.info(f"🔄 {message} ({current}/{total})")
        
        # Call progress callback if available
        if callback:
            try:
                callback(level, current, total, message)
            except Exception as e:
                self.logger.warning(f"⚠️ Progress callback error: {str(e)}")