"""
File: smartcash/dataset/preprocessor/core/validator.py
Deskripsi: Simplified validator wrapper menggunakan consolidated ValidationCore
"""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils import (
    ValidationCore, PathManager, FileOperations,
    create_validation_core, create_path_manager, create_file_operations
)

class ValidationEngine:
    """🔍 Simplified validation engine menggunakan consolidated utils"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Initialize consolidated components
        self.validation_core = create_validation_core(config.get('validation', {}))
        self.path_manager = create_path_manager(config)
        self.file_ops = create_file_operations(config)
        
        # Configuration
        self.preprocessing_config = config.get('preprocessing', {})
        self.validation_config = self.preprocessing_config.get('validation', {})
    
    def validate_split(self, split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Validate split menggunakan consolidated validation"""
        try:
            # Check jika validation disabled
            if not self.validation_config.get('enabled', True):
                if progress_callback:
                    progress_callback("current", 100, 100, "Validation disabled")
                return {
                    'is_valid': True,
                    'message': "✅ Validation disabled in config",
                    'stats': {'total_images': 0, 'valid_images': 0}
                }
            
            # Get source paths
            img_dir, label_dir = self.path_manager.get_source_paths(split)
            
            if not img_dir.exists() or not label_dir.exists():
                if progress_callback:
                    progress_callback("current", 0, 100, f"Directories not found for {split}")
                return {
                    'is_valid': False,
                    'message': f"❌ Directories not found: {img_dir} or {label_dir}",
                    'stats': {'total_images': 0, 'valid_images': 0}
                }
            
            # Scan image files
            if progress_callback:
                progress_callback("current", 10, 100, f"Scanning {split} images...")
            
            image_files = self.file_ops.scan_images(img_dir)
            
            if not image_files:
                if progress_callback:
                    progress_callback("current", 0, 100, f"No images in {split}")
                return {
                    'is_valid': False,
                    'message': f"❌ No images found in {img_dir}",
                    'stats': {'total_images': 0, 'valid_images': 0}
                }
            
            # Batch validation dengan progress
            if progress_callback:
                progress_callback("current", 20, 100, f"Validating {len(image_files)} files...")
            
            def validation_progress(level, current, total, message):
                if progress_callback:
                    progress = 20 + int((current / total) * 70)  # 20-90%
                    progress_callback("current", progress, 100, message)
            
            validation_results = self.validation_core.batch_validate_pairs(
                image_files, validation_progress
            )
            
            # Compile statistics
            total_files = len(validation_results)
            valid_files = sum(1 for result in validation_results.values() if result.is_valid)
            
            # Enhanced stats
            stats = {
                'total_images': total_files,
                'valid_images': valid_files,
                'invalid_images': total_files - valid_files,
                'validation_rate': f"{(valid_files / total_files) * 100:.1f}%" if total_files > 0 else "0%",
                'class_distribution': self._extract_class_distribution(validation_results),
                'common_errors': self._extract_common_errors(validation_results)
            }
            
            if progress_callback:
                progress_callback("current", 100, 100, f"Validation complete for {split}")
            
            success = valid_files == total_files
            return {
                'is_valid': success,
                'message': f"✅ Validation passed: {valid_files}/{total_files} valid" if success else f"⚠️ Validation partial: {valid_files}/{total_files} valid",
                'stats': stats
            }
            
        except Exception as e:
            if progress_callback:
                progress_callback("current", 0, 100, f"Validation error: {str(e)}")
            return {
                'is_valid': False,
                'message': f"❌ Validation error: {str(e)}",
                'stats': {'total_images': 0, 'valid_images': 0}
            }
    
    def _extract_class_distribution(self, validation_results: Dict) -> Dict[int, int]:
        """📊 Extract class distribution dari validation results"""
        distribution = {}
        
        for result in validation_results.values():
            if result.is_valid and 'label_stats' in result.stats:
                label_stats = result.stats['label_stats']
                if 'class_distribution' in label_stats:
                    for class_id, count in label_stats['class_distribution'].items():
                        distribution[class_id] = distribution.get(class_id, 0) + count
        
        return distribution
    
    def _extract_common_errors(self, validation_results: Dict) -> List[str]:
        """📋 Extract common errors dari validation results"""
        error_counts = {}
        
        for result in validation_results.values():
            if result.errors:
                for error in result.errors:
                    # Simplify error message
                    simplified = self._simplify_error_message(error)
                    error_counts[simplified] = error_counts.get(simplified, 0) + 1
        
        # Return top 5 most common errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{error} ({count}x)" for error, count in sorted_errors[:5]]
    
    def _simplify_error_message(self, error: str) -> str:
        """🔧 Simplify error message untuk grouping"""
        # Extract error type
        if "tidak ditemukan" in error.lower():
            return "File tidak ditemukan"
        elif "corrupt" in error.lower() or "tidak dapat dibaca" in error.lower():
            return "File corrupt"
        elif "terlalu kecil" in error.lower():
            return "Ukuran terlalu kecil"
        elif "terlalu besar" in error.lower():
            return "Ukuran terlalu besar"
        elif "koordinat" in error.lower():
            return "Koordinat tidak valid"
        elif "format" in error.lower():
            return "Format tidak valid"
        else:
            return "Error lainnya"

def create_validation_engine(config: Dict[str, Any]) -> ValidationEngine:
    """🏭 Factory untuk create ValidationEngine"""
    return ValidationEngine(config)