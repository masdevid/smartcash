"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Backend integration utilities untuk preprocessing operations
"""

from typing import Dict, Any, Tuple

def validate_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """Validate apakah dataset ready untuk preprocessing"""
    try:
        from smartcash.dataset.utils.path_validator import get_path_validator
        from pathlib import Path
        
        data_dir = config.get('data', {}).get('dir', 'data')
        if not data_dir or not isinstance(data_dir, str):
            return False, "Path dataset tidak valid"
        
        data_path = Path(data_dir)
        if not data_path.exists():
            return False, f"Directory dataset tidak ditemukan: {data_dir}"
        
        # Check minimal structure
        required_splits = ['train']
        missing_splits = []
        
        for split in required_splits:
            split_path = data_path / split
            if not split_path.exists():
                missing_splits.append(split)
                continue
                
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                missing_splits.append(f"{split} (images/labels)")
        
        if missing_splits:
            return False, f"Struktur dataset tidak lengkap: {', '.join(missing_splits)}"
        
        # Use validator untuk detail analysis
        validator = get_path_validator(logger)
        result = validator.validate_dataset_structure(data_dir)
        
        if not result.get('valid', False):
            issues = result.get('issues', ['Unknown error'])
            return False, f"Dataset tidak valid: {issues[0] if issues else 'No images found'}"
        
        total_images = result.get('total_images', 0)
        if total_images == 0:
            return False, "Dataset tidak memiliki gambar"
            
        return True, f"Dataset siap: {total_images:,} gambar"
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error validasi dataset: {str(e)}")
        return False, f"Error validasi dataset: {str(e)[:100]}..."

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int]:
    """Check apakah preprocessed data sudah ada"""
    try:
        from pathlib import Path
        
        preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        if not preprocessed_dir.exists():
            return False, 0
        
        total_files = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for split in ['train', 'valid', 'test']:
            split_images_dir = preprocessed_dir / split / 'images'
            if split_images_dir.exists():
                split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
                total_files += len(split_files)
        
        return total_files > 0, total_files
        
    except Exception:
        return False, 0

def create_backend_preprocessor(ui_config: Dict[str, Any], logger=None):
    """Create preprocessor instance dari UI config"""
    try:
        from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
        
        # Convert UI config ke backend format
        backend_config = _convert_ui_to_backend_config(ui_config)
        
        return PreprocessingManager(backend_config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating backend preprocessor: {str(e)}")
        return None

def create_backend_checker(logger=None):
    """Create dataset checker instance"""
    try:
        from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
        return DatasetChecker(logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating dataset checker: {str(e)}")
        return None

def create_backend_cleanup_service(config: Dict[str, Any], logger=None):
    """Create cleanup service instance"""
    try:
        from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
        return CleanupExecutor(config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating cleanup service: {str(e)}")
        return None

def _convert_ui_to_backend_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI config ke backend service format"""
    preprocessing = ui_config.get('preprocessing', {})
    performance = ui_config.get('performance', {})
    
    return {
        'data': ui_config.get('data', {}),
        'preprocessing': preprocessing,
        'performance': performance,
        'cleanup': ui_config.get('cleanup', {}),
        
        # Derived settings
        'img_size': preprocessing.get('normalization', {}).get('target_size', [640, 640]),
        'normalize': preprocessing.get('normalization', {}).get('enabled', True),
        'num_workers': performance.get('num_workers', 8),
        'split': preprocessing.get('target_split', 'all'),
        'force_reprocess': preprocessing.get('force_reprocess', False)
    }