"""
File: smartcash/dataset/preprocessor/api/preprocessing_api.py
Deskripsi: Main preprocessing API dengan clean interface
"""

from typing import Dict, Any, List, Union, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from ..service import PreprocessingService
from ..config.validator import validate_preprocessing_config
from ..config.defaults import get_default_config

def preprocess_dataset(config: Dict[str, Any] = None,
                      progress_callback: Optional[Callable] = None,
                      ui_components: Dict[str, Any] = None,
                      splits: List[str] = None) -> Dict[str, Any]:
    """🚀 Main preprocessing function dengan YOLO normalization
    
    Args:
        config: Preprocessing configuration
        progress_callback: Callback dengan signature (level, current, total, message)
        ui_components: UI components untuk progress integration
        splits: Override target splits dari config
        
    Returns:
        Dict dengan processing results
    """
    try:
        logger = get_logger(__name__)
        
        # Validate dan merge config
        base_config = get_default_config()
        if config:
            merged_config = validate_preprocessing_config(config)
        else:
            merged_config = base_config
        
        # Override splits jika provided
        if splits:
            merged_config['preprocessing']['target_splits'] = splits
        
        # Create service dengan UI integration
        from ..utils.progress_bridge import create_preprocessing_bridge
        
        # Setup progress bridge dengan UI components
        if ui_components and progress_callback:
            bridge = create_preprocessing_bridge(ui_components)
            bridge.register_callback(progress_callback)
            service = PreprocessingService(merged_config)
            service.progress_bridge = bridge
        else:
            service = PreprocessingService(merged_config, progress_callback)
        
        result = service.preprocess_dataset()
        
        # Enhance result dengan config info
        if result['success']:
            result['configuration'] = {
                'target_splits': merged_config['preprocessing']['target_splits'],
                'normalization_preset': 'custom' if config else 'default',
                'output_format': 'npy + txt labels'
            }
        
        return result
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Preprocessing API error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Preprocessing error: {str(e)}",
            'stats': {}
        }

def get_preprocessing_status(config: Dict[str, Any] = None,
                           ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 Get preprocessing status dan readiness check"""
    try:
        # Validate config
        if config:
            merged_config = validate_preprocessing_config(config)
        else:
            merged_config = get_default_config()
        
        service = PreprocessingService(merged_config)
        return service.get_preprocessing_status()
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Status check error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Status error: {str(e)}",
            'service_ready': False
        }

def validate_dataset_structure(data_dir: Union[str, Path],
                             splits: List[str] = None,
                             auto_fix: bool = True) -> Dict[str, Any]:
    """✅ Validate dataset structure dengan auto-fix
    
    Args:
        data_dir: Base data directory
        splits: Target splits untuk validation
        auto_fix: Auto-create missing directories
        
    Returns:
        Dict dengan validation results
    """
    try:
        from ..validation.directory_validator import DirectoryValidator
        
        validator = DirectoryValidator({'auto_fix': auto_fix})
        result = validator.validate_structure(data_dir, splits)
        
        return {
            'success': result['is_valid'],
            'message': f"✅ Structure valid" if result['is_valid'] else f"❌ Structure issues found",
            'details': result
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Structure validation error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Validation error: {str(e)}"
        }

def validate_filenames(data_dir: Union[str, Path],
                      splits: List[str] = None,
                      auto_rename: bool = False) -> Dict[str, Any]:
    """📝 Validate dan optionally rename files ke research format
    
    Args:
        data_dir: Base data directory
        splits: Target splits
        auto_rename: Auto-rename invalid filenames
        
    Returns:
        Dict dengan validation/rename results
    """
    try:
        from ..validation.filename_validator import FilenameValidator
        from ..core.file_processor import FileProcessor
        
        validator = FilenameValidator()
        fp = FileProcessor()
        
        splits = splits or ['train', 'valid', 'test']
        total_processed = 0
        total_renamed = 0
        results_by_split = {}
        
        for split in splits:
            split_path = Path(data_dir) / split / 'images'
            if not split_path.exists():
                continue
            
            # Scan image files
            image_files = fp.scan_files(split_path)
            
            if auto_rename:
                rename_result = validator.rename_invalid_files(image_files)
                results_by_split[split] = rename_result['stats']
                total_renamed += rename_result['stats']['renamed']
            else:
                # Just validate
                validations = validator.batch_validate([f.name for f in image_files])
                invalid_count = sum(1 for v in validations.values() if v['needs_rename'])
                results_by_split[split] = {
                    'total': len(image_files),
                    'invalid': invalid_count,
                    'valid': len(image_files) - invalid_count
                }
            
            total_processed += len(image_files)
        
        action = "renamed" if auto_rename else "validated"
        message = f"✅ {action.capitalize()} {total_processed} files"
        if auto_rename and total_renamed > 0:
            message += f", {total_renamed} files renamed"
        
        return {
            'success': True,
            'message': message,
            'action': action,
            'total_processed': total_processed,
            'total_renamed': total_renamed if auto_rename else None,
            'by_split': results_by_split
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Filename validation error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Validation error: {str(e)}"
        }

def get_preprocessing_preview(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """👀 Preview preprocessing tanpa execute
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        Dict dengan preview information
    """
    try:
        # Validate config
        if config:
            merged_config = validate_preprocessing_config(config)
        else:
            merged_config = get_default_config()
        
        service = PreprocessingService(merged_config)
        
        # Get input statistics
        from ..api.stats_api import get_dataset_stats
        input_stats = get_dataset_stats(
            merged_config['data']['dir'],
            merged_config['preprocessing']['target_splits']
        )
        
        # Calculate processing estimates
        if input_stats['success']:
            total_files = sum(
                split_data['file_counts'].get('raw', 0)
                for split_data in input_stats['by_split'].values()
            )
            
            # Rough time estimates (based on file processing)
            estimated_time_seconds = total_files * 0.5  # 0.5s per file estimate
            
            preview = {
                'success': True,
                'configuration': {
                    'target_splits': merged_config['preprocessing']['target_splits'],
                    'normalization': merged_config['preprocessing']['normalization'],
                    'output_dir': merged_config['data']['preprocessed_dir']
                },
                'input_summary': {
                    'total_files': total_files,
                    'by_split': {
                        split: data['file_counts'].get('raw', 0)
                        for split, data in input_stats['by_split'].items()
                    }
                },
                'estimates': {
                    'processing_time_minutes': round(estimated_time_seconds / 60, 1),
                    'output_files': total_files,  # 1:1 mapping
                    'estimated_output_size_mb': total_files * 2.5  # Rough estimate
                }
            }
        else:
            preview = {
                'success': False,
                'message': "❌ Cannot generate preview - input data not accessible"
            }
        
        return preview
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Preview error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Preview error: {str(e)}"
        }

def get_preprocessing_api():
    """Get preprocessing API service instance"""
    try:
        from ..service import PreprocessingService
        config = get_default_config()
        return PreprocessingService(config)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ API creation error: {str(e)}")
        return None

def start_preprocessing_operation(config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """Start a preprocessing operation"""
    return preprocess_dataset(config, **kwargs)