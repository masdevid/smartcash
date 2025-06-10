"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Enhanced backend integration dengan multi-split, validasi, dan aspect ratio support
"""

from typing import Dict, Any, Tuple, List

def validate_enhanced_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """Enhanced validation dengan multi-split dan aspect ratio awareness"""
    try:
        from smartcash.dataset.utils.path_validator import get_path_validator
        from pathlib import Path
        
        data_dir = config.get('data', {}).get('dir', 'data')
        if not data_dir or not isinstance(data_dir, str):
            return False, "❌ Path dataset tidak valid"
        
        data_path = Path(data_dir)
        if not data_path.exists():
            return False, f"❌ Directory dataset tidak ditemukan: {data_dir}"
        
        # Enhanced multi-split validation
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        available_splits = []
        missing_splits = []
        
        for split in target_splits:
            split_path = data_path / split
            if not split_path.exists():
                missing_splits.append(split)
                continue
                
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                missing_splits.append(f"{split} (images/labels)")
                continue
            
            # Check image count
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_count = len([f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions])
            
            if image_count > 0:
                available_splits.append(split)
            else:
                missing_splits.append(f"{split} (no images)")
        
        if missing_splits:
            return False, f"❌ Split tidak lengkap: {', '.join(missing_splits)}"
        
        if not available_splits:
            return False, "❌ Tidak ada split valid yang tersedia"
        
        # Enhanced validation dengan validator
        validator = get_path_validator(logger)
        result = validator.validate_enhanced_dataset_structure(data_dir, target_splits)
        
        if not result.get('valid', False):
            issues = result.get('issues', ['Unknown validation error'])
            return False, f"❌ Dataset validation gagal: {issues[0] if issues else 'No images found'}"
        
        total_images = result.get('total_images', 0)
        if total_images == 0:
            return False, "❌ Dataset tidak memiliki gambar valid"
        
        # Enhanced success message dengan split info
        split_info = f"{len(available_splits)} split ({', '.join(available_splits)})"
        return True, f"✅ Dataset ready: {total_images:,} gambar dalam {split_info}"
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error enhanced validation: {str(e)}")
        return False, f"❌ Error validation: {str(e)[:100]}..."

def check_enhanced_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int, Dict[str, int]]:
    """Enhanced check dengan split breakdown details"""
    try:
        from pathlib import Path
        
        preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        if not preprocessed_dir.exists():
            return False, 0, {}
        
        total_files = 0
        split_breakdown = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Check semua possible splits
        for split in ['train', 'valid', 'test']:
            split_images_dir = preprocessed_dir / split / 'images'
            if split_images_dir.exists():
                split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
                split_count = len(split_files)
                if split_count > 0:
                    split_breakdown[split] = split_count
                    total_files += split_count
        
        return total_files > 0, total_files, split_breakdown
        
    except Exception:
        return False, 0, {}

def create_enhanced_backend_preprocessor(ui_config: Dict[str, Any], logger=None):
    """Create enhanced preprocessor dengan multi-split dan aspect ratio support"""
    try:
        from smartcash.dataset.preprocessor.core.enhanced_preprocessing_manager import EnhancedPreprocessingManager
        
        # Convert UI config ke enhanced backend format
        backend_config = _convert_ui_to_enhanced_backend_config(ui_config)
        
        return EnhancedPreprocessingManager(backend_config, logger)
        
    except ImportError:
        # Fallback ke standard preprocessor dengan enhanced config
        try:
            from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
            backend_config = _convert_ui_to_enhanced_backend_config(ui_config)
            return PreprocessingManager(backend_config, logger)
        except Exception as e:
            if logger:
                logger.error(f"❌ Error creating preprocessor: {str(e)}")
            return None
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced preprocessor: {str(e)}")
        return None

def create_enhanced_backend_checker(config: Dict[str, Any], logger=None):
    """Create enhanced dataset checker dengan multi-split awareness"""
    try:
        from smartcash.dataset.preprocessor.operations.enhanced_dataset_checker import EnhancedDatasetChecker
        return EnhancedDatasetChecker(config, logger)
        
    except ImportError:
        # Fallback ke standard checker
        try:
            from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
            return DatasetChecker(logger)
        except Exception as e:
            if logger:
                logger.error(f"❌ Error creating dataset checker: {str(e)}")
            return None
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced checker: {str(e)}")
        return None

def create_enhanced_backend_cleanup_service(config: Dict[str, Any], logger=None):
    """Create enhanced cleanup service dengan multi-split preservation"""
    try:
        from smartcash.dataset.preprocessor.operations.enhanced_cleanup_executor import EnhancedCleanupExecutor
        return EnhancedCleanupExecutor(config, logger)
        
    except ImportError:
        # Fallback ke standard cleanup
        try:
            from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
            return CleanupExecutor(config, logger)
        except Exception as e:
            if logger:
                logger.error(f"❌ Error creating cleanup service: {str(e)}")
            return None
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced cleanup service: {str(e)}")
        return None

def _convert_ui_to_enhanced_backend_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI config ke enhanced backend service format"""
    preprocessing = ui_config.get('preprocessing', {})
    normalization = preprocessing.get('normalization', {})
    validation = preprocessing.get('validation', {})
    performance = ui_config.get('performance', {})
    
    # Extract enhanced target_size dari params
    target_size = normalization.get('target_size', [640, 640])
    if isinstance(target_size, list) and len(target_size) >= 2:
        img_size = target_size
    else:
        img_size = [640, 640]
    
    # Extract enhanced target_splits
    target_splits = preprocessing.get('target_splits', ['train', 'valid'])
    if isinstance(target_splits, str):
        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
    
    # Enhanced backend config dengan semua fitur baru
    return {
        'data': ui_config.get('data', {}),
        'preprocessing': preprocessing,
        'performance': performance,
        'cleanup': ui_config.get('cleanup', {}),
        
        # Enhanced backend compatibility format
        'img_size': img_size,
        'target_splits': target_splits,  # Multi-split support
        'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True),  # Aspect ratio support
        'normalize': normalization.get('enabled', True),
        'normalization_method': normalization.get('method', 'minmax'),
        'batch_size': performance.get('batch_size', 32),  # Batch processing
        'force_reprocess': preprocessing.get('force_reprocess', False),
        
        # Enhanced validation settings
        'validation_enabled': validation.get('enabled', True),
        'move_invalid': validation.get('move_invalid', True),
        'invalid_dir': validation.get('invalid_dir', 'data/invalid'),
        'validation_config': validation,
        
        # Output settings dengan enhanced structure
        'output_dir': preprocessing.get('output_dir', 'data/preprocessed'),
        'save_visualizations': preprocessing.get('save_visualizations', False),
        
        # Enhanced threading configuration
        'threading': performance.get('threading', {
            'io_workers': 8,
            'cpu_workers': None,
            'parallel_threshold': 100,
            'batch_processing': True
        })
    }

def validate_enhanced_backend_compatibility(config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Validate compatibility dengan enhanced backend services"""
    compatibility_report = {
        'enhanced_preprocessor': False,
        'enhanced_checker': False,
        'enhanced_cleanup': False,
        'fallback_available': False,
        'features_supported': {
            'multi_split': False,
            'aspect_ratio': False,
            'enhanced_validation': False,
            'batch_processing': False
        }
    }
    
    # Test enhanced preprocessor
    try:
        from smartcash.dataset.preprocessor.core.enhanced_preprocessing_manager import EnhancedPreprocessingManager
        compatibility_report['enhanced_preprocessor'] = True
        compatibility_report['features_supported']['multi_split'] = True
        compatibility_report['features_supported']['aspect_ratio'] = True
        compatibility_report['features_supported']['batch_processing'] = True
    except ImportError:
        pass
    
    # Test enhanced checker
    try:
        from smartcash.dataset.preprocessor.operations.enhanced_dataset_checker import EnhancedDatasetChecker
        compatibility_report['enhanced_checker'] = True
        compatibility_report['features_supported']['enhanced_validation'] = True
    except ImportError:
        pass
    
    # Test enhanced cleanup
    try:
        from smartcash.dataset.preprocessor.operations.enhanced_cleanup_executor import EnhancedCleanupExecutor
        compatibility_report['enhanced_cleanup'] = True
    except ImportError:
        pass
    
    # Test fallback services
    try:
        from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
        from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
        from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
        compatibility_report['fallback_available'] = True
    except ImportError:
        pass
    
    # Log compatibility report
    if logger:
        enhanced_count = sum([
            compatibility_report['enhanced_preprocessor'],
            compatibility_report['enhanced_checker'], 
            compatibility_report['enhanced_cleanup']
        ])
        
        if enhanced_count == 3:
            logger.success("✅ Full enhanced backend compatibility")
        elif enhanced_count > 0:
            logger.warning(f"⚠️ Partial enhanced compatibility ({enhanced_count}/3 services)")
        elif compatibility_report['fallback_available']:
            logger.info("ℹ️ Using fallback backend services")
        else:
            logger.error("❌ No compatible backend services found")
    
    return compatibility_report

# Backward compatibility aliases
validate_dataset_ready = validate_enhanced_dataset_ready
check_preprocessed_exists = check_enhanced_preprocessed_exists
create_backend_preprocessor = create_enhanced_backend_preprocessor
create_backend_checker = create_enhanced_backend_checker
create_backend_cleanup_service = create_enhanced_backend_cleanup_service