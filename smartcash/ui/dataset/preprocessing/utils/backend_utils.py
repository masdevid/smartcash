"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Updated backend integration dengan enhanced preprocessor compatibility
"""

from typing import Dict, Any, Tuple, List

def validate_enhanced_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """🔍 Enhanced validation dengan comprehensive checks"""
    try:
        # Use enhanced preprocessor validation
        from smartcash.dataset.preprocessor import validate_dataset
        
        # Get target splits dari config
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        all_valid = True
        total_images = 0
        validation_messages = []
        
        # Validate each target split
        for split in target_splits:
            result = validate_dataset(config, split)
            
            if result.get('success', False):
                summary = result.get('summary', {})
                valid_images = summary.get('valid_images', 0)
                total_imgs = summary.get('total_images', 0)
                
                total_images += valid_images
                validation_messages.append(f"✅ {split}: {valid_images}/{total_imgs} valid")
                
                if logger:
                    logger.info(f"✅ {split} validation: {valid_images} gambar valid")
            else:
                all_valid = False
                error_msg = result.get('message', f'Validation failed untuk {split}')
                validation_messages.append(f"❌ {split}: {error_msg}")
                
                if logger:
                    logger.error(f"❌ {split} validation failed: {error_msg}")
        
        if not all_valid:
            return False, f"❌ Validation issues: {'; '.join(validation_messages)}"
        
        if total_images == 0:
            return False, "❌ Tidak ada gambar valid ditemukan di semua splits"
        
        success_msg = f"✅ Dataset ready: {total_images:,} gambar valid dalam {len(target_splits)} splits"
        return True, success_msg
        
    except Exception as e:
        error_msg = f"❌ Error validation: {str(e)}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def check_enhanced_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int, Dict[str, int]]:
    """📊 Enhanced check dengan detailed split breakdown"""
    try:
        from pathlib import Path
        
        output_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        if not output_dir.exists():
            return False, 0, {}
        
        total_files = 0
        split_breakdown = {}
        
        # Check for both .npy and regular image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']
        
        # Get target splits dari config
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        # Check each target split
        for split in target_splits:
            split_images_dir = output_dir / split / 'images'
            if split_images_dir.exists():
                split_files = []
                for ext in image_extensions:
                    split_files.extend(list(split_images_dir.glob(f'*{ext}')))
                
                split_count = len(split_files)
                if split_count > 0:
                    split_breakdown[split] = split_count
                    total_files += split_count
        
        return total_files > 0, total_files, split_breakdown
        
    except Exception:
        return False, 0, {}

def create_enhanced_backend_preprocessor(ui_config: Dict[str, Any], logger=None):
    """🏭 Create enhanced preprocessor dengan dual progress support"""
    try:
        from smartcash.dataset.preprocessor import create_preprocessing_service
        
        # Convert UI config ke backend format
        backend_config = _convert_ui_to_enhanced_backend_config(ui_config)
        
        # Create service dengan enhanced features
        service = create_preprocessing_service(backend_config)
        
        # Add compatibility methods jika tidak ada
        if not hasattr(service, 'preprocess_dataset'):
            service.preprocess_dataset = service.preprocess_and_visualize
        
        return service
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced preprocessor: {str(e)}")
        return None

def create_enhanced_backend_checker(config: Dict[str, Any], logger=None):
    """🔍 Create enhanced dataset checker"""
    try:
        from smartcash.dataset.preprocessor import validate_dataset, get_preprocessing_statistics
        
        class EnhancedChecker:
            def __init__(self, config, logger):
                self.config = config
                self.logger = logger
            
            def check_dataset(self) -> Dict[str, Any]:
                """Enhanced dataset check dengan comprehensive stats"""
                try:
                    # Get target splits
                    target_splits = self.config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
                    if isinstance(target_splits, str):
                        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
                    
                    results = {}
                    total_valid = 0
                    
                    for split in target_splits:
                        result = validate_dataset(self.config, split)
                        results[split] = result
                        
                        if result.get('success'):
                            total_valid += result.get('summary', {}).get('valid_images', 0)
                    
                    # Get comprehensive statistics
                    stats_result = get_preprocessing_statistics(self.config)
                    
                    return {
                        'success': True,
                        'message': f"✅ Dataset check completed: {total_valid} valid images",
                        'results': results,
                        'statistics': stats_result.get('stats', {}),
                        'total_valid_images': total_valid
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Check failed: {str(e)}"
                    }
        
        return EnhancedChecker(config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced checker: {str(e)}")
        return None

def create_enhanced_backend_cleanup_service(config: Dict[str, Any], logger=None):
    """🧹 Create enhanced cleanup service"""
    try:
        from smartcash.dataset.preprocessor import cleanup_preprocessed_data
        
        class EnhancedCleanupService:
            def __init__(self, config, logger):
                self.config = config
                self.logger = logger
            
            def cleanup_preprocessed_data(self) -> Dict[str, Any]:
                """Enhanced cleanup dengan detailed reporting"""
                try:
                    # Check existing data sebelum cleanup
                    has_data, file_count, split_breakdown = check_enhanced_preprocessed_exists(self.config)
                    
                    if not has_data:
                        return {
                            'success': True,
                            'message': "ℹ️ Tidak ada preprocessed data untuk dibersihkan",
                            'stats': {'files_removed': 0, 'splits_cleaned': 0}
                        }
                    
                    # Perform cleanup
                    result = cleanup_preprocessed_data(self.config)
                    
                    if result.get('success'):
                        return {
                            'success': True,
                            'message': f"🧹 Cleanup berhasil: {file_count} files removed",
                            'stats': {
                                'files_removed': file_count,
                                'splits_cleaned': len(split_breakdown),
                                'split_breakdown': split_breakdown
                            }
                        }
                    else:
                        return {
                            'success': False,
                            'message': result.get('message', 'Cleanup failed')
                        }
                        
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Cleanup error: {str(e)}"
                    }
        
        return EnhancedCleanupService(config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating enhanced cleanup service: {str(e)}")
        return None

def _convert_ui_to_enhanced_backend_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """🔄 Convert UI config ke enhanced backend format dengan full compatibility"""
    preprocessing = ui_config.get('preprocessing', {})
    normalization = preprocessing.get('normalization', {})
    validation = preprocessing.get('validation', {})
    performance = ui_config.get('performance', {})
    
    # Enhanced target_splits handling
    target_splits = preprocessing.get('target_splits', ['train', 'valid'])
    if isinstance(target_splits, str):
        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
    
    # Enhanced target_size handling
    target_size = normalization.get('target_size', [640, 640])
    if isinstance(target_size, list) and len(target_size) >= 2:
        img_size = target_size
    else:
        img_size = [640, 640]
    
    # Enhanced backend config yang kompatibel dengan semua features
    return {
        'data': ui_config.get('data', {'dir': 'data'}),
        'preprocessing': {
            **preprocessing,
            'target_splits': target_splits,
            'normalization': {
                **normalization,
                'target_size': img_size,
                'enabled': normalization.get('enabled', True),
                'method': normalization.get('method', 'minmax'),
                'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True)
            },
            'validation': {
                **validation,
                'enabled': validation.get('enabled', True),
                'move_invalid': validation.get('move_invalid', True),
                'invalid_dir': validation.get('invalid_dir', 'data/invalid')
            },
            'output': {
                'output_dir': preprocessing.get('output_dir', 'data/preprocessed'),
                'create_npy': preprocessing.get('output', {}).get('create_npy', True),
                'organize_by_split': True
            }
        },
        'performance': {
            **performance,
            'batch_size': performance.get('batch_size', 32),
            'threading': performance.get('threading', {
                'io_workers': 8,
                'cpu_workers': None,
                'parallel_threshold': 100,
                'batch_processing': True
            })
        },
        'cleanup': ui_config.get('cleanup', {}),
        
        # Legacy compatibility keys
        'img_size': img_size,
        'normalize': normalization.get('enabled', True),
        'normalization_method': normalization.get('method', 'minmax'),
        'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True),
        'validation_enabled': validation.get('enabled', True),
        'move_invalid': validation.get('move_invalid', True),
        'invalid_dir': validation.get('invalid_dir', 'data/invalid'),
        'output_dir': preprocessing.get('output_dir', 'data/preprocessed'),
        'force_reprocess': preprocessing.get('force_reprocess', False)
    }

def validate_enhanced_backend_compatibility(config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """🔍 Validate enhanced backend compatibility"""
    compatibility_report = {
        'enhanced_preprocessor': False,
        'dual_progress_support': False,
        'multi_split_support': False,
        'augmentor_consistency': False,
        'features_supported': {}
    }
    
    try:
        # Test enhanced preprocessor
        from smartcash.dataset.preprocessor import create_preprocessing_service, check_preprocessing_compatibility
        
        compatibility_report['enhanced_preprocessor'] = True
        
        # Check enhanced features
        compat_check = check_preprocessing_compatibility()
        compatibility_report['dual_progress_support'] = compat_check.get('dual_progress_tracker', False)
        compatibility_report['augmentor_consistency'] = compat_check.get('augmentor_consistency', False)
        compatibility_report['features_supported'] = compat_check.get('enhanced_features', [])
        
        # Test multi-split support
        test_config = _convert_ui_to_enhanced_backend_config(config)
        if 'target_splits' in test_config.get('preprocessing', {}):
            compatibility_report['multi_split_support'] = True
        
        # Log compatibility status
        if logger:
            feature_count = len(compatibility_report['features_supported'])
            logger.info(f"🔍 Backend compatibility: {feature_count} enhanced features available")
            
            if compatibility_report['enhanced_preprocessor']:
                logger.success("✅ Enhanced preprocessor available")
            if compatibility_report['dual_progress_support']:
                logger.success("✅ Dual progress tracker support")
            if compatibility_report['augmentor_consistency']:
                logger.success("✅ Augmentor consistency support")
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Compatibility check error: {str(e)}")
    
    return compatibility_report

# Backward compatibility aliases
validate_dataset_ready = validate_enhanced_dataset_ready
check_preprocessed_exists = check_enhanced_preprocessed_exists
create_backend_preprocessor = create_enhanced_backend_preprocessor
create_backend_checker = create_enhanced_backend_checker
create_backend_cleanup_service = create_enhanced_backend_cleanup_service