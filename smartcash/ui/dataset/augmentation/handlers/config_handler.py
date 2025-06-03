"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler dengan unified logging dan parameter alignment yang diperbaiki
"""

from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui

def save_configuration(ui_components: Dict[str, Any]):
    """Save config dengan parameter alignment"""
    try:
        config = _extract_ui_config(ui_components)
        
        if not _validate_config(config):
            log_to_ui(ui_components, 'Config tidak valid - periksa parameter', 'error', '❌ ')
            return
        
        config_manager = get_config_manager()
        success = config_manager.save_config(config, 'augmentation_config')
        
        if success:
            _update_cache(ui_components, config)
            log_to_ui(ui_components, 'Config berhasil disimpan dengan parameter alignment', 'success', '✅ ')
        else:
            log_to_ui(ui_components, 'Gagal menyimpan config', 'error', '❌ ')
        
    except Exception as e:
        log_to_ui(ui_components, f'Config save error: {str(e)}', 'error', '❌ ')

def reset_configuration(ui_components: Dict[str, Any]):
    """Reset config dengan research defaults"""
    try:
        default_config = _get_default_config()
        _apply_config_to_ui(ui_components, default_config)
        
        config_manager = get_config_manager()
        config_manager.save_config(default_config, 'augmentation_config')
        _update_cache(ui_components, default_config)
        
        log_to_ui(ui_components, 'Config direset ke research defaults', 'success', '🔄 ')
        
    except Exception as e:
        log_to_ui(ui_components, f'Config reset error: {str(e)}', 'error', '❌ ')

def _extract_ui_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan parameter alignment"""
    try:
        from smartcash.dataset.augmentor.config import extract_ui_config
        return extract_ui_config(ui_components)
    except ImportError:
        return _manual_extraction(ui_components)

def _manual_extraction(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Manual extraction dengan research parameters sesuai dengan augmentation_config.yaml"""
    from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
    from datetime import datetime
    
    # Extract augmentation types dari UI
    aug_types = _extract_aug_types(ui_components)
    
    # Parameter dasar
    basic_params = {
        'enabled': True,
        'num_variations': _get_widget_value_safe(ui_components, 'num_variations', 3),
        'target_count': _get_widget_value_safe(ui_components, 'target_count', 500),
        'output_prefix': _get_widget_value_safe(ui_components, 'output_prefix', 'aug'),
        'process_bboxes': True,
        'output_dir': 'data/augmented',
        'validate_results': True,
        'resume': False,
        'num_workers': _get_widget_value_safe(ui_components, 'num_workers', 4),
        'balance_classes': _get_widget_value_safe(ui_components, 'balance_classes', True),
        'move_to_preprocessed': True
    }
    
    # Parameter augmentasi posisi
    position_params = {
        'fliplr': _get_widget_value_safe(ui_components, 'fliplr', 0.5),
        'degrees': _get_widget_value_safe(ui_components, 'degrees', 15),
        'translate': _get_widget_value_safe(ui_components, 'translate', 0.15),
        'scale': _get_widget_value_safe(ui_components, 'scale', 0.15),
        'shear_max': _get_widget_value_safe(ui_components, 'shear_max', 10)
    }
    
    # Parameter augmentasi pencahayaan
    lighting_params = {
        'hsv_h': _get_widget_value_safe(ui_components, 'hsv_h', 0.025),
        'hsv_s': _get_widget_value_safe(ui_components, 'hsv_s', 0.7),
        'hsv_v': _get_widget_value_safe(ui_components, 'hsv_v', 0.4),
        'contrast': [0.7, 1.3],  # Default from YAML
        'brightness': [0.7, 1.3],  # Default from YAML
        'blur': _get_widget_value_safe(ui_components, 'blur', 0.2),
        'noise': _get_widget_value_safe(ui_components, 'noise', 0.1)
    }
    
    # Pengaturan cleanup
    cleanup_params = {
        'backup_enabled': _get_widget_value_safe(ui_components, 'backup_enabled', True),
        'backup_dir': 'data/backup/augmentation',
        'backup_count': _get_widget_value_safe(ui_components, 'backup_count', 5),
        'patterns': ['aug_*', '*_augmented*']
    }
    
    # Pengaturan visualisasi
    visualization_params = {
        'enabled': _get_widget_value_safe(ui_components, 'visualization_enabled', True),
        'sample_count': _get_widget_value_safe(ui_components, 'sample_count', 5),
        'save_visualizations': _get_widget_value_safe(ui_components, 'save_visualizations', True),
        'vis_dir': 'visualizations/augmentation',
        'show_original': True,
        'show_bboxes': True
    }
    
    # Pengaturan performa
    performance_params = {
        'num_workers': _get_widget_value_safe(ui_components, 'num_workers', 4),
        'batch_size': _get_widget_value_safe(ui_components, 'batch_size', 16),
        'use_gpu': _get_widget_value_safe(ui_components, 'use_gpu', True)
    }
    
    # Metadata
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Struktur config sesuai dengan augmentation_config.yaml
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        'augmentation': {
            'types': aug_types,
            **basic_params,
            'position': position_params,
            'lighting': lighting_params
        },
        'cleanup': cleanup_params,
        'visualization': visualization_params,
        'performance': performance_params
    }

def _extract_aug_types(ui_components: Dict[str, Any]) -> list:
    """Extract augmentation types dengan fallback strategies"""
    widget = ui_components.get('augmentation_types')
    if widget and hasattr(widget, 'value') and widget.value:
        return list(widget.value)
    
    for alt_name in ['types_widget', 'aug_types', 'augmentation_type']:
        widget = ui_components.get(alt_name)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
    
    return ['combined']  # Research default

def _validate_config(config: Dict[str, Any]) -> bool:
    """Validate config untuk research compatibility"""
    try:
        aug_config = config.get('augmentation', {})
        
        if aug_config.get('num_variations', 0) <= 0:
            return False
        if aug_config.get('target_count', 0) <= 0:
            return False
        if not aug_config.get('types'):
            return False
        
        ranges = {
            'fliplr': (0.0, 1.0), 'degrees': (0, 45), 'translate': (0.0, 0.5), 'scale': (0.0, 0.5),
            'hsv_h': (0.0, 0.1), 'hsv_s': (0.0, 1.0), 'brightness': (0.0, 1.0), 'contrast': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in ranges.items():
            value = aug_config.get(param)
            if value is not None and not (min_val <= value <= max_val):
                return False
        
        return True
        
    except Exception:
        return False

def _get_default_config() -> Dict[str, Any]:
    """Default config sesuai dengan augmentation_config.yaml"""
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'config_version': '1.0',
        'updated_at': current_time,
        '_base_': 'base_config.yaml',
        
        # Konfigurasi augmentasi utama
        'augmentation': {
            # Parameter dasar
            'enabled': True,
            'types': ['combined', 'position', 'lighting'],
            'num_variations': 3,
            'target_count': 1000,
            'output_prefix': 'aug',
            'process_bboxes': True,
            'output_dir': 'data/augmented',
            'validate_results': True,
            'resume': False,
            'balance_classes': True,
            'move_to_preprocessed': True,
            
            # Parameter augmentasi posisi
            'position': {
                'fliplr': 0.5,
                'degrees': 15,
                'translate': 0.15,
                'scale': 0.15,
                'shear_max': 10
            },
            
            # Parameter augmentasi pencahayaan
            'lighting': {
                'hsv_h': 0.025,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'contrast': [0.7, 1.3],
                'brightness': [0.7, 1.3],
                'blur': 0.2,
                'noise': 0.1
            }
        },
        
        # Pengaturan pengelolaan data augmentasi
        'cleanup': {
            'backup_enabled': True,
            'backup_dir': 'data/backup/augmentation',
            'backup_count': 5,
            'patterns': ['aug_*', '*_augmented*']
        },
        
        # Pengaturan visualisasi
        'visualization': {
            'enabled': True,
            'sample_count': 5,
            'save_visualizations': True,
            'vis_dir': 'visualizations/augmentation',
            'show_original': True,
            'show_bboxes': True
        },
        
        # Pengaturan performa
        'performance': {
            'num_workers': 4,
            'batch_size': 16,
            'use_gpu': True
        }
    }

def _apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Apply config ke UI dengan parameter alignment sesuai dengan struktur augmentation_config.yaml"""
    aug_config = config.get('augmentation', {})
    position_config = aug_config.get('position', {})
    lighting_config = aug_config.get('lighting', {})
    cleanup_config = config.get('cleanup', {})
    visualization_config = config.get('visualization', {})
    performance_config = config.get('performance', {})
    
    # Ekstraksi parameter dasar dari config
    basic_mappings = {
        'num_variations': aug_config.get('num_variations', 3),
        'target_count': aug_config.get('target_count', 1000),
        'output_prefix': aug_config.get('output_prefix', 'aug'),
        'balance_classes': aug_config.get('balance_classes', True),
        'num_workers': performance_config.get('num_workers', 4),
        'batch_size': performance_config.get('batch_size', 16),
        'use_gpu': performance_config.get('use_gpu', True)
    }
    
    # Ekstraksi parameter posisi dari config
    position_mappings = {
        'fliplr': position_config.get('fliplr', 0.5),
        'degrees': position_config.get('degrees', 15),
        'translate': position_config.get('translate', 0.15),
        'scale': position_config.get('scale', 0.15),
        'shear_max': position_config.get('shear_max', 10)
    }
    
    # Ekstraksi parameter pencahayaan dari config
    lighting_mappings = {
        'hsv_h': lighting_config.get('hsv_h', 0.025),
        'hsv_s': lighting_config.get('hsv_s', 0.7),
        'hsv_v': lighting_config.get('hsv_v', 0.4),
        'blur': lighting_config.get('blur', 0.2),
        'noise': lighting_config.get('noise', 0.1)
    }
    
    # Ekstraksi parameter cleanup dari config
    cleanup_mappings = {
        'backup_enabled': cleanup_config.get('backup_enabled', True),
        'backup_count': cleanup_config.get('backup_count', 5)
    }
    
    # Ekstraksi parameter visualisasi dari config
    visualization_mappings = {
        'visualization_enabled': visualization_config.get('enabled', True),
        'sample_count': visualization_config.get('sample_count', 5),
        'save_visualizations': visualization_config.get('save_visualizations', True)
    }
    
    # Gabungkan semua parameter untuk diaplikasikan ke UI
    all_mappings = {
        **basic_mappings,
        **position_mappings,
        **lighting_mappings,
        **cleanup_mappings,
        **visualization_mappings
    }
    
    # Terapkan semua parameter ke UI components
    for widget_key, value in all_mappings.items():
        _set_widget_value_safe(ui_components, widget_key, value)
    
    # Set jenis augmentasi
    aug_types = aug_config.get('types', ['combined'])
    _set_augmentation_types(ui_components, aug_types)
    
    # Log info untuk user
    log_to_ui(ui_components, f'Konfigurasi augmentasi berhasil dimuat dengan {len(all_mappings)} parameter', 'info', '📊 ')

def _update_cache(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update cache dengan new config"""
    ui_components['config'] = config
    ui_components['config_cache_valid'] = True

def _set_augmentation_types(ui_components: Dict[str, Any], types: list) -> None:
    """Set augmentation types dengan multiple strategies"""
    for widget_key in ['augmentation_types', 'types_widget', 'aug_types']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = list(types)
                return
            except Exception:
                continue

def _get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safe widget value extraction dengan type consistency"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = widget.value
            if isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            return value
        except Exception:
            pass
    return default

def _set_widget_value_safe(ui_components: Dict[str, Any], key: str, value: Any) -> None:
    """Safe widget value setting"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            widget.value = value
        except Exception:
            pass

def _load_preprocessing_config_safe() -> Dict[str, Any]:
    """Load preprocessing config dengan fallback ke default scaler structure"""
    try:
        config_manager = get_config_manager()
        preprocessing_config = config_manager.get_config('preprocessing_config')
        
        if preprocessing_config and 'preprocessing' in preprocessing_config:
            return preprocessing_config['preprocessing']
        
        return {'normalization': {'scaler': 'minmax'}}
        
    except Exception:
        # Fallback jika config tidak bisa dimuat
        return {'normalization': {'scaler': 'minmax'}}