"""
File: smartcash/ui/dataset/preprocessing/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi preprocessing sesuai dengan preprocessing_config.yaml
"""

from typing import Dict, Any

def update_preprocessing_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config preprocessing sesuai dengan preprocessing_config.yaml"""
    preprocessing_config = config.get('preprocessing', {})
    validate_config = preprocessing_config.get('validate', {})
    normalization_config = preprocessing_config.get('normalization', {})
    analysis_config = preprocessing_config.get('analysis', {})
    balance_config = preprocessing_config.get('balance', {})
    balance_methods = balance_config.get('methods', {})
    augmentation_ref = config.get('augmentation_reference', {})
    cleanup_config = config.get('cleanup', {})
    performance_config = config.get('performance', {})
    
    # One-liner component update dengan safe access
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update settings dengan mapping approach
    field_mappings = [
        # Preprocessing basic settings
        ('output_dir', preprocessing_config, 'output_dir', 'data/preprocessed'),
        ('save_visualizations', preprocessing_config, 'save_visualizations', True),
        ('vis_dir', preprocessing_config, 'vis_dir', 'visualizations/preprocessing'),
        ('sample_size', preprocessing_config, 'sample_size', 500),
        
        # Validation settings
        ('validate_enabled', validate_config, 'enabled', True),
        ('fix_issues', validate_config, 'fix_issues', True),
        ('move_invalid', validate_config, 'move_invalid', True),
        ('visualize', validate_config, 'visualize', True),
        ('check_image_quality', validate_config, 'check_image_quality', True),
        ('check_labels', validate_config, 'check_labels', True),
        ('check_coordinates', validate_config, 'check_coordinates', True),
        
        # Normalization settings
        ('normalization_enabled', normalization_config, 'enabled', True),
        ('normalization_dropdown', normalization_config, 'method', 'minmax'),
        ('preserve_aspect_ratio', normalization_config, 'preserve_aspect_ratio', True),
        ('normalize_pixel_values', normalization_config, 'normalize_pixel_values', True),
        
        # Analysis settings
        ('analysis_enabled', analysis_config, 'enabled', True),
        ('class_balance', analysis_config, 'class_balance', True),
        ('image_size_distribution', analysis_config, 'image_size_distribution', True),
        ('bbox_statistics', analysis_config, 'bbox_statistics', True),
        ('layer_balance', analysis_config, 'layer_balance', True),
        
        # Balance settings
        ('balance_enabled', balance_config, 'enabled', False),
        ('target_distribution', balance_config, 'target_distribution', 'auto'),
        ('undersampling', balance_methods, 'undersampling', False),
        ('oversampling', balance_methods, 'oversampling', True),
        ('augmentation', balance_methods, 'augmentation', True),
        ('min_samples_per_class', balance_config, 'min_samples_per_class', 100),
        ('max_samples_per_class', balance_config, 'max_samples_per_class', 1000),
        
        # Augmentation reference settings
        ('use_augmentation_for_preprocessing', augmentation_ref, 'use_for_preprocessing', True),
        ('preprocessing_variations', augmentation_ref, 'preprocessing_variations', 3),
        
        # Cleanup settings
        ('backup_dir', cleanup_config, 'backup_dir', 'data/backup/preprocessing'),
        ('backup_enabled', cleanup_config, 'backup_enabled', True),
        ('auto_cleanup_preprocessed', cleanup_config, 'auto_cleanup_preprocessed', False),
        
        # Performance settings
        ('worker_slider', performance_config, 'num_workers', 8),
        ('batch_size', performance_config, 'batch_size', 32),
        ('use_gpu', performance_config, 'use_gpu', True),
        ('compression_level', performance_config, 'compression_level', 90),
        ('max_memory_usage_gb', performance_config, 'max_memory_usage_gb', 4.0),
        ('use_mixed_precision', performance_config, 'use_mixed_precision', True)
    ]
    
    # Apply all mappings dengan one-liner approach
    [safe_update(component_key, source_config.get(config_key, default_value)) for component_key, source_config, config_key, default_value in field_mappings]
    
    # Special handling untuk target_size (resolution)
    try:
        target_size = normalization_config.get('target_size', [640, 640])
        if isinstance(target_size, list) and len(target_size) >= 2:
            resolution_str = f"{target_size[0]}x{target_size[1]}"
            safe_update('resolution_dropdown', resolution_str)
    except Exception:
        safe_update('resolution_dropdown', '640x640')  # Default fallback


def reset_preprocessing_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI components ke default konfigurasi preprocessing"""
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        config_manager = get_config_manager()
        default_config = get_default_preprocessing_config()
        update_preprocessing_ui(ui_components, default_config)
    except Exception:
        # Fallback to basic reset jika config manager gagal
        _apply_basic_defaults(ui_components)


def _apply_basic_defaults(ui_components: Dict[str, Any]) -> None:
    """Apply basic defaults ke UI components jika config manager tidak tersedia"""
    basic_defaults = {
        'resolution_dropdown': '640x640',
        'normalization_dropdown': 'minmax',
        'worker_slider': 8,
        'batch_size': 32,
        'normalize_pixel_values': True,
        'preserve_aspect_ratio': True,
        'save_visualizations': True,
        'sample_size': 500,
        'use_gpu': True,
        'validate_enabled': True,
        'fix_issues': True,
        'analysis_enabled': True
    }
    
    for key, value in basic_defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            ui_components[key].value = value
