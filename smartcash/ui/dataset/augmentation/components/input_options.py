"""
File: smartcash/ui/dataset/augmentation/components/input_options.py
Deskripsi: Consolidated form inputs dengan backend mapping dan validation integration
"""

from typing import Dict, Any
import ipywidgets as widgets

def create_augmentation_form_inputs(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create consolidated form inputs untuk augmentation dengan backend compatibility
    
    Args:
        config: Optional config untuk initial values
        
    Returns:
        Dictionary berisi semua form widgets dan metadata
    """
    config = config or {}
    aug_config = config.get('augmentation', {})
    
    # Import komponen widgets
    from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
    from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
    from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
    
    # Create widget groups
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    types_options = create_augmentation_types_widget()
    
    # Additional preprocessing options untuk backend integration
    normalization_options = _create_normalization_options(config.get('preprocessing', {}))
    
    # Consolidated widget mapping untuk easy access
    all_widgets = {
        # Basic options
        **basic_options['widgets'],
        # Advanced options
        **advanced_options['widgets'],
        # Types dan split
        **types_options['widgets'],
        # Normalization
        **normalization_options['widgets']
    }
    
    # Validation consolidation
    validation_info = _consolidate_validation_info(
        basic_options.get('validation', {}),
        advanced_options.get('validation', {}),
        types_options.get('validation', {}),
        normalization_options.get('validation', {})
    )
    
    # Backend mapping consolidation
    backend_mapping = _consolidate_backend_mapping(
        advanced_options.get('backend_mapping', {}),
        normalization_options.get('backend_mapping', {})
    )
    
    return {
        'widgets': all_widgets,
        'groups': {
            'basic': basic_options,
            'advanced': advanced_options,
            'types': types_options,
            'normalization': normalization_options
        },
        'validation': validation_info,
        'backend_mapping': backend_mapping,
        
        # Form state management
        'form_state': {
            'dirty': False,
            'last_validated': None,
            'validation_errors': [],
            'validation_warnings': []
        },
        
        # Quick access untuk common operations
        'quick_access': {
            'required_fields': ['num_variations', 'target_count', 'augmentation_types', 'target_split'],
            'position_params': ['fliplr', 'degrees', 'translate', 'scale'],
            'lighting_params': ['hsv_h', 'hsv_s', 'brightness', 'contrast'],
            'normalization_params': ['norm_method', 'denormalize'],
            'all_numeric': ['num_variations', 'target_count', 'fliplr', 'degrees', 'translate', 'scale', 'hsv_h', 'hsv_s', 'brightness', 'contrast']
        }
    }

def _create_normalization_options(preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create normalization options untuk backend preprocessing integration"""
    norm_config = preprocessing_config.get('normalization', {})
    
    # Normalization method dengan backend compatibility
    norm_method = widgets.Dropdown(
        options=[
            ('MinMax (0-1): YOLOv5 compatible', 'minmax'),
            ('Standard (z-score): Statistical normalization', 'standard'), 
            ('ImageNet: Transfer learning preset', 'imagenet'),
            ('None: Raw values (0-255)', 'none')
        ],
        value=norm_config.get('method', 'minmax'),
        description='Norm Method:',
        disabled=False,
        layout=widgets.Layout(width='auto'),
        style={'description_width': '100px'}
    )
    
    # Denormalization option untuk save format
    denormalize = widgets.Checkbox(
        value=norm_config.get('denormalize', False),
        description='Denormalize setelah preprocessing (save as uint8)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='5px 0')
    )
    
    # Target size (readonly - fixed untuk YOLO)
    target_size_display = widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: #f5f5f5; 
                    border-radius: 3px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #ddd;">
            <strong>🎯 Target Size:</strong> 640x640 (Fixed untuk YOLOv5 compatibility)
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    # Normalization info panel
    norm_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #fff3e0; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #ff980040;">
            <strong style="color: #f57c00;">🔧 Normalization Backend:</strong><br>
            • <strong style="color: #f57c00;">MinMax:</strong> OpenCV + NumPy array [0.0, 1.0]<br>
            • <strong style="color: #f57c00;">Standard:</strong> Scikit-learn StandardScaler compatible<br>
            • <strong style="color: #f57c00;">ImageNet:</strong> Torchvision transforms preset<br>
            • <strong style="color: #f57c00;">Denormalize:</strong> Convert float32 → uint8 untuk compatibility
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #f57c00; margin: 8px 0;'>🔧 Preprocessing Options</h6>"),
        norm_method,
        denormalize,
        target_size_display,
        norm_info
    ], layout=widgets.Layout(padding='8px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'norm_method': norm_method,
            'denormalize': denormalize
        },
        'validation': {
            'ranges': {},
            'required': ['norm_method'],
            'defaults': {
                'norm_method': 'minmax',
                'denormalize': False
            }
        },
        'backend_mapping': {
            'preprocessing': {
                'normalization_method': 'norm_method',
                'denormalize_output': 'denormalize',
                'target_size': [640, 640]  # Fixed
            }
        }
    }

def _consolidate_validation_info(*validation_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate validation info dari multiple widget groups"""
    consolidated = {
        'ranges': {},
        'required': [],
        'defaults': {},
        'backend_constraints': [],
        'validation_rules': []
    }
    
    for val_dict in validation_dicts:
        if not val_dict:
            continue
            
        # Merge ranges
        consolidated['ranges'].update(val_dict.get('ranges', {}))
        
        # Merge required fields
        consolidated['required'].extend(val_dict.get('required', []))
        
        # Merge defaults
        consolidated['defaults'].update(val_dict.get('defaults', {}))
        
        # Merge backend constraints
        if 'backend_constraints' in val_dict:
            consolidated['backend_constraints'].append(val_dict['backend_constraints'])
    
    # Remove duplicates dari required
    consolidated['required'] = list(set(consolidated['required']))
    
    # Add consolidated validation rules
    consolidated['validation_rules'] = [
        ('numeric_ranges', 'Validasi range untuk parameter numerik'),
        ('required_fields', 'Validasi field wajib diisi'),
        ('backend_compatibility', 'Validasi kompatibilitas dengan backend service'),
        ('type_consistency', 'Validasi konsistensi tipe data'),
        ('logical_constraints', 'Validasi constraint logis antar parameter')
    ]
    
    return consolidated

def _consolidate_backend_mapping(*mapping_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate backend mapping dari multiple components"""
    consolidated = {
        'service_mapping': {},
        'parameter_transformation': {},
        'api_compatibility': {}
    }
    
    for mapping_dict in mapping_dicts:
        if not mapping_dict:
            continue
            
        # Deep merge mapping dictionaries
        for section, section_mapping in mapping_dict.items():
            if section not in consolidated['service_mapping']:
                consolidated['service_mapping'][section] = {}
            consolidated['service_mapping'][section].update(section_mapping)
    
    # Parameter transformation rules untuk backend
    consolidated['parameter_transformation'] = {
        'float_precision': {
            'hsv_h': 3,      # 0.001 precision
            'hsv_s': 2,      # 0.01 precision  
            'brightness': 2, # 0.01 precision
            'contrast': 2,   # 0.01 precision
            'translate': 2,  # 0.01 precision
            'scale': 2       # 0.01 precision
        },
        'int_constraints': {
            'num_variations': {'min': 1, 'max': 10},
            'target_count': {'min': 100, 'max': 2000},
            'degrees': {'min': 0, 'max': 30}
        },
        'boolean_fields': ['balance_classes', 'denormalize'],
        'list_fields': ['augmentation_types']
    }
    
    # API compatibility information
    consolidated['api_compatibility'] = {
        'albumentations': ['fliplr', 'degrees', 'translate', 'scale'],
        'opencv': ['hsv_h', 'hsv_s', 'brightness', 'contrast'],
        'numpy': ['norm_method', 'denormalize'],
        'service_api': ['num_variations', 'target_count', 'augmentation_types', 'target_split']
    }
    
    return consolidated

def validate_form_inputs(form_data: Dict[str, Any], validation_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive form validation dengan backend compatibility check
    
    Args:
        form_data: Form data yang akan divalidasi
        validation_info: Validation rules dan constraints
        
    Returns:
        Validation result dengan errors dan warnings
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'backend_compatible': True,
        'summary': {}
    }
    
    # Required fields validation
    required = validation_info.get('required', [])
    for field in required:
        if field not in form_data or form_data[field] is None:
            result['valid'] = False
            result['errors'].append(f"❌ Field '{field}' wajib diisi")
        elif isinstance(form_data[field], (list, tuple)) and len(form_data[field]) == 0:
            result['valid'] = False
            result['errors'].append(f"❌ Field '{field}' harus memiliki minimal 1 pilihan")
    
    # Range validation untuk numeric fields
    ranges = validation_info.get('ranges', {})
    for field, (min_val, max_val) in ranges.items():
        if field in form_data:
            value = form_data[field]
            try:
                if not (min_val <= float(value) <= max_val):
                    result['valid'] = False
                    result['errors'].append(f"❌ {field}: {value} harus antara {min_val}-{max_val}")
            except (ValueError, TypeError):
                result['valid'] = False
                result['errors'].append(f"❌ {field}: nilai tidak valid")
    
    # Backend compatibility warnings
    if 'augmentation_types' in form_data:
        types = form_data['augmentation_types']
        if isinstance(types, (list, tuple)) and len(types) > 3:
            result['warnings'].append("⚠️ >3 augmentation types: processing time akan lebih lama")
    
    # Parameter consistency checks
    if 'degrees' in form_data and form_data['degrees'] > 20:
        result['warnings'].append("⚠️ Rotasi >20°: mungkin terlalu ekstrem untuk mata uang")
    
    if 'brightness' in form_data and 'contrast' in form_data:
        if form_data['brightness'] > 0.3 and form_data['contrast'] > 0.3:
            result['warnings'].append("⚠️ Brightness + Contrast tinggi: hasil mungkin tidak realistis")
    
    # Summary generation
    result['summary'] = {
        'total_fields': len(form_data),
        'required_filled': len([f for f in required if f in form_data and form_data[f] is not None]),
        'total_required': len(required),
        'error_count': len(result['errors']),
        'warning_count': len(result['warnings'])
    }
    
    return result

def extract_backend_compatible_config(form_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract config yang compatible dengan backend service
    
    Args:
        form_inputs: Dictionary dari form input results
        
    Returns:
        Backend-compatible configuration
    """
    widgets_dict = form_inputs.get('widgets', {})
    backend_mapping = form_inputs.get('backend_mapping', {})
    
    # Helper untuk extract widget values
    get_value = lambda key, default: getattr(widgets_dict.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Build config sesuai backend API
    config = {
        'augmentation': {
            # Core parameters
            'num_variations': get_value('num_variations', 3),
            'target_count': get_value('target_count', 500),
            'output_prefix': get_value('output_prefix', 'aug'),
            'balance_classes': get_value('balance_classes', True),
            'target_split': get_value('target_split', 'train'),
            'types': list(get_value('augmentation_types', ['combined'])),
            
            # Position parameters (Albumentations compatible)
            'position': {
                'horizontal_flip': get_value('fliplr', 0.5),
                'rotation_limit': get_value('degrees', 10),
                'translate_limit': get_value('translate', 0.1),
                'scale_limit': get_value('scale', 0.1)
            },
            
            # Lighting parameters (OpenCV compatible)
            'lighting': {
                'hsv_h_limit': get_value('hsv_h', 0.015),
                'hsv_s_limit': get_value('hsv_s', 0.7),
                'brightness_limit': get_value('brightness', 0.2),
                'contrast_limit': get_value('contrast', 0.2)
            }
        },
        
        'preprocessing': {
            'normalization': {
                'method': get_value('norm_method', 'minmax'),
                'denormalize': get_value('denormalize', False),
                'target_size': [640, 640]  # Fixed untuk YOLOv5
            }
        },
        
        # Backend metadata
        'backend_info': {
            'extracted_at': __import__('datetime').datetime.now().isoformat(),
            'api_version': '2.0',
            'compatibility': {
                'albumentations': '>=1.3.0',
                'opencv': '>=4.8.0',
                'numpy': '>=1.21.0'
            }
        }
    }
    
    return config