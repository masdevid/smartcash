"""
File: smartcash/ui/dataset/augmentation/utils/backend_utils.py
Deskripsi: Backend integration utilities untuk service layer
"""

from typing import Dict, Any, Optional

def create_service_from_ui(ui_components: Dict[str, Any]):
    """Create augmentation service dari UI components"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui as create_service
        return create_service(ui_components)
    except ImportError as e:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Service import error: {str(e)}", 'error')
        return None
    except Exception as e:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Service creation error: {str(e)}", 'error')
        return None

def create_service_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create service config dari UI components"""
    from smartcash.ui.dataset.augmentation.utils.ui_utils import get_widget_value_safe, extract_augmentation_types
    
    return {
        'num_variations': get_widget_value_safe(ui_components, 'num_variations', 3),
        'target_count': get_widget_value_safe(ui_components, 'target_count', 500),
        'output_prefix': get_widget_value_safe(ui_components, 'output_prefix', 'aug'),
        'balance_classes': get_widget_value_safe(ui_components, 'balance_classes', True),
        'target_split': get_widget_value_safe(ui_components, 'target_split', 'train'),
        'types': extract_augmentation_types(ui_components),
        'position': {
            'fliplr': get_widget_value_safe(ui_components, 'fliplr', 0.5),
            'degrees': get_widget_value_safe(ui_components, 'degrees', 10),
            'translate': get_widget_value_safe(ui_components, 'translate', 0.1),
            'scale': get_widget_value_safe(ui_components, 'scale', 0.1)
        },
        'lighting': {
            'hsv_h': get_widget_value_safe(ui_components, 'hsv_h', 0.015),
            'hsv_s': get_widget_value_safe(ui_components, 'hsv_s', 0.7),
            'brightness': get_widget_value_safe(ui_components, 'brightness', 0.2),
            'contrast': get_widget_value_safe(ui_components, 'contrast', 0.2)
        }
    }

def validate_service_config(ui_components: Dict[str, Any]) -> bool:
    """Validate UI config untuk service compatibility"""
    try:
        config = create_service_config(ui_components)
        
        # Basic validation
        if config['num_variations'] <= 0 or config['target_count'] <= 0:
            return False
        
        if not config['types']:
            return False
        
        # Range validation
        ranges = {
            'fliplr': (0.0, 1.0),
            'degrees': (0, 30),
            'translate': (0.0, 0.25),
            'scale': (0.0, 0.25),
            'hsv_h': (0.0, 0.05),
            'hsv_s': (0.0, 1.0),
            'brightness': (0.0, 0.4),
            'contrast': (0.0, 0.4)
        }
        
        # Check position ranges
        for param, (min_val, max_val) in ranges.items():
            if param in config['position']:
                value = config['position'][param]
                if not (min_val <= value <= max_val):
                    return False
        
        # Check lighting ranges
        for param, (min_val, max_val) in ranges.items():
            if param in config['lighting']:
                value = config['lighting'][param]
                if not (min_val <= value <= max_val):
                    return False
        
        return True
        
    except Exception:
        return False

def get_dataset_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get dataset status untuk check operations"""
    try:
        from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
        from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
        
        data_location = get_best_data_location()
        
        # Check raw dataset
        raw_info = detect_split_structure(data_location)
        
        # Check augmented dataset
        aug_info = detect_split_structure(f"{data_location}/augmented")
        
        # Check preprocessed dataset
        prep_info = detect_split_structure(f"{data_location}/preprocessed")
        
        return {
            'data_location': data_location,
            'raw': raw_info,
            'augmented': aug_info,
            'preprocessed': prep_info,
            'ready_for_augmentation': raw_info['status'] == 'success' and raw_info.get('total_images', 0) > 0
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'ready_for_augmentation': False
        }