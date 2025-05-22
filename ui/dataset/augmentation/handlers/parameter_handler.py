"""
File: smartcash/ui/dataset/augmentation/handlers/parameter_handler.py
Deskripsi: Handler untuk ekstraksi dan validasi parameter augmentasi (SRP)
"""

from typing import Dict, Any, Tuple, List

def extract_and_validate_parameters(ui_components: Dict[str, Any], ui_logger) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Ekstrak dan validasi parameter augmentasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        ui_logger: UI Logger bridge
        
    Returns:
        Tuple (is_valid, error_message, validated_params)
    """
    try:
        ui_logger.info("📋 Mengekstrak parameter dari UI...")
        
        # Extract parameter menggunakan helper
        params = _extract_parameters_from_ui(ui_components)
        
        # Validasi parameter
        is_valid, error_message = _validate_parameters(params, ui_logger)
        
        if is_valid:
            ui_logger.success("✅ Parameter validation berhasil")
            return True, "", params
        else:
            return False, error_message, {}
            
    except Exception as e:
        error_msg = f"Error ekstraksi parameter: {str(e)}"
        ui_logger.error(f"❌ {error_msg}")
        return False, error_msg, {}

def _extract_parameters_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstrak parameter dari UI components."""
    # Helper untuk mendapatkan nilai widget
    def get_widget_value(keys: List[str], default):
        for key in keys:
            if key in ui_components and hasattr(ui_components[key], 'value'):
                return ui_components[key].value
        return default
    
    # Ekstrak parameter dasar
    params = {
        'types': _extract_augmentation_types(ui_components),
        'split': get_widget_value(['target_split', 'split_target', 'split'], 'train'),
        'num_variations': get_widget_value(['num_variations', 'variations'], 2),
        'target_count': get_widget_value(['target_count', 'count'], 500),
        'output_prefix': get_widget_value(['output_prefix', 'prefix'], 'aug_'),
        'balance_classes': get_widget_value(['balance_classes'], False),
        'validate_results': get_widget_value(['validate_results'], True),
        'process_bboxes': True
    }
    
    # Ekstrak path configuration
    params.update({
        'data_dir': ui_components.get('data_dir', 'data'),
        'augmented_dir': ui_components.get('augmented_dir', 'data/augmented'),
        'output_dir': ui_components.get('output_dir', 'data/augmented')
    })
    
    return params

def _extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Ekstrak jenis augmentasi dari UI."""
    # Cek berbagai kemungkinan nama widget
    for key in ['augmentation_types', 'aug_types', 'types']:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            aug_types = ui_components[key].value
            if isinstance(aug_types, (tuple, list)):
                return list(aug_types)
            elif aug_types:
                return [aug_types]
    
    # Default fallback
    return ['combined']

def _validate_parameters(params: Dict[str, Any], ui_logger) -> Tuple[bool, str]:
    """Validasi parameter augmentasi."""
    # Validasi jenis augmentasi
    if not params.get('types'):
        return False, "Jenis augmentasi tidak boleh kosong"
    
    # Validasi num_variations
    if params.get('num_variations', 0) <= 0:
        return False, "Jumlah variasi harus lebih dari 0"
    
    if params.get('num_variations', 0) > 10:
        return False, "Jumlah variasi maksimal 10"
    
    # Validasi target_count
    if params.get('target_count', 0) <= 0:
        return False, "Target count harus lebih dari 0"
    
    # Validasi output_prefix
    prefix = params.get('output_prefix', '')
    if not prefix or not str(prefix).strip():
        return False, "Output prefix tidak boleh kosong"
    
    # Validasi split
    if params.get('split') not in ['train', 'valid', 'test']:
        return False, f"Split tidak valid: {params.get('split')}"
    
    # Log parameter yang berhasil divalidasi
    ui_logger.debug(f"📊 Types: {params['types']}")
    ui_logger.debug(f"📂 Split: {params['split']}")
    ui_logger.debug(f"🔢 Variations: {params['num_variations']}")
    ui_logger.debug(f"🎯 Target: {params['target_count']}")
    
    return True, ""