"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: Updated UI utilities dengan HSV support dan enhanced validation
"""

from typing import Dict, Any, Union, List
from IPython.display import display, clear_output
import ipywidgets as widgets
import logging

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Enhanced logging dengan fallback chain"""
    try:
        # Priority 1: UI Logger
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Priority 2: Log widget dengan enhanced styling
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            color_map = {
                'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 
                'error': '#dc3545', 'debug': '#6c757d'
            }
            color = color_map.get(level, '#007bff')
            emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌', 'debug': '🔍'}
            emoji = emoji_map.get(level, 'ℹ️')
            
            html = f"""
            <div style="color: {color}; margin: 2px 0; padding: 4px; 
                        font-family: monospace; font-size: 13px;
                        border-left: 3px solid {color}; padding-left: 8px;">
                {emoji} {message}
            </div>
            """
            
            with widget:
                display(HTML(html))
            return
            
    except Exception:
        pass

# DEPRECATED: log_to_accordion has been removed as UILogger already handles log redirection to accordion
# Use UILogger directly for all logging needs:
# logger = ui_components.get('logger')
# if logger:
#     logger.info/error/warning/success(message)
#
# This function is kept as a stub for backward compatibility but will be removed in future versions
def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """DEPRECATED: Use UILogger directly instead"""
    # Get logger from ui_components
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        # Use UILogger directly
        getattr(logger, level)(message)
        return True
        
    # Fallback to log_to_ui if logger not available
    log_to_ui(ui_components, message, level)
    return False

def get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Enhanced safe widget value extraction dengan backend compatibility"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = getattr(widget, 'value')
            
            # Enhanced type preservation dan validation
            if isinstance(default, bool) and not isinstance(value, bool):
                return bool(value)
            elif isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            elif isinstance(default, list) and not isinstance(value, list):
                return [value] if value else []
            elif isinstance(default, str) and not isinstance(value, str):
                return str(value) if value is not None else default
            
            return value
        except Exception:
            pass
    return default

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Enhanced augmentation types extraction dengan validation"""
    types_widget = ui_components.get('augmentation_types')
    if types_widget and hasattr(types_widget, 'value'):
        try:
            value = getattr(types_widget, 'value')
            if isinstance(value, (list, tuple)) and value:
                # Validate tegen available types
                valid_types = ['combined', 'position', 'lighting', 'geometric', 'color', 'noise']
                filtered_types = [t for t in value if t in valid_types]
                return filtered_types if filtered_types else ['combined']
        except Exception:
            pass
    
    # Fallback ke combined jika tidak ada selection valid
    return ['combined']

def validate_form_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced form validation dengan HSV parameters dan cleanup target"""
    validation_result = {'valid': True, 'errors': [], 'warnings': [], 'backend_compatible': True}
    
    # Basic validation rules dengan enhanced ranges
    validations = [
        ('num_variations', lambda x: 1 <= x <= 10, "Jumlah variasi harus antara 1-10"),
        ('target_count', lambda x: 100 <= x <= 2000, "Target count harus antara 100-2000"),
        ('cleanup_target', lambda x: x in ['augmented', 'samples', 'both'], "Cleanup target harus valid"),
        ('augmentation_types', lambda x: x and len(x) > 0, "Pilih minimal 1 jenis augmentasi"),
        ('target_split', lambda x: x in ['train', 'valid', 'test'], "Target split harus train, valid, atau test")
    ]
    
    for key, validator, error_msg in validations:
        value = get_widget_value_safe(ui_components, key)
        try:
            if not validator(value):
                validation_result['valid'] = False
                validation_result['backend_compatible'] = False
                validation_result['errors'].append(f"❌ {error_msg}")
        except Exception:
            validation_result['valid'] = False
            validation_result['backend_compatible'] = False
            validation_result['errors'].append(f"❌ Error validating {key}")
    
    # Advanced parameter validation dengan HSV
    _validate_position_parameters(ui_components, validation_result)
    _validate_lighting_parameters_with_hsv(ui_components, validation_result)
    _validate_cleanup_parameters(ui_components, validation_result)
    
    # Backend compatibility check
    if validation_result['valid']:
        _check_backend_compatibility(ui_components, validation_result)
    
    return validation_result

def _validate_position_parameters(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate position parameters dengan realistic ranges"""
    fliplr = get_widget_value_safe(ui_components, 'fliplr', 0.5)
    degrees = get_widget_value_safe(ui_components, 'degrees', 12)
    translate = get_widget_value_safe(ui_components, 'translate', 0.08)
    scale = get_widget_value_safe(ui_components, 'scale', 0.04)
    
    if fliplr > 0.8:
        validation_result['warnings'].append("⚠️ Flip probability sangat tinggi (>80%) - hasil mungkin tidak natural")
    
    if degrees > 20:
        validation_result['warnings'].append("⚠️ Rotasi >20° mungkin terlalu ekstrem untuk uang kertas")
    
    if translate > 0.15 or scale > 0.15:
        validation_result['warnings'].append("⚠️ Translate/Scale >15% mungkin mengubah proporsi terlalu drastis")

def _validate_lighting_parameters_with_hsv(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate lighting parameters dengan HSV support"""
    brightness = get_widget_value_safe(ui_components, 'brightness', 0.2)
    contrast = get_widget_value_safe(ui_components, 'contrast', 0.15)
    hsv_h = get_widget_value_safe(ui_components, 'hsv_h', 10)
    hsv_s = get_widget_value_safe(ui_components, 'hsv_s', 15)
    
    if brightness > 0.3 or contrast > 0.3:
        validation_result['warnings'].append("⚠️ Brightness/Contrast >30% mungkin menghasilkan gambar tidak realistis")
    
    if brightness < 0.05 and contrast < 0.05:
        validation_result['warnings'].append("⚠️ Variasi pencahayaan sangat rendah - augmentasi mungkin tidak efektif")
    
    # HSV validation
    if hsv_h > 25:
        validation_result['warnings'].append("⚠️ HSV Hue >25 mungkin mengubah warna terlalu drastis")
    
    if hsv_s > 40:
        validation_result['warnings'].append("⚠️ HSV Saturation >40 mungkin menghasilkan warna tidak natural")

def _validate_cleanup_parameters(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Validate cleanup parameters"""
    cleanup_target = get_widget_value_safe(ui_components, 'cleanup_target', 'both')
    target_split = get_widget_value_safe(ui_components, 'target_split', 'train')
    
    if cleanup_target == 'both':
        validation_result['warnings'].append("⚠️ Cleanup 'both' akan menghapus semua file augmented dan samples")
    
    if target_split == 'test' and cleanup_target in ['augmented', 'both']:
        validation_result['warnings'].append("⚠️ Cleanup pada test split tidak direkomendasikan")

def _check_backend_compatibility(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Check backend service compatibility"""
    try:
        backend_ready = ui_components.get('backend_ready', False)
        service_integration = ui_components.get('service_integration', False)
        
        if not (backend_ready and service_integration):
            validation_result['warnings'].append("⚠️ Backend service tidak sepenuhnya terintegrasi")
            validation_result['backend_compatible'] = False
    except Exception:
        validation_result['backend_compatible'] = False

def update_button_states(ui_components: Dict[str, Any], state: str = 'ready'):
    """Enhanced button state management dengan backend integration"""
    button_configs = {
        'ready': {'text_suffix': '', 'style': 'primary', 'disabled': False},
        'processing': {'text_suffix': ' 🔄', 'style': 'warning', 'disabled': True},
        'error': {'text_suffix': ' ❌', 'style': 'danger', 'disabled': False},
        'success': {'text_suffix': ' ✅', 'style': 'success', 'disabled': False},
        'validating': {'text_suffix': ' 🔍', 'style': 'info', 'disabled': True}
    }
    
    config = button_configs.get(state, button_configs['ready'])
    button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button', 'generate_button']
    
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            # Preserve original properties
            if not hasattr(button, '_original_description'):
                button._original_description = button.description
                button._original_style = getattr(button, 'button_style', 'primary')
            
            # Update properties
            button.disabled = config['disabled']
            
            if state == 'ready':
                button.description = button._original_description
                button.button_style = button._original_style
            else:
                button.description = button._original_description + config['text_suffix']
                if hasattr(button, 'button_style'):
                    button.button_style = config['style']

def clear_ui_outputs(ui_components: Dict[str, Any]):
    """Enhanced output clearing dengan logging"""
    output_keys = ['log_output', 'status', 'confirmation_area']
    cleared_count = 0
    
    for key in output_keys:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            try:
                widget.clear_output(wait=True)
                cleared_count += 1
            except Exception:
                pass
    
    # Log clearing action
    if cleared_count > 0:
        log_to_ui(ui_components, f"🧹 UI outputs cleared ({cleared_count} widgets)", "info")

def show_validation_errors(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Enhanced validation error display dengan backend integration"""
    if not validation_result.get('errors') and not validation_result.get('warnings'):
        return
    
    # Build comprehensive error message
    error_sections = []
    
    if validation_result.get('errors'):
        error_sections.append("<strong style='color: #dc3545;'>❌ Errors:</strong>")
        for error in validation_result['errors']:
            error_sections.append(f"<div style='margin-left: 15px; color: #dc3545;'>{error}</div>")
    
    if validation_result.get('warnings'):
        error_sections.append("<strong style='color: #ffc107;'>⚠️ Warnings:</strong>")
        for warning in validation_result['warnings']:
            error_sections.append(f"<div style='margin-left: 15px; color: #ffc107;'>{warning}</div>")
    
    # Backend compatibility status
    if not validation_result.get('backend_compatible', True):
        error_sections.append("<strong style='color: #17a2b8;'>ℹ️ Backend Status:</strong>")
        error_sections.append("<div style='margin-left: 15px; color: #17a2b8;'>Fallback mode - beberapa fitur mungkin terbatas</div>")
    
    error_html = f"""
    <div style="padding: 12px; background-color: #f8f9fa; border-radius: 6px; 
                margin: 8px 0; border-left: 4px solid #dc3545;">
        <h6 style="margin: 0 0 8px 0; color: #333;">🔍 Form Validation Results</h6>
        {"<br>".join(error_sections)}
    </div>
    """
    
    # Display dalam confirmation area atau fallback ke log
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output()
        with confirmation_area:
            display(HTML(error_html))
    else:
        log_to_ui(ui_components, "Validation issues detected - check form inputs", 'warning')

def handle_ui_error(ui_components: Dict[str, Any], error_message: str, show_in_confirmation: bool = True):
    """Enhanced error handling dengan logging"""
    error_html = f"""
    <div style="padding: 12px; background-color: #f8d7da; color: #721c24; 
                border-radius: 6px; margin: 8px 0; border-left: 4px solid #dc3545;">
        <strong>❌ Error:</strong> {error_message}
        <br><small style="color: #856404;">💡 Coba refresh cell atau check console untuk detail</small>
    </div>
    """
    
    if show_in_confirmation:
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output()
            with confirmation_area:
                display(HTML(error_html))
            return
    
    # Fallback ke log_to_ui
    log_to_ui(ui_components, error_message, 'error')

def create_config_summary_html(config: Dict[str, Any]) -> str:
    """Enhanced config summary dengan HSV dan cleanup target"""
    aug_config = config.get('augmentation', {})
    cleanup_config = config.get('cleanup', {})
    backend_config = config.get('backend', {})
    
    summary_items = [
        f"🎯 Variations: {aug_config.get('num_variations', 2)}",
        f"📊 Target Count: {aug_config.get('target_count', 500)}",
        f"🔄 Types: {', '.join(aug_config.get('types', ['combined']))}",
        f"📂 Split: {aug_config.get('target_split', 'train')}",
        f"⚖️ Balance Classes: {'Yes' if aug_config.get('balance_classes', True) else 'No'}",
        f"🧹 Cleanup: {cleanup_config.get('default_target', 'both')}",
        f"🔧 Backend: {'Enabled' if backend_config.get('service_enabled', True) else 'Disabled'}"
    ]
    
    return f"""
    <div style="padding: 10px; background-color: #e3f2fd; border-radius: 6px; 
                margin: 8px 0; border-left: 4px solid #2196f3;">
        <strong style="color: #1976d2;">📋 Configuration Summary:</strong><br>
        <div style="margin-top: 6px; font-size: 13px;">
            {' | '.join(summary_items)}
        </div>
    </div>
    """

def show_config_summary(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Enhanced config summary display"""
    summary_html = create_config_summary_html(config)
    
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        with confirmation_area:
            display(HTML(summary_html))
    else:
        log_to_ui(ui_components, "Config summary ready - check form values", 'info')

def update_preview_status(ui_components: Dict[str, Any], status: str, message: str):
    """Update preview status dengan styling"""
    preview_status = ui_components.get('preview_status')
    if preview_status:
        status_colors = {'generating': '#007bff', 'success': '#28a745', 'error': '#dc3545', 'info': '#17a2b8'}
        emoji_map = {'generating': '🔄', 'success': '✅', 'error': '❌', 'info': 'ℹ️'}
        color = status_colors.get(status, '#666')
        emoji = emoji_map.get(status, 'ℹ️')
        preview_status.value = f"<div style='text-align: center; color: {color}; font-size: 12px; margin: 4px 0;'>{emoji} {message}</div>"

def load_preview_image(ui_components: Dict[str, Any], image_path: str = '/data/aug_preview.jpg'):
    """Load preview image dari file path"""
    try:
        import os
        
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            preview_image = ui_components.get('preview_image')
            if preview_image and hasattr(preview_image, 'value'):
                preview_image.value = image_data
                update_preview_status(ui_components, 'success', 'Preview loaded')
                return True
        else:
            update_preview_status(ui_components, 'error', 'Preview file not found')
            return False
            
    except Exception as e:
        update_preview_status(ui_components, 'error', f'Load error: {str(e)}')
        return False

# Enhanced one-liner utilities dengan HSV dan preview support
safe_get_value = lambda ui_components, key, default=None: get_widget_value_safe(ui_components, key, default)
safe_log = lambda ui_components, msg, level='info': log_to_ui(ui_components, msg, level)
clear_outputs = lambda ui_components: clear_ui_outputs(ui_components)
update_buttons = lambda ui_components, state: update_button_states(ui_components, state)
show_error = lambda ui_components, msg: handle_ui_error(ui_components, msg)
validate_form = lambda ui_components: validate_form_inputs(ui_components)
extract_types = lambda ui_components: extract_augmentation_types(ui_components)
update_preview = lambda ui_components, status, msg: update_preview_status(ui_components, status, msg)
load_preview = lambda ui_components, path='/data/aug_preview.jpg': load_preview_image(ui_components, path)

# Enhanced progress utilities dengan backend integration
show_progress_safe = lambda ui_components, operation: _safe_progress_operation(ui_components, 'show', operation)
update_progress_safe = lambda ui_components, level, pct, msg: _safe_progress_operation(ui_components, 'update', level, pct, msg)
complete_progress_safe = lambda ui_components, msg: _safe_progress_operation(ui_components, 'complete', msg)
error_progress_safe = lambda ui_components, msg: _safe_progress_operation(ui_components, 'error', msg)

def _safe_progress_operation(ui_components: Dict[str, Any], operation: str, *args):
    """Safe progress operation dengan logging"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            return
        
        # Calculate percentage
        if operation == 'show':
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show()
        elif operation == 'update':
            if len(args) >= 3:  # level, pct, msg
                level, pct, msg = args[0], args[1], args[2]
                if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(pct, msg)
                elif level == 'step' and hasattr(progress_tracker, 'update_step'):
                    progress_tracker.update_step(pct, msg)
                elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(pct, msg)
        elif operation in ['complete', 'error']:
            method = getattr(progress_tracker, operation, None)
            if method:
                method(args[0] if args else '')
        
        # Log operation
        if operation in ['complete', 'error']:
            level = 'success' if operation == 'complete' else 'error'
            log_to_ui(ui_components, args[0] if args else f'Operation {operation}', level)
                
    except Exception:
        pass  # Silent fail untuk compatibility