"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: Enhanced UI utilities dengan unified logging dan parameter extraction yang DRY
"""

from typing import Dict, Any, List
from IPython.display import display, HTML

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Unified logging ke UI dengan fallback chain yang robust"""
    try:
        # Priority 1: UI Logger dengan level method
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Priority 2: Log widget dengan HTML styling
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545', 'debug': '#6c757d'}
            color = color_map.get(level, '#007bff')
            
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px; font-family: monospace;">{message}</div>'
            
            with widget:
                display(HTML(html))
            return
            
    except Exception:
        pass
    
    # Fallback print dengan emoji indicators
    emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌', 'debug': '🔍'}
    emoji = emoji_map.get(level, 'ℹ️')
    print(f"{emoji} {message}")

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Backward compatibility untuk log_to_accordion pattern"""
    log_to_ui(ui_components, message, level)

def get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe widget value extraction dengan type preservation"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = getattr(widget, 'value')
            # Type preservation untuk common types
            if isinstance(default, bool) and not isinstance(value, bool):
                return bool(value)
            elif isinstance(default, (int, float)) and isinstance(value, (int, float)):
                return type(default)(value)
            return value
        except Exception:
            pass
    return default

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Extract augmentation types dengan validation dan fallback"""
    types_widget = ui_components.get('augmentation_types')
    if types_widget and hasattr(types_widget, 'value'):
        try:
            value = getattr(types_widget, 'value')
            if isinstance(value, (list, tuple)) and value:
                return list(value)
        except Exception:
            pass
    
    # Fallback ke combined jika tidak ada selection
    return ['combined']

def validate_form_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate form inputs dengan detailed feedback"""
    validation_result = {'valid': True, 'errors': [], 'warnings': []}
    
    # Basic validation rules
    validations = [
        ('num_variations', lambda x: 1 <= x <= 10, "Jumlah variasi harus antara 1-10"),
        ('target_count', lambda x: 100 <= x <= 2000, "Target count harus antara 100-2000"),
        ('output_prefix', lambda x: x and len(x.strip()) > 0, "Output prefix tidak boleh kosong"),
        ('augmentation_types', lambda x: x and len(x) > 0, "Pilih minimal 1 jenis augmentasi")
    ]
    
    for key, validator, error_msg in validations:
        value = get_widget_value_safe(ui_components, key)
        try:
            if not validator(value):
                validation_result['valid'] = False
                validation_result['errors'].append(f"❌ {error_msg}")
        except Exception:
            validation_result['valid'] = False
            validation_result['errors'].append(f"❌ Error validating {key}")
    
    # Advanced validations dengan warnings
    fliplr = get_widget_value_safe(ui_components, 'fliplr', 0.5)
    if fliplr > 0.8:
        validation_result['warnings'].append("⚠️ Flip probability tinggi (>80%) - mungkin terlalu agresif")
    
    brightness = get_widget_value_safe(ui_components, 'brightness', 0.2)
    contrast = get_widget_value_safe(ui_components, 'contrast', 0.2)
    if brightness > 0.3 or contrast > 0.3:
        validation_result['warnings'].append("⚠️ Brightness/Contrast tinggi - hasil mungkin tidak realistis")
    
    return validation_result

def update_button_states(ui_components: Dict[str, Any], state: str = 'ready'):
    """Update button states dengan visual feedback yang konsisten"""
    button_configs = {
        'ready': {'text_suffix': '', 'style': 'primary', 'disabled': False},
        'processing': {'text_suffix': ' (Processing...)', 'style': 'warning', 'disabled': True},
        'error': {'text_suffix': ' (Error)', 'style': 'danger', 'disabled': False},
        'success': {'text_suffix': ' (✓)', 'style': 'success', 'disabled': False}
    }
    
    config = button_configs.get(state, button_configs['ready'])
    button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            # Preserve original text jika belum ada
            if not hasattr(button, '_original_description'):
                button._original_description = button.description
            
            # Update properties
            button.disabled = config['disabled']
            if hasattr(button, 'button_style'):
                button.button_style = config['style'] if state != 'ready' else getattr(button, '_original_style', 'primary')
            if config['text_suffix']:
                button.description = button._original_description + config['text_suffix']
            else:
                button.description = button._original_description

def clear_ui_outputs(ui_components: Dict[str, Any]):
    """Clear semua output widgets dengan safe error handling"""
    output_keys = ['log_output', 'status', 'confirmation_area']
    for key in output_keys:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            try:
                widget.clear_output(wait=True)
            except Exception:
                pass

def show_validation_errors(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Show validation errors dengan styled formatting"""
    if not validation_result.get('errors') and not validation_result.get('warnings'):
        return
    
    error_html = ""
    
    # Errors dengan red styling
    for error in validation_result.get('errors', []):
        error_html += f'<div style="color: #dc3545; margin: 2px 0; padding: 4px;">{error}</div>'
    
    # Warnings dengan orange styling
    for warning in validation_result.get('warnings', []):
        error_html += f'<div style="color: #ffc107; margin: 2px 0; padding: 4px;">{warning}</div>'
    
    # Display dalam confirmation area atau fallback ke log
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output()
        with confirmation_area:
            display(HTML(error_html))
    else:
        log_to_ui(ui_components, "Validation issues detected - check form inputs", 'warning')

def handle_ui_error(ui_components: Dict[str, Any], error_message: str, show_in_confirmation: bool = True):
    """Handle UI errors dengan consistent styling dan placement"""
    error_html = f"""
    <div style="padding: 10px; background-color: #f8d7da; color: #721c24; 
                border-radius: 4px; margin: 5px 0; border-left: 4px solid #dc3545;">
        <strong>❌ Error:</strong> {error_message}
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
    """Create HTML summary dari config untuk display"""
    aug_config = config.get('augmentation', {})
    
    summary_items = [
        f"🎯 Variations: {aug_config.get('num_variations', 3)}",
        f"📊 Target Count: {aug_config.get('target_count', 500)}",
        f"🔄 Types: {', '.join(aug_config.get('types', ['combined']))}",
        f"📂 Split: {aug_config.get('target_split', 'train')}",
        f"⚖️ Balance Classes: {'Yes' if aug_config.get('balance_classes', True) else 'No'}"
    ]
    
    return f"""
    <div style="padding: 8px; background-color: #e3f2fd; border-radius: 4px; margin: 5px 0;">
        <strong>📋 Config Summary:</strong><br>
        {' | '.join(summary_items)}
    </div>
    """

def show_config_summary(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Show config summary dalam confirmation area"""
    summary_html = create_config_summary_html(config)
    
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        with confirmation_area:
            display(HTML(summary_html))
    else:
        log_to_ui(ui_components, "Config summary ready - check form values", 'info')

# One-liner utilities untuk common operations
safe_get_value = lambda ui_components, key, default=None: get_widget_value_safe(ui_components, key, default)
safe_log = lambda ui_components, msg, level='info': log_to_ui(ui_components, msg, level)
clear_outputs = lambda ui_components: clear_ui_outputs(ui_components)
update_buttons = lambda ui_components, state: update_button_states(ui_components, state)
show_error = lambda ui_components, msg: handle_ui_error(ui_components, msg)
validate_form = lambda ui_components: validate_form_inputs(ui_components)

# Progress utilities yang reusable
show_progress_safe = lambda ui_components, operation: ui_components.get('progress_tracker', {}).get('show_container', lambda x: None)(operation)
update_progress_safe = lambda ui_components, level, pct, msg: ui_components.get('progress_tracker', {}).get('update_progress', lambda *args: None)(level, pct, msg)
complete_progress_safe = lambda ui_components, msg: ui_components.get('progress_tracker', {}).get('complete_operation', lambda x: None)(msg)
error_progress_safe = lambda ui_components, msg: ui_components.get('progress_tracker', {}).get('error_operation', lambda x: None)(msg)