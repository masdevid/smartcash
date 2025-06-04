"""
File: smartcash/ui/components/check_uncheck_buttons.py
Deskripsi: Fixed check/uncheck buttons dengan proper event binding dan debugging
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Callable
from smartcash.ui.utils.constants import ICONS, COLORS

def create_check_uncheck_buttons(target_prefix: str = "package", button_width: str = '120px',
                                container_width: str = '100%', show_count: bool = True) -> Dict[str, Any]:
    """Create check all dan uncheck all buttons dengan enhanced debugging"""
    
    check_all_button = widgets.Button(
        description="✅ Check All", button_style='success', tooltip=f'Check semua {target_prefix}',
        layout=widgets.Layout(width=button_width, height='32px', margin='0 5px 0 0')
    )
    
    uncheck_all_button = widgets.Button(
        description="❌ Uncheck All", button_style='', tooltip=f'Uncheck semua {target_prefix}',
        layout=widgets.Layout(width=button_width, height='32px', margin='0 5px 0 0')
    )
    
    count_display = widgets.HTML("", layout=widgets.Layout(margin='0 0 0 10px')) if show_count else None
    
    button_list = [check_all_button, uncheck_all_button]
    count_display and button_list.append(count_display)
    
    container = widgets.HBox(button_list, layout=widgets.Layout(
        width=container_width, justify_content='flex-start', align_items='center', margin='5px 0'
    ))
    
    return {
        'container': container,
        'check_all_button': check_all_button,
        'uncheck_all_button': uncheck_all_button,
        'count_display': count_display,
        'target_prefix': target_prefix
    }

def setup_check_uncheck_handlers(ui_components: Dict[str, Any], check_uncheck_components: Dict[str, Any],
                               checkbox_filter: Callable[[str], bool] = None, 
                               update_callback: Callable[[int], None] = None) -> None:
    """Setup handlers dengan enhanced debugging dan error handling"""
    
    target_prefix = check_uncheck_components['target_prefix']
    default_filter = lambda key: key.startswith(f'{target_prefix}_') and key.endswith('_checkbox')
    checkbox_filter = checkbox_filter or default_filter
    
    # Find checkboxes dengan debugging
    all_checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
    print(f"🔍 Found {len(all_checkboxes)} checkboxes for {target_prefix}: {list(all_checkboxes.keys())[:5]}...")
    
    def check_all_handler(button=None):
        """Check semua checkboxes dengan debugging"""
        try:
            checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
            print(f"🔄 Check all: Found {len(checkboxes)} checkboxes")
            
            for key, checkbox in checkboxes.items():
                try:
                    checkbox.value = True
                except Exception as e:
                    print(f"⚠️ Error setting {key}: {str(e)}")
            
            _update_count_display(check_uncheck_components, len(checkboxes), len(checkboxes))
            update_callback and update_callback(len(checkboxes))
            print(f"✅ Check all completed: {len(checkboxes)} checkboxes")
            
        except Exception as e:
            print(f"💥 Check all error: {str(e)}")
    
    def uncheck_all_handler(button=None):
        """Uncheck semua checkboxes dengan debugging"""
        try:
            checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
            print(f"🔄 Uncheck all: Found {len(checkboxes)} checkboxes")
            
            for key, checkbox in checkboxes.items():
                try:
                    checkbox.value = False
                except Exception as e:
                    print(f"⚠️ Error unsetting {key}: {str(e)}")
            
            _update_count_display(check_uncheck_components, 0, len(checkboxes))
            update_callback and update_callback(0)
            print(f"✅ Uncheck all completed: {len(checkboxes)} checkboxes")
            
        except Exception as e:
            print(f"💥 Uncheck all error: {str(e)}")
    
    # Clear existing handlers dan bind new ones dengan debugging
    check_button = check_uncheck_components['check_all_button']
    uncheck_button = check_uncheck_components['uncheck_all_button']
    
    try:
        # Clear existing handlers
        check_button._click_handlers.callbacks.clear()
        uncheck_button._click_handlers.callbacks.clear()
        
        # Bind new handlers
        check_button.on_click(check_all_handler)
        uncheck_button.on_click(uncheck_all_handler)
        
        print(f"✅ Handlers bound for {target_prefix}: check={len(check_button._click_handlers.callbacks)}, uncheck={len(uncheck_button._click_handlers.callbacks)}")
        
    except Exception as e:
        print(f"💥 Handler binding error: {str(e)}")
    
    # Setup initial count display
    _update_initial_count(ui_components, check_uncheck_components, checkbox_filter)

def _update_count_display(check_uncheck_components: Dict[str, Any], checked_count: int, total_count: int) -> None:
    """Update count display dengan enhanced styling"""
    count_display = check_uncheck_components.get('count_display')
    if count_display:
        try:
            percentage = (checked_count / total_count * 100) if total_count > 0 else 0
            color = COLORS.get('success', '#28a745') if checked_count == total_count else COLORS.get('warning', '#ffc107') if checked_count > 0 else COLORS.get('muted', '#6c757d')
            
            count_display.value = f"""
            <span style="color: {color}; font-weight: bold; margin-left: 10px; 
                         padding: 2px 6px; background: rgba(248,249,250,0.8); 
                         border-radius: 3px; border: 1px solid {color};">
                {checked_count}/{total_count} ({percentage:.0f}%)
            </span>
            """
        except Exception as e:
            print(f"⚠️ Count display update error: {str(e)}")

def _update_initial_count(ui_components: Dict[str, Any], check_uncheck_components: Dict[str, Any], 
                         checkbox_filter: Callable[[str], bool]) -> None:
    """Update initial count dengan debugging"""
    try:
        checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
        checked_count = sum(1 for checkbox in checkboxes.values() if getattr(checkbox, 'value', False))
        
        _update_count_display(check_uncheck_components, checked_count, len(checkboxes))
        print(f"📊 Initial count: {checked_count}/{len(checkboxes)} checkboxes checked")
        
    except Exception as e:
        print(f"⚠️ Initial count error: {str(e)}")

def create_package_check_uncheck_buttons(ui_components: Dict[str, Any], show_count: bool = True) -> Dict[str, Any]:
    """Factory function untuk package-specific check/uncheck buttons dengan enhanced filter"""
    
    print(f"🚀 Creating package check/uncheck buttons...")
    
    check_uncheck_components = create_check_uncheck_buttons('package', show_count=show_count)
    
    # Enhanced package filter dengan debugging
    def package_checkbox_filter(key: str) -> bool:
        """Enhanced filter untuk package checkboxes dengan debugging"""
        # List semua potential package keys
        package_indicators = ['smartcash_core', 'notebook_deps', 'file_utils', 'yaml_parser',  # Core
                             'pytorch', 'torchvision', 'yolov5', 'ultralytics',  # ML/AI
                             'pandas', 'numpy', 'opencv', 'pillow', 'matplotlib']  # Data Processing
        
        is_package = (key.endswith('_checkbox') and 
                     (any(indicator in key for indicator in package_indicators) or
                      any(category in key for category in ['core', 'ml', 'data']) or
                      key.startswith('package_'))) and key != 'auto_analyze_checkbox'
        
        return is_package
    
    # Debug: Find all potential checkboxes
    all_keys = list(ui_components.keys())
    checkbox_keys = [k for k in all_keys if k.endswith('_checkbox')]
    package_keys = [k for k in checkbox_keys if package_checkbox_filter(k)]
    
    print(f"🔍 All checkbox keys ({len(checkbox_keys)}): {checkbox_keys}")
    print(f"🎯 Package checkbox keys ({len(package_keys)}): {package_keys}")
    
    # Package-specific update callback dengan debugging
    def package_update_callback(selected_count: int):
        """Callback untuk update package count dengan debugging"""
        try:
            if 'update_package_count' in ui_components and callable(ui_components['update_package_count']):
                ui_components['update_package_count'](selected_count)
                print(f"📊 Package count callback: {selected_count}")
        except Exception as e:
            print(f"⚠️ Package callback error: {str(e)}")
    
    # Setup handlers dengan enhanced filter
    setup_check_uncheck_handlers(ui_components, check_uncheck_components, 
                                package_checkbox_filter, package_update_callback)
    
    return check_uncheck_components

def update_check_uncheck_count(check_uncheck_components: Dict[str, Any], ui_components: Dict[str, Any],
                              checkbox_filter: Callable[[str], bool] = None) -> None:
    """Manual update count display dengan debugging"""
    
    try:
        target_prefix = check_uncheck_components['target_prefix']
        default_filter = lambda key: key.startswith(f'{target_prefix}_') and key.endswith('_checkbox')
        checkbox_filter = checkbox_filter or default_filter
        
        _update_initial_count(ui_components, check_uncheck_components, checkbox_filter)
        
    except Exception as e:
        print(f"⚠️ Manual count update error: {str(e)}")

def debug_checkbox_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Debug function untuk check status semua checkboxes"""
    
    all_checkboxes = {k: v for k, v in ui_components.items() if k.endswith('_checkbox') and hasattr(v, 'value')}
    
    status = {
        'total_checkboxes': len(all_checkboxes),
        'checked_count': sum(1 for cb in all_checkboxes.values() if getattr(cb, 'value', False)),
        'checkbox_details': {}
    }
    
    for key, checkbox in all_checkboxes.items():
        try:
            status['checkbox_details'][key] = {
                'value': getattr(checkbox, 'value', None),
                'description': getattr(checkbox, 'description', 'No description'),
                'has_on_click': hasattr(checkbox, 'on_click')
            }
        except Exception as e:
            status['checkbox_details'][key] = {'error': str(e)}
    
    return status

def validate_check_uncheck_setup(ui_components: Dict[str, Any], check_uncheck_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate check/uncheck setup dengan comprehensive check"""
    
    validation = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    try:
        # Check buttons exist
        check_button = check_uncheck_components.get('check_all_button')
        uncheck_button = check_uncheck_components.get('uncheck_all_button')
        
        if not check_button or not hasattr(check_button, 'on_click'):
            validation['issues'].append("Check button invalid")
            validation['valid'] = False
        
        if not uncheck_button or not hasattr(uncheck_button, 'on_click'):
            validation['issues'].append("Uncheck button invalid")
            validation['valid'] = False
        
        # Check handler binding
        check_handlers = len(check_button._click_handlers.callbacks) if check_button else 0
        uncheck_handlers = len(uncheck_button._click_handlers.callbacks) if uncheck_button else 0
        
        validation['stats'].update({
            'check_handlers': check_handlers,
            'uncheck_handlers': uncheck_handlers,
            'has_count_display': 'count_display' in check_uncheck_components
        })
        
        if check_handlers == 0 or uncheck_handlers == 0:
            validation['issues'].append(f"Missing handlers: check={check_handlers}, uncheck={uncheck_handlers}")
            validation['valid'] = False
        
        # Check target checkboxes
        target_prefix = check_uncheck_components.get('target_prefix', 'package')
        
        if target_prefix == 'package':
            package_checkboxes = debug_checkbox_status(ui_components)
            validation['stats']['package_checkboxes'] = package_checkboxes['total_checkboxes']
            validation['stats']['checked_packages'] = package_checkboxes['checked_count']
            
            if package_checkboxes['total_checkboxes'] == 0:
                validation['issues'].append("No package checkboxes found")
                validation['valid'] = False
        
    except Exception as e:
        validation['issues'].append(f"Validation error: {str(e)}")
        validation['valid'] = False
    
    return validation

# One-liner utilities untuk debugging
get_checkbox_count = lambda ui_components: len([k for k in ui_components.keys() if k.endswith('_checkbox')])
get_package_checkbox_count = lambda ui_components: len([k for k in ui_components.keys() if k.endswith('_checkbox') and any(cat in k for cat in ['core', 'ml', 'data'])])
debug_handlers = lambda check_uncheck_components: (check_uncheck_components.get('check_all_button', {}).get('_click_handlers', {}).get('callbacks', []), check_uncheck_components.get('uncheck_all_button', {}).get('_click_handlers', {}).get('callbacks', []))
force_check_all = lambda ui_components: [setattr(v, 'value', True) for k, v in ui_components.items() if k.endswith('_checkbox') and hasattr(v, 'value')]
force_uncheck_all = lambda ui_components: [setattr(v, 'value', False) for k, v in ui_components.items() if k.endswith('_checkbox') and hasattr(v, 'value')]