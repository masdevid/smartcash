"""
File: smartcash/ui/components/check_uncheck_buttons.py
Deskripsi: Komponen check/uncheck all buttons yang reusable dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Callable
from smartcash.ui.utils.constants import ICONS, COLORS

def create_check_uncheck_buttons(target_prefix: str = "package", button_width: str = '120px',
                                container_width: str = '100%', show_count: bool = True) -> Dict[str, Any]:
    """Create check all dan uncheck all buttons dengan one-liner style"""
    
    check_all_button = widgets.Button(
        description="✅ Check All", button_style='success', tooltip=f'Check semua {target_prefix}',
        layout=widgets.Layout(width=button_width, height='32px', margin='0 5px 0 0')
    )
    
    uncheck_all_button = widgets.Button(
        description="❌ Uncheck All", button_style='', tooltip=f'Uncheck semua {target_prefix}',
        layout=widgets.Layout(width=button_width, height='32px', margin='0 5px 0 0')
    )
    
    # Count display jika diminta
    count_display = widgets.HTML("", layout=widgets.Layout(margin='0 0 0 10px')) if show_count else None
    
    # Container dengan button arrangement
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
    """Setup handlers untuk check/uncheck buttons dengan flexible filtering"""
    
    target_prefix = check_uncheck_components['target_prefix']
    default_filter = lambda key: key.startswith(f'{target_prefix}_') and key.endswith('_checkbox')
    checkbox_filter = checkbox_filter or default_filter
    
    def check_all_handler(button=None):
        """Check semua checkboxes dengan filter - one-liner"""
        checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
        [setattr(checkbox, 'value', True) for checkbox in checkboxes.values()]
        _update_count_display(check_uncheck_components, len(checkboxes), len(checkboxes))
        update_callback and update_callback(len(checkboxes))
    
    def uncheck_all_handler(button=None):
        """Uncheck semua checkboxes dengan filter - one-liner"""
        checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
        [setattr(checkbox, 'value', False) for checkbox in checkboxes.values()]
        _update_count_display(check_uncheck_components, 0, len(checkboxes))
        update_callback and update_callback(0)
    
    # Bind handlers
    check_uncheck_components['check_all_button'].on_click(check_all_handler)
    check_uncheck_components['uncheck_all_button'].on_click(uncheck_all_handler)
    
    # Setup initial count display
    _update_initial_count(ui_components, check_uncheck_components, checkbox_filter)

def _update_count_display(check_uncheck_components: Dict[str, Any], checked_count: int, total_count: int) -> None:
    """Update count display dengan one-liner formatting"""
    count_display = check_uncheck_components.get('count_display')
    if count_display:
        percentage = (checked_count / total_count * 100) if total_count > 0 else 0
        count_display.value = f"""
        <span style="color: {COLORS.get('success', '#28a745')}; font-weight: bold; margin-left: 10px;">
            {checked_count}/{total_count} ({percentage:.0f}%)
        </span>
        """

def _update_initial_count(ui_components: Dict[str, Any], check_uncheck_components: Dict[str, Any], 
                         checkbox_filter: Callable[[str], bool]) -> None:
    """Update initial count display berdasarkan current state"""
    checkboxes = {k: v for k, v in ui_components.items() if checkbox_filter(k) and hasattr(v, 'value')}
    checked_count = sum(1 for checkbox in checkboxes.values() if getattr(checkbox, 'value', False))
    _update_count_display(check_uncheck_components, checked_count, len(checkboxes))

def create_package_check_uncheck_buttons(ui_components: Dict[str, Any], show_count: bool = True) -> Dict[str, Any]:
    """Factory function untuk package-specific check/uncheck buttons"""
    
    check_uncheck_components = create_check_uncheck_buttons('package', show_count=show_count)
    
    # Package-specific filter
    def package_checkbox_filter(key: str) -> bool:
        """Filter untuk package checkboxes - one-liner pattern matching"""
        return (key.endswith('_checkbox') and 
                any(category in key for category in ['core', 'ml', 'data', 'ui', 'dev']) and
                key != 'auto_analyze_checkbox')
    
    # Package-specific update callback
    def package_update_callback(selected_count: int):
        """Callback untuk update package count di UI lain jika ada"""
        if 'update_package_count' in ui_components and callable(ui_components['update_package_count']):
            ui_components['update_package_count'](selected_count)
    
    # Setup handlers dengan package-specific logic
    setup_check_uncheck_handlers(ui_components, check_uncheck_components, 
                                package_checkbox_filter, package_update_callback)
    
    return check_uncheck_components

def create_category_check_uncheck_buttons(ui_components: Dict[str, Any], category_name: str) -> Dict[str, Any]:
    """Factory function untuk category-specific check/uncheck buttons"""
    
    check_uncheck_components = create_check_uncheck_buttons(f'{category_name}_package', button_width='100px')
    
    # Category-specific filter
    def category_checkbox_filter(key: str) -> bool:
        """Filter untuk category-specific checkboxes - one-liner"""
        return key.startswith(f'{category_name}_') and key.endswith('_checkbox')
    
    setup_check_uncheck_handlers(ui_components, check_uncheck_components, category_checkbox_filter)
    
    return check_uncheck_components

def update_check_uncheck_count(check_uncheck_components: Dict[str, Any], ui_components: Dict[str, Any],
                              checkbox_filter: Callable[[str], bool] = None) -> None:
    """Manual update count display jika checkbox values berubah dari external"""
    
    target_prefix = check_uncheck_components['target_prefix']
    default_filter = lambda key: key.startswith(f'{target_prefix}_') and key.endswith('_checkbox')
    checkbox_filter = checkbox_filter or default_filter
    
    _update_initial_count(ui_components, check_uncheck_components, checkbox_filter)

def create_smart_check_uncheck_buttons(ui_components: Dict[str, Any], 
                                     smart_filters: Dict[str, Callable[[str], bool]] = None) -> Dict[str, Any]:
    """Create smart check/uncheck buttons dengan multiple filter options"""
    
    smart_filters = smart_filters or {
        'all': lambda key: key.endswith('_checkbox'),
        'packages': lambda key: 'package' in key and key.endswith('_checkbox'),
        'settings': lambda key: 'setting' in key and key.endswith('_checkbox')
    }
    
    # Create dropdown untuk filter selection
    filter_dropdown = widgets.Dropdown(
        options=list(smart_filters.keys()),
        value='packages',
        description='Target:',
        style={'description_width': '50px'},
        layout=widgets.Layout(width='120px', margin='0 10px 0 0')
    )
    
    check_uncheck_components = create_check_uncheck_buttons('smart', button_width='100px')
    
    def smart_check_handler(button=None):
        """Smart check dengan dynamic filter - one-liner"""
        current_filter = smart_filters[filter_dropdown.value]
        checkboxes = {k: v for k, v in ui_components.items() if current_filter(k) and hasattr(v, 'value')}
        [setattr(checkbox, 'value', True) for checkbox in checkboxes.values()]
        _update_count_display(check_uncheck_components, len(checkboxes), len(checkboxes))
    
    def smart_uncheck_handler(button=None):
        """Smart uncheck dengan dynamic filter - one-liner"""
        current_filter = smart_filters[filter_dropdown.value]
        checkboxes = {k: v for k, v in ui_components.items() if current_filter(k) and hasattr(v, 'value')}
        [setattr(checkbox, 'value', False) for checkbox in checkboxes.values()]
        _update_count_display(check_uncheck_components, 0, len(checkboxes))
    
    # Update container dengan dropdown
    new_container = widgets.HBox([
        filter_dropdown,
        check_uncheck_components['check_all_button'],
        check_uncheck_components['uncheck_all_button'],
        check_uncheck_components['count_display']
    ], layout=check_uncheck_components['container'].layout)
    
    check_uncheck_components['container'] = new_container
    check_uncheck_components['filter_dropdown'] = filter_dropdown
    
    # Bind smart handlers
    check_uncheck_components['check_all_button'].on_click(smart_check_handler)
    check_uncheck_components['uncheck_all_button'].on_click(smart_uncheck_handler)
    
    # Update count saat filter berubah
    def on_filter_change(change):
        """Update count saat filter berubah - one-liner"""
        current_filter = smart_filters[change['new']]
        checkboxes = {k: v for k, v in ui_components.items() if current_filter(k) and hasattr(v, 'value')}
        checked_count = sum(1 for checkbox in checkboxes.values() if getattr(checkbox, 'value', False))
        _update_count_display(check_uncheck_components, checked_count, len(checkboxes))
    
    filter_dropdown.observe(on_filter_change, names='value')
    
    return check_uncheck_components

# Utility functions untuk integration dengan existing components
def add_check_uncheck_to_existing_ui(ui_components: Dict[str, Any], insert_after_key: str = None,
                                   target_type: str = 'package') -> None:
    """Add check/uncheck buttons ke existing UI layout"""
    
    if target_type == 'package':
        check_uncheck_components = create_package_check_uncheck_buttons(ui_components)
    elif target_type == 'smart':
        check_uncheck_components = create_smart_check_uncheck_buttons(ui_components)
    else:
        check_uncheck_components = create_check_uncheck_buttons(target_type)
        setup_check_uncheck_handlers(ui_components, check_uncheck_components)
    
    # Add ke ui_components
    ui_components.update({
        'check_uncheck_container': check_uncheck_components['container'],
        'check_all_button': check_uncheck_components['check_all_button'],
        'uncheck_all_button': check_uncheck_components['uncheck_all_button'],
        'check_count_display': check_uncheck_components.get('count_display')
    })
    
    # Update main UI jika ada insert point
    if insert_after_key and insert_after_key in ui_components:
        # Logic untuk insert ke UI layout bisa ditambahkan sesuai kebutuhan
        pass

# One-liner utilities
get_checked_count = lambda ui_components, filter_fn: sum(1 for k, v in ui_components.items() if filter_fn(k) and hasattr(v, 'value') and v.value)
get_total_checkboxes = lambda ui_components, filter_fn: len([k for k in ui_components.keys() if filter_fn(k) and hasattr(ui_components[k], 'value')])
toggle_all_checkboxes = lambda ui_components, filter_fn, state: [setattr(v, 'value', state) for k, v in ui_components.items() if filter_fn(k) and hasattr(v, 'value')]