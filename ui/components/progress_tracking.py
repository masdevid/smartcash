"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Enhanced progress tracking dengan dynamic color dan visibility controls
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union

def create_progress_tracking_container() -> Dict[str, Any]:
    """
    Buat progress tracking container dengan kontrol dinamis.
    
    Returns:
        Dictionary berisi semua komponen progress dengan control methods
    """
    # Overall progress (primary)
    overall_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Overall:',
        bar_style='info',
        layout=widgets.Layout(width='100%', height='20px', visibility='hidden')
    )
    
    overall_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='2px 0', visibility='hidden')
    )
    
    # Step progress
    step_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Step:',
        bar_style='info',
        layout=widgets.Layout(width='100%', height='20px', visibility='hidden')
    )
    
    step_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='2px 0', visibility='hidden')
    )
    
    # Current progress (detailed)
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Current:',
        bar_style='info',
        layout=widgets.Layout(width='100%', height='20px', visibility='hidden')
    )
    
    current_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='2px 0', visibility='hidden')
    )
    
    # Container
    container = widgets.VBox([
        widgets.HTML("<h4>📊 Progress</h4>"),
        overall_progress, overall_label,
        step_progress, step_label,
        current_progress, current_label
    ], layout=widgets.Layout(margin='10px 0', padding='10px', display='none'))
    
    components = {
        'container': container,
        'overall_progress': overall_progress,
        'overall_label': overall_label,
        'step_progress': step_progress,
        'step_label': step_label,
        'current_progress': current_progress,
        'current_label': current_label,
        'progress_bar': overall_progress,  # Alias untuk backward compatibility
    }
    
    # Add control methods
    components.update(_create_control_methods(components))
    
    return components

def _create_control_methods(components: Dict[str, Any]) -> Dict[str, Any]:
    """Buat control methods untuk dynamic management."""
    
    def show_container():
        """Show progress container."""
        components['container'].layout.display = 'block'
        components['container'].layout.visibility = 'visible'
    
    def hide_container():
        """Hide progress container."""
        components['container'].layout.display = 'none'
        components['container'].layout.visibility = 'hidden'
    
    def set_visibility(progress_type: str, visible: bool):
        """Set visibility untuk specific progress type."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        progress_key = f'{progress_type}_progress'
        label_key = f'{progress_type}_label'
        
        visibility = 'visible' if visible else 'hidden'
        display = 'block' if visible else 'none'
        
        if progress_key in components:
            components[progress_key].layout.visibility = visibility
            components[progress_key].layout.display = display
        if label_key in components:
            components[label_key].layout.visibility = visibility
            components[label_key].layout.display = display
    
    def set_color(progress_type: str, color_style: str):
        """Set color untuk specific progress type."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        progress_key = f'{progress_type}_progress'
        valid_styles = ['info', 'success', 'warning', 'danger', '']
        
        if progress_key in components and color_style in valid_styles:
            components[progress_key].bar_style = color_style
    
    def update_progress(progress_type: str, value: int, message: str = "", color_style: str = None):
        """Update progress dengan optional color dan message."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        progress_key = f'{progress_type}_progress'
        label_key = f'{progress_type}_label'
        
        # Update value
        if progress_key in components:
            components[progress_key].value = max(0, min(100, value))
            
            # Update color jika disediakan
            if color_style:
                set_color(progress_type, color_style)
        
        # Update label
        if label_key in components and message:
            components[label_key].value = f"<div style='color: #495057; font-size: 13px;'>{message}</div>"
    
    def reset_all():
        """Reset semua progress bars."""
        for progress_type in ['overall', 'step', 'current']:
            update_progress(progress_type, 0, "", 'info')
            set_visibility(progress_type, False)
        hide_container()
    
    def show_for_operation(operation: str):
        """Show progress bars sesuai operation type."""
        operation_configs = {
            'download': {'overall': True, 'step': True, 'current': False},
            'check': {'overall': True, 'step': False, 'current': False},
            'cleanup': {'overall': True, 'step': False, 'current': True},
            'save': {'overall': False, 'step': False, 'current': False},
            'all': {'overall': True, 'step': True, 'current': True}
        }
        
        config = operation_configs.get(operation, operation_configs['all'])
        
        show_container()
        for progress_type, should_show in config.items():
            set_visibility(progress_type, should_show)
            if should_show:
                set_color(progress_type, 'info')
    
    def complete_operation(message: str = "Selesai"):
        """Complete operation dengan success state."""
        for progress_type in ['overall', 'step', 'current']:
            progress_key = f'{progress_type}_progress'
            if (progress_key in components and 
                components[progress_key].layout.visibility == 'visible'):
                update_progress(progress_type, 100, message, 'success')
    
    def error_operation(message: str = "Error"):
        """Set error state untuk visible progress bars."""
        for progress_type in ['overall', 'step', 'current']:
            progress_key = f'{progress_type}_progress'
            if (progress_key in components and 
                components[progress_key].layout.visibility == 'visible'):
                update_progress(progress_type, 0, f"❌ {message}", 'danger')
    
    return {
        'show_container': show_container,
        'hide_container': hide_container,
        'set_visibility': set_visibility,
        'set_color': set_color,
        'update_progress': update_progress,
        'reset_all': reset_all,
        'show_for_operation': show_for_operation,
        'complete_operation': complete_operation,
        'error_operation': error_operation
    }

# Legacy compatibility functions
def update_overall_progress(ui_components: Dict[str, Any], progress: int, total: int, message: str):
    """Legacy function untuk backward compatibility."""
    if 'update_progress' in ui_components:
        percentage = int((progress / max(total, 1)) * 100)
        ui_components['update_progress']('overall', percentage, message)

def update_step_progress(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str):
    """Legacy function untuk backward compatibility."""
    if 'update_progress' in ui_components:
        percentage = int((step / max(total_steps, 1)) * 100)
        ui_components['update_progress']('step', percentage, f"Step {step}/{total_steps}: {step_name}")

def update_current_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Legacy function untuk backward compatibility."""
    if 'update_progress' in ui_components:
        percentage = int((current / max(total, 1)) * 100)
        ui_components['update_progress']('current', percentage, message)

def show_progress_for_operation(ui_components: Dict[str, Any], operation: str):
    """Show progress sesuai operation type."""
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation'](operation)

def complete_progress_operation(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress operation."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](message)

def error_progress_operation(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state untuk progress."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)