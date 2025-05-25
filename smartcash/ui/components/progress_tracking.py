"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Fixed progress tracking dengan visible progress bars dan tqdm compatibility
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union
from tqdm.auto import tqdm
import time
import threading
from IPython.display import HTML, display

def create_progress_tracking_container() -> Dict[str, Any]:
    """
    Buat progress tracking container dengan visible progress bars.
    
    Returns:
        Dictionary berisi semua komponen progress dengan control methods
    """
    # Progress status HTML widget
    status_widget = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='5px 0', visibility='visible')
    )
    
    # Container untuk tqdm bars
    tqdm_container = widgets.Output(
        layout=widgets.Layout(margin='5px 0', visibility='visible')
    )
    
    # Main container yang ALWAYS visible
    container = widgets.VBox([
        widgets.HTML("<h4>📊 Progress</h4>"),
        status_widget,
        tqdm_container
    ], layout=widgets.Layout(
        margin='10px 0', 
        padding='10px', 
        display='block',  # Always block display
        visibility='visible'  # Always visible
    ))
    
    # Progress state management
    progress_state = {
        'overall_bar': None,
        'step_bar': None, 
        'current_bar': None,
        'active_bars': set(),
        'operation_type': None,
        'container': container,
        'status_widget': status_widget,
        'tqdm_container': tqdm_container
    }
    
    # Create control methods
    components = {
        'container': container,
        'progress_container': container,
        'status_widget': status_widget,
        'tqdm_container': tqdm_container,
        '_progress_state': progress_state
    }
    
    components.update(_create_tqdm_control_methods(progress_state))
    
    return components

def _create_tqdm_control_methods(state: Dict[str, Any]) -> Dict[str, Any]:
    """Buat control methods untuk tqdm progress management."""
    
    def show_container():
        """Show progress container - always visible."""
        state['container'].layout.display = 'block'
        state['container'].layout.visibility = 'visible'
        state['status_widget'].layout.visibility = 'visible'
        state['tqdm_container'].layout.visibility = 'visible'
    
    def hide_container():
        """Hide progress container."""
        state['container'].layout.display = 'none'
        state['container'].layout.visibility = 'hidden'
        _cleanup_all_bars(state)
    
    def show_for_operation(operation: str):
        """Show progress untuk operation dengan visible tqdm bars."""
        show_container()
        state['operation_type'] = operation
        
        # Clear existing bars
        _cleanup_all_bars(state)
        
        # Configuration per operation
        operation_configs = {
            'download': {'overall': True, 'step': True, 'current': True},
            'check': {'overall': True, 'step': False, 'current': False},
            'cleanup': {'overall': True, 'step': True, 'current': False},
            'save': {'overall': False, 'step': False, 'current': False},
            'all': {'overall': True, 'step': True, 'current': True}
        }
        
        config = operation_configs.get(operation, operation_configs['all'])
        
        # Create bars dengan proper visibility
        with state['tqdm_container']:
            if config['overall']:
                state['overall_bar'] = tqdm(
                    total=100, 
                    desc="📊 Overall", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='blue', 
                    position=0,
                    leave=True  # Keep bar visible
                )
                state['active_bars'].add('overall')
            
            if config['step']:
                state['step_bar'] = tqdm(
                    total=100, 
                    desc="🔄 Step", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='green', 
                    position=1,
                    leave=True  # Keep bar visible
                )
                state['active_bars'].add('step')
            
            if config['current']:
                state['current_bar'] = tqdm(
                    total=100, 
                    desc="⚡ Current", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='orange', 
                    position=2,
                    leave=True  # Keep bar visible
                )
                state['active_bars'].add('current')
    
    def update_progress(progress_type: str, value: int, message: str = "", color_style: str = None):
        """Update progress dengan visible tqdm bars dan bounds checking."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        value = max(0, min(100, int(value))) if value is not None else 0
        bar_key = f'{progress_type}_bar'
        
        if bar_key in state and state[bar_key] is not None:
            bar = state[bar_key]
            
            # Reset dan update bar value
            bar.reset(total=100)
            bar.update(value)
            
            # Update description dengan message
            if message:
                emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                emoji = emoji_map.get(progress_type, '📊')
                bar.set_description(f"{emoji} {message}")
            
            # Force refresh
            bar.refresh()
        
        # Update status widget jika ada message
        if message:
            _update_status_message(state, message, color_style)
    
    def complete_operation(message: str = "Selesai"):
        """Complete operation dengan persistent success state."""
        # Set semua active bars ke 100%
        for bar_type in state['active_bars']:
            bar_key = f'{bar_type}_bar'
            if state.get(bar_key):
                bar = state[bar_key]
                bar.reset(total=100)
                bar.update(100)
                
                emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                emoji = emoji_map.get(bar_type, '📊')
                bar.set_description(f"✅ {emoji} {message}")
                bar.refresh()
        
        _update_status_message(state, f"✅ {message}", 'success')
        
        # Delayed cleanup untuk visual feedback
        def delayed_cleanup():
            time.sleep(4)
            if state['active_bars']:
                _cleanup_all_bars(state)
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
    
    def error_operation(message: str = "Error"):
        """Set error state untuk progress."""
        for bar_type in state['active_bars']:
            bar_key = f'{bar_type}_bar'
            if state.get(bar_key):
                bar = state[bar_key]
                
                emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                emoji = emoji_map.get(bar_type, '📊')
                bar.set_description(f"❌ {emoji} {message}")
                bar.colour = 'red'
                bar.refresh()
        
        _update_status_message(state, f"❌ {message}", 'error')
    
    def reset_all():
        """Reset semua progress bars."""
        _cleanup_all_bars(state)
        state['operation_type'] = None
        # Don't hide container, keep it visible
    
    return {
        'show_container': show_container,
        'hide_container': hide_container,
        'show_for_operation': show_for_operation,
        'update_progress': update_progress,
        'complete_operation': complete_operation,
        'error_operation': error_operation,
        'reset_all': reset_all
    }

def _cleanup_all_bars(state: Dict[str, Any]) -> None:
    """Cleanup semua tqdm bars dengan proper closing."""
    bars_to_close = ['overall_bar', 'step_bar', 'current_bar']
    
    for bar_key in bars_to_close:
        if bar_key in state and state[bar_key] is not None:
            try:
                state[bar_key].close()
            except Exception:
                pass
            state[bar_key] = None
    
    state['active_bars'].clear()
    
    # Clear tqdm container output
    if 'tqdm_container' in state:
        state['tqdm_container'].clear_output(wait=True)

def _update_status_message(state: Dict[str, Any], message: str, style: str = None) -> None:
    """Update status message widget dengan colors."""
    if 'status_widget' not in state:
        return
    
    color_map = {
        'success': '#28a745',
        'info': '#007bff', 
        'warning': '#ffc107',
        'error': '#dc3545',
        'danger': '#dc3545'
    }
    
    color = color_map.get(style, '#495057')
    
    html_content = f"""
    <div style="color: {color}; font-size: 14px; font-weight: 500; margin: 5px 0;">
        {message}
    </div>
    """
    
    state['status_widget'].value = html_content
    state['status_widget'].layout.visibility = 'visible'

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
    """Show progress untuk operation type."""
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