"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Enhanced progress tracking dengan full width bars (100% width, 20px height) dan context-aware display
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union
from tqdm.auto import tqdm
import time
import threading
from IPython.display import HTML, display

def create_progress_tracking_container() -> Dict[str, Any]:
    """Create enhanced progress tracking dengan full width bars dan context-aware display."""
    
    # Progress status dengan enhanced styling
    status_widget = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='8px 0', visibility='visible', width='100%')
    )
    
    # Container untuk tqdm bars dengan full width
    tqdm_container = widgets.Output(
        layout=widgets.Layout(
            margin='5px 0', 
            visibility='visible',
            width='100%',
            max_width='100%'
        )
    )
    
    # Main container dengan enhanced styling
    container = widgets.VBox([
        widgets.HTML("<h4 style='color: #333; margin: 10px 0 5px 0;'>📊 Progress</h4>"),
        status_widget,
        tqdm_container
    ], layout=widgets.Layout(
        margin='10px 0', 
        padding='15px', 
        display='block',
        border='1px solid #28a745',
        border_radius='8px',
        background_color='#f8fff8',
        width='100%',
        max_width='100%',
        overflow='hidden'
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
    
    components.update(_create_enhanced_control_methods(progress_state))
    
    return components

def _create_enhanced_control_methods(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced control methods dengan context-aware progress bars."""
    
    def show_container():
        """Show progress container."""
        state['container'].layout.display = 'block'
        state['container'].layout.visibility = 'visible'
    
    def hide_container():
        """Hide progress container."""
        state['container'].layout.display = 'none'
        state['container'].layout.visibility = 'hidden'
        _cleanup_all_bars(state)
    
    def show_for_operation(operation: str):
        """Show progress dengan context-aware bars (1 bar untuk cleanup/check, 2-3 untuk preprocessing)."""
        show_container()
        state['operation_type'] = operation
        
        # Clear existing bars
        _cleanup_all_bars(state)
        
        # Context-aware operation configurations
        operation_configs = {
            'check': {'bars': ['overall']},  # Single bar untuk check
            'cleanup': {'bars': ['overall']},  # Single bar untuk cleanup
            'preprocessing': {'bars': ['overall', 'step']},  # 2 bars untuk preprocessing
            'download': {'bars': ['overall', 'step', 'current']},  # 3 bars untuk download (kompleks)
            'augmentation': {'bars': ['overall', 'step']},  # 2 bars untuk augmentation
            'training': {'bars': ['overall', 'step', 'current']},  # 3 bars untuk training
        }
        
        config = operation_configs.get(operation, {'bars': ['overall']})
        
        # Create context-aware bars dengan full width styling
        with state['tqdm_container']:
            if 'overall' in config['bars']:
                state['overall_bar'] = tqdm(
                    total=100, 
                    desc="📊 Overall Progress",
                    bar_format='{desc}\n{percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='#28a745', 
                    position=0,
                    ncols=80,  # Full width
                    ascii=False, 
                    mininterval=0.05,
                    maxinterval=0.1, 
                    smoothing=0.3
                )
                state['active_bars'].add('overall')
            
            if 'step' in config['bars']:
                state['step_bar'] = tqdm(
                    total=100, 
                    desc="🔄 Step Progress",
                    bar_format='{desc}\n{percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='#17a2b8', 
                    position=1,
                    ncols=80,  # Full width
                    ascii=False, 
                    mininterval=0.05,
                    maxinterval=0.1, 
                    smoothing=0.3
                )
                state['active_bars'].add('step')
            
            if 'current' in config['bars']:
                state['current_bar'] = tqdm(
                    total=100, 
                    desc="⚡ Current Operation",
                    bar_format='{desc}\n{percentage:3.0f}%|{bar}| {n}/{total}',
                    colour='#ffc107', 
                    position=2,
                    ncols=80,  # Full width
                    ascii=False, 
                    mininterval=0.05,
                    maxinterval=0.1, 
                    smoothing=0.3
                )
                state['active_bars'].add('current')
    
    def update_progress(progress_type: str, value: int, message: str = "", color_style: str = None):
        """Update progress dengan enhanced full width bars."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        value = max(0, min(100, value))
        bar_key = f'{progress_type}_bar'
        
        if bar_key in state and state[bar_key] is not None:
            bar = state[bar_key]
            
            # Smooth progress updates
            diff = value - bar.n
            if diff > 0:
                bar.update(diff)
            elif diff < 0:
                bar.reset(total=100)
                bar.update(value)
            
            # Update description dengan message
            if message:
                emoji_map = {
                    'overall': '📊', 
                    'step': '🔄', 
                    'current': '⚡'
                }
                emoji = emoji_map.get(progress_type, '📊')
                bar.set_description(f"{emoji} {message}")
        
        # Update status widget
        if message:
            _update_status_message(state, message, color_style)
    
    def complete_operation(message: str = "Selesai"):
        """Complete operation dengan green success bars."""
        # Set all bars to 100% dengan green success
        for bar_type in state['active_bars']:
            bar_key = f'{bar_type}_bar'
            if state.get(bar_key):
                bar = state[bar_key]
                bar.n = 100
                bar.colour = '#28a745'  # Green
                bar.refresh()
                
                emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                emoji = emoji_map.get(bar_type, '📊')
                bar.set_description(f"✅ {emoji} {message}")
        
        _update_status_message(state, f"✅ {message}", 'success')
        
        # Delayed cleanup dengan longer delay untuk better visual feedback
        def delayed_cleanup():
            time.sleep(4)
            if state['active_bars']:
                _cleanup_all_bars(state)
        
        threading.Thread(target=delayed_cleanup, daemon=True).start()
    
    def error_operation(message: str = "Error"):
        """Set error state dengan red bars."""
        for bar_type in state['active_bars']:
            bar_key = f'{bar_type}_bar'
            if state.get(bar_key):
                bar = state[bar_key]
                
                emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                emoji = emoji_map.get(bar_type, '📊')
                bar.set_description(f"❌ {emoji} {message}")
                bar.colour = '#dc3545'  # Red
                bar.refresh()
        
        _update_status_message(state, f"❌ {message}", 'error')
    
    def reset_all():
        """Reset all progress dan hide container."""
        _cleanup_all_bars(state)
        hide_container()
        state['operation_type'] = None
    
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
    """Cleanup all tqdm bars dengan proper error handling."""
    bars_to_close = ['overall_bar', 'step_bar', 'current_bar']
    
    for bar_key in bars_to_close:
        if bar_key in state and state[bar_key] is not None:
            try:
                state[bar_key].close()
            except Exception:
                pass
            state[bar_key] = None
    
    state['active_bars'].clear()
    
    if 'tqdm_container' in state:
        state['tqdm_container'].clear_output(wait=True)

def _update_status_message(state: Dict[str, Any], message: str, style: str = None) -> None:
    """Update status message dengan enhanced styling."""
    if 'status_widget' not in state:
        return
    
    color_map = {
        'success': '#28a745',
        'info': '#007bff', 
        'warning': '#ffc107',
        'error': '#dc3545'
    }
    
    color = color_map.get(style, '#495057')
    
    html_content = f"""
    <div style="color: {color}; font-size: 14px; font-weight: 500; margin: 8px 0; 
                padding: 10px; background: rgba(40, 167, 69, 0.1); 
                border-radius: 6px; border-left: 4px solid {color};
                width: 100%; box-sizing: border-box;">
        {message}
    </div>
    """
    
    state['status_widget'].value = html_content
    state['status_widget'].layout.visibility = 'visible'

# Legacy compatibility functions
def update_overall_progress(ui_components: Dict[str, Any], progress: int, total: int, message: str):
    """Legacy compatibility function."""
    if 'update_progress' in ui_components:
        percentage = int((progress / max(total, 1)) * 100)
        ui_components['update_progress']('overall', percentage, message)

def update_step_progress(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str):
    """Legacy compatibility function."""
    if 'update_progress' in ui_components:
        percentage = int((step / max(total_steps, 1)) * 100)
        ui_components['update_progress']('step', percentage, f"Step {step}/{total_steps}: {step_name}")

def update_current_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Legacy compatibility function."""
    if 'update_progress' in ui_components:
        percentage = int((current / max(total, 1)) * 100)
        ui_components['update_progress']('current', percentage, message)

def show_progress_for_operation(ui_components: Dict[str, Any], operation: str):
    """Show progress for operation."""
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation'](operation)

def complete_progress_operation(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress operation."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](message)

def error_progress_operation(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state for progress."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)