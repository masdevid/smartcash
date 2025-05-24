"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Fixed progress tracking dengan enhanced tqdm integration dan proper visibility control
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union
from tqdm.auto import tqdm
import time
from IPython.display import HTML, display

def create_progress_tracking_container() -> Dict[str, Any]:
    """
    Buat progress tracking container dengan enhanced tqdm dan proper visibility control.
    
    Returns:
        Dictionary berisi semua komponen progress dengan control methods
    """
    # Progress status HTML widget
    status_widget = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='5px 0', visibility='hidden')
    )
    
    # Container untuk tqdm bars
    tqdm_container = widgets.Output(
        layout=widgets.Layout(margin='5px 0', visibility='hidden')
    )
    
    # Main container dengan improved visibility control
    container = widgets.VBox([
        widgets.HTML("<h4>📊 Progress</h4>"),
        status_widget,
        tqdm_container
    ], layout=widgets.Layout(
        margin='10px 0', 
        padding='10px', 
        display='none',  # Start hidden
        visibility='hidden'
    ))
    
    # Enhanced progress state management
    progress_state = {
        'overall_bar': None,
        'step_bar': None, 
        'current_bar': None,
        'active_bars': set(),
        'operation_type': None,
        'container': container,
        'status_widget': status_widget,
        'tqdm_container': tqdm_container,
        'is_visible': False,
        'last_update_time': 0
    }
    
    # Create enhanced control methods
    components = {
        'container': container,
        'progress_container': container,  # Alias untuk compatibility
        'status_widget': status_widget,
        'tqdm_container': tqdm_container,
        '_progress_state': progress_state
    }
    
    components.update(_create_enhanced_control_methods(progress_state))
    
    return components

def _create_enhanced_control_methods(state: Dict[str, Any]) -> Dict[str, Any]:
    """Buat enhanced control methods dengan proper error handling."""
    
    def show_container():
        """Show progress container dengan proper visibility control."""
        try:
            state['container'].layout.display = 'block'
            state['container'].layout.visibility = 'visible'
            state['is_visible'] = True
        except Exception:
            pass
    
    def hide_container():
        """Hide progress container dan cleanup bars."""
        try:
            state['container'].layout.display = 'none'
            state['container'].layout.visibility = 'hidden'
            state['is_visible'] = False
            _cleanup_all_bars(state)
        except Exception:
            pass
    
    def show_for_operation(operation: str):
        """Show progress sesuai operation type dengan enhanced bar configuration."""
        try:
            show_container()
            state['operation_type'] = operation
            
            # Clear existing bars
            _cleanup_all_bars(state)
            
            # Enhanced configuration per operation
            operation_configs = {
                'download': {'overall': True, 'step': True, 'current': True},  # Full 3-level
                'check': {'overall': True, 'step': False, 'current': False},   # Simple check
                'cleanup': {'overall': True, 'step': True, 'current': True},   # Full 3-level  
                'save': {'overall': False, 'step': False, 'current': False},   # No progress
                'all': {'overall': True, 'step': True, 'current': True}        # Full 3-level
            }
            
            config = operation_configs.get(operation, operation_configs['all'])
            
            # Create bars sesuai config dengan error handling
            with state['tqdm_container']:
                if config['overall']:
                    try:
                        state['overall_bar'] = tqdm(
                            total=100, desc="📊 Overall", 
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                            colour='blue', position=0, leave=True
                        )
                        state['active_bars'].add('overall')
                    except Exception:
                        pass
                
                if config['step']:
                    try:
                        state['step_bar'] = tqdm(
                            total=100, desc="🔄 Step", 
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                            colour='green', position=1, leave=True
                        )
                        state['active_bars'].add('step')
                    except Exception:
                        pass
                
                if config['current']:
                    try:
                        state['current_bar'] = tqdm(
                            total=100, desc="⚡ Current", 
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                            colour='orange', position=2, leave=True
                        )
                        state['active_bars'].add('current')
                    except Exception:
                        pass
        except Exception:
            # Fallback: tetap show container meski ada error
            show_container()
    
    def update_progress(progress_type: str, value: int, message: str = "", color_style: str = None):
        """Update progress dengan enhanced error handling dan throttling."""
        try:
            # Throttling untuk prevent excessive updates
            current_time = time.time()
            if current_time - state.get('last_update_time', 0) < 0.1:  # Max 10 updates per second
                return
            state['last_update_time'] = current_time
            
            if progress_type not in ['overall', 'step', 'current']:
                return
            
            # Ensure container is visible
            if not state.get('is_visible', False):
                show_container()
                
            value = max(0, min(100, int(value)))
            bar_key = f'{progress_type}_bar'
            
            if bar_key in state and state[bar_key] is not None:
                try:
                    bar = state[bar_key]
                    
                    # Update bar value dengan smooth transition
                    current_value = getattr(bar, 'n', 0)
                    if value > current_value:
                        diff = value - current_value
                        bar.update(diff)
                    elif value < current_value:
                        # Reset dan set ke nilai baru
                        bar.reset(total=100)
                        if value > 0:
                            bar.update(value)
                    
                    # Update description dengan message
                    if message:
                        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                        emoji = emoji_map.get(progress_type, '📊')
                        bar.set_description(f"{emoji} {message}")
                        
                except Exception:
                    pass
            
            # Update status widget jika ada message
            if message:
                _update_status_message(state, message, color_style)
                
        except Exception:
            # Silent fail untuk prevent disruption
            pass
    
    def complete_operation(message: str = "Selesai"):
        """Complete operation dengan enhanced success state."""
        try:
            # Set semua active bars ke 100% dengan success message
            for bar_type in list(state['active_bars']):
                bar_key = f'{bar_type}_bar'
                if state.get(bar_key):
                    try:
                        bar = state[bar_key]
                        bar.n = 100
                        bar.refresh()
                        
                        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                        emoji = emoji_map.get(bar_type, '📊')
                        bar.set_description(f"✅ {emoji} {message}")
                        bar.colour = 'green'
                    except Exception:
                        pass
            
            _update_status_message(state, f"✅ {message}", 'success')
            
            # Delayed cleanup dengan enhanced approach
            def delayed_cleanup():
                try:
                    time.sleep(2)  # Show success state for 2 seconds
                    if state.get('is_visible', False):  # Only cleanup if still visible
                        _cleanup_all_bars(state)
                except Exception:
                    pass
            
            # Use thread-safe approach for Colab
            import threading
            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()
            
        except Exception:
            pass
    
    def error_operation(message: str = "Error"):
        """Set error state dengan enhanced error display."""
        try:
            # Set semua active bars ke error state
            for bar_type in list(state['active_bars']):
                bar_key = f'{bar_type}_bar'
                if state.get(bar_key):
                    try:
                        bar = state[bar_key]
                        
                        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
                        emoji = emoji_map.get(bar_type, '📊')
                        bar.set_description(f"❌ {emoji} {message}")
                        bar.colour = 'red'
                        bar.refresh()
                    except Exception:
                        pass
            
            _update_status_message(state, f"❌ {message}", 'error')
        except Exception:
            pass
    
    def reset_all():
        """Reset semua progress bars dan hide container."""
        try:
            _cleanup_all_bars(state)
            hide_container()
            state['operation_type'] = None
        except Exception:
            pass
    
    def debug_state():
        """Debug function untuk troubleshooting progress state."""
        return {
            'is_visible': state.get('is_visible', False),
            'operation_type': state.get('operation_type'),
            'active_bars': list(state.get('active_bars', set())),
            'container_display': getattr(state['container'].layout, 'display', 'unknown'),
            'container_visibility': getattr(state['container'].layout, 'visibility', 'unknown')
        }
    
    return {
        'show_container': show_container,
        'hide_container': hide_container,
        'show_for_operation': show_for_operation,
        'update_progress': update_progress,
        'complete_operation': complete_operation,
        'error_operation': error_operation,
        'reset_all': reset_all,
        'debug_state': debug_state
    }

def _cleanup_all_bars(state: Dict[str, Any]) -> None:
    """Cleanup semua tqdm bars dengan enhanced error handling."""
    bars_to_close = ['overall_bar', 'step_bar', 'current_bar']
    
    for bar_key in bars_to_close:
        if bar_key in state and state[bar_key] is not None:
            try:
                bar = state[bar_key]
                if hasattr(bar, 'close'):
                    bar.close()
            except Exception:
                pass
            finally:
                state[bar_key] = None
    
    state['active_bars'].clear()
    
    # Clear tqdm container output
    if 'tqdm_container' in state:
        try:
            state['tqdm_container'].clear_output(wait=True)
        except Exception:
            pass

def _update_status_message(state: Dict[str, Any], message: str, style: str = None) -> None:
    """Update status message widget dengan enhanced styling."""
    if 'status_widget' not in state:
        return
    
    try:
        # Color mapping
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
    except Exception:
        pass

# Legacy compatibility functions untuk backward compatibility
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