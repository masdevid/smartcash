"""
File: smartcash/ui/components/progress_tracker.py
Deskripsi: Optimized progress tracker dengan enhanced flexibility dan backward compatibility
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable, Union, List
from IPython.display import display, HTML

def create_progress_tracker(tracker_type: str = "single", height: str = "120px", width: str = "100%", 
                          show_percentage: bool = True, show_eta: bool = False, 
                          auto_hide_delay: float = 3.0, theme: str = "default") -> Dict[str, Any]:
    """
    Create flexible progress tracker dengan multiple configurations
    
    Args:
        tracker_type: 'single', 'dual', 'multi' untuk different progress tracking modes
        height: Height of progress container
        width: Width of progress container
        show_percentage: Show percentage display
        show_eta: Show estimated time of arrival
        auto_hide_delay: Auto hide delay after completion (0 = no auto hide)
        theme: 'default', 'minimal', 'detailed' styling themes
    """
    
    if tracker_type == "dual":
        return _create_dual_progress_tracker(height, width, show_percentage, show_eta, auto_hide_delay, theme)
    elif tracker_type == "multi":
        return _create_multi_progress_tracker(height, width, show_percentage, show_eta, auto_hide_delay, theme)
    else:
        return _create_single_progress_tracker(height, width, show_percentage, show_eta, auto_hide_delay, theme)

def _create_single_progress_tracker(height: str, width: str, show_percentage: bool, 
                                   show_eta: bool, auto_hide_delay: float, theme: str) -> Dict[str, Any]:
    """Create single progress tracker dengan enhanced features"""
    
    # Theme configurations
    themes = {
        "default": {"bg_color": "#e9ecef", "bar_color": "#007bff", "text_color": "#333"},
        "minimal": {"bg_color": "#f8f9fa", "bar_color": "#28a745", "text_color": "#666"},
        "detailed": {"bg_color": "#e9ecef", "bar_color": "#6f42c1", "text_color": "#212529"}
    }
    current_theme = themes.get(theme, themes["default"])
    
    # Progress bar
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Progress:',
                                     bar_style='', style={'bar_color': current_theme["bar_color"]},
                                     layout=widgets.Layout(width='100%', margin='2px 0'))
    
    # Progress message
    progress_message = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='2px 0'))
    
    # Optional components
    components = [progress_bar, progress_message]
    
    if show_percentage:
        percentage_display = widgets.HTML(value="0%", layout=widgets.Layout(width='60px', text_align='right'))
        progress_row = widgets.HBox([progress_bar, percentage_display], 
                                   layout=widgets.Layout(width='100%', align_items='center'))
        components = [progress_row, progress_message]
    
    if show_eta:
        eta_display = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='2px 0'))
        components.append(eta_display)
    
    # Container
    container = widgets.VBox(components, layout=widgets.Layout(width=width, height=height, 
                                                              visibility='hidden', padding='10px',
                                                              border='1px solid #ddd', border_radius='4px'))
    
    # State tracking
    state = {'visible': False, 'start_time': None, 'current_operation': None}
    
    def show_for_operation(operation_name: str) -> None:
        """Show tracker untuk operation"""
        import time
        state.update({'visible': True, 'start_time': time.time(), 'current_operation': operation_name})
        container.layout.visibility = 'visible'
        progress_bar.value = 0
        progress_bar.bar_style = ''
        _update_display(0, f"Memulai {operation_name}...")
    
    def update_progress(category: str, value: int, message: str = "") -> None:
        """Update progress dengan enhanced display"""
        if not state['visible']: return
        
        progress_bar.value = min(max(value, 0), 100)
        _update_display(value, message)
        
        if show_eta and state['start_time']:
            _update_eta_display(value)
    
    def complete_operation(final_message: str) -> None:
        """Complete operation dengan success styling"""
        if not state['visible']: return
        
        progress_bar.value = 100
        progress_bar.bar_style = 'success'
        _update_display(100, final_message, "success")
        
        if auto_hide_delay > 0:
            import threading
            threading.Timer(auto_hide_delay, reset_all).start()
    
    def error_operation(error_message: str) -> None:
        """Handle error dengan error styling"""
        if not state['visible']: return
        
        progress_bar.bar_style = 'danger'
        _update_display(progress_bar.value, error_message, "error")
        
        if auto_hide_delay > 0:
            import threading
            threading.Timer(auto_hide_delay, reset_all).start()
    
    def reset_all() -> None:
        """Reset tracker ke initial state"""
        state.update({'visible': False, 'start_time': None, 'current_operation': None})
        container.layout.visibility = 'hidden'
        progress_bar.value = 0
        progress_bar.bar_style = ''
        _update_display(0, "")
    
    def _update_display(value: int, message: str, msg_type: str = "info") -> None:
        """Update display components"""
        colors = {"info": current_theme["text_color"], "success": "#28a745", "error": "#dc3545"}
        color = colors.get(msg_type, colors["info"])
        
        progress_message.value = f"<div style='color: {color}; font-size: 14px;'>{message}</div>"
        
        if show_percentage and 'percentage_display' in locals():
            percentage_display.value = f"<div style='color: {color}; font-weight: bold;'>{value}%</div>"
    
    def _update_eta_display(current_progress: int) -> None:
        """Update ETA display"""
        if not show_eta or current_progress <= 0: return
        
        import time
        elapsed = time.time() - state['start_time']
        estimated_total = (elapsed / current_progress) * 100
        eta_seconds = estimated_total - elapsed
        
        if eta_seconds > 0:
            eta_display.value = f"<div style='color: #666; font-size: 12px;'>ETA: {eta_seconds:.0f}s</div>"
    
    # Enhanced API methods
    def update_status(message: str, msg_type: str = "info") -> None:
        """Update status message tanpa mengubah progress"""
        _update_display(progress_bar.value, message, msg_type)
    
    def set_theme(new_theme: str) -> None:
        """Change theme dinamically"""
        nonlocal current_theme
        current_theme = themes.get(new_theme, themes["default"])
        progress_bar.style.bar_color = current_theme["bar_color"]
    
    return {
        'container': container, 'progress_bar': progress_bar, 'progress_message': progress_message,
        'show_for_operation': show_for_operation, 'update_progress': update_progress,
        'complete_operation': complete_operation, 'error_operation': error_operation,
        'reset_all': reset_all, 'update_status': update_status, 'set_theme': set_theme,
        'state': state, 'type': 'single'
    }

def _create_dual_progress_tracker(height: str, width: str, show_percentage: bool, 
                                 show_eta: bool, auto_hide_delay: float, theme: str) -> Dict[str, Any]:
    """Create dual progress tracker (overall + current step)"""
    
    # Create two single trackers
    overall_tracker = _create_single_progress_tracker("60px", "100%", show_percentage, False, 0, theme)
    current_tracker = _create_single_progress_tracker("60px", "100%", show_percentage, show_eta, 0, theme)
    
    # Always visible containers
    overall_tracker['container'].layout.visibility = 'visible'
    current_tracker['container'].layout.visibility = 'visible'
    
    # Combined container
    container = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; margin-bottom: 5px;'>Overall Progress</div>"),
        overall_tracker['container'],
        widgets.HTML("<div style='font-weight: bold; margin: 10px 0 5px 0;'>Current Step</div>"),
        current_tracker['container']
    ], layout=widgets.Layout(width=width, height=height, visibility='hidden',
                           padding='10px', border='1px solid #ddd', border_radius='4px'))
    
    # Dual tracker state
    dual_state = {'visible': False, 'current_operation': None}
    
    def show_for_operation(operation_name: str, **kwargs) -> None:
        """Show dual tracker dengan flexible parameters"""
        dual_state.update({'visible': True, 'current_operation': operation_name})
        container.layout.visibility = 'visible'
        overall_tracker['show_for_operation'](operation_name)
        current_tracker['reset_all']()
    
    def update_progress(category: str, value: int, message: str = "") -> None:
        """Update appropriate tracker based on category"""
        if category == "overall":
            overall_tracker['update_progress']('overall', value, message)
        else:
            current_tracker['update_progress']('current', value, message)
    
    def complete_operation(final_message: str) -> None:
        """Complete dual operation"""
        overall_tracker['complete_operation'](final_message)
        current_tracker['complete_operation']("Step completed")
        
        if auto_hide_delay > 0:
            import threading
            threading.Timer(auto_hide_delay, reset_all).start()
    
    def error_operation(error_message: str) -> None:
        """Handle dual error"""
        overall_tracker['error_operation'](error_message)
        current_tracker['error_operation']("Step failed")
    
    def reset_all() -> None:
        """Reset dual tracker"""
        dual_state.update({'visible': False, 'current_operation': None})
        container.layout.visibility = 'hidden'
        overall_tracker['reset_all']()
        current_tracker['reset_all']()
    
    return {
        'container': container, 'overall_tracker': overall_tracker, 'current_tracker': current_tracker,
        'show_for_operation': show_for_operation, 'update_progress': update_progress,
        'complete_operation': complete_operation, 'error_operation': error_operation,
        'reset_all': reset_all, 'state': dual_state, 'type': 'dual'
    }

def _create_multi_progress_tracker(height: str, width: str, show_percentage: bool, 
                                  show_eta: bool, auto_hide_delay: float, theme: str) -> Dict[str, Any]:
    """Create multi-step progress tracker"""
    
    # Step tracking
    steps_container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    current_step_tracker = _create_single_progress_tracker("60px", "100%", show_percentage, show_eta, 0, theme)
    current_step_tracker['container'].layout.visibility = 'visible'
    
    container = widgets.VBox([
        widgets.HTML("<div style='font-weight: bold; margin-bottom: 10px;'>Multi-Step Progress</div>"),
        steps_container,
        widgets.HTML("<div style='font-weight: bold; margin: 10px 0 5px 0;'>Current Step</div>"),
        current_step_tracker['container']
    ], layout=widgets.Layout(width=width, height=height, visibility='hidden',
                           padding='10px', border='1px solid #ddd', border_radius='4px'))
    
    # Multi tracker state
    multi_state = {'visible': False, 'steps': [], 'current_step': 0}
    
    def show_for_operation(operation_name: str, steps: List[str] = None) -> None:
        """Show multi tracker dengan step list"""
        steps = steps or ["Step 1", "Step 2", "Step 3"]
        multi_state.update({'visible': True, 'steps': steps, 'current_step': 0})
        container.layout.visibility = 'visible'
        _update_steps_display()
        current_step_tracker['show_for_operation'](steps[0] if steps else "Unknown Step")
    
    def update_progress(category: str, value: int, message: str = "") -> None:
        """Update current step progress"""
        current_step_tracker['update_progress']('current', value, message)
    
    def next_step(step_name: str = None) -> None:
        """Move to next step"""
        multi_state['current_step'] += 1
        _update_steps_display()
        step_name = step_name or multi_state['steps'][multi_state['current_step']] if multi_state['current_step'] < len(multi_state['steps']) else "Final Step"
        current_step_tracker['show_for_operation'](step_name)
    
    def complete_operation(final_message: str) -> None:
        """Complete multi operation"""
        current_step_tracker['complete_operation'](final_message)
        _update_steps_display(completed=True)
        
        if auto_hide_delay > 0:
            import threading
            threading.Timer(auto_hide_delay, reset_all).start()
    
    def error_operation(error_message: str) -> None:
        """Handle multi error"""
        current_step_tracker['error_operation'](error_message)
    
    def reset_all() -> None:
        """Reset multi tracker"""
        multi_state.update({'visible': False, 'steps': [], 'current_step': 0})
        container.layout.visibility = 'hidden'
        steps_container.children = []
        current_step_tracker['reset_all']()
    
    def _update_steps_display(completed: bool = False) -> None:
        """Update steps visualization"""
        steps_html = []
        for i, step in enumerate(multi_state['steps']):
            if i < multi_state['current_step'] or completed:
                status = "✅"
                color = "#28a745"
            elif i == multi_state['current_step']:
                status = "🔄"
                color = "#007bff"
            else:
                status = "⏳"
                color = "#6c757d"
            
            steps_html.append(f"<div style='color: {color}; margin: 2px 0;'>{status} {step}</div>")
        
        steps_container.children = [widgets.HTML("".join(steps_html))]
    
    return {
        'container': container, 'current_step_tracker': current_step_tracker, 'steps_container': steps_container,
        'show_for_operation': show_for_operation, 'update_progress': update_progress, 'next_step': next_step,
        'complete_operation': complete_operation, 'error_operation': error_operation,
        'reset_all': reset_all, 'state': multi_state, 'type': 'multi'
    }

# Backward compatibility functions
def create_dual_progress_tracker(height: str = "180px", width: str = "100%") -> Dict[str, Any]:
    """Backward compatibility untuk create_dual_progress_tracker"""
    return create_progress_tracker("dual", height, width)

def create_single_progress_tracker(height: str = "120px", width: str = "100%") -> Dict[str, Any]:
    """Backward compatibility untuk create_single_progress_tracker"""
    return create_progress_tracker("single", height, width)

def create_three_progress_tracker(height: str = "200px", width: str = "100%") -> Dict[str, Any]:
    """Backward compatibility untuk create_three_progress_tracker - alias untuk multi"""
    return create_progress_tracker("multi", height, width)