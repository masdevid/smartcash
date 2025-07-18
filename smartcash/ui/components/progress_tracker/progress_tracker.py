"""
Progress tracker component for displaying operation progress with multiple levels.
"""

import time
from typing import Dict, List, Optional, Callable, Any
import ipywidgets as widgets
from smartcash.ui.components.base_component import BaseUIComponent
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel, ProgressBarConfig
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager

class ProgressTracker(BaseUIComponent):
    """A progress tracker component that displays progress with multiple levels."""
    
    def __init__(self, 
                 component_name: str = "progress_tracker",
                 config: Optional[ProgressConfig] = None,
                 **kwargs):
        """Initialize the progress tracker.
        
        Args:
            component_name: Unique name for this component
            config: Progress tracker configuration
            **kwargs: Additional arguments to pass to BaseUIComponent
        """
        self._config = config or ProgressConfig()
        self.callback_manager = CallbackManager()
        self.tqdm_manager = None
        self._active_levels = [config.name for config in self._config.get_level_configs() 
                             if config.visible]
        self._current_step_index = 0
        self._is_complete = False
        self._is_error = False
        
        # Initialize base class
        super().__init__(component_name, **kwargs)
        
        # Register default callbacks
        self._register_default_callbacks()
    
    def _create_ui_components(self) -> None:
        """Create and initialize UI components."""
        # Create main container with modern styling (no padding)
        self._ui_components['container'] = widgets.VBox(
            [],
            layout=widgets.Layout(
                display='block',
                width='100%',
                margin='8px 0',
                padding='0px',
                border='1px solid rgba(255, 255, 255, 0.18)',
                border_radius='12px',
                background_color='rgba(255, 255, 255, 0.95)',
                box_shadow='0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)',
                backdrop_filter='blur(10px)',
                min_height='60px',
                height='auto',
                overflow='visible',
                box_sizing='border-box'
            )
        )
        
        # Create header widget with modern styling
        self._ui_components['header'] = widgets.HTML(
            self._create_modern_header(self._config.operation),
            layout=widgets.Layout(width='100%', margin='0 0 8px 0')
        )
        
        # Create status widget with modern styling
        self._ui_components['status'] = widgets.HTML(
            self._create_modern_status("Ready", "info"),
            layout=widgets.Layout(width='100%', margin='0 0 8px 0')
        )
        
        # Create output widgets for progress bars with compact layout
        bar_height = '24px'
        bar_layout = widgets.Layout(
            width='100%',
            margin='4px 0',
            height=bar_height,
            min_height=bar_height,
            overflow='visible'
        )
        
        # Create output widgets for each level
        self._ui_components['overall_output'] = widgets.Output(layout=bar_layout)
        self._ui_components['step_output'] = widgets.Output(layout=bar_layout)
        self._ui_components['current_output'] = widgets.Output(layout=bar_layout)
        
        # Initialize TQDM manager before assembling container
        self.tqdm_manager = TqdmManager(self)
        
        # Assemble the container
        self._assemble_container()
        
        # Initialize progress bars
        self.tqdm_manager.initialize_bars(self._config.get_level_configs())
    
    @property
    def progress_bar(self):
        """Backward compatibility property to return the main progress bar.
        
        Note: This is maintained for backward compatibility with existing code.
        New code should use the progress_bars property instead.
        """
        # Return the first available progress bar for backward compatibility
        if hasattr(self, 'tqdm_manager') and self.tqdm_manager and hasattr(self.tqdm_manager, 'bars'):
            for bar in self.tqdm_manager.bars.values():
                if bar is not None:
                    return bar
        return None
        
    @property
    def status_label(self):
        """Backward compatibility property to return status label."""
        return self._ui_components.get('status')
    
    def _update_container_height(self) -> None:
        """Update container height based on visible components with compact spacing."""
        if not hasattr(self, '_ui_components') or 'container' not in self._ui_components:
            return
            
        # Compact spacing for minimal height
        base_height = 50  # px for header + status + compact padding
        bar_height = 28   # px per progress bar with compact spacing
        
        # Count visible progress bars based on level
        visible_bars = self._config.level.value
        
        # Calculate total height with compact spacing
        total_height = base_height + (visible_bars * bar_height)
        
        # Update container height
        container = self._ui_components['container']
        container.layout.height = f'{total_height}px'
        container.layout.min_height = f'{total_height}px'
    
    def _assemble_container(self) -> None:
        """Assemble the container with appropriate widgets based on config level."""
        container = self._ui_components['container']
        
        # Always include header and status
        children = [
            self._ui_components['header'],
            self._ui_components['status']
        ]
        
        # Add progress bars based on config level
        level_configs = self._config.get_level_configs()
        for config in level_configs:
            if config.visible:
                output_key = f"{config.name}_output"
                if output_key in self._ui_components:
                    children.append(self._ui_components[output_key])
        
        # Update container children
        container.children = children
    
    def _create_modern_header(self, operation: str) -> str:
        """Create modern styled header HTML."""
        return f"""
        <div style='
            display: flex;
            align-items: center;
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        '>
            <div style='
                width: 4px;
                height: 32px;
                background: linear-gradient(180deg, #007bff 0%, #28a745 100%);
                border-radius: 2px;
                margin-right: 16px;
            '></div>
            <h3 style='
                font-size: 18px;
                font-weight: 700;
                margin: 0;
                padding: 0;
                line-height: 1.3;
                color: #1a1a1a;
                letter-spacing: -0.01em;
            '>
                {operation}
            </h3>
        </div>
        """
    
    def _create_modern_status(self, message: str, style: str = "info") -> str:
        """Create modern styled status HTML with adaptive logging level."""
        color_map = {
            'success': '#16a34a', 'info': '#0ea5e9', 
            'warning': '#f59e0b', 'error': '#ef4444'
        }
        color = color_map.get(style, '#64748b')
        
        # Adaptive background based on logging level
        bg_colors = {
            'success': 'linear-gradient(135deg, rgba(22, 163, 74, 0.08) 0%, rgba(22, 163, 74, 0.04) 100%)',
            'info': 'linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%)',
            'warning': 'linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.04) 100%)',
            'error': 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%)'
        }
        bg_color = bg_colors.get(style, 'linear-gradient(135deg, rgba(100, 116, 139, 0.05) 0%, rgba(100, 116, 139, 0.02) 100%)')
        
        # Status icon based on style
        icons = {
            'success': '✅',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌'
        }
        icon = icons.get(style, '📋')
        
        # Clean message from existing emojis
        import re
        clean_message = re.sub(r'^[✅❌⚠️ℹ️🚀📋]\s*', '', message).strip()
        
        rgb_color = self._hex_to_rgb(color)
        
        return f"""
        <div style="
            display: flex;
            align-items: center;
            width: 100%;
            color: {color};
            font-size: 14px;
            font-weight: 600;
            margin: 0;
            padding: 16px 20px;
            background: {bg_color};
            border-radius: 12px;
            border: 1px solid rgba({rgb_color}, 0.2);
            box-sizing: border-box;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba({rgb_color}, 0.1);
        ">
            <span style="
                font-size: 16px;
                margin-right: 12px;
                opacity: 0.9;
            ">{icon}</span>
            <span style="flex: 1;">{clean_message}</span>
        </div>
        """
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to RGB values for CSS."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"
        return "100, 116, 139"  # Default gray
    
    def _register_default_callbacks(self) -> None:
        """Register default callbacks."""
        self.on_complete(lambda: self._delayed_hide())
        self.on_progress_update(self._sync_progress_state)
    
    def _sync_progress_state(self, level: str, progress: int, message: str) -> None:
        """Synchronize progress state with UI."""
        if self.tqdm_manager and hasattr(self.tqdm_manager, 'update_bar'):
            self.tqdm_manager.update_bar(level, progress, message)
    
    def _delayed_hide(self, delay: int = 1) -> None:
        """Hide the progress tracker after a delay."""
        if self._config.auto_hide and self._config.auto_hide_delay > 0:
            import threading
            def hide_after_delay():
                time.sleep(delay)
                self.hide()
            threading.Thread(target=hide_after_delay, daemon=True).start()
    
    # Public API methods
    def show(self, operation: str = None, steps: List[str] = None, 
             step_weights: Dict[str, int] = None, level: ProgressLevel = None,
             auto_hide: bool = None) -> None:
        """Show the progress tracker with optional configuration overrides."""
        if not self._initialized:
            self.initialize()
            
        if operation:
            self._config.operation = operation
        if steps:
            self._config.steps = steps
        if step_weights:
            self._config.step_weights = step_weights
        if level and level != self._config.level:
            self._config.level = level
            self._active_levels = [config.name for config in self._config.get_level_configs() 
                                 if config.visible]
            self._assemble_container()
        if auto_hide is not None:
            self._config.auto_hide = auto_hide
        
        self._ui_components['container'].layout.display = 'flex'
        self._ui_components['container'].layout.visibility = 'visible'
        self._update_container_height()
    
    def hide(self) -> None:
        """Hide the progress tracker and reset display state."""
        if self._initialized and 'container' in self._ui_components:
            self._ui_components['container'].layout.display = 'none'
            self._ui_components['container'].layout.visibility = 'hidden'
            
        # Reset display state to allow new instances
        self._display_active = False
        if hasattr(self, '_last_display_handle'):
            delattr(self, '_last_display_handle')
    
    def update_status(self, message: str, level: str = "current", style: str = "info") -> None:
        """Update the status message with adaptive styling based on logging level."""
        if not self._initialized:
            self.initialize()
            
        # Update main status widget with modern styling
        if 'status' in self._ui_components:
            self._ui_components['status'].value = self._create_modern_status(message, style)
            
        # Also update specific level if provided
        if level in self._ui_components:
            self._ui_components[level].value = message
    
    def set_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """Update progress for a specific level."""
        try:
            # Ensure tqdm_manager is initialized
            if not hasattr(self, 'tqdm_manager') or not self.tqdm_manager:
                self.tqdm_manager = TqdmManager(self)
                self.tqdm_manager.initialize_bars(self._config.get_level_configs())
            
            # Map level to valid progress bar names
            level_map = {
                'primary': 'overall',
                'secondary': 'step',
                'tertiary': 'current'
            }
            level = level_map.get(level, level)
            
            if level not in ['overall', 'step', 'current']:
                level = 'overall'  # Default to overall if invalid level provided
            
            # Update the status message with modern styling
            if message and 'status' in self._ui_components:
                # Determine style based on progress value
                if progress == 100:
                    style = "success"
                elif progress > 0:
                    style = "info"
                else:
                    style = "info"
                self._ui_components['status'].value = self._create_modern_status(message, style)
            
            # Update the progress bar
            self.tqdm_manager.update_bar(level, progress, message)
            
            # Force a display update if in notebook environment
            if hasattr(self, '_last_displayed') and 'IPython' in globals():
                try:
                    from IPython.display import display
                    display(self._ui_components['container'], 
                          display_id=getattr(self, '_last_displayed', True))
                except ImportError:
                    pass  # Not in a notebook environment
                
        except Exception as e:
            print(f"Progress update error: {e}")
            
    def show(self, operation: str = None, steps: list = None, 
             step_weights: dict = None, level = None,
             auto_hide: bool = None):
        """Display the progress tracker with enhanced duplicate prevention."""
        try:
            # Prevent multiple instances by checking for existing display
            if hasattr(self, '_display_active') and self._display_active:
                return getattr(self, '_last_display_handle', None)
            
            container = self._ui_components.get('container')
            if not container:
                return None
            
            # Update configuration if provided
            if operation:
                self._config.operation = operation
                if 'header' in self._ui_components:
                    self._ui_components['header'].value = self._create_modern_header(operation)
            
            # Mark as active to prevent duplicates
            self._display_active = True
            
            # Create a unique display ID to prevent conflicts
            display_id = f"progress_{id(self)}_{hash(operation or self._config.operation)}"
            self._last_displayed = display_id
            
            # Ensure all UI components are properly initialized
            if not hasattr(self, 'tqdm_manager') or not self.tqdm_manager:
                self.tqdm_manager = TqdmManager(self)
                self.tqdm_manager.initialize_bars(self._config.get_level_configs())
            
            # Show container
            container.layout.display = 'flex'
            container.layout.visibility = 'visible'
            
            # Check if we're in a notebook environment
            try:
                from IPython.display import display, clear_output
                is_notebook = True
            except ImportError:
                is_notebook = False
            
            if is_notebook:
                # Clear any existing display to prevent duplicates
                if hasattr(self, '_last_display_handle'):
                    try:
                        clear_output(wait=True)
                    except:
                        pass
                
                # Display with unique ID
                display_handle = display(container, display_id=display_id)
                self._last_display_handle = display_handle
                return display_handle
            else:
                # In script mode, just print progress
                print("\n" + "="*50)
                print(f"Progress: {self._config.operation}")
                print("="*50)
                return None
                
        except Exception as e:
            print(f"Error displaying progress: {e}")
            self._display_active = False
            return None
    
    def complete(self, message: str = "Completed!") -> None:
        """Mark the operation as complete with modern status styling."""
        if self.tqdm_manager:
            self.tqdm_manager.set_all_complete(message, self._config.get_level_configs())
        
        # Update status with success styling
        if 'status' in self._ui_components:
            self._ui_components['status'].value = self._create_modern_status(message, "success")
            
        self._is_complete = True
        self.callback_manager.trigger('complete')
    
    def error(self, message: str = "An error occurred!") -> None:
        """Mark the operation as failed with modern error styling."""
        if self.tqdm_manager:
            self.tqdm_manager.set_all_error(message)
        
        # Update status with error styling
        if 'status' in self._ui_components:
            self._ui_components['status'].value = self._create_modern_status(message, "error")
            
        self._is_error = True
        self.callback_manager.trigger('error', message)
    
    def reset(self) -> None:
        """Reset the progress tracker to its initial state."""
        if self.tqdm_manager:
            self.tqdm_manager.initialize_bars(self._config.get_level_configs())
        
        # Reset status with modern styling
        if 'status' in self._ui_components:
            self._ui_components['status'].value = self._create_modern_status("Ready", "info")
            
        self._is_complete = False
        self._is_error = False
        self._current_step_index = 0
        self._display_active = False
        self.callback_manager.trigger('reset')
    
    # Callback registration methods
    def on_progress_update(self, callback: Callable[[str, int, str], None]) -> str:
        """Register a callback for progress updates."""
        return self.callback_manager.register('progress_update', callback)
    
    def on_step_complete(self, callback: Callable[[str, int], None]) -> str:
        """Register a callback for step completion."""
        return self.callback_manager.register('step_complete', callback)
    
    def on_complete(self, callback: Callable[[], None]) -> str:
        """Register a callback for operation completion."""
        return self.callback_manager.register('complete', callback)
    
    def on_error(self, callback: Callable[[str], None]) -> str:
        """Register a callback for error events."""
        return self.callback_manager.register('error', callback)
    
    def on_reset(self, callback: Callable[[], None]) -> str:
        """Register a callback for reset events."""
        return self.callback_manager.register('reset', callback)
    
    def remove_callback(self, callback_id: str) -> None:
        """Remove a registered callback."""
        self.callback_manager.unregister(callback_id)
    
    # Backward compatibility properties
    @property
    def container(self):
        """Backward compatibility property for container."""
        return self._ui_components.get('container')
    
    @property
    def status_widget(self):
        """Backward compatibility property for status widget."""
        return self._ui_components.get('status')
    
    @property
    def step_info_widget(self):
        """Backward compatibility property (always None)."""
        return None
    
    @property
    def progress_bars(self):
        """Backward compatibility property for progress bars."""
        if not hasattr(self, 'tqdm_manager') or not self.tqdm_manager:
            return {'main': None}
            
        if not hasattr(self.tqdm_manager, 'tqdm_bars'):
            return {'main': None}
            
        if self._config.level == ProgressLevel.SINGLE and self.tqdm_manager.tqdm_bars:
            first_key = next(iter(self.tqdm_manager.tqdm_bars))
            return {'main': self.tqdm_manager.tqdm_bars.get(first_key)}
        
        return self.tqdm_manager.tqdm_bars

# Backward compatibility functions
def create_progress_tracker(config: Optional[ProgressConfig] = None):
    """Legacy function to create a progress tracker."""
    return ProgressTracker("legacy_progress_tracker", config)

def update_progress(tracker, progress: int, message: str = "", 
                  level: str = "primary") -> None:
    """Legacy function to update progress."""
    if hasattr(tracker, 'set_progress'):
        tracker.set_progress(progress, level, message)
    elif hasattr(tracker, 'update_bar') and hasattr(tracker, 'tqdm_bars'):
        tracker.update_bar(level, progress, message)

def complete_progress(tracker, message: str = "Completed!") -> None:
    """Legacy function to complete progress."""
    if hasattr(tracker, 'complete'):
        tracker.complete(message)
    elif hasattr(tracker, 'set_all_complete') and hasattr(tracker, '_config'):
        tracker.set_all_complete(message, tracker._config.get_level_configs())

def error_progress(tracker, message: str = "An error occurred!") -> None:
    """Legacy function to indicate error in progress."""
    if hasattr(tracker, 'error'):
        tracker.error(message)
    elif hasattr(tracker, 'set_all_error'):
        tracker.set_all_error(message)
