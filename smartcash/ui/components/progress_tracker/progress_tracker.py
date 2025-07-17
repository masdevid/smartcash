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
        # Create main container with auto height
        self._ui_components['container'] = widgets.VBox(
            [],
            layout=widgets.Layout(
                display='block',
                width='100%',
                margin='10px 0',
                padding='15px',
                border='1px solid #e0e0e0',
                border_radius='8px',
                background_color='#f8fff8',
                min_height='100px',
                height='auto',
                overflow='visible',
                box_sizing='content-box'
            )
        )
        
        # Create header widget
        self._ui_components['header'] = widgets.HTML(
            f"<h4>{self._config.operation}</h4>",
            layout=widgets.Layout(width='100%', margin='0 0 10px 0')
        )
        
        # Create status widget
        self._ui_components['status'] = widgets.HTML(
            "<i>Initializing...</i>",
            layout=widgets.Layout(width='100%', margin='0 0 8px 0')
        )
        
        # Create output widgets for progress bars with consistent height
        bar_height = '30px'
        bar_layout = widgets.Layout(
            width='100%',
            margin='5px 0',
            height=bar_height,
            min_height=bar_height,
            overflow='hidden'
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
        """Update container height based on visible components."""
        if not hasattr(self, '_ui_components') or 'container' not in self._ui_components:
            return
            
        # Base height for header and status
        base_height = 60  # px for header + status + padding
        bar_height = 30   # px per progress bar
        
        # Count visible progress bars based on level
        visible_bars = self._config.level.value
        
        # Calculate total height
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
        """Hide the progress tracker."""
        if self._initialized and 'container' in self._ui_components:
            self._ui_components['container'].layout.display = 'none'
            self._ui_components['container'].layout.visibility = 'hidden'
    
    def update_status(self, message: str, level: str = "current") -> None:
        """Update the status message for a specific level."""
        if not self._initialized:
            self.initialize()
            
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
            
            # Update the status message
            if message and 'status' in self._ui_components:
                self._ui_components['status'].value = message
            
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
            
    def show(self):
        """Display the progress tracker and return the display ID."""
        try:
            container = self._ui_components.get('container')
            if not container:
                return None
                
            # Create a unique display ID
            display_id = f"progress_{id(container)}"
            self._last_displayed = display_id
            
            # Ensure all UI components are properly initialized
            if not hasattr(self, 'tqdm_manager') or not self.tqdm_manager:
                self.tqdm_manager = TqdmManager(self)
                self.tqdm_manager.initialize_bars(self._config.get_level_configs())
            
            # Check if we're in a notebook environment
            try:
                from IPython.display import display
                is_notebook = True
            except ImportError:
                is_notebook = False
            
            if is_notebook:
                # In notebook, use IPython display
                display_handle = display(container, display_id=display_id, display=True)
                display(container, display_id=display_id)  # Force initial display
                return display_handle
            else:
                # In script mode, just print progress
                print("\n" + "="*50)
                print(f"Progress: {self._config.operation}")
                print("="*50)
                return None
                
        except Exception as e:
            print(f"Error displaying progress: {e}")
            return None
    
    def complete(self, message: str = "Completed!") -> None:
        """Mark the operation as complete."""
        if self.tqdm_manager:
            self.tqdm_manager.set_all_complete(message, self._config.get_level_configs())
        self._is_complete = True
        self.callback_manager.trigger('complete')
    
    def error(self, message: str = "An error occurred!") -> None:
        """Mark the operation as failed."""
        if self.tqdm_manager:
            self.tqdm_manager.set_all_error(message)
        self._is_error = True
        self.callback_manager.trigger('error', message)
    
    def reset(self) -> None:
        """Reset the progress tracker to its initial state."""
        if self.tqdm_manager:
            self.tqdm_manager.initialize_bars(self._config.get_level_configs())
        self._is_complete = False
        self._is_error = False
        self._current_step_index = 0
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
