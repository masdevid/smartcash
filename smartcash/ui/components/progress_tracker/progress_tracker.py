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
        # Create main container
        self._ui_components['container'] = widgets.VBox(
            [],
            layout=widgets.Layout(
                display='none',
                width='100%',
                margin='10px 0',
                padding='15px',
                border='1px solid #e0e0e0',
                border_radius='8px',
                background_color='#f8fff8',
                min_height=self._config.get_container_height(),
                max_height='400px',
                overflow='hidden',
                box_sizing='border-box'
            )
        )
        
        # Create header widget
        self._ui_components['header'] = widgets.HTML(
            "", 
            layout=widgets.Layout(width='100%', margin='0 0 10px 0')
        )
        
        # Create status widget
        self._ui_components['status'] = widgets.HTML(
            "",
            layout=widgets.Layout(width='100%', margin='0 0 8px 0')
        )
        
        # Create output widgets for progress bars
        self._ui_components['overall_output'] = widgets.Output(
            layout=widgets.Layout(width='100%', margin='1px 0', 
                               min_height='25px', max_height='30px')
        )
        self._ui_components['step_output'] = widgets.Output(
            layout=widgets.Layout(width='100%', margin='1px 0', 
                               min_height='25px', max_height='30px')
        )
        self._ui_components['current_output'] = widgets.Output(
            layout=widgets.Layout(width='100%', margin='1px 0', 
                               min_height='25px', max_height='30px')
        )
        
        # Assemble the container
        self._assemble_container()
        
        # Initialize TQDM manager
        self.tqdm_manager = TqdmManager(self)
        self.tqdm_manager.initialize_bars(self._config.get_level_configs())
    
    def _assemble_container(self) -> None:
        """Assemble the container with appropriate widgets based on config level."""
        container = self._ui_components['container']
        container.children = [
            self._ui_components['header'],
            self._ui_components['status']
        ]
        
        # Add progress outputs based on level
        if self._config.level.value >= 2:  # DUAL or TRIPLE
            container.children += (self._ui_components['overall_output'],)
        
        if self._config.level.value >= 3:  # TRIPLE
            container.children += (
                self._ui_components['step_output'],
                self._ui_components['current_output']
            )
        elif self._config.level.value == 2:  # DUAL
            container.children += (self._ui_components['current_output'],)
        elif self._config.level.value == 1:  # SINGLE
            container.children += (self._ui_components['overall_output'],)
    
    def _register_default_callbacks(self) -> None:
        """Register default callbacks."""
        self.on_complete(lambda: self._delayed_hide())
        self.on_progress_update(self._sync_progress_state)
    
    def _sync_progress_state(self, level: str, progress: int, message: str) -> None:
        """Synchronize progress state with UI."""
        if self.tqdm_manager:
            self.tqdm_manager.update_bar(level, progress, message)
    
    def _delayed_hide(self, delay: int = 1) -> None:
        """Hide the progress tracker after a delay."""
        if self._config.auto_hide and self._config.auto_hide_delay > 0:
            time.sleep(delay)
            self.hide()
    
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
    
    def set_progress(self, progress: int, level: str = "primary", 
                    message: str = "") -> None:
        """Update progress for a specific level."""
        if self.tqdm_manager:
            self.tqdm_manager.update_bar(level, progress, message)
    
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
