"""
File: smartcash/ui/setup/env_config/utils/dual_progress_tracker.py
Deskripsi: Simple and clean progress tracker for environment setup
"""

from enum import Enum, auto
import ipywidgets as widgets
from ipywidgets import VBox, FloatProgress, HTML
from typing import Dict, Any, Callable, Optional

class SetupStage(Enum):
    """Stages for setup progress tracking"""
    DRIVE_MOUNT = auto()
    FOLDER_SETUP = auto()
    SYMLINK_SETUP = auto()
    CONFIG_SYNC = auto()
    ENV_SETUP = auto()
    COMPLETE = auto()

class DualProgressTracker:
    """Simple and clean progress tracker for environment setup
    
    Features:
    - Single progress bar with status text
    - Stage tracking with automatic progress calculation
    - Callback support for progress updates
    - Error and completion handling
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, logger=None):
        """Initialize the progress tracker with optional UI components and logger"""
        self.logger = logger
        self.components = ui_components if ui_components is not None else {}
        self.current_stage = None
        self.overall_progress = 0
        self.stage_progress = 0
        self.callbacks = []
        
        # Initialize container first
        self.container = None
        
        # Create UI elements
        self._create_ui()
        
        # Ensure we have a valid container
        if self.container is None:
            self.container = VBox()
            
        # Register with UI components
        self.components['progress_container'] = self.container
        self.components['progress_tracker'] = self
    
    def _create_ui(self):
        """Create and configure the UI elements"""
        # Progress bar
        self.progress_bar = FloatProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            orientation='horizontal',
            layout={
                'width': '100%',
                'visibility': 'visible',
                'display': 'flex'
            }
        )
        
        # Status text
        self.status_text = HTML(
            value="<i>Ready to start setup...</i>",
            layout={
                'width': '100%',
                'visibility': 'visible',
                'display': 'block'
            }
        )
        
        # Main container with visible layout
        self.container = VBox(
            [self.progress_bar, self.status_text],
            layout={
                'width': '100%',
                'visibility': 'visible',
                'display': 'flex',
                'flex_flow': 'column',
                'align_items': 'stretch',
                'padding': '10px',
                'border': '1px solid #e0e0e0',
                'border_radius': '5px',
                'margin': '5px 0'
            }
        )
    
    def update_stage(self, stage: SetupStage, message: Optional[str] = None):
        """Update the current stage of the setup process"""
        self.current_stage = stage
        stage_name = stage.name.replace('_', ' ').title()
        status = message or f"{stage_name}..."
        self._update_status(status)
        self.update_progress(0, status)
    
    def update_progress(self, progress: float, message: Optional[str] = None):
        """Update the progress of the current stage"""
        # Update progress values
        self.stage_progress = max(0, min(100, progress))
        self.progress_bar.value = self.stage_progress
        
        # Update status if message provided
        if message:
            self._update_status(message)
        
        # Calculate overall progress if we have a current stage
        self._update_overall_progress()
        
        # Notify callbacks
        self._notify_callbacks(message or "")
    
    def update_within_stage(self, current: int, total: int, message: Optional[str] = None):
        """
        Update progress within the current stage using item counts
        
        Args:
            current: Current item number (1-based, can be string or int)
            total: Total number of items in this stage (can be string or int)
            message: Optional status message
        """
        try:
            # Convert to integers if they're strings
            current = int(current)
            total = int(total)
            
            if total <= 0:
                return
                
            progress = (current / total) * 100
            self.update_progress(progress, message or f"Processing item {current} of {total}")
        except (TypeError, ValueError) as e:
            error_msg = f"Invalid progress values - current: {current}, total: {total}"
            if self.logger:
                self.logger.error(f"{error_msg}: {e}")
            self.error(error_msg)
    
    def complete_stage(self, message: Optional[str] = None):
        """Mark the current stage as complete"""
        if self.current_stage is None:
            return
            
        # Update UI
        stage_name = self.current_stage.name.replace('_', ' ').title()
        status = message or f"{stage_name} completed"
        self._update_status(f"✓ {status}")
        
        # Set progress to 100%
        self.stage_progress = 100
        self.progress_bar.value = 100
        
        # Update overall progress
        self._update_overall_progress()
        
        # Notify callbacks
        self._notify_callbacks(status)
    
    def complete(self, message: str = "Setup completed successfully!"):
        """Mark the entire setup as complete"""
        self.overall_progress = 100
        self.stage_progress = 100
        self.progress_bar.value = 100
        self._update_status(f"✓ {message}")
        self._notify_callbacks(message)
    
    def error(self, message: str):
        """Report an error in the setup process"""
        self._update_status(f"❌ Error: {message}")
        if self.logger:
            self.logger.error(message)
    
    def add_callback(self, callback: Callable[[float, float, str], None]):
        """Add a callback function to be called on progress updates"""
        if callable(callback) and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def _update_status(self, message: str):
        """Update the status text with proper HTML formatting"""
        self.status_text.value = f"<b>{message}</b>"
    
    def _update_overall_progress(self):
        """Calculate and update the overall progress based on current stage"""
        if self.current_stage is None:
            return
            
        stage_count = len(SetupStage)
        stage_idx = list(SetupStage).index(self.current_stage)
        stage_weight = 100 / stage_count
        
        base_progress = (stage_idx / stage_count) * 100
        stage_contribution = (self.stage_progress / 100) * stage_weight
        self.overall_progress = min(100, base_progress + stage_contribution)
    
    def _notify_callbacks(self, message: str):
        """Notify all registered callbacks with current progress"""
        for callback in self.callbacks:
            try:
                callback(self.overall_progress, self.stage_progress, message)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in progress callback: {e}")
    
    @property
    def progress_container(self):
        """Get the progress container widget"""
        return self.container
    
    @property
    def ui_components(self) -> Dict[str, Any]:
        """Get the UI components dictionary"""
        return self.components
        
    def show(self):
        """Make the progress container visible"""
        if not hasattr(self.container, 'layout'):
            return
            
        # Ensure the container and its children are visible
        for widget in [self.container, self.progress_bar, self.status_text]:
            if hasattr(widget, 'layout'):
                if hasattr(widget.layout, 'visibility'):
                    widget.layout.visibility = 'visible'
                if hasattr(widget.layout, 'display'):
                    widget.layout.display = 'flex' if widget == self.container else 'block'
        
        # Ensure the container is properly sized
        if hasattr(self.container.layout, 'width'):
            self.container.layout.width = '100%'
        if hasattr(self.container.layout, 'height') and self.container.layout.height is None:
            self.container.layout.height = 'auto'
            
    def hide(self):
        """Hide the progress container"""
        if hasattr(self.container, 'layout'):
            if hasattr(self.container.layout, 'visibility'):
                self.container.layout.visibility = 'hidden'
            elif hasattr(self.container.layout, 'display'):
                self.container.layout.display = 'none'

# Backward compatibility
track_setup_progress = DualProgressTracker
