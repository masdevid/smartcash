"""
File: smartcash/ui/dataset/augment/handlers/augment_ui_handler.py
Description: Main UI handler for augment module following core inheritance patterns

This handler inherits from the core BaseHandler and implements augment-specific
UI operations while preserving all original business logic and error handling.
"""

from typing import Dict, Any, Optional, Callable
import logging
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.errors.decorators import handle_errors
from ..configs.augment_config_handler import AugmentConfigHandler
from ..constants import (
    SUCCESS_MESSAGES, ERROR_MESSAGES, WARNING_MESSAGES,
    AugmentationOperation, ProcessingPhase
)


class AugmentUIHandler(BaseHandler):
    """
    Main UI handler for augment module with core inheritance patterns.
    
    Features:
    - 🏗️ Inherits from core BaseHandler
    - 🎨 Preserved original business logic
    - 🔄 Centralized operation management
    - ✅ Comprehensive error handling
    - 📊 Real-time UI updates
    - 📝 Activity logging and progress tracking
    """
    
    @handle_ui_errors(error_component_title="Augment UI Handler Initialization Error")
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Initialize augment UI handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(
            module_name='augment',
            parent_module='dataset', 
            ui_components=ui_components
        )
        
        # Initialize config handler
        self.config_handler = AugmentConfigHandler()
        
        # Operation state tracking
        self._current_operation: Optional[str] = None
        self._operation_progress: float = 0.0
        self._operation_phase: str = "Idle"
        
        # Button references for state management
        self._buttons: Dict[str, Any] = {}
        
        self.logger.info("🎨 AugmentUIHandler initialized with core inheritance")
    
    @handle_ui_errors(error_component_title="Handler Setup Error")
    def setup_handlers(self) -> None:
        """
        Setup all UI handlers and event bindings.
        
        Features:
        - Button event handlers
        - Configuration synchronization
        - Progress tracking
        - Error boundary setup
        """
        self._setup_button_handlers()
        self._setup_config_handlers()
        self._setup_progress_tracking()
        self._check_initial_state()
        
        self.logger.info("✅ All handlers setup completed")
    
    def _setup_button_handlers(self) -> None:
        """Setup button event handlers with preserved business logic."""
        button_mapping = {
            'augment_button': self.handle_augment,
            'check_button': self.handle_check,
            'cleanup_button': self.handle_cleanup, 
            'preview_button': self.handle_preview
        }
        
        for button_key, handler in button_mapping.items():
            button = self.ui_components.get(button_key)
            if button:
                button.on_click(lambda b, h=handler: h())
                self._buttons[button_key] = button
                self.logger.debug(f"📌 Setup handler for {button_key}")
    
    def _setup_config_handlers(self) -> None:
        """Setup configuration change handlers."""
        # Monitor form changes for real-time updates
        form_widgets = [
            'num_variations_slider', 'target_count_slider', 'intensity_slider',
            'target_split_dropdown', 'cleanup_target_dropdown', 'balance_classes_checkbox',
            'augmentation_types_select', 'preview_mode_checkbox', 'custom_mode_checkbox'
        ]
        
        for widget_key in form_widgets:
            widget = self.ui_components.get(widget_key)
            if widget:
                widget.observe(self._on_config_change, names='value')
                self.logger.debug(f"📝 Setup config handler for {widget_key}")
    
    def _setup_progress_tracking(self) -> None:
        """Setup progress tracking for operation summary."""
        # Initialize summary container if available
        if 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            if 'status' in update_methods:
                update_methods['status']('pending', 'Ready for operations')
                self.logger.debug("📊 Progress tracking initialized")
    
    def _check_initial_state(self) -> None:
        """Check initial state and update UI accordingly."""
        try:
            # Get current configuration
            config = self.config_handler.get_config()
            
            # Update summary with initial stats
            self._update_dataset_stats()
            self._log_activity("UI Handler initialized and ready")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Initial state check failed: {e}")
    
    @handle_errors(error_msg="Augmentation operation failed", reraise=False)
    def handle_augment(self) -> None:
        """
        Handle augmentation operation with preserved business logic.
        
        Features:
        - Configuration validation
        - Progress tracking
        - Error handling
        - UI state management
        """
        self.logger.info("🚀 Starting augmentation operation")
        
        try:
            # Update UI state
            self._set_operation_state(AugmentationOperation.AUGMENT, "Starting augmentation...")
            self._update_button_states(augment_running=True)
            
            # Validate configuration
            config = self._get_current_config()
            is_valid, errors = self.config_handler.validate_config(config)
            
            if not is_valid:
                self._handle_validation_errors(errors)
                return
            
            # Simulate augmentation process (replace with actual backend call)
            self._simulate_augmentation_process(config)
            
            # Success handling
            self._set_operation_state(None, SUCCESS_MESSAGES['augmentation_complete'])
            self._log_activity("Augmentation completed successfully")
            self.logger.info("✅ Augmentation operation completed")
            
        except Exception as e:
            self._handle_operation_error("augmentation", e)
        finally:
            self._update_button_states(augment_running=False)
    
    @handle_errors(error_msg="Check operation failed", reraise=False)
    def handle_check(self) -> None:
        """Handle dataset check operation."""
        self.logger.info("🔍 Starting check operation")
        
        try:
            self._set_operation_state(AugmentationOperation.CHECK, "Checking dataset...")
            
            # Simulate check process
            self._simulate_check_process()
            
            self._set_operation_state(None, SUCCESS_MESSAGES['check_complete'])
            self._log_activity("Dataset check completed")
            self.logger.info("✅ Check operation completed")
            
        except Exception as e:
            self._handle_operation_error("check", e)
    
    @handle_errors(error_msg="Cleanup operation failed", reraise=False)
    def handle_cleanup(self) -> None:
        """Handle cleanup operation."""
        self.logger.info("🗑️ Starting cleanup operation")
        
        try:
            self._set_operation_state(AugmentationOperation.CLEANUP, "Cleaning up files...")
            
            # Get cleanup target from UI
            cleanup_target = self.ui_components.get('cleanup_target_dropdown', {}).get('value', 'both')
            
            # Show warning
            self._log_activity(f"⚠️ {WARNING_MESSAGES['cleanup_warning']}")
            
            # Simulate cleanup process
            self._simulate_cleanup_process(cleanup_target)
            
            self._set_operation_state(None, SUCCESS_MESSAGES['cleanup_complete'])
            self._log_activity("Cleanup operation completed")
            self.logger.info("✅ Cleanup operation completed")
            
        except Exception as e:
            self._handle_operation_error("cleanup", e)
    
    @handle_errors(error_msg="Preview operation failed", reraise=False)
    def handle_preview(self) -> None:
        """Handle preview operation."""
        self.logger.info("👁️ Starting preview operation")
        
        try:
            self._set_operation_state(AugmentationOperation.PREVIEW, "Generating preview...")
            
            # Simulate preview generation
            self._simulate_preview_process()
            
            self._set_operation_state(None, SUCCESS_MESSAGES['preview_ready'])
            self._log_activity("Preview generated successfully")
            self.logger.info("✅ Preview operation completed")
            
        except Exception as e:
            self._handle_operation_error("preview", e)
    
    def _on_config_change(self, change) -> None:
        """Handle configuration changes."""
        try:
            # Update configuration in real-time
            config = self._get_current_config()
            
            # Validate new configuration
            is_valid, errors = self.config_handler.validate_config(config)
            
            if not is_valid:
                self.logger.warning(f"⚠️ Configuration validation failed: {errors}")
            else:
                self._update_dataset_stats()
                self.logger.debug("📝 Configuration updated successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Configuration change error: {e}")
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Extract current configuration from UI components."""
        return self.config_handler.extract_ui_config(self.ui_components)
    
    def _set_operation_state(self, operation: Optional[AugmentationOperation], message: str) -> None:
        """Update operation state and UI."""
        self._current_operation = operation.value if operation else None
        
        # Update operation summary if available
        if 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            if operation:
                status = 'processing'
            else:
                status = 'success' if 'completed' in message or 'ready' in message else 'info'
            
            if 'status' in update_methods:
                update_methods['status'](status, message)
    
    def _update_button_states(self, augment_running: bool = False) -> None:
        """Update button states during operations."""
        for button_key, button in self._buttons.items():
            if button_key == 'augment_button':
                button.disabled = augment_running
            else:
                button.disabled = augment_running
    
    def _update_dataset_stats(self) -> None:
        """Update dataset statistics in summary."""
        if 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            if 'dataset_stats' in update_methods:
                # Simulate stats (replace with actual data)
                update_methods['dataset_stats'](1250, 2500, 12)
    
    def _log_activity(self, message: str) -> None:
        """Log activity to summary container."""
        if 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            if 'activity' in update_methods:
                update_methods['activity'](message)
    
    def _handle_validation_errors(self, errors: list) -> None:
        """Handle configuration validation errors."""
        error_msg = "; ".join(errors)
        self._set_operation_state(None, f"❌ Validation failed: {error_msg}")
        self._log_activity(f"Configuration errors: {error_msg}")
        self.logger.error(f"❌ Validation errors: {errors}")
    
    def _handle_operation_error(self, operation: str, error: Exception) -> None:
        """Handle operation errors."""
        error_msg = ERROR_MESSAGES.get(f'{operation}_failed', f'{operation} operation failed')
        self._set_operation_state(None, f"❌ {error_msg}: {str(error)}")
        self._log_activity(f"Error in {operation}: {str(error)}")
        self.logger.error(f"❌ {operation} error: {error}")
    
    # Simulation methods (replace with actual backend calls)
    def _simulate_augmentation_process(self, config: Dict[str, Any]) -> None:
        """Simulate augmentation process with progress updates."""
        import time
        
        phases = [
            (ProcessingPhase.VALIDATION, 0.2),
            (ProcessingPhase.PROCESSING, 0.7), 
            (ProcessingPhase.FINALIZATION, 0.1)
        ]
        
        for phase, weight in phases:
            self._update_progress(phase.value, weight)
            time.sleep(0.5)  # Simulate processing time
    
    def _simulate_check_process(self) -> None:
        """Simulate check process."""
        self._update_dataset_stats()
        self._log_activity("Dataset structure validated")
    
    def _simulate_cleanup_process(self, target: str) -> None:
        """Simulate cleanup process."""
        self._log_activity(f"Cleaning up {target} files...")
        # Simulate cleanup logic
    
    def _simulate_preview_process(self) -> None:
        """Simulate preview generation."""
        self._log_activity("Preview images generated")
        # Simulate preview logic
    
    def _update_progress(self, phase: str, progress: float) -> None:
        """Update progress in summary container."""
        if 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            if 'progress' in update_methods:
                update_methods['progress'](progress, phase)