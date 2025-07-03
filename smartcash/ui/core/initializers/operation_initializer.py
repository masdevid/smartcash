"""
File: smartcash/ui/core/initializers/operation_initializer.py
Deskripsi: Operation initializer dengan fail-fast principle
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.initializers.base_initializer import BaseInitializer

class OperationInitializer(BaseInitializer):
    """Initializer untuk operation dengan fail-fast principle."""
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: str = None,
                 enable_progress: bool = True,
                 enable_summary: bool = True,
                 enable_dialogs: bool = True):
        super().__init__(module_name, parent_module)
        self._enable_progress = enable_progress
        self._enable_summary = enable_summary
        self._enable_dialogs = enable_dialogs
        self._progress_tracker = None
        
        self.logger.debug(f"🔧 Initialized OperationInitializer for {self.full_module_name}")
    
    def create_operation_ui(self, title: str, description: str, 
                           custom_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create operation UI dengan fail-fast validation."""
        ui_components = {}
        
        # Create components - fail-fast if any critical component fails
        ui_components.update(self._create_header_components(title, description))
        ui_components.update(self._create_action_components())
        
        if self._enable_progress:
            ui_components.update(self._create_progress_components())
        
        ui_components.update(self._create_log_components())
        
        if self._enable_summary:
            ui_components.update(self._create_summary_components())
        
        if self._enable_dialogs:
            ui_components.update(self._create_dialog_components())
        
        if custom_components:
            ui_components.update(custom_components)
        
        ui_components.update(self._create_main_container(ui_components))
        
        self.logger.info(f"✅ Created operation UI with {len(ui_components)} components")
        return ui_components
    
    def _create_header_components(self, title: str, description: str) -> Dict[str, Any]:
        """Create header components."""
        try:
            from smartcash.ui.components.header import create_header
            
            header = create_header(title=title, subtitle=description, show_divider=True)
            if header is None:
                raise RuntimeError("create_header returned None")
            
            return {'header': header, 'header_container': header}
        except ImportError:
            raise ImportError("Header components module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create header components: {str(e)}")
    
    def _create_action_components(self) -> Dict[str, Any]:
        """Create action components."""
        try:
            from smartcash.ui.components.action_container import create_action_container
            
            action_buttons = [
                {"button_id": "start", "text": "▶️ Start", "style": "primary", "order": 1},
                {"button_id": "cancel", "text": "⏹️ Cancel", "style": "danger", "order": 2}
            ]
            
            action_container = create_action_container(
                buttons=action_buttons, title="Actions", alignment="center"
            )
            if action_container is None:
                raise RuntimeError("create_action_container returned None")
            
            return {
                'action_container': action_container,
                'start_button': action_container.get('buttons', {}).get('start'),
                'cancel_button': action_container.get('buttons', {}).get('cancel')
            }
        except ImportError:
            raise ImportError("Action container module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create action components: {str(e)}")
    
    def _create_progress_components(self) -> Dict[str, Any]:
        """Create progress components."""
        try:
            from smartcash.ui.components.progress_tracker import ProgressTracker
            
            self._progress_tracker = ProgressTracker()
            if self._progress_tracker is None:
                raise RuntimeError("ProgressTracker creation returned None")
            
            return {
                'progress_tracker': self._progress_tracker,
                'progress_bar': self._progress_tracker
            }
        except ImportError:
            raise ImportError("Progress tracker module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create progress components: {str(e)}")
    
    def _create_log_components(self) -> Dict[str, Any]:
        """Create log components."""
        try:
            from smartcash.ui.components.log_accordion import create_log_accordion
            
            log_accordion = create_log_accordion()
            if log_accordion is None:
                raise RuntimeError("create_log_accordion returned None")
            
            # Expand accordion by default
            if isinstance(log_accordion, dict) and 'log_accordion' in log_accordion:
                accordion_widget = log_accordion['log_accordion']
                if hasattr(accordion_widget, 'selected_index'):
                    accordion_widget.selected_index = 0
            
            return {
                'log_accordion': log_accordion,
                'log_output': log_accordion,
                'log_components': log_accordion
            }
        except ImportError:
            raise ImportError("Log accordion module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create log components: {str(e)}")
    
    def _create_summary_components(self) -> Dict[str, Any]:
        """Create summary components."""
        try:
            from smartcash.ui.components.summary_container import create_summary_container
            
            summary_container = create_summary_container(
                title="Operation Summary", theme="info", icon="📋"
            )
            if summary_container is None:
                raise RuntimeError("create_summary_container returned None")
            
            return {
                'summary_container': summary_container,
                'summary_output': summary_container
            }
        except ImportError:
            raise ImportError("Summary container module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create summary components: {str(e)}")
    
    def _create_dialog_components(self) -> Dict[str, Any]:
        """Create dialog components."""
        try:
            from smartcash.ui.components.dialog import create_confirmation_area
            
            confirmation_area = create_confirmation_area()
            if confirmation_area is None:
                raise RuntimeError("create_confirmation_area returned None")
            
            return {
                'confirmation_area': confirmation_area,
                'dialog_area': confirmation_area,
                'error_area': confirmation_area
            }
        except ImportError:
            raise ImportError("Dialog components module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create dialog components: {str(e)}")
    
    def _create_main_container(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Create main container."""
        try:
            from smartcash.ui.components.main_container import create_main_container
            
            container_components = {}
            
            if 'header' in ui_components:
                container_components['header_container'] = ui_components['header']
            if 'action_container' in ui_components:
                container_components['form_container'] = ui_components['action_container']
            if 'summary_container' in ui_components:
                container_components['summary_container'] = ui_components['summary_container']
            
            main_container = create_main_container(**container_components)
            if main_container is None:
                raise RuntimeError("create_main_container returned None")
            
            return {'main_container': main_container, 'ui': main_container}
        except ImportError:
            raise ImportError("Main container module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to create main container: {str(e)}")
    
    def update_progress(self, value: float, message: Optional[str] = None) -> None:
        """Update progress tracker dengan fail-fast."""
        if self._progress_tracker is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if not hasattr(self._progress_tracker, 'update'):
            raise RuntimeError("Progress tracker has no update method")
        
        try:
            self._progress_tracker.update(value, message)
        except Exception as e:
            raise RuntimeError(f"Failed to update progress: {str(e)}")
    
    def reset_progress(self) -> None:
        """Reset progress tracker dengan fail-fast."""
        if self._progress_tracker is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if not hasattr(self._progress_tracker, 'reset'):
            raise RuntimeError("Progress tracker has no reset method")
        
        try:
            self._progress_tracker.reset()
        except Exception as e:
            raise RuntimeError(f"Failed to reset progress: {str(e)}")
    
    def show_confirmation(self, message: str, on_confirm: Callable, 
                         on_cancel: Optional[Callable] = None, title: str = "Konfirmasi") -> None:
        """Show confirmation dialog dengan fail-fast."""
        try:
            from smartcash.ui.components.dialog import show_confirmation_dialog
            
            show_confirmation_dialog(
                title=title, message=message, on_confirm=on_confirm, on_cancel=on_cancel
            )
        except ImportError:
            raise ImportError("Dialog module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to show confirmation dialog: {str(e)}")
    
    def show_info_dialog(self, title: str, message: str) -> None:
        """Show info dialog dengan fail-fast."""
        try:
            from smartcash.ui.components.dialog import show_info_dialog
            
            show_info_dialog(title, message)
        except ImportError:
            raise ImportError("Dialog module not available")
        except Exception as e:
            raise RuntimeError(f"Failed to show info dialog: {str(e)}")
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize operation."""
        return {
            'success': True,
            'message': f"Operation initializer ready for {self.full_module_name}",
            'module': self.full_module_name
        }