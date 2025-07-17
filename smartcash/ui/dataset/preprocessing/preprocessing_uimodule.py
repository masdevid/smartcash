# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/preprocessing_uimodule.py
Description: Final implementation of the Preprocessing Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Optional, Callable, Tuple

from smartcash.ui.core.base_ui_module import BaseUIModule

from .components.preprocessing_ui import create_preprocessing_ui_components
from .configs.preprocessing_config_handler import PreprocessingConfigHandler
from .configs.preprocessing_defaults import get_default_config


class PreprocessingUIModule(BaseUIModule):
    """
    Preprocessing UI Module.
    """

    def __init__(self):
        """Initialize the Preprocessing UI module."""
        super().__init__(
            module_name='preprocessing',
            parent_module='dataset'
        )
        self._required_components = [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'operation_container'
        ]
        self.logger.debug("PreprocessingUIModule initialized.")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_config()

    def create_config_handler(self, config: Dict[str, Any]) -> PreprocessingConfigHandler:
        """Creates a configuration handler instance."""
        return PreprocessingConfigHandler(config)

    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the UI components for the module."""
        return create_preprocessing_ui_components(config=config)

    def _register_default_operations(self) -> None:
        """Register default operation handlers for the preprocessing module.
        
        This method registers button handlers for all preprocessing operations.
        It handles both base button IDs and _button suffixed variants for compatibility.
        """
        super()._register_default_operations()
        
        # Log all available UI components for debugging
        self.logger.debug(f"All UI components: {list(self._ui_components.keys())}")
        
        # Define button handlers - we'll register both base and _button suffixed variants
        button_handlers = {
            'preprocess': self._operation_preprocess,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup,
            'save': self._operation_save,
            'reset': self._operation_reset
        }
        
        # Get all available button widgets from UI components
        button_widgets = {}
        
        # 1. First, collect all button-like widgets from UI components
        for key, widget in self._ui_components.items():
            # Check if it's a button or a button-like widget
            is_button = (
                key.endswith('_button') or  # Ends with _button
                key in button_handlers or   # Matches a known button ID
                (hasattr(widget, 'description') and hasattr(widget, 'on_click'))  # Has button-like attributes
            )
            
            if is_button and widget is not None:
                button_widgets[key] = widget
                self.logger.debug(f"Found button in UI components: {key} ({type(widget).__name__})")
        
        # 2. Check for buttons in the action container
        action_container = self._ui_components.get('action_container')
        if action_container:
            self.logger.debug(f"Action container type: {type(action_container).__name__}")
            
            # Try to get buttons from action container's buttons attribute
            if hasattr(action_container, 'buttons') and action_container.buttons:
                self.logger.debug(f"Action container buttons structure: {action_container.buttons}")
                
                # Helper function to add button with logging
                def add_button(btn_id, widget, source):
                    if btn_id and widget is not None:
                        if btn_id not in button_widgets:
                            button_widgets[btn_id] = widget
                            self.logger.debug(f"Added {source} button: {btn_id} ({type(widget).__name__})")
                        return True
                    return False
                
                # Check for action buttons
                if hasattr(action_container.buttons, 'get'):
                    # Handle action buttons
                    if 'action' in action_container.buttons and isinstance(action_container.buttons['action'], dict):
                        for btn_id, widget in action_container.buttons['action'].items():
                            add_button(btn_id, widget, 'action')
                    
                    # Handle primary button
                    if 'primary' in action_container.buttons and action_container.buttons['primary'] is not None:
                        primary_btn = action_container.buttons['primary']
                        if hasattr(primary_btn, 'description'):
                            btn_id = primary_btn.description.lower().replace(' ', '_')
                            add_button(btn_id, primary_btn, 'primary')
            
            # Try to find buttons in children if not found in buttons attribute
            if not button_widgets and hasattr(action_container, 'children'):
                for child in action_container.children:
                    if hasattr(child, 'description') and hasattr(child, 'on_click'):
                        btn_id = child.description.lower().replace(' ', '_')
                        if btn_id:
                            button_widgets[btn_id] = child
                            self.logger.debug(f"Found button in action container children: {btn_id}")
        
        # Log all found button widgets
        self.logger.info(f"Found {len(button_widgets)} button widgets: {list(button_widgets.keys())}")
        
        # Register handlers for each button variant that exists
        registered_handlers = set()
        
        for button_id, handler in button_handlers.items():
            # Try to register base ID handler
            if button_id in button_widgets:
                try:
                    widget = button_widgets[button_id]
                    if hasattr(widget, 'on_click'):
                        self.register_button_handler(button_id, handler)
                        registered_handlers.add(button_id)
                        self.logger.info(f"✅ Registered handler for button: {button_id} ({type(widget).__name__})")
                    else:
                        self.logger.warning(f"⚠️ Widget {button_id} is not clickable ({type(widget).__name__})")
                except Exception as e:
                    self.logger.error(f"❌ Failed to register handler for button '{button_id}': {str(e)}")
            
            # Try to register _button suffixed handler
            button_id_suffixed = f"{button_id}_button"
            if button_id_suffixed in button_widgets and button_id_suffixed not in registered_handlers:
                try:
                    widget = button_widgets[button_id_suffixed]
                    if hasattr(widget, 'on_click'):
                        self.register_button_handler(button_id_suffixed, handler)
                        registered_handlers.add(button_id_suffixed)
                        self.logger.info(f"✅ Registered handler for button: {button_id_suffixed} ({type(widget).__name__})")
                    else:
                        self.logger.warning(f"⚠️ Widget {button_id_suffixed} is not clickable ({type(widget).__name__})")
                except Exception as e:
                    self.logger.error(f"❌ Failed to register handler for button '{button_id_suffixed}': {str(e)}")
            
            # If neither variant was found, log a warning
            if button_id not in registered_handlers and button_id_suffixed not in registered_handlers:
                self.logger.warning(f"⚠️ Button not found or not clickable: {button_id} or {button_id_suffixed}")
        
        # Log summary of registered handlers
        if registered_handlers:
            self.logger.info(f"✅ Successfully registered {len(registered_handlers)} button handlers: {', '.join(registered_handlers)}")
        else:
            self.logger.warning("⚠️ No button handlers were registered!")
        
        # Setup button handlers
        try:
            self._setup_button_handlers()
            self.logger.debug("Button handlers setup completed")
        except Exception as e:
            self.logger.error(f"❌ Failed to setup button handlers: {str(e)}")
    
    def _handle_button_click(self, handler, button_id):
        """Handle button click events with error handling."""
        try:
            self.logger.debug(f"Button clicked: {button_id}")
            handler(None)  # Pass None as the button parameter
        except Exception as e:
            self.logger.error(f"Error in button '{button_id}' handler: {str(e)}")
            # You might want to show an error message to the user here

    def _initialize_progress_display(self) -> None:
        """Initialize progress display components."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container:
                self.progress_display = operation_container.get_widget('progress_display')
                if self.progress_display:
                    self.progress_display.set_visibility(False)
        except (KeyError, AttributeError) as e:
            self.logger.warning(f"Could not initialize progress display: {e}")

    def _operation_preprocess(self, button) -> None:
        """Handles the preprocess data operation."""
        self.run_operation('preprocess')

    def _operation_check(self, button) -> None:
        """Handles the check data operation."""
        self.run_operation('check')

    def _operation_cleanup(self, button) -> None:
        """Handles the cleanup data operation."""
        self.run_operation('cleanup')
        
    def _operation_save(self, button) -> None:
        """Handles the save button click event."""
        try:
            self.logger.info("Save button clicked")
            # Add save logic here
            self.show_success("Settings saved successfully!")
        except Exception as e:
            self.logger.error(f"Error in save operation: {str(e)}")
            self.show_error(f"Failed to save settings: {str(e)}")
    
    def _operation_reset(self, button) -> None:
        """Handles the reset button click event."""
        try:
            self.logger.info("Reset button clicked")
            # Add reset logic here
            self.show_info("Settings have been reset to default values")
        except Exception as e:
            self.logger.error(f"Error in reset operation: {str(e)}")
            self.show_error(f"Failed to reset settings: {str(e)}")

    def _run_operation(
        self, 
        operation_name: str, 
        **kwargs
    ) -> None:
        """Run a preprocessing operation."""
        if not hasattr(self, 'progress_display') or not self.progress_display:
            self._initialize_progress_display()
        super()._run_operation(operation_name, **kwargs)




    def _operation_check(self, button: Any) -> None:
        """Handles the check button click event."""
        self.log("Memulai operasi pemeriksaan...", 'info')
        self._run_operation(self._execute_check_operation, button)

    def _operation_cleanup(self, button: Any) -> None:
        """Handles the cleanup button click event by showing a confirmation dialog."""
        self.log("Tombol pembersihan diklik.", 'info')
        preprocessed_files, _ = self._get_preprocessed_data_stats()

        op_container = self.get_component('operation_container')
        if not op_container or not hasattr(op_container, 'show_dialog'):
            self.log("Komponen dialog tidak tersedia, menjalankan pembersihan secara langsung.", 'warning')
            self._run_operation(self._execute_cleanup_operation, button)
            return

        if preprocessed_files == 0:
            self.log("Tidak ada data yang sudah diproses untuk dibersihkan.", 'info')
            op_container.show_dialog(
                title="Tidak Ada untuk Dibersihkan",
                message="Tidak ada file yang dihasilkan oleh proses preprocessing yang ditemukan.",
                confirm_text="OK",
                on_cancel=None # No cancel button
            )
            return

        self.log(f"Menampilkan dialog konfirmasi untuk membersihkan {preprocessed_files} file...", 'info')
        message = (
            f"Anda akan menghapus {preprocessed_files} file yang telah diproses.\n\n"
            f"Tindakan ini tidak dapat diurungkan. Lanjutkan?"
        )
        op_container.show_dialog(
            title="Konfirmasi Pembersihan",
            message=message,
            on_confirm=lambda: self._run_operation(self._execute_cleanup_operation, button),
            confirm_text=f"Ya, Hapus {preprocessed_files} File",
            cancel_text="Batal",
            danger_mode=True
        )

    def _get_preprocessed_data_stats(self) -> Tuple[int, int]:
        """Gets the count of preprocessed and raw files from the backend."""
        try:
            self.log("Mengecek status dari backend...", 'info')
            status = get_preprocessing_status(config=self.get_current_config())

            if not status.get('service_ready'):
                self.log("Layanan backend tidak siap, mengasumsikan tidak ada file.", 'warning')
                return False

            stats = status.get('file_statistics', {}).get('train', {})
            preprocessed_files = stats.get('preprocessed_files', 0)
            raw_images = stats.get('raw_images', 0)

            self.log(f"File terdeteksi: {preprocessed_files} diproses, {raw_images} mentah.", 'info')
            return preprocessed_files, raw_images

        except Exception as e:
            self.show_error_dialog("Gagal Memeriksa Status", f"Tidak dapat memeriksa keberadaan file yang diproses: {e}")
            return 0, 0 # Fail safe

    # ==================== OPERATION EXECUTION METHODS ====================

    def _execute_preprocess_operation(self) -> None:
        """Executes the preprocessing operation after a confirmation dialog."""
        from .operations.preprocess_operation import PreprocessOperationHandler
        
        try:
            preprocessed_files, raw_images = self._get_preprocessed_data_stats()

            message = (
                f"Anda akan memulai pra-pemrosesan.\n\n"
                f"- Gambar mentah terdeteksi: {raw_images}\n"
                f"- File yang sudah diproses: {preprocessed_files}\n\n"
                f"Proses ini mungkin menimpa file yang ada. Lanjutkan?"
            )

            def on_confirm():
                handler = PreprocessOperationHandler(
                    ui_module=self,
                    config=self.get_current_config(),
                    callbacks={'on_success': self._update_operation_summary}
                )
                handler.execute()

            self.show_dialog(
                title="Konfirmasi Pra-pemrosesan",
                message=message,
                on_confirm=on_confirm,
                confirm_text="Lanjutkan",
                danger_mode=True
            )
        except Exception as e:
            self.show_error_dialog("Gagal Memeriksa Status", f"Tidak dapat memeriksa status pra-pemrosesan: {e}")

    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        updater = self.get_component('operation_summary_updater')
        if updater and callable(updater):
            self.log(f"Memperbarui ringkasan operasi.", 'debug')
            updater(content, title="Ringkasan Operasi", icon="📊", visible=True)
        else:
            self.log("Komponen updater ringkasan operasi tidak ditemukan atau tidak dapat dipanggil.", 'warning')

    def _execute_check_operation(self) -> None:
        """Executes the check operation."""
        from .operations.check_operation import CheckOperationHandler
        handler = CheckOperationHandler(
            ui_module=self,
            config=self.get_current_config(),
            callbacks={'on_success': self._update_operation_summary}
        )
        handler.execute()

    def _execute_cleanup_operation(self) -> None:
        """Executes the cleanup operation."""
        from .operations.cleanup_operation import CleanupOperationHandler
        handler = CleanupOperationHandler(
            ui_module=self,
            config=self.get_current_config(),
            callbacks={'on_success': self._update_operation_summary}
        )
        handler.execute()


def initialize_preprocessing_ui(display: bool = True, **kwargs: Any) -> PreprocessingUIModule:
    """
    Initializes and optionally displays the Preprocessing UI Module.

    Args:
        display: If True, the UI will be displayed in the output.
        **kwargs: Additional arguments to pass to the module.

    Returns:
        An instance of the PreprocessingUIModule.
    """
    module = PreprocessingUIModule(**kwargs)
    module.initialize()
    if display:
        module.display_ui()
    return module
