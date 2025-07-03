"""
File: smartcash/ui/dataset/augmentation/handlers/operation/preview.py
Deskripsi: Preview operation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging
import os
import ipywidgets as widgets
from IPython.display import display, Image

# Import base operation handler
from smartcash.ui.dataset.augmentation.handlers.operation.base_operation import BaseOperationHandler

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors


class PreviewOperationHandler(BaseOperationHandler):
    """Preview operation handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for preview generation operation:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize preview operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger.debug("PreviewOperationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute preview operation dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Clear UI outputs
        self.clear_ui_outputs()
        
        # Extract config if not provided
        if not config:
            config_handler = self.ui_components.get('config_handler')
            config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Generate preview image
        result = self._generate_preview_image(config)
        
        return result
    
    @handle_ui_errors(log_error=True)
    def check_and_load_existing_preview(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check dan load existing preview image saat UI initialization
        
        Args:
            config: Dictionary konfigurasi augmentation
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get preview path from config
        preview_path = config.get('augmentation', {}).get('preview_path', '')
        
        # Check if preview path exists
        if preview_path and os.path.exists(preview_path):
            self.logger.info(f"🔍 Loading existing preview dari {preview_path}")
            
            # Load preview image
            result = self._load_preview_to_widget(preview_path)
            
            # Update preview status
            if result.get('status'):
                self._update_preview_status("success", "Preview loaded successfully")
            else:
                self._update_preview_status("error", result.get('message', 'Failed to load preview'))
            
            return result
        else:
            self.log_info("ℹ️ Tidak ada preview yang tersedia")
            self._update_preview_status("info", "No preview available")
            
            return {'status': False, 'message': 'No preview available'}
    
    @handle_ui_errors(log_error=True)
    def _generate_preview_image(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate preview image dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Set button states
        self.disable_all_buttons(self.ui_components)
        
        # Create progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("Generating preview...")
        
        # Log preview generation
        self.logger.info("📷 Memulai pembuatan preview...")
        
        # Generate preview using backend_utils
        from smartcash.ui.dataset.augmentation.utils.backend_utils import create_live_preview
        preview_result = create_live_preview(config)
        
        # Check result status - using 'status' key for consistency (not 'success')
        if preview_result.get('status'):
            # Get preview path
            preview_path = preview_result.get('preview_path', '')
            
            # Load preview image
            load_result = self._load_preview_to_widget(preview_path)
            
            # Update preview status
            if load_result.get('status'):
                self._update_preview_status("success", "Preview generated successfully")
                
                # Update progress tracker
                if 'progress_tracker' in self.ui_components:
                    self.ui_components['progress_tracker'].complete("Preview generated successfully")
                
                # Update config with preview path
                config_handler = self.ui_components.get('config_handler')
                if config_handler:
                    config = config_handler.get_config()
                    if 'augmentation' not in config:
                        config['augmentation'] = {}
                    
                    config['augmentation']['preview_path'] = preview_path
                    config_handler.save_config(config)
                
                # Update summary panel
                self.update_operation_summary({
                    'status': True,
                    'message': 'Preview generated successfully',
                    'preview_path': preview_path
                })
                
                # Reset button states
                self.enable_all_buttons(self.ui_components)
                
                return {'status': True, 'message': 'Preview generated successfully', 'preview_path': preview_path}
            else:
                self._update_preview_status("error", load_result.get('message', 'Failed to load preview'))
                
                # Update progress tracker
                if 'progress_tracker' in self.ui_components:
                    self.ui_components['progress_tracker'].error("Failed to load preview")
                
                # Update summary panel
                self.update_operation_summary({
                    'status': False,
                    'message': load_result.get('message', 'Failed to load preview')
                })
                
                # Reset button states
                self.enable_all_buttons(self.ui_components)
                
                return load_result
        else:
            # Update preview status
            self._update_preview_status("error", preview_result.get('message', 'Failed to generate preview'))
            
            # Update progress tracker
            if 'progress_tracker' in self.ui_components:
                self.ui_components['progress_tracker'].error(preview_result.get('message', 'Failed to generate preview'))
            
            # Update summary panel
            self.update_operation_summary({
                'status': False,
                'message': preview_result.get('message', 'Failed to generate preview')
            })
            
            # Reset button states
            self.enable_all_buttons(self.ui_components)
            
            return preview_result
    
    @handle_ui_errors(log_error=True)
    def _load_preview_to_widget(self, preview_path: str) -> Dict[str, Any]:
        """Load preview image ke widget dengan centralized error handling
        
        Args:
            preview_path: Path ke preview image
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # Check if preview path exists
        if not os.path.exists(preview_path):
            return {'status': False, 'message': f'Preview path tidak ditemukan: {preview_path}'}
        
        # Get preview image widget
        preview_image = self.ui_components.get('preview_image')
        if not preview_image:
            return {'status': False, 'message': 'Preview image widget tidak ditemukan'}
        
        try:
            # Load image
            with open(preview_path, 'rb') as f:
                image_data = f.read()
            
            # Display image
            preview_image.value = image_data
            
            # Log success
            self.log_info(f"✅ Preview image berhasil dimuat dari {preview_path}")
            
            return {'status': True, 'message': 'Preview image loaded successfully'}
        except Exception as e:
            # Log error
            self.log_error(f"❌ Error loading preview image: {str(e)}")
            
            return {'status': False, 'message': f'Error loading preview image: {str(e)}'}
    
    @handle_ui_errors(log_error=True)
    def _update_preview_status(self, status: str, message: str) -> None:
        """Update preview status dengan centralized error handling
        
        Args:
            status: Status preview (success, error, info)
            message: Message to display
        """
        # Get preview status widget
        preview_status = self.ui_components.get('preview_status')
        if not preview_status:
            return
        
        # Determine style based on status - using 'status' key for consistency across all handlers
        style = ""
        if status == "success" or status == "true":
            style = "color: #5cb85c;"
        elif status == "error" or status == "false":
            style = "color: #d9534f;"
        elif status == "info":
            style = "color: #0275d8;"
        
        # Update preview status
        preview_status.value = f"<div style='{style}'>{message}</div>"
    
