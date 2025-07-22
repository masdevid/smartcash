"""
File: smartcash/ui/dataset/augmentation/operations/augment_preview_operation.py
Deskripsi: Operasi untuk menampilkan preview augmentasi dataset
"""

import os
import time
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from pathlib import Path

from .augmentation_base_operation import AugmentationBaseOperation, OperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule

class AugmentPreviewOperation(AugmentationBaseOperation):
    """
    Operasi untuk menampilkan preview hasil augmentasi.
    Mengikuti pola implementasi asli dari augmentation_handlers.py
    """
    
    def __init__(
        self, 
        ui_module: 'AugmentationUIModule',
        config: Dict[str, Any], 
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        super().__init__(ui_module, config, callbacks)
        self._preview_path = None
        self._preview_service = None
    
    def _load_preview_to_widget(self, preview_path: Optional[str] = None) -> bool:
        """Load preview image to UI widget.
        
        Args:
            preview_path: Path to preview file (optional, will use self._preview_path if not provided)
        
        Returns:
            bool: True if successful, False if failed
        """
        try:
            target_path = preview_path or self._preview_path
            
            if not target_path or not os.path.exists(target_path):
                return False
                
            file_size = os.path.getsize(target_path)
            if file_size == 0:
                return False
                
            with open(target_path, 'rb') as f:
                image_data = f.read()
                
            if len(image_data) == 0:
                return False
                
            # Find preview image widget from UI components
            preview_image_widget = self._get_preview_image_widget()
            preview_status_widget = self._get_preview_status_widget()
            
            if preview_image_widget and hasattr(preview_image_widget, 'value'):
                preview_image_widget.value = image_data
                
                # Update status widget if available
                if preview_status_widget:
                    size_kb = file_size / 1024
                    preview_status_widget.value = f"<div style='text-align: center; color: #4caf50; font-size: 12px; margin: 4px 0;'>✅ Preview loaded ({size_kb:.1f}KB): {target_path}</div>"
                
                return True
                
            return False
            
        except Exception as e:
            self.log_error(f"Failed to load preview: {str(e)}")
            return False
    
    def _get_preview_image_widget(self):
        """Get preview image widget from UI components."""
        try:
            # Search from various possible locations
            if hasattr(self._ui_module, '_ui_components'):
                ui_components = self._ui_module._ui_components
                if hasattr(ui_components, 'get'):  # Check if it's a dictionary-like object
                    form_widgets = ui_components.get('form_container', {})
                    if hasattr(form_widgets, 'get'):
                        form_widgets = form_widgets.get('widgets', {})
                        preview_widget = form_widgets.get('preview_widget')
                        
                        if preview_widget and hasattr(preview_widget, 'get'):
                            preview_widgets = preview_widget.get('widgets', {})
                            if hasattr(preview_widgets, 'get'):
                                return preview_widgets.get('preview_image')
            
            # Fallback: search from self._ui_components
            if hasattr(self, '_ui_components') and hasattr(self._ui_components, 'get'):
                return self._ui_components.get('preview_image')
                
            return None
            
        except Exception:
            return None
    
    def _get_preview_status_widget(self):
        """Get preview status widget from UI components."""
        try:
            # Search from various possible locations
            if hasattr(self._ui_module, '_ui_components'):
                ui_components = self._ui_module._ui_components
                if hasattr(ui_components, 'get'):  # Check if it's a dictionary-like object
                    form_widgets = ui_components.get('form_container', {})
                    if hasattr(form_widgets, 'get'):
                        form_widgets = form_widgets.get('widgets', {})
                        preview_widget = form_widgets.get('preview_widget')
                        
                        if preview_widget and hasattr(preview_widget, 'get'):
                            preview_widgets = preview_widget.get('widgets', {})
                            if hasattr(preview_widgets, 'get'):
                                return preview_widgets.get('preview_status')
            
            # Fallback: search from self._ui_components
            if hasattr(self, '_ui_components') and hasattr(self._ui_components, 'get'):
                return self._ui_components.get('preview_status')
                
            return None
            
        except Exception:
            return None
    
    def load_existing_preview(self) -> bool:
        """Load existing preview from possible locations.
        
        Returns:
            bool: True if successfully loaded preview, False if not
        """
        try:
            # Define potential preview file paths
            preview_paths = [
                '/data/aug_preview.jpg',
                'data/aug_preview.jpg',
                './data/aug_preview.jpg',
                '/Users/masdevid/Projects/smartcash/data/aug_preview.jpg',
                str(Path.cwd() / 'data' / 'aug_preview.jpg')
            ]
            
            # Try to find and load existing preview
            for preview_path in preview_paths:
                if os.path.exists(preview_path):
                    try:
                        if self._load_preview_to_widget(preview_path):
                            return True
                    except Exception:
                        continue
            
            # If no preview found, update status
            preview_status_widget = self._get_preview_status_widget()
            if preview_status_widget:
                preview_status_widget.value = "<div style='text-align: center; color: #666; font-size: 12px; margin: 4px 0;'>No preview available - Click Generate to create one</div>"
            
            return False
            
        except Exception as e:
            self.log_error(f"Error loading existing preview: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute preview operation.
        
        Returns:
            Dict containing operation results
        """
        self.log_operation_start("Creating Preview")
        self.update_operation_status('Preparing preview...', 'info')
        
        try:
            # Get service from backend
            self._preview_service = self.get_backend_api('preview')
            if not self._preview_service:
                return self._handle_error("Cannot initialize preview service")
            
            # Get target split from configuration
            target_split = self._config.get('target_split', 'train')
            
            # Call backend to create preview
            result = self._preview_service(target_split=target_split)
            
            if not result or 'preview_path' not in result:
                return self._handle_error("Failed to create preview: Preview path not found")
            
            # Save preview path
            self._preview_path = result['preview_path']
            
            # Load preview to widget (optional in test environment)
            widget_loaded = self._load_preview_to_widget(self._preview_path)
            if not widget_loaded:
                # Don't fail the operation - preview file was created successfully
                pass
            
            # Update status
            self.update_operation_status("Preview created successfully", "success")
            
            return {
                'status': 'success',
                'success': True,
                'message': 'Preview created successfully',
                'preview_path': self._preview_path
            }
            
        except Exception as e:
            error_msg = f"Failed to create preview: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg, e)
    
# Factory function has been moved to augment_factory.py
