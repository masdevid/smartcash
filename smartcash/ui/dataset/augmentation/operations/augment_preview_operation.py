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
        """
        Memuat gambar preview ke widget UI.
        
        Args:
            preview_path: Path ke file preview (optional, akan menggunakan self._preview_path jika tidak ada)
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Gunakan path yang diberikan atau self._preview_path
            target_path = preview_path or self._preview_path
            
            if not target_path or not os.path.exists(target_path):
                self.log_warning(f"Path preview tidak valid: {target_path}")
                return False
                
            file_size = os.path.getsize(target_path)
            if file_size == 0:
                self.log_warning("File preview kosong")
                return False
                
            with open(target_path, 'rb') as f:
                image_data = f.read()
                
            if len(image_data) == 0:
                self.log_warning("Tidak ada data gambar yang terbaca")
                return False
                
            # Cari preview image widget dari UI components
            preview_image_widget = self._get_preview_image_widget()
            preview_status_widget = self._get_preview_status_widget()
            
            if preview_image_widget and hasattr(preview_image_widget, 'value'):
                preview_image_widget.value = image_data
                self.log_info(f"Preview berhasil dimuat ke widget dari: {target_path}")
                
                # Update status widget jika ada
                if preview_status_widget:
                    size_kb = file_size / 1024
                    preview_status_widget.value = f"<div style='text-align: center; color: #4caf50; font-size: 12px; margin: 4px 0;'>✅ Preview loaded ({size_kb:.1f}KB): {target_path}</div>"
                
                return True
                
            self.log_warning("Preview image widget tidak ditemukan")
            return False
            
        except Exception as e:
            self.log_error(f"Gagal memuat preview: {str(e)}")
            return False
    
    def _get_preview_image_widget(self):
        """Mendapatkan widget gambar preview dari UI components."""
        try:
            # Cari dari berbagai lokasi yang mungkin
            if hasattr(self._ui_module, '_ui_components') and self._ui_module._ui_components:
                form_widgets = self._ui_module._ui_components.get('form_container', {}).get('widgets', {})
                preview_widget = form_widgets.get('preview_widget')
                
                if preview_widget and isinstance(preview_widget, dict):
                    preview_widgets = preview_widget.get('widgets', {})
                    return preview_widgets.get('preview_image')
            
            # Fallback: cari dari self._ui_components
            return self._ui_components.get('preview_image')
            
        except Exception as e:
            self.log_warning(f"Error getting preview image widget: {e}")
            return None
    
    def _get_preview_status_widget(self):
        """Mendapatkan widget status preview dari UI components."""
        try:
            # Cari dari berbagai lokasi yang mungkin
            if hasattr(self._ui_module, '_ui_components') and self._ui_module._ui_components:
                form_widgets = self._ui_module._ui_components.get('form_container', {}).get('widgets', {})
                preview_widget = form_widgets.get('preview_widget')
                
                if preview_widget and isinstance(preview_widget, dict):
                    preview_widgets = preview_widget.get('widgets', {})
                    return preview_widgets.get('preview_status')
            
            # Fallback: cari dari self._ui_components
            return self._ui_components.get('preview_status')
            
        except Exception as e:
            self.log_warning(f"Error getting preview status widget: {e}")
            return None
    
    def load_existing_preview(self) -> bool:
        """
        Memuat preview yang sudah ada dari berbagai lokasi yang mungkin.
        Fungsi ini diintegrasikan dari _load_existing_preview di UI module.
        
        Returns:
            bool: True jika berhasil memuat preview, False jika tidak
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
                            self.log_info(f"Existing preview loaded successfully from: {preview_path}")
                            return True
                    except Exception as e:
                        self.log_warning(f"Failed to load preview from {preview_path}: {e}")
                        continue
            
            # If no preview found, update status
            preview_status_widget = self._get_preview_status_widget()
            if preview_status_widget:
                preview_status_widget.value = "<div style='text-align: center; color: #666; font-size: 12px; margin: 4px 0;'>No preview available - Click Generate to create one</div>"
            
            self.log_info("No existing preview found in standard locations")
            return False
            
        except Exception as e:
            self.log_error(f"Error loading existing preview: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """
        Eksekusi operasi preview.
        
        Returns:
            Dict berisi hasil operasi
        """
        self.log_operation_start("Membuat Preview")
        self.update_operation_status('Menyiapkan preview...', 'info')
        
        try:
            # Dapatkan service dari backend
            self._preview_service = self.get_backend_api('create_live_preview')
            if not self._preview_service:
                return self._handle_error("Tidak dapat menginisialisasi layanan preview")
            
            # Dapatkan target split dari konfigurasi
            target_split = self._config.get('target_split', 'train')
            self.log_info(f"Membuat preview untuk split: {target_split}")
            
            # Panggil backend untuk membuat preview
            result = self._preview_service(
                target_split=target_split,
                config=self._config
            )
            
            if not result or 'preview_path' not in result:
                return self._handle_error("Gagal membuat preview: Path preview tidak ditemukan")
            
            # Simpan path preview
            self._preview_path = result['preview_path']
            
            # Muat preview ke widget
            if not self._load_preview_to_widget(self._preview_path):
                return self._handle_error("Gagal memuat preview ke UI")
            
            # Update status
            self.update_operation_status("Preview berhasil dibuat", "success")
            
            return {
                'status': 'success',
                'success': True,
                'message': 'Preview berhasil dibuat',
                'preview_path': self._preview_path
            }
            
        except Exception as e:
            error_msg = f"Gagal membuat preview: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg, e)
    
# Factory function has been moved to augment_factory.py
