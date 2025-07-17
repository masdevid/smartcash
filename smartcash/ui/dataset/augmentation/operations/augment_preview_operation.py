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
    
    def _load_preview_to_widget(self) -> bool:
        """
        Memuat gambar preview ke widget UI.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if not self._preview_path or not os.path.exists(self._preview_path):
                self.log_warning("Path preview tidak valid")
                return False
                
            file_size = os.path.getsize(self._preview_path)
            if file_size == 0:
                self.log_warning("File preview kosong")
                return False
                
            with open(self._preview_path, 'rb') as f:
                image_data = f.read()
                
            if len(image_data) == 0:
                self.log_warning("Tidak ada data gambar yang terbaca")
                return False
                
            preview_image = self._ui_components.get('preview_image')
            if preview_image and hasattr(preview_image, 'value'):
                preview_image.value = image_data
                self.log_info("Preview berhasil dimuat ke widget")
                return True
                
            return False
            
        except Exception as e:
            self.log_error(f"Gagal memuat preview: {str(e)}")
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
            if not self._load_preview_to_widget():
                return self._handle_error("Gagal memuat preview ke UI")
            
            # Update status
            self.update_operation_status("Preview berhasil dibuat", "success")
            
            return {
                'status': 'success',
                'message': 'Preview berhasil dibuat',
                'preview_path': self._preview_path
            }
            
        except Exception as e:
            error_msg = f"Gagal membuat preview: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg, e)
    
# Factory function has been moved to augment_factory.py
