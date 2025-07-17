"""
file_path: smartcash/ui/dataset/visualization/operations/refresh_operation.py

Operasi untuk memperbarui visualisasi.
"""
from typing import Dict, Any, Optional
from smartcash.ui.dataset.visualization.operations.visualization_base_operation import VisualizationBaseOperation

class RefreshVisualizationOperation(VisualizationBaseOperation):
    """Operasi untuk memperbarui visualisasi."""
    
    def __init__(self, ui_module, config=None, callbacks=None):
        """Inisialisasi operasi refresh.
        
        Args:
            ui_module: Referensi ke modul UI yang memanggil operasi ini
            config: Konfigurasi untuk operasi
            callbacks: Callback untuk event operasi
        """
        super().__init__(
            ui_module=ui_module,
            config=config or {},
            callbacks=callbacks
        )
        self.name = "refresh_visualization"
        self.description = "Memperbarui tampilan visualisasi dengan pengaturan saat ini"
    
    def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementasi eksekusi operasi refresh.
        
        Returns:
            Dict berisi hasil operasi
        """
        self.logger.info("Memulai proses refresh visualisasi...")
        
        try:
            # Dapatkan tipe visualisasi saat ini
            viz_type = self.ui_module.components.get('visualization_type', 'bar')
            
            # Perbarui visualisasi
            self.ui_module.update_visualization(viz_type)
            
            self.logger.info("Visualisasi berhasil diperbarui")
            return {
                "success": True,
                "message": "Visualisasi berhasil diperbarui",
                "visualization_type": viz_type
            }
            
        except Exception as e:
            error_msg = f"Gagal memperbarui visualisasi: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }
