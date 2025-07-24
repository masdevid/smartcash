"""
file_path: smartcash/ui/dataset/visualization/operations/load_augmented_operation.py

Operasi untuk memuat sampel data augmented.
"""
from typing import Dict, Any, Optional, List
from smartcash.ui.dataset.visualization.operations.visualization_base_operation import VisualizationBaseOperation

class LoadAugmentedOperation(VisualizationBaseOperation):
    """Operasi untuk memuat sampel data augmented."""
    
    def __init__(self, ui_module, config=None, callbacks=None):
        """Inisialisasi operasi load augmented.
        
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
        self.name = "load_augmented"
        self.description = "Memuat sampel data yang telah diaugmentasi"
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the load augmented operation.
        
        Returns:
            Dict[str, Any]: Operation result with success status and details
        """
        try:
            return self._execute_impl()
        except Exception as e:
            self.logger.error(f"Load augmented operation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to load augmented samples: {str(e)}",
                "error": str(e)
            }
    
    def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementasi eksekusi operasi load augmented.
        
        Returns:
            Dict berisi hasil operasi dan data sampel
        """
        self.logger.info("Memulai proses memuat sampel data augmented...")
        
        try:
            # Dapatkan direktori data dari konfigurasi dengan fallback
            data_dir = self._ui_module._config.get('data_dir', 'data')
            if not data_dir:
                # Fallback to default data directory
                data_dir = 'data'
                self.logger.warning("Using default data directory: 'data'")
            
            # Dapatkan layanan sampel
            samples_service = self.get_backend_api('samples_service')
            if not samples_service:
                raise RuntimeError("Layanan sampel tidak tersedia")
            
            # Muat sampel augmented
            result = samples_service['get_samples'](
                data_dir=data_dir,
                split='train',  # Use valid split parameter instead of sample_type
                max_samples=10  # Use max_samples instead of limit
            )
            
            if not result.get('success'):
                raise RuntimeError(result.get('message', 'Gagal memuat sampel augmented'))
            
            # Simpan sampel yang dimuat
            samples = result.get('samples', [])
            self._ui_module._datasets['augmented'] = samples
            
            # Perbarui visualisasi dengan sampel augmented
            self._ui_module.update_visualization('augmented_samples')
            
            self.logger.info(f"Berhasil memuat {len(samples)} sampel data augmented")
            return {
                "success": True,
                "message": f"Berhasil memuat {len(samples)} sampel data augmented",
                "sample_count": len(samples),
                "samples": samples[:5]  # Kembalikan 5 sampel pertama untuk ditampilkan
            }
            
        except Exception as e:
            error_msg = f"Gagal memuat sampel data augmented: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }
