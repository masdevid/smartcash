"""
File: smartcash/utils/training/training_callbacks.py
Author: Alfrida Sabar
Deskripsi: Sistem callback untuk proses training dengan event-handler pattern
"""

from typing import Dict, List, Callable, Any

class TrainingCallbacks:
    """
    Sistem callback untuk proses training dengan pendekatan event-handler.
    Mendukung berbagai tipe callback pada berbagai titik dalam training pipeline.
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi sistem callback.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger
        
        # Daftar event yang didukung
        self.supported_events = [
            'batch_end',    # Setelah setiap batch
            'epoch_end',    # Setelah setiap epoch
            'training_end', # Setelah proses training selesai
            'validation_end',  # Setelah validasi
            'checkpoint_saved', # Setelah checkpoint disimpan
        ]
        
        # Dictionary untuk menyimpan callback
        self.callbacks = {event: [] for event in self.supported_events}
    
    def register(self, event: str, callback: Callable) -> bool:
        """
        Register callback function untuk event tertentu.
        
        Args:
            event: Nama event yang valid
            callback: Fungsi callback yang akan dipanggil
            
        Returns:
            Bool yang menunjukkan keberhasilan registrasi
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            return True
        else:
            if self.logger:
                self.logger.warning(f"⚠️ Event tidak didukung: {event}")
            return False
    
    def unregister(self, event: str, callback: Callable) -> bool:
        """
        Unregister callback function dari event tertentu.
        
        Args:
            event: Nama event
            callback: Fungsi callback yang akan dihapus
            
        Returns:
            Bool yang menunjukkan keberhasilan penghapusan
        """
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            return True
        return False
    
    def clear(self, event: str = None) -> None:
        """
        Hapus semua callback untuk event tertentu atau semua event.
        
        Args:
            event: Nama event (optional, jika None akan menghapus semua)
        """
        if event is None:
            # Reset semua callback
            self.callbacks = {event: [] for event in self.supported_events}
        elif event in self.callbacks:
            # Reset callback untuk event tertentu
            self.callbacks[event] = []
    
    def trigger(self, event: str, **kwargs) -> None:
        """
        Jalankan semua callback untuk event tertentu.
        
        Args:
            event: Nama event
            **kwargs: Parameter yang akan diteruskan ke callback
        """
        if event not in self.callbacks:
            return
            
        for callback in self.callbacks[event]:
            try:
                callback(**kwargs)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Error pada callback {event}: {str(e)}")
    
    def get_callback_count(self, event: str = None) -> Dict[str, int]:
        """
        Dapatkan jumlah callback yang terdaftar.
        
        Args:
            event: Nama event (optional)
            
        Returns:
            Dict berisi jumlah callback per event atau untuk event tertentu
        """
        if event is not None:
            if event in self.callbacks:
                return {event: len(self.callbacks[event])}
            return {event: 0}
        
        return {event: len(callbacks) for event, callbacks in self.callbacks.items()}
    
    def add_default_callbacks(self) -> None:
        """
        Tambahkan callback default untuk monitoring.
        """
        # Progress tracking untuk batch
        def batch_progress(batch, loss, batch_size, **kwargs):
            if self.logger and batch % 20 == 0:
                self.logger.info(f"🔄 Batch {batch}: loss={loss:.4f}")
        
        # Callback untuk mencatat metrik validasi
        def validation_metrics(val_loss, metrics, **kwargs):
            if self.logger:
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                self.logger.info(f"📊 Validasi: loss={val_loss:.4f}, {metrics_str}")
        
        # Tambahkan callback default
        self.register('batch_end', batch_progress)
        self.register('validation_end', validation_metrics)