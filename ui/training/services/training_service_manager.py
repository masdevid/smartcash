"""
File: smartcash/ui/training/services/training_service_manager.py
Deskripsi: Manager untuk mengelola integrasi antara backend training dan UI training
"""

from typing import Dict, Any, Optional, Callable
import os
from pathlib import Path

from smartcash.common.logger import get_logger, SmartCashLogger
from smartcash.common.config.manager import SimpleConfigManager as ConfigManager, get_config_manager
from smartcash.ui.utils.notification_manager import NotificationManager


class TrainingServiceManager:
    """Manager untuk mengelola integrasi antara backend training dan UI training"""
    
    def __init__(
        self, 
        ui_components: Dict[str, Any] = None, 
        config: Dict[str, Any] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi training service manager
        
        Args:
            ui_components: Komponen UI untuk training
            config: Konfigurasi training
            logger: Logger untuk logging
        """
        self.ui_components = ui_components or {}
        self.config = config or {}
        self.logger = logger or get_logger(__name__)
        self._notification_manager = None
        
        # Inisialisasi notification manager jika ui_components tersedia
        if ui_components:
            self._notification_manager = NotificationManager(ui_components)
        
        # Inisialisasi config manager
        self.config_manager = get_config_manager()
        
        self.logger.info("🔄 Training service manager diinisialisasi")
    
    def register_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Register UI components untuk training
        
        Args:
            ui_components: Komponen UI untuk training
        """
        self.ui_components = ui_components
        
        # Inisialisasi notification manager
        self._notification_manager = NotificationManager(ui_components)
        
        self.logger.info("✅ UI components berhasil diregister")
    
    def register_model_manager(self, model_manager) -> None:
        """Register model manager untuk training
        
        Args:
            model_manager: Model manager untuk training
        """
        if not model_manager:
            self.logger.error("❌ Model manager tidak valid")
            return
        
        # Simpan model manager ke ui_components
        self.ui_components['model_manager'] = model_manager
        
        # Buat training service dari model manager
        self._create_training_service(model_manager)
        
        self.logger.info(f"✅ Model manager ({model_manager.model_type}) berhasil diregister")
    
    def register_config(self, config: Dict[str, Any]) -> None:
        """Register konfigurasi training
        
        Args:
            config: Konfigurasi training
        """
        self.config = config
        
        # Update config ke training service
        training_service = self.ui_components.get('training_service')
        if training_service:
            training_service.config = config.get('training', {})
        
        self.logger.info("✅ Konfigurasi training berhasil diregister")
    
    def _create_training_service(self, model_manager) -> None:
        """Buat training service dari model manager
        
        Args:
            model_manager: Model manager untuk training
        """
        try:
            # Dapatkan training service dari model manager
            backend_training_service = model_manager.get_training_service()
            
            # Buat adapter untuk training service
            from smartcash.ui.training.adapters import TrainingServiceAdapter
            training_service = TrainingServiceAdapter(backend_training_service, self.logger)
            
            # Simpan training service dan komponen terkait ke ui_components
            self.ui_components.update({
                'training_service': training_service,
                'checkpoint_service': backend_training_service.checkpoint_service,
                'metrics_tracker': backend_training_service.metrics_tracker,
                'backend_training_service': backend_training_service
            })
            
            self.logger.info(f"✅ Training service berhasil dibuat untuk model {model_manager.model_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat training service: {str(e)}")
    
    def validate_training_readiness(self) -> bool:
        """Validasi kesiapan training
        
        Returns:
            bool: True jika siap training, False jika tidak
        """
        model_manager = self.ui_components.get('model_manager')
        training_service = self.ui_components.get('training_service')
        
        # Basic checks
        if not model_manager:
            self._notify_error("Model manager tidak tersedia")
            return False
        
        if not training_service:
            self._notify_error("Training service tidak tersedia")
            return False
        
        # Check if model is built
        if not hasattr(model_manager, 'model') or not model_manager.model:
            try:
                self._notify_info("Membangun model...")
                model_manager.build_model()
            except Exception as e:
                self._notify_error(f"Gagal build model: {str(e)}")
                return False
        
        # Check dataset
        if not self._validate_dataset(model_manager):
            return False
        
        self._notify_success("Model siap untuk training")
        return True
    
    def _validate_dataset(self, model_manager) -> bool:
        """Validasi dataset untuk training
        
        Args:
            model_manager: Model manager untuk training
            
        Returns:
            bool: True jika dataset valid, False jika tidak
        """
        # Check dataset di model manager
        if hasattr(model_manager, 'get_data_loaders'):
            try:
                train_loader, val_loader = model_manager.get_data_loaders()
                if not train_loader:
                    self._notify_error("Dataset training tidak tersedia")
                    return False
                
                self._notify_info(f"Dataset training tersedia: {len(train_loader)} batch")
                return True
            except Exception as e:
                self._notify_error(f"Error saat memuat dataset: {str(e)}")
                return False
        
        # Check dataset di config
        if hasattr(model_manager, 'config') and 'dataset' in model_manager.config:
            dataset_config = model_manager.config['dataset']
            if not dataset_config:
                self._notify_error("Konfigurasi dataset tidak tersedia")
                return False
            
            self._notify_info("Konfigurasi dataset tersedia")
            return True
        
        self._notify_warning("Dataset tidak terdeteksi, akan menggunakan dummy dataset")
        return True
    
    def start_training(self) -> bool:
        """Mulai training dengan konfigurasi dari UI
        
        Returns:
            bool: True jika training berhasil dimulai, False jika gagal
        """
        # Validasi kesiapan training
        if not self.validate_training_readiness():
            return False
        
        training_service = self.ui_components.get('training_service')
        if not training_service:
            self._notify_error("Training service tidak tersedia")
            return False
        
        # Mulai training
        try:
            self._notify_process_start("Memulai training...")
            success = training_service.start_training()
            
            if success:
                self._notify_process_complete("Training berhasil dimulai")
            else:
                self._notify_process_error("Gagal memulai training")
            
            return success
        except Exception as e:
            self._notify_process_error(f"Error saat memulai training: {str(e)}")
            return False
    
    def stop_training(self) -> None:
        """Hentikan training yang sedang berjalan"""
        training_service = self.ui_components.get('training_service')
        if not training_service:
            return
        
        self._notify_warning("Menghentikan training...")
        training_service.stop_training()
    
    def reset_training_state(self) -> None:
        """Reset state training setelah selesai atau error"""
        training_service = self.ui_components.get('training_service')
        if not training_service:
            return
        
        training_service.reset_training_state()
        self._notify_info("Training state direset")
    
    def _notify_info(self, message: str) -> None:
        """Notifikasi informasi
        
        Args:
            message: Pesan informasi
        """
        if self._notification_manager:
            self._notification_manager.notify_info(message)
        self.logger.info(f"ℹ️ {message}")
    
    def _notify_success(self, message: str) -> None:
        """Notifikasi sukses
        
        Args:
            message: Pesan sukses
        """
        if self._notification_manager:
            self._notification_manager.notify_success(message)
        self.logger.success(f"✅ {message}")
    
    def _notify_warning(self, message: str) -> None:
        """Notifikasi peringatan
        
        Args:
            message: Pesan peringatan
        """
        if self._notification_manager:
            self._notification_manager.notify_warning(message)
        self.logger.warning(f"⚠️ {message}")
    
    def _notify_error(self, message: str) -> None:
        """Notifikasi error
        
        Args:
            message: Pesan error
        """
        if self._notification_manager:
            self._notification_manager.notify_error(message)
        self.logger.error(f"❌ {message}")
    
    def _notify_process_start(self, message: str) -> None:
        """Notifikasi awal proses
        
        Args:
            message: Pesan awal proses
        """
        if self._notification_manager:
            self._notification_manager.notify_process_start(message)
        self.logger.info(f"🚀 {message}")
    
    def _notify_process_complete(self, message: str) -> None:
        """Notifikasi akhir proses
        
        Args:
            message: Pesan akhir proses
        """
        if self._notification_manager:
            self._notification_manager.notify_process_complete(message)
        self.logger.success(f"✅ {message}")
    
    def _notify_process_error(self, message: str) -> None:
        """Notifikasi error proses
        
        Args:
            message: Pesan error proses
        """
        if self._notification_manager:
            self._notification_manager.notify_process_error(message)
        self.logger.error(f"❌ {message}")
    
    def update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar
        
        Args:
            current: Nilai progress saat ini
            total: Nilai progress total
            message: Pesan progress
        """
        if self._notification_manager:
            self._notification_manager.update_progress(current, total, message)
    
    def update_status(self, message: str, status_type: str = 'info') -> None:
        """Update status panel
        
        Args:
            message: Pesan status
            status_type: Tipe status (info, success, warning, error)
        """
        if self._notification_manager:
            self._notification_manager.update_status(message, status_type)
