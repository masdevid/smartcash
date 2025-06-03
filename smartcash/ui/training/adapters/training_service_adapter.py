"""
File: smartcash/ui/training/adapters/training_service_adapter.py
Deskripsi: Adapter untuk mengintegrasikan TrainingService dari backend dengan UI training
"""

from typing import Dict, Any, Optional, Callable, List, Union
import torch
from torch.utils.data import DataLoader

from smartcash.model.service.training_service import TrainingService
from smartcash.common.logger import get_logger


class TrainingServiceAdapter:
    """Adapter untuk mengintegrasikan TrainingService dari backend dengan UI training"""
    
    def __init__(self, training_service: TrainingService, logger=None):
        """Inisialisasi adapter dengan training service dari backend
        
        Args:
            training_service: Instance TrainingService dari backend
            logger: Logger untuk logging (opsional)
        """
        self.training_service = training_service
        self.logger = logger or get_logger(__name__)
        self.progress_callback = None
        self.metrics_callback = None
        self.checkpoint_callback = None
        self.config = {}
        
        # Inisialisasi state
        self._training_running = False
        self._stop_requested = False
        
        self.logger.info("🔄 Training service adapter diinisialisasi")
    
    def set_progress_callbacks(self, progress_callback: Callable, metrics_callback: Callable, checkpoint_callback: Callable) -> None:
        """Set callback untuk progress, metrics, dan checkpoint
        
        Args:
            progress_callback: Callback untuk progress tracking
            metrics_callback: Callback untuk metrics reporting
            checkpoint_callback: Callback untuk checkpoint progress
        """
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        self.checkpoint_callback = checkpoint_callback
        
        # Buat callback dictionary untuk TrainingService
        callback_dict = {
            'progress': progress_callback,
            'metrics': metrics_callback,
            'epoch_end': lambda epoch, metrics, is_best: metrics_callback(epoch, metrics),
            'training_end': lambda final_metrics, total_time: self._handle_training_end(final_metrics, total_time),
            'training_error': lambda error_message, phase: self._handle_training_error(error_message, phase)
        }
        
        # Set callback ke training service
        self.training_service.set_callback(callback_dict)
        
        # Set checkpoint callback
        if hasattr(self.training_service, 'checkpoint_service') and self.training_service.checkpoint_service:
            self.training_service.checkpoint_service.set_progress_callback(checkpoint_callback)
        
        self.logger.info("✅ Progress callbacks berhasil diatur")
    
    def start_training(self) -> bool:
        """Mulai training dengan konfigurasi dari UI
        
        Returns:
            bool: True jika training berhasil, False jika gagal
        """
        if self._training_running:
            self.logger.warning("⚠️ Training sudah berjalan")
            return False
        
        try:
            # Dapatkan model manager dari training service
            model_manager = self.training_service.model_manager
            if not model_manager:
                self.logger.error("❌ Model manager tidak tersedia")
                return False
            
            # Dapatkan dataset dari model manager
            train_loader, val_loader = self._get_data_loaders(model_manager)
            if not train_loader:
                self.logger.error("❌ Dataset tidak tersedia")
                return False
            
            # Dapatkan konfigurasi training
            config = self._get_training_config()
            
            # Set flag training running
            self._training_running = True
            self._stop_requested = False
            
            # Jalankan training
            self.logger.info(f"🚀 Memulai training dengan {len(train_loader)} batch")
            result = self.training_service.train(
                train_loader=train_loader,
                val_loader=val_loader,
                **config
            )
            
            # Reset flag training running
            self._training_running = False
            
            return True
            
        except Exception as e:
            self._training_running = False
            self.logger.error(f"❌ Error saat memulai training: {str(e)}")
            return False
    
    def stop_training(self) -> None:
        """Hentikan training yang sedang berjalan"""
        if not self._training_running:
            return
        
        self._stop_requested = True
        self.training_service.stop_training()
        self.logger.info("🛑 Training dihentikan")
    
    def reset_training_state(self) -> None:
        """Reset state training setelah selesai atau error"""
        self._training_running = False
        self._stop_requested = False
        
        # Reset progress tracker
        if hasattr(self.training_service, '_progress_tracker'):
            self.training_service._progress_tracker = self.training_service._progress_tracker.__class__()
        
        # Reset metrics tracker
        if hasattr(self.training_service, '_metrics_tracker'):
            self.training_service._metrics_tracker = self.training_service._metrics_tracker.__class__()
        
        self.logger.info("🔄 Training state direset")
    
    def _get_data_loaders(self, model_manager) -> tuple:
        """Dapatkan data loader dari model manager
        
        Args:
            model_manager: Model manager yang berisi dataset
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        try:
            # Coba dapatkan dataset dari model manager
            if hasattr(model_manager, 'get_data_loaders'):
                return model_manager.get_data_loaders()
            
            # Fallback ke dataset dari atribut
            if hasattr(model_manager, 'train_loader') and hasattr(model_manager, 'val_loader'):
                return model_manager.train_loader, model_manager.val_loader
            
            # Fallback ke dataset dari config
            if hasattr(model_manager, 'config') and 'dataset' in model_manager.config:
                dataset_config = model_manager.config['dataset']
                if 'train_loader' in dataset_config and 'val_loader' in dataset_config:
                    return dataset_config['train_loader'], dataset_config['val_loader']
            
            self.logger.warning("⚠️ Tidak dapat menemukan dataset, mencoba fallback ke dummy dataset")
            
            # Fallback ke dummy dataset
            return self._create_dummy_dataset(), None
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mendapatkan dataset: {str(e)}")
            return None, None
    
    def _create_dummy_dataset(self) -> DataLoader:
        """Buat dummy dataset untuk testing
        
        Returns:
            DataLoader: Dummy dataset loader
        """
        # Buat dummy tensor
        dummy_data = torch.randn(16, 3, 640, 640)
        dummy_target = torch.randn(16, 7, 5)  # 7 kelas, 5 koordinat (x, y, w, h, conf)
        
        # Buat dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 16
            
            def __getitem__(self, idx): return dummy_data[idx], dummy_target[idx]
        
        # Buat dummy dataloader
        return torch.utils.data.DataLoader(DummyDataset(), batch_size=4, shuffle=True)
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi training dari model manager atau default
        
        Returns:
            Dict[str, Any]: Konfigurasi training
        """
        # Default config
        default_config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'lr_scheduler': 'cosine',
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.001,
            'save_best': True,
            'save_interval': 10
        }
        
        # Gabungkan dengan config dari model manager
        if hasattr(self.training_service.model_manager, 'config'):
            model_config = self.training_service.model_manager.config
            if 'training' in model_config:
                training_config = model_config['training']
                default_config.update(training_config)
        
        # Gabungkan dengan config dari UI
        default_config.update(self.config)
        
        return default_config
    
    def _handle_training_end(self, final_metrics: Dict[str, float], total_time: float) -> None:
        """Handle training end callback
        
        Args:
            final_metrics: Metrik akhir training
            total_time: Total waktu training
        """
        self._training_running = False
        
        # Log hasil training
        self.logger.info(f"✅ Training selesai dalam {total_time:.2f} detik")
        for metric_name, metric_value in final_metrics.items():
            self.logger.info(f"📊 {metric_name}: {metric_value:.4f}")
    
    def _handle_training_error(self, error_message: str, phase: str) -> None:
        """Handle training error callback
        
        Args:
            error_message: Pesan error
            phase: Fase training saat error terjadi
        """
        self._training_running = False
        self.logger.error(f"❌ Training error pada fase {phase}: {error_message}")
