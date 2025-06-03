"""
File: smartcash/ui/training/services/training_service_manager.py
Deskripsi: Manager untuk mengkoordinasikan training service dengan UI components
"""

import torch
from typing import Dict, Any, Optional, Callable
from smartcash.ui.training.adapters.ui_training_callback import UITrainingCallback
from smartcash.ui.training.utils.training_progress_utils import update_training_progress, update_chart_display, update_metrics_display

class TrainingServiceManager:
    """Manager untuk mengkoordinasikan training service dengan UI components"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger):
        self.ui_components = ui_components
        self.config = config
        self.logger = logger
        self.model_manager = None
        self.training_service = None
        self.is_training = False
        
        # Setup UI callback
        self.ui_callback = UITrainingCallback(ui_components, logger)
        
    def register_model_manager(self, model_manager) -> None:
        """Register model manager untuk training"""
        self.model_manager = model_manager
        self.logger.info("✅ Model manager terdaftar untuk training")
        
    def register_config(self, config: Dict[str, Any]) -> None:
        """Register konfigurasi untuk training"""
        self.config.update(config)
        
    def start_training(self, training_config: Dict[str, Any]) -> None:
        """Start training dengan konfigurasi yang diberikan"""
        try:
            if self.is_training:
                self.logger.warning("⚠️ Training sudah berjalan")
                return
                
            # Validasi model manager
            if not self.model_manager:
                raise ValueError("Model manager belum terdaftar")
                
            # Setup training service
            self._setup_training_service()
            
            # Validasi data loader
            train_loader, val_loader = self._setup_data_loaders()
            
            # Update UI state
            self.is_training = True
            self._update_ui_training_state(True)
            
            # Start training
            self.logger.info("🚀 Memulai training dengan model service...")
            
            result = self.training_service.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=training_config.get('epochs', 100),
                learning_rate=training_config.get('learning_rate', 0.001),
                weight_decay=training_config.get('weight_decay', 0.0005),
                early_stopping=training_config.get('early_stopping', True),
                patience=training_config.get('patience', 10),
                save_best=training_config.get('save_best', True),
                save_interval=training_config.get('save_interval', 10),
                config=training_config
            )
            
            # Training selesai
            self._on_training_complete(result)
            
        except Exception as e:
            self._on_training_error(str(e))
            
    def stop_training(self) -> None:
        """Stop training yang sedang berjalan"""
        try:
            if not self.is_training:
                self.logger.warning("⚠️ Tidak ada training yang sedang berjalan")
                return
                
            if self.training_service:
                self.training_service.stop_training()
                
            self._update_ui_training_state(False)
            self.is_training = False
            self.logger.info("🛑 Training dihentikan")
            
        except Exception as e:
            self.logger.error(f"❌ Error menghentikan training: {str(e)}")
            
    def reset_metrics(self) -> None:
        """Reset metrics dan chart"""
        try:
            if self.training_service and hasattr(self.training_service, 'metrics_tracker'):
                self.training_service.metrics_tracker.reset()
                
            # Reset UI displays
            chart_output = self.ui_components.get('chart_output')
            chart_output and chart_output.clear_output(wait=True)
            
            metrics_output = self.ui_components.get('metrics_output')
            metrics_output and metrics_output.clear_output(wait=True)
            
            self.logger.success("✅ Metrics berhasil direset")
            
        except Exception as e:
            self.logger.error(f"❌ Error reset metrics: {str(e)}")
            
    def _setup_training_service(self) -> None:
        """Setup training service dengan callback UI"""
        try:
            # Get training service dari model manager
            self.training_service = self.model_manager.get_training_service(
                callback=self.ui_callback
            )
            
            if not self.training_service:
                raise ValueError("Gagal mendapatkan training service dari model manager")
                
            self.logger.info("✅ Training service berhasil disetup")
            
        except Exception as e:
            raise ValueError(f"Error setup training service: {str(e)}")
            
    def _setup_data_loaders(self) -> tuple:
        """Setup data loaders untuk training"""
        try:
            # Untuk sementara, buat dummy data loaders
            # TODO: Integrate dengan dataset module yang sebenarnya
            
            batch_size = self.config.get('model', {}).get('batch_size', 16)
            
            # Dummy train loader
            train_data = torch.utils.data.TensorDataset(
                torch.randn(100, 3, 640, 640),  # Dummy images
                torch.randint(0, 7, (100, 6))   # Dummy targets [img_idx, class, x, y, w, h]
            )
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True
            )
            
            # Dummy val loader
            val_data = torch.utils.data.TensorDataset(
                torch.randn(20, 3, 640, 640),
                torch.randint(0, 7, (20, 6))
            )
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=batch_size, shuffle=False
            )
            
            self.logger.info(f"📊 Data loaders siap: Train={len(train_loader)}, Val={len(val_loader)} batches")
            return train_loader, val_loader
            
        except Exception as e:
            raise ValueError(f"Error setup data loaders: {str(e)}")
            
    def _update_ui_training_state(self, is_training: bool) -> None:
        """Update UI state berdasarkan status training"""
        # Update button states
        start_button = self.ui_components.get('start_button')
        stop_button = self.ui_components.get('stop_button')
        
        if start_button:
            start_button.disabled = is_training
            start_button.description = "🔄 Training..." if is_training else "🚀 Mulai Training"
            
        if stop_button:
            stop_button.disabled = not is_training
            
        # Show/hide progress container
        progress_container = self.ui_components.get('progress_container')
        if progress_container and hasattr(progress_container, 'layout'):
            progress_container.layout.display = 'flex' if is_training else 'none'
            
    def _on_training_complete(self, result: Dict[str, Any]) -> None:
        """Handler ketika training selesai"""
        self.is_training = False
        self._update_ui_training_state(False)
        
        # Update final metrics
        update_metrics_display(
            self.ui_components.get('metrics_output'),
            result.get('metrics', {})
        )
        
        # Log hasil training
        best_epoch = result.get('best_epoch', -1)
        best_metric = result.get('best_metric', 0)
        total_time = result.get('total_time', 0)
        
        self.logger.success(f"🎉 Training selesai!")
        self.logger.info(f"   • Best epoch: {best_epoch}")
        self.logger.info(f"   • Best metric: {best_metric:.4f}")
        self.logger.info(f"   • Total waktu: {total_time:.2f}s")
        
    def _on_training_error(self, error_message: str) -> None:
        """Handler ketika terjadi error training"""
        self.is_training = False
        self._update_ui_training_state(False)
        self.logger.error(f"❌ Training error: {error_message}")
        
    def get_training_status(self) -> Dict[str, Any]:
        """Get status training saat ini"""
        status = {
            'is_training': self.is_training,
            'has_model_manager': self.model_manager is not None,
            'has_training_service': self.training_service is not None
        }
        
        if self.training_service and hasattr(self.training_service, 'get_training_progress'):
            status.update(self.training_service.get_training_progress())
            
        return status