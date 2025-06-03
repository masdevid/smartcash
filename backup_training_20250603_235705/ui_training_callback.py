"""
File: smartcash/ui/training/adapters/ui_training_callback.py
Deskripsi: Adapter untuk mengkonversi training events menjadi UI updates
"""

import time
from typing import Dict, Any, List
from smartcash.model.service.callback_interfaces import TrainingCallback, MetricsCallback, ProgressCallback
from smartcash.ui.training.utils.training_progress_utils import update_training_progress, update_chart_display, update_metrics_display

class UITrainingCallback(TrainingCallback, MetricsCallback, ProgressCallback):
    """Callback adapter untuk mengkonversi training events menjadi UI updates"""
    
    def __init__(self, ui_components: Dict[str, Any], logger):
        self.ui_components = ui_components
        self.logger = logger
        self.start_time = 0
        self.epoch_start_time = 0
        
        # Chart data storage
        self.chart_data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    # TrainingCallback interface
    def on_training_start(self, total_epochs: int, total_batches: int, config: Dict[str, Any]) -> None:
        self.start_time = time.time()
        self.logger.info(f"🚀 Training dimulai: {total_epochs} epochs, {total_batches} batches per epoch")
        
        # Show progress container
        progress_container = self.ui_components.get('progress_container')
        if progress_container and hasattr(progress_container, 'layout'):
            progress_container.layout.display = 'flex'
            
        # Initialize progress
        update_training_progress(
            self.ui_components, 
            'overall', 
            0, 
            total_epochs, 
            "🚀 Training dimulai..."
        )
        
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.epoch_start_time = time.time()
        self.logger.info(f"📅 Epoch {epoch + 1}/{total_epochs} dimulai")
        
        update_training_progress(
            self.ui_components,
            'overall',
            epoch,
            total_epochs,
            f"📅 Epoch {epoch + 1}/{total_epochs}"
        )
        
    def on_batch_end(self, batch: int, total_batches: int, metrics: Dict[str, float]) -> None:
        # Update batch progress (hanya setiap 10 batch untuk mengurangi spam)
        if batch % 10 == 0 or batch == total_batches - 1:
            update_training_progress(
                self.ui_components,
                'step',
                batch + 1,
                total_batches,
                f"Batch {batch + 1}/{total_batches} - Loss: {metrics.get('loss', 0):.4f}"
            )
            
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        epoch_time = time.time() - self.epoch_start_time
        elapsed_time = time.time() - self.start_time
        
        # Log epoch completion
        status_emoji = "🏆" if is_best else "✅"
        self.logger.info(f"{status_emoji} Epoch {epoch + 1} selesai ({epoch_time:.2f}s)")
        
        # Update chart data
        self.chart_data['epochs'].append(epoch + 1)
        self.chart_data['train_loss'].append(metrics.get('loss', 0))
        if 'val_loss' in metrics:
            self.chart_data['val_loss'].append(metrics['val_loss'])
        if 'learning_rate' in metrics:
            self.chart_data['learning_rate'].append(metrics['learning_rate'])
            
        # Update displays
        update_chart_display(
            self.ui_components.get('chart_output'),
            self.chart_data
        )
        
        update_metrics_display(
            self.ui_components.get('metrics_output'),
            metrics
        )
        
    def on_training_end(self, final_metrics: Dict[str, float], total_time: float) -> None:
        self.logger.success(f"🎉 Training selesai! Total waktu: {total_time:.2f}s")
        
        # Hide progress container
        progress_container = self.ui_components.get('progress_container')
        if progress_container and hasattr(progress_container, 'layout'):
            progress_container.layout.display = 'none'
            
        # Final metrics update
        update_metrics_display(
            self.ui_components.get('metrics_output'),
            final_metrics.get('current', {})
        )
        
    def on_validation_start(self, epoch: int) -> None:
        update_training_progress(
            self.ui_components,
            'step',
            0,
            1,
            f"🔍 Validating epoch {epoch + 1}..."
        )
        
    def on_validation_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        val_loss = metrics.get('loss', 0)
        self.logger.info(f"🔍 Validation epoch {epoch + 1}: loss={val_loss:.4f}")
        
    def on_training_error(self, error_message: str, phase: str) -> None:
        self.logger.error(f"❌ Training error [{phase}]: {error_message}")
        
        # Hide progress container
        progress_container = self.ui_components.get('progress_container')
        if progress_container and hasattr(progress_container, 'layout'):
            progress_container.layout.display = 'none'
            
    # MetricsCallback interface
    def update_metrics(self, metrics: Dict[str, float], phase: str = "train") -> None:
        # Update real-time metrics display
        if phase in ['train', 'val']:
            update_metrics_display(
                self.ui_components.get('metrics_output'),
                metrics
            )
            
    def update_learning_rate(self, lr: float) -> None:
        self.logger.debug(f"📉 Learning rate: {lr:.6f}")
        
    def update_loss_breakdown(self, loss_components: Dict[str, float]) -> None:
        components_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
        self.logger.debug(f"📊 Loss breakdown: {components_str}")
        
    def update_prediction_samples(self, samples: List[Dict[str, Any]]) -> None:
        # TODO: Implement prediction samples visualization
        pass
        
    def update_inference_time(self, inference_time: float) -> None:
        self.logger.debug(f"⏱️ Inference time: {inference_time:.6f}s")
        
    # ProgressCallback interface
    def update_progress(self, current: int, total: int, message: str, phase: str = "general") -> None:
        update_training_progress(
            self.ui_components,
            phase,
            current,
            total,
            message
        )
        
    def update_status(self, status: str, phase: str = "general") -> None:
        self.logger.info(f"ℹ️ Status [{phase}]: {status}")
        
    def update_stage(self, stage: str, substage: str = None) -> None:
        stage_msg = f"{stage}" + (f" - {substage}" if substage else "")
        self.logger.info(f"🔄 Stage: {stage_msg}")
        
    def on_complete(self, success: bool, message: str) -> None:
        emoji = "✅" if success else "❌"
        self.logger.info(f"{emoji} Complete: {message}")
        
    def on_error(self, error_message: str, phase: str) -> None:
        self.logger.error(f"❌ Error [{phase}]: {error_message}")