"""TrainingService untuk modul training SmartCash"""

import torch
import time
import numpy as np
from typing import Dict, Optional, Any, Callable
from pathlib import Path
from sklearn.metrics import confusion_matrix

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import ModelTrainingError
from smartcash.model.service.progress_tracker import ProgressTracker
from smartcash.model.service.metrics_tracker import MetricsTracker
from smartcash.model.service.checkpoint_service import CheckpointService
from smartcash.model.service.callback_interfaces import CallbackType
from smartcash.common.environment import EnvironmentManager
from smartcash.model.manager import ModelManager

class TrainingService:
    """Simplified training service dengan UI integration dan one-liner style"""
    
    def __init__(self, model_manager=None, checkpoint_service: Optional[CheckpointService] = None, 
                 logger=None, callback: Optional[CallbackType] = None):
        self.model_manager = model_manager
        self.checkpoint_service = checkpoint_service
        self.logger = logger or get_logger(__name__)
        self.progress_tracker = ProgressTracker()
        self.metrics_tracker = MetricsTracker()
        self.set_callback(callback)
        
        # Training state dengan one-liner initialization
        self.is_training = self.should_stop = False
        self.current_epoch = self.best_epoch = 0
        self.best_metric = float('inf')
        self.start_time = 0
        
        self.logger.info("✨ TrainingService initialized")
    
    def set_callback(self, callback: Optional[CallbackType]) -> None:
        """Set callback untuk semua trackers dengan one-liner delegation"""
        if not callback: return
        
        # Set callbacks untuk semua trackers
        if hasattr(callback, 'update_progress'):
            self.progress_tracker.set_callback(callback)
        if hasattr(callback, 'update_metrics'):
            self.metrics_tracker.set_callback(callback)
        if self.checkpoint_service and hasattr(callback, 'update_progress'):
            self.checkpoint_service.set_progress_callback(callback)
        
        self._callback = callback
        self.logger.debug(f"🔄 Callback diatur: {type(callback).__name__}")
    
    def train(self, train_loader, val_loader=None, epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 0.0005, early_stopping: bool = True, patience: int = 10,
              save_best: bool = True, save_interval: int = 0, checkpoint_dir: str = "runs/train/checkpoints",
              resume_from: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simplified training loop dengan progress tracking"""
        
        # Validation dan initialization
        self.model_manager or self._raise_error("❌ Model manager tidak tersedia")
        self.model_manager.model or self.model_manager.build_model()
        
        # Initialize training state
        self.is_training, self.should_stop = True, False
        self.current_epoch = self.best_epoch = 0
        self.best_metric, self.start_time = float('inf'), time.time()
        
        # Setup training configuration
        full_config = {'epochs': epochs, 'learning_rate': learning_rate, 'weight_decay': weight_decay, 
                      'early_stopping': early_stopping, 'patience': patience, 'save_best': save_best,
                      'save_interval': save_interval, 'checkpoint_dir': checkpoint_dir, **(config or {})}
        
        try:
            # Progress initialization
            self.progress_tracker.update(0, 100, "🔄 Mempersiapkan training...")
            self.progress_tracker.update_stage("preparation")
            
            # Setup components
            model = self.model_manager.model
            device = next(model.parameters()).device
            optimizer = self.model_manager.get_optimizer(learning_rate, weight_decay)
            scheduler = self.model_manager.get_scheduler(optimizer, epochs)
            
            # Initialize checkpoint service
            self.checkpoint_service = self.checkpoint_service or CheckpointService(
                checkpoint_dir=checkpoint_dir, save_best=save_best, save_interval=save_interval,
                metric_name="val_loss", mode="min", logger=self.logger
            )
            
            # Resume dari checkpoint jika diperlukan
            start_epoch = self._handle_resume(resume_from, model, optimizer, scheduler) if resume_from else 0
            self.current_epoch = start_epoch
            
            # Training notification
            self._notify_training_start(epochs, len(train_loader), full_config)
            
            # Main training loop
            no_improve_count = 0
            for epoch in range(start_epoch, epochs):
                if self.should_stop: break
                
                self.current_epoch = epoch
                self._notify_epoch_start(epoch, epochs)
                
                # Training dan validation
                train_metrics = self._train_epoch(model, train_loader, optimizer, epoch, epochs)
                val_metrics = self._validate_epoch(model, val_loader, epoch, epochs) if val_loader else {}
                
                # Learning rate update
                if scheduler:
                    scheduler.step()
                    self.metrics_tracker.update_learning_rate(optimizer.param_groups[0]['lr'])
                
                # Combined metrics dan best model checking
                epoch_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
                is_best, no_improve_count = self._check_best_model(val_metrics, epoch, no_improve_count)
                
                # Epoch end notification
                self._notify_epoch_end(epoch, epoch_metrics, is_best)
                
                # Checkpoint saving
                self._save_checkpoints(model, optimizer, scheduler, epoch, epoch_metrics, is_best, save_best, save_interval)
                
                # Evaluasi periodik
                if epoch % 5 == 0:
                    self._evaluate_model(model, val_loader)
                
                # Early stopping check
                if early_stopping and no_improve_count >= patience:
                    self.logger.info(f"🛑 Early stopping: {patience} epochs tanpa peningkatan")
                    break
            
            # Final checkpoint dan completion
            self._save_final_checkpoint(model, optimizer, scheduler, epoch_metrics)
            total_time = time.time() - self.start_time
            final_metrics = self.metrics_tracker.get_metrics_summary()
            self._notify_training_end(final_metrics, total_time)
            
            self.is_training = False
            return self._create_training_result(total_time, final_metrics)
            
        except Exception as e:
            error_msg = f"❌ Training error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "training")
            self._notify_training_error(error_msg, "training")
            self.is_training = False
            raise ModelTrainingError(error_msg)
    
    def _train_epoch(self, model, train_loader, optimizer, epoch, epochs) -> Dict[str, float]:
        """Train single epoch dengan progress tracking"""
        model.train()
        batch_losses = []
        
        self.progress_tracker.update(0, len(train_loader), f"🔄 Training epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            if self.should_stop: break
            
            # Device transfer dan forward pass
            device = next(model.parameters()).device
            data, targets = data.to(device), self._transfer_targets_to_device(targets, device)
            
            # Training step
            optimizer.zero_grad()
            outputs = model(data)
            loss_dict = self._compute_loss(model, outputs, targets)
            loss = loss_dict.get('loss', loss_dict.get('total_loss', list(loss_dict.values())[0]))
            
            loss.backward()
            optimizer.step()
            
            # Metrics tracking
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            batch_metrics = {"loss": batch_loss, **{k: v.item() if hasattr(v, 'item') else v for k, v in loss_dict.items() if k != 'loss'}}
            
            self.metrics_tracker.update(batch_metrics, "train_batch")
            self._notify_batch_end(batch_idx, len(train_loader), batch_metrics)
            
            # Progress update
            batch_idx % max(1, len(train_loader) // 20) == 0 and self.progress_tracker.update(
                batch_idx + 1, len(train_loader), f"🔄 Epoch {epoch+1}/{epochs} - {((batch_idx + 1) / len(train_loader) * 100):.1f}% (loss: {batch_loss:.4f})"
            )
        
        # Epoch metrics
        metrics = {"loss": np.mean(batch_losses)}
        self.metrics_tracker.update(metrics, "train_epoch")
        self.progress_tracker.update(len(train_loader), len(train_loader), f"✅ Epoch {epoch+1}/{epochs} selesai (loss: {metrics['loss']:.4f})")
        return metrics
    
    def _validate_epoch(self, model, val_loader, epoch, epochs) -> Dict[str, float]:
        """Validate single epoch dengan progress tracking"""
        model.eval()
        batch_losses = []
        
        self.progress_tracker.update(0, len(val_loader), f"🔄 Validating epoch {epoch+1}/{epochs}")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                if self.should_stop: break
                
                device = next(model.parameters()).device
                data, targets = data.to(device), self._transfer_targets_to_device(targets, device)
                
                outputs = model(data)
                loss_dict = self._compute_loss(model, outputs, targets)
                loss = loss_dict.get('loss', loss_dict.get('total_loss', list(loss_dict.values())[0]))
                batch_losses.append(loss.item())
                
                # Progress update
                batch_idx % max(1, len(val_loader) // 10) == 0 and self.progress_tracker.update(
                    batch_idx + 1, len(val_loader), f"🔄 Validation {epoch+1}/{epochs} - {((batch_idx + 1) / len(val_loader) * 100):.1f}%"
                )
        
        metrics = {"loss": np.mean(batch_losses)}
        self.metrics_tracker.update(metrics, "val")
        self.progress_tracker.update(len(val_loader), len(val_loader), f"✅ Validation {epoch+1}/{epochs} selesai (loss: {metrics['loss']:.4f})")
        self._notify_validation_end(epoch, metrics)
        return metrics
    
    def _check_best_model(self, val_metrics, epoch, no_improve_count):
        """Check dan update best model dengan one-liner logic"""
        if not val_metrics or "loss" not in val_metrics: return False, no_improve_count
        
        val_loss = val_metrics["loss"]
        if val_loss < self.best_metric:
            improvement = (self.best_metric - val_loss) / self.best_metric * 100 if self.best_metric != float('inf') else 100
            self.best_metric, self.best_epoch = val_loss, epoch
            self.logger.info(f"🏆 New best model: val_loss={val_loss:.4f} (improvement: {improvement:.2f}%)")
            return True, 0
        return False, no_improve_count + 1
    
    def _save_checkpoints(self, model, optimizer, scheduler, epoch, metrics, is_best, save_best, save_interval):
        """Save checkpoints berdasarkan kondisi"""
        save_conditions = [
            (is_best and save_best, f"epoch_{epoch:03d}.pt", True),
            (save_interval > 0 and (epoch + 1) % save_interval == 0, f"epoch_{epoch:03d}.pt", False)
        ]
        
        [self.checkpoint_service.save_checkpoint(model, path, optimizer, scheduler, epoch, metrics, is_best=is_best_flag)
         for condition, path, is_best_flag in save_conditions if condition]
    
    def _handle_resume(self, resume_path, model, optimizer, scheduler):
        """Handle resume dari checkpoint"""
        self.logger.info(f"🔄 Melanjutkan dari: {Path(resume_path).name}")
        _, metadata = self.checkpoint_service.load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = metadata.get("epoch", 0) + 1 if metadata else 0
        self.logger.info(f"🔄 Melanjutkan dari epoch {start_epoch}")
        return start_epoch
    
    def _compute_loss(self, model, outputs, targets):
        """Compute loss dengan error handling"""
        if hasattr(model, 'compute_loss'): return model.compute_loss(outputs, targets)[1] if isinstance(model.compute_loss(outputs, targets), tuple) else {'loss': model.compute_loss(outputs, targets)}
        return {'loss': getattr(model, 'loss_function', torch.nn.CrossEntropyLoss())(outputs, targets)}
    
    def _transfer_targets_to_device(self, targets, device):
        """Transfer targets ke device dengan type handling"""
        if isinstance(targets, torch.Tensor): return targets.to(device)
        if isinstance(targets, dict): return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        return targets
    
    def _save_final_checkpoint(self, model, optimizer, scheduler, metrics):
        """Save final checkpoint"""
        self.checkpoint_service.save_checkpoint(model, f"epoch_{self.current_epoch:03d}.pt", optimizer, scheduler, self.current_epoch, metrics)
    
    def _create_training_result(self, total_time, final_metrics):
        """Create training result dictionary"""
        return {
            "best_epoch": self.best_epoch, "best_metric": self.best_metric, "total_epochs": self.current_epoch + 1,
            "total_time": total_time, "metrics": final_metrics,
            "best_checkpoint": self.checkpoint_service.get_best_checkpoint_path(),
            "last_checkpoint": self.checkpoint_service.get_last_checkpoint_path()
        }
    
    def stop_training(self) -> None:
        """Stop training yang sedang berjalan"""
        if not self.is_training:
            self.logger.warning("⚠️ Tidak ada training yang sedang berjalan")
        else:
            setattr(self, 'should_stop', True)
            self.logger.info("🛑 Menghentikan training...")
            self.progress_tracker.update_status("Menghentikan training...")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress dengan status lengkap"""
        return {
            "is_training": self.is_training, "current_epoch": self.current_epoch, "best_epoch": self.best_epoch,
            "best_metric": self.best_metric, "progress": self.progress_tracker.get_status(),
            "metrics": self.metrics_tracker.get_metrics_summary(),
            "elapsed_time": time.time() - self.start_time if self.is_training else 0
        }
    
    # Notification methods dengan one-liner delegation ke callback
    def _notify_training_start(self, total_epochs, total_batches, config):
        """Notify training start dengan error protection"""
        try:
            if hasattr(self._callback, 'on_training_start'):
                self._callback.on_training_start(total_epochs, total_batches, config)
            if isinstance(self._callback, dict) and 'training_start' in self._callback:
                self._callback['training_start'](total_epochs, total_batches, config)
        except Exception as e: self.logger.warning(f"⚠️ Training start callback error: {str(e)}")
    
    def _notify_epoch_start(self, epoch, total_epochs):
        """Notify epoch start"""
        try:
            if hasattr(self._callback, 'on_epoch_start'):
                self._callback.on_epoch_start(epoch, total_epochs)
            if isinstance(self._callback, dict) and 'epoch_start' in self._callback:
                self._callback['epoch_start'](epoch, total_epochs)
        except Exception as e: self.logger.warning(f"⚠️ Epoch start callback error: {str(e)}")
    
    def _notify_batch_end(self, batch, total_batches, metrics):
        """Notify batch end"""
        try:
            if hasattr(self._callback, 'on_batch_end'):
                self._callback.on_batch_end(batch, total_batches, metrics)
            if isinstance(self._callback, dict) and 'batch_end' in self._callback:
                self._callback['batch_end'](batch, total_batches, metrics)
        except Exception as e: self.logger.warning(f"⚠️ Batch end callback error: {str(e)}")
    
    def _notify_epoch_end(self, epoch, metrics, is_best=False):
        """Notify epoch end"""
        try:
            if hasattr(self._callback, 'on_epoch_end'):
                self._callback.on_epoch_end(epoch, metrics, is_best)
            if isinstance(self._callback, dict) and 'epoch_end' in self._callback:
                self._callback['epoch_end'](epoch, metrics, is_best)
        except Exception as e: self.logger.warning(f"⚠️ Epoch end callback error: {str(e)}")
    
    def _notify_validation_end(self, epoch, metrics):
        """Notify validation end"""
        try:
            if hasattr(self._callback, 'on_validation_end'):
                self._callback.on_validation_end(epoch, metrics)
            if isinstance(self._callback, dict) and 'validation_end' in self._callback:
                self._callback['validation_end'](epoch, metrics)
        except Exception as e: self.logger.warning(f"⚠️ Validation end callback error: {str(e)}")
    
    def _notify_training_end(self, final_metrics, total_time):
        """Notify training end"""
        try:
            if hasattr(self._callback, 'on_training_end'):
                self._callback.on_training_end(final_metrics, total_time)
            if isinstance(self._callback, dict) and 'training_end' in self._callback:
                self._callback['training_end'](final_metrics, total_time)
        except Exception as e: self.logger.warning(f"⚠️ Training end callback error: {str(e)}")
    
    def _notify_training_error(self, error_message, phase):
        """Notify training error"""
        try:
            if hasattr(self._callback, 'on_training_error'):
                self._callback.on_training_error(error_message, phase)
            if isinstance(self._callback, dict) and 'training_error' in self._callback:
                self._callback['training_error'](error_message, phase)
        except Exception as e: self.logger.warning(f"⚠️ Training error callback error: {str(e)}")
    
    def _raise_error(self, message): raise ModelTrainingError(message)
    
    # One-liner utilities dan properties
    is_training_running = lambda self: self.is_training
    get_current_epoch = lambda self: self.current_epoch
    get_best_epoch = lambda self: self.best_epoch
    get_best_metric = lambda self: self.best_metric
    get_elapsed_time = lambda self: time.time() - self.start_time if self.is_training else 0
    get_checkpoint_service = lambda self: self.checkpoint_service
    
    def _evaluate_model(self, model, val_loader):
        """Lakukan evaluasi model dan hasilkan confusion matrix"""
        try:
            # Inisialisasi placeholder (akan diganti dengan implementasi aktual)
            y_true = []
            y_pred = []
            
            # Lakukan evaluasi
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(val_loader):
                    device = next(model.parameters()).device
                    data, targets = data.to(device), self._transfer_targets_to_device(targets, device)
                    
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            
            # Hitung confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Panggil callback untuk update UI
            if hasattr(self._callback, 'on_confusion_matrix'):
                self._callback.on_confusion_matrix(cm)
                
            return cm
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            if hasattr(self._callback, 'on_training_error'):
                self._callback.on_training_error(str(e), "evaluation")


class TrainingService:
    """Service untuk menangani proses training"""
    
    def __init__(self, config: dict, env_manager: EnvironmentManager):
        self.config = config
        self.env_manager = env_manager
        self.is_paused = False
        self.is_stopped = False
        
        # Inisialisasi model manager
        self.model_manager = ModelManager(
            config=config,
            pretrained_models_path=env_manager.get_model_dir()
        )
        self.model = self.model_manager.build_model()
        
        self.progress_callback = None
        self.metrics_callback = None
        self.evaluation_callback = None
        
    def set_progress_callback(self, callback: Callable[[dict], None]):
        """Set callback untuk update progress"""
        self.progress_callback = callback
        
    def set_metrics_callback(self, callback: Callable[[dict], None]):
        """Set callback untuk update metrics"""
        self.metrics_callback = callback
        
    def set_evaluation_callback(self, callback: Callable[[np.ndarray, list], None]):
        """Set callback untuk hasil evaluasi"""
        self.evaluation_callback = callback
        
    def start(self):
        """Main training loop tanpa threading"""
        try:
            # Setup training configuration
            full_config = {'epochs': self.config['epochs'], 'learning_rate': self.config['learning_rate'], 
                          'weight_decay': self.config['weight_decay'], 'early_stopping': self.config['early_stopping'], 
                          'patience': self.config['patience'], 'save_best': self.config['save_best'],
                          'save_interval': self.config['save_interval'], 'checkpoint_dir': self.config['checkpoint_dir']}
            
            # Training notification
            self._notify_training_start(full_config['epochs'], len(self.config['train_loader']), full_config)
            
            # Main training loop
            no_improve_count = 0
            for epoch in range(self.config['epochs']):
                if self.is_stopped:
                    break
                
                # Pause handling
                while self.is_paused:
                    time.sleep(0.1)
                
                # Update progress
                self._update_progress(epoch)
                
                # Training step (simulasi)
                time.sleep(1)
                
                # Update metrics (simulasi)
                metrics = self._generate_metrics(epoch)
                if self.metrics_callback:
                    self.metrics_callback(metrics)
                
                # Evaluasi periodik
                if epoch % self.config['eval_interval'] == 0:
                    self._run_evaluation()
            
            # Evaluasi final
            self._run_evaluation()
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            
    def _update_progress(self, epoch: int):
        """Update progress training"""
        if self.progress_callback:
            progress_data = {
                'epoch_progress': (epoch + 1) / self.config['epochs'],
                'batch_progress': 0.75,  # Contoh
                'overall_progress': (epoch + 0.75) / self.config['epochs'],
                'epoch_label': f"Epoch {epoch+1}/{self.config['epochs']}",
                'batch_label': f"Batch 120/160",
                'overall_label': f"{int((epoch + 0.75) / self.config['epochs'] * 100)}%"
            }
            self.progress_callback(progress_data)
            
    def _generate_metrics(self, epoch: int) -> dict:
        """Generate metrics simulasi"""
        return {
            'mAP': 0.85 - 0.01 * epoch,
            'Loss': 0.2 + 0.01 * epoch,
            'Akurasi': 0.92 - 0.005 * epoch,
            'Presisi': 0.89 - 0.003 * epoch,
            'F1': 0.87 - 0.002 * epoch,
            'Waktu Inferensi': 45 + epoch
        }
        
    def _run_evaluation(self):
        """Jalankan evaluasi menggunakan ModelManager"""
        if not self.model_manager:
            return
        
        # Dapatkan validation loader
        val_loader = self.env_manager.get_dataloader('validation')
        
        # Jalankan evaluasi
        cm = self.model_manager.evaluate_model(val_loader)
        
        # Dapatkan class labels
        classes = self.config.get('class_labels', 
            ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000'])
        
        if self.evaluation_callback:
            self.evaluation_callback(cm, classes)
            
    def pause(self):
        """Jeda training"""
        self.is_paused = True
        
    def resume(self):
        """Lanjutkan training"""
        self.is_paused = False
        
    def stop(self):
        """Hentikan training"""
        self.is_stopped = True
        self.is_paused = False  # Pastikan thread tidak stuck