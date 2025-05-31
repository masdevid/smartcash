"""
File: smartcash/model/service/training_service.py
Deskripsi: Implementasi training service untuk model YOLOv5 + EfficientNet
"""

import os
import torch
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import ModelTrainingError
from smartcash.model.service.progress_tracker import ProgressTracker
from smartcash.model.service.metrics_tracker import MetricsTracker
from smartcash.model.service.checkpoint_service import CheckpointService
from smartcash.model.service.callback_interfaces import TrainingCallback, CallbackType

class TrainingService:
    """Training service untuk model YOLOv5 + EfficientNet dengan progress tracking dan UI integration"""
    
    def __init__(
        self,
        model_manager = None,
        checkpoint_service: Optional[CheckpointService] = None,
        logger = None,
        callback: Optional[CallbackType] = None
    ):
        self.model_manager = model_manager
        self.checkpoint_service = checkpoint_service
        self.logger = logger or get_logger(__name__)
        self.progress_tracker = ProgressTracker()
        self.metrics_tracker = MetricsTracker()
        self.set_callback(callback)
        
        # Training state
        self.is_training = False
        self.should_stop = False
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.best_epoch = -1
        self.start_time = 0
        
        self.logger.info("✨ TrainingService initialized")
    
    def set_callback(self, callback: Optional[CallbackType]) -> None:
        """Set callback untuk training, progress, dan metrics tracking"""
        if callback is None: return
        
        # Set callback untuk progress tracker
        if hasattr(callback, 'update_progress'):
            self.progress_tracker.set_callback(callback)
        elif isinstance(callback, dict) and 'progress' in callback:
            self.progress_tracker.set_callback(callback)
        
        # Set callback untuk metrics tracker
        if hasattr(callback, 'update_metrics'):
            self.metrics_tracker.set_callback(callback)
        elif isinstance(callback, dict) and 'metrics' in callback:
            self.metrics_tracker.set_callback(callback)
        
        # Set callback untuk checkpoint service
        if self.checkpoint_service:
            if hasattr(callback, 'update_progress'):
                self.checkpoint_service.set_progress_callback(callback)
            elif isinstance(callback, dict) and 'progress' in callback:
                self.checkpoint_service.set_progress_callback(callback)
        
        # Simpan callback lengkap
        self._callback = callback
        self.logger.debug(f"🔄 Callback diatur: {type(callback).__name__}")
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        lr_scheduler: str = "cosine",
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.001,
        save_best: bool = True,
        save_interval: int = 0,
        checkpoint_dir: str = "runs/train/checkpoints",
        resume_from: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train model dengan progress tracking dan UI integration"""
        try:
            # Validasi model manager
            if self.model_manager is None:
                raise ModelTrainingError("❌ Model manager tidak tersedia untuk training")
            
            # Validasi model
            if not hasattr(self.model_manager, 'model') or self.model_manager.model is None:
                self.logger.info("🔄 Model belum diinisialisasi, membangun model...")
                self.model_manager.build_model()
            
            # Inisialisasi state training
            self.is_training = True
            self.should_stop = False
            self.current_epoch = 0
            self.best_metric = float('inf')
            self.best_epoch = -1
            self.start_time = time.time()
            
            # Gabungkan config
            full_config = {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "lr_scheduler": lr_scheduler,
                "early_stopping": early_stopping,
                "patience": patience,
                "min_delta": min_delta,
                "save_best": save_best,
                "save_interval": save_interval,
                "checkpoint_dir": checkpoint_dir
            }
            if config:
                full_config.update(config)
            
            # Update progress
            self.progress_tracker.update(0, 100, "🔄 Mempersiapkan training...")
            self.progress_tracker.update_stage("preparation")
            
            # Inisialisasi checkpoint service jika belum ada
            if self.checkpoint_service is None:
                self.checkpoint_service = CheckpointService(
                    checkpoint_dir=checkpoint_dir,
                    save_best=save_best,
                    save_interval=save_interval,
                    metric_name="val_loss",
                    mode="min",
                    logger=self.logger
                )
                if hasattr(self._callback, 'update_progress'):
                    self.checkpoint_service.set_progress_callback(self._callback)
            
            # Siapkan model, optimizer, dan scheduler
            model = self.model_manager.model
            device = next(model.parameters()).device
            
            # Buat optimizer
            optimizer = self.model_manager.get_optimizer(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            # Buat scheduler
            scheduler = self.model_manager.get_scheduler(
                optimizer=optimizer,
                epochs=epochs
            )
            
            # Resume dari checkpoint jika diperlukan
            start_epoch = 0
            if resume_from:
                self.logger.info(f"🔄 Melanjutkan training dari checkpoint: {resume_from}")
                self.progress_tracker.update_status(f"Melanjutkan dari checkpoint: {Path(resume_from).name}")
                
                # Load checkpoint
                _, metadata = self.checkpoint_service.load_checkpoint(
                    path=resume_from,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                # Update start epoch
                if metadata and "epoch" in metadata:
                    start_epoch = metadata["epoch"] + 1
                    self.current_epoch = start_epoch
                    self.logger.info(f"🔄 Melanjutkan dari epoch {start_epoch}")
            
            # Notifikasi awal training
            self._notify_training_start(epochs, len(train_loader), full_config)
            
            # Training loop
            no_improve_count = 0
            for epoch in range(start_epoch, epochs):
                if self.should_stop:
                    self.logger.info("🛑 Training dihentikan oleh pengguna")
                    break
                
                # Update state
                self.current_epoch = epoch
                
                # Notifikasi awal epoch
                self._notify_epoch_start(epoch, epochs)
                
                # Training epoch
                train_metrics = self._train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    epoch=epoch,
                    epochs=epochs
                )
                
                # Validation epoch jika val_loader tersedia
                val_metrics = {}
                if val_loader:
                    self._notify_validation_start(epoch)
                    val_metrics = self._validate_epoch(
                        model=model,
                        val_loader=val_loader,
                        epoch=epoch,
                        epochs=epochs
                    )
                
                # Update learning rate
                if scheduler:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    self.metrics_tracker.update_learning_rate(current_lr)
                
                # Gabungkan metrics
                epoch_metrics = {**train_metrics}
                if val_metrics:
                    epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Check apakah ini best model
                is_best = False
                if val_metrics and "loss" in val_metrics:
                    val_loss = val_metrics["loss"]
                    if val_loss < self.best_metric:
                        improvement = (self.best_metric - val_loss) / self.best_metric * 100 if self.best_metric != float('inf') else 100
                        self.best_metric = val_loss
                        self.best_epoch = epoch
                        is_best = True
                        no_improve_count = 0
                        self.logger.info(f"🏆 New best model: val_loss = {val_loss:.4f} (improved by {improvement:.2f}%)")
                    else:
                        no_improve_count += 1
                
                # Notifikasi akhir epoch
                self._notify_epoch_end(epoch, epoch_metrics, is_best)
                
                # Simpan checkpoint
                if is_best and save_best:
                    self.checkpoint_service.save_checkpoint(
                        model=model,
                        path=f"epoch_{epoch:03d}.pt",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=epoch_metrics,
                        is_best=True
                    )
                elif save_interval > 0 and (epoch + 1) % save_interval == 0:
                    self.checkpoint_service.save_checkpoint(
                        model=model,
                        path=f"epoch_{epoch:03d}.pt",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=epoch_metrics
                    )
                
                # Early stopping
                if early_stopping and no_improve_count >= patience:
                    self.logger.info(f"🛑 Early stopping triggered after {patience} epochs tanpa peningkatan")
                    self.progress_tracker.update_status(f"Early stopping triggered (no improvement for {patience} epochs)")
                    break
            
            # Simpan checkpoint terakhir
            self.checkpoint_service.save_checkpoint(
                model=model,
                path=f"epoch_{self.current_epoch:03d}.pt",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=self.current_epoch,
                metrics=epoch_metrics
            )
            
            # Notifikasi akhir training
            total_time = time.time() - self.start_time
            final_metrics = self.metrics_tracker.get_metrics_summary()
            self._notify_training_end(final_metrics, total_time)
            
            # Update state
            self.is_training = False
            
            # Return hasil training
            return {
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "total_epochs": self.current_epoch + 1,
                "total_time": total_time,
                "metrics": final_metrics,
                "best_checkpoint": self.checkpoint_service.get_best_checkpoint_path(),
                "last_checkpoint": self.checkpoint_service.get_last_checkpoint_path()
            }
            
        except Exception as e:
            error_msg = f"❌ Error during training: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "training")
            self._notify_training_error(error_msg, "training")
            self.is_training = False
            raise ModelTrainingError(error_msg)
    
    def _train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        epochs: int
    ) -> Dict[str, float]:
        """Train satu epoch"""
        # Set model ke mode training
        model.train()
        
        # Inisialisasi metrics
        metrics = {"loss": 0.0}
        batch_losses = []
        
        # Update progress
        self.progress_tracker.update(0, len(train_loader), f"🔄 Training epoch {epoch+1}/{epochs}")
        self.progress_tracker.update_stage("training", f"epoch_{epoch+1}")
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            if self.should_stop:
                break
            
            # Pindahkan data ke device
            device = next(model.parameters()).device
            data = data.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, dict):
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            
            # Hitung loss
            if hasattr(model, 'compute_loss'):
                loss_dict = model.compute_loss(outputs, targets)
                loss = loss_dict['loss']
                loss_items = {k: v.item() for k, v in loss_dict.items() if k != 'loss'}
            else:
                # Fallback untuk model tanpa compute_loss
                criterion = getattr(model, 'loss_function', torch.nn.CrossEntropyLoss())
                loss = criterion(outputs, targets)
                loss_items = {}
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            
            # Update batch metrics
            batch_metrics = {"loss": batch_loss, **loss_items}
            self.metrics_tracker.update(batch_metrics, "train_batch")
            
            # Update loss breakdown
            if loss_items:
                self.metrics_tracker.update_loss_breakdown(loss_items)
            
            # Notifikasi batch end
            self._notify_batch_end(batch_idx, len(train_loader), batch_metrics)
            
            # Update progress setiap beberapa batch
            if batch_idx % max(1, len(train_loader) // 20) == 0:
                progress_pct = (batch_idx + 1) / len(train_loader) * 100
                self.progress_tracker.update(
                    batch_idx + 1, 
                    len(train_loader), 
                    f"🔄 Training epoch {epoch+1}/{epochs} - {progress_pct:.1f}% (loss: {batch_loss:.4f})"
                )
        
        # Hitung rata-rata metrics
        metrics["loss"] = np.mean(batch_losses)
        
        # Update epoch metrics
        self.metrics_tracker.update(metrics, "train_epoch")
        
        # Update progress
        self.progress_tracker.update(
            len(train_loader), 
            len(train_loader), 
            f"✅ Epoch {epoch+1}/{epochs} selesai (loss: {metrics['loss']:.4f})"
        )
        
        return metrics
    
    def _validate_epoch(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        epochs: int
    ) -> Dict[str, float]:
        """Validate satu epoch"""
        # Set model ke mode eval
        model.eval()
        
        # Inisialisasi metrics
        metrics = {"loss": 0.0}
        batch_losses = []
        
        # Update progress
        self.progress_tracker.update(0, len(val_loader), f"🔄 Validating epoch {epoch+1}/{epochs}")
        self.progress_tracker.update_stage("validation", f"epoch_{epoch+1}")
        
        # Validation loop
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                if self.should_stop:
                    break
                
                # Pindahkan data ke device
                device = next(model.parameters()).device
                data = data.to(device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                
                # Forward pass
                outputs = model(data)
                
                # Hitung loss
                if hasattr(model, 'compute_loss'):
                    loss_dict = model.compute_loss(outputs, targets)
                    loss = loss_dict['loss']
                    loss_items = {k: v.item() for k, v in loss_dict.items() if k != 'loss'}
                else:
                    # Fallback untuk model tanpa compute_loss
                    criterion = getattr(model, 'loss_function', torch.nn.CrossEntropyLoss())
                    loss = criterion(outputs, targets)
                    loss_items = {}
                
                # Update metrics
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                # Update progress setiap beberapa batch
                if batch_idx % max(1, len(val_loader) // 10) == 0:
                    progress_pct = (batch_idx + 1) / len(val_loader) * 100
                    self.progress_tracker.update(
                        batch_idx + 1, 
                        len(val_loader), 
                        f"🔄 Validating epoch {epoch+1}/{epochs} - {progress_pct:.1f}% (loss: {batch_loss:.4f})"
                    )
        
        # Hitung rata-rata metrics
        metrics["loss"] = np.mean(batch_losses)
        
        # Update epoch metrics
        self.metrics_tracker.update(metrics, "val")
        
        # Update progress
        self.progress_tracker.update(
            len(val_loader), 
            len(val_loader), 
            f"✅ Validation epoch {epoch+1}/{epochs} selesai (loss: {metrics['loss']:.4f})"
        )
        
        # Notifikasi validation end
        self._notify_validation_end(epoch, metrics)
        
        return metrics
    
    def stop_training(self) -> None:
        """Hentikan training yang sedang berjalan"""
        if not self.is_training:
            self.logger.warning("⚠️ Tidak ada training yang sedang berjalan")
            return
        
        self.should_stop = True
        self.logger.info("🛑 Menghentikan training...")
        self.progress_tracker.update_status("Menghentikan training...")
    
    def is_training_running(self) -> bool:
        """Check apakah training sedang berjalan"""
        return self.is_training
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Dapatkan progress training saat ini"""
        return {
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "progress": self.progress_tracker.get_status(),
            "metrics": self.metrics_tracker.get_metrics_summary(),
            "elapsed_time": time.time() - self.start_time if self.is_training else 0
        }
    
    # Notifikasi callback
    def _notify_training_start(self, total_epochs: int, total_batches: int, config: Dict[str, Any]) -> None:
        """Notifikasi awal training"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_training_start'):
                self._callback.on_training_start(total_epochs, total_batches, config)
            elif isinstance(self._callback, dict) and 'training_start' in self._callback:
                self._callback['training_start'](total_epochs, total_batches, config)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil training_start callback: {str(e)}")
    
    def _notify_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Notifikasi awal epoch"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_epoch_start'):
                self._callback.on_epoch_start(epoch, total_epochs)
            elif isinstance(self._callback, dict) and 'epoch_start' in self._callback:
                self._callback['epoch_start'](epoch, total_epochs)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil epoch_start callback: {str(e)}")
    
    def _notify_batch_end(self, batch: int, total_batches: int, metrics: Dict[str, float]) -> None:
        """Notifikasi akhir batch"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_batch_end'):
                self._callback.on_batch_end(batch, total_batches, metrics)
            elif isinstance(self._callback, dict) and 'batch_end' in self._callback:
                self._callback['batch_end'](batch, total_batches, metrics)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil batch_end callback: {str(e)}")
    
    def _notify_epoch_end(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Notifikasi akhir epoch"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_epoch_end'):
                self._callback.on_epoch_end(epoch, metrics, is_best)
            elif isinstance(self._callback, dict) and 'epoch_end' in self._callback:
                self._callback['epoch_end'](epoch, metrics, is_best)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil epoch_end callback: {str(e)}")
    
    def _notify_validation_start(self, epoch: int) -> None:
        """Notifikasi awal validation"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_validation_start'):
                self._callback.on_validation_start(epoch)
            elif isinstance(self._callback, dict) and 'validation_start' in self._callback:
                self._callback['validation_start'](epoch)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil validation_start callback: {str(e)}")
    
    def _notify_validation_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Notifikasi akhir validation"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_validation_end'):
                self._callback.on_validation_end(epoch, metrics)
            elif isinstance(self._callback, dict) and 'validation_end' in self._callback:
                self._callback['validation_end'](epoch, metrics)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil validation_end callback: {str(e)}")
    
    def _notify_training_end(self, final_metrics: Dict[str, float], total_time: float) -> None:
        """Notifikasi akhir training"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_training_end'):
                self._callback.on_training_end(final_metrics, total_time)
            elif isinstance(self._callback, dict) and 'training_end' in self._callback:
                self._callback['training_end'](final_metrics, total_time)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil training_end callback: {str(e)}")
    
    def _notify_training_error(self, error_message: str, phase: str) -> None:
        """Notifikasi error training"""
        if not hasattr(self, '_callback') or self._callback is None:
            return
        
        try:
            if hasattr(self._callback, 'on_training_error'):
                self._callback.on_training_error(error_message, phase)
            elif isinstance(self._callback, dict) and 'training_error' in self._callback:
                self._callback['training_error'](error_message, phase)
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil training_error callback: {str(e)}")
    
    # One-liner utilities
    get_current_epoch = lambda self: self.current_epoch
    get_best_epoch = lambda self: self.best_epoch
    get_best_metric = lambda self: self.best_metric
    get_elapsed_time = lambda self: time.time() - self.start_time if self.is_training else 0
    get_checkpoint_service = lambda self: self.checkpoint_service
