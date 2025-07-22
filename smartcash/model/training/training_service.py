"""
File: smartcash/model/training/training_service.py
Deskripsi: Main training orchestrator dengan progress tracking dan UI integration
"""

import torch
import torch.nn as nn
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from torch.cuda.amp import autocast
import yaml

from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.model.training.metrics_tracker import MetricsTracker
from smartcash.model.training.optimizer_factory import OptimizerFactory, WarmupScheduler
from smartcash.model.training.loss_manager import LossManager
from smartcash.model.training.utils.training_progress_bridge import TrainingProgressBridge
from smartcash.model.training.utils.early_stopping import create_early_stopping
from smartcash.model.utils.device_utils import setup_device, model_to_device


class SimpleCallbackLogger:
    """Simple logger adapter for callback functions."""
    
    def __init__(self, callback_func):
        self.callback = callback_func
    
    def info(self, message):
        if self.callback:
            self.callback(message)
    
    def error(self, message):
        if self.callback:
            self.callback(f"❌ {message}")
    
    def warning(self, message):
        if self.callback:
            self.callback(f"⚠️ {message}")

class TrainingService:
    """Main training orchestrator dengan comprehensive progress tracking dan UI integration"""
    
    def __init__(self, model_api, config: Optional[Dict[str, Any]] = None,
                 ui_components: Optional[Dict[str, Any]] = None,
                 progress_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None):
        """
        Initialize training service
        
        Args:
            model_api: SmartCashModelAPI instance dari Fase 1
            config: Training configuration
            ui_components: UI components untuk integration
            progress_callback: Callback untuk progress updates
            metrics_callback: Callback untuk metrics updates
        """
        self.model_api = model_api
        self.config = config or self._load_training_config()
        self.ui_components = ui_components or {}
        
        # Setup device
        device_config = self.config.get('device', {'auto_detect': True, 'preferred': 'cuda'})
        self.device = setup_device(device_config)
        
        # Initialize components
        self.data_factory = DataLoaderFactory(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        self.optimizer_factory = OptimizerFactory(self.config)
        self.loss_manager = LossManager(self.config)
        
        # Setup progress tracking
        self.progress_bridge = TrainingProgressBridge(
            ui_components=ui_components,
            progress_callback=progress_callback,
            metrics_callback=metrics_callback
        )
        
        # Setup early stopping
        self.early_stopping = create_early_stopping(self.config)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_stopped = False
        
        # Logging - handle both logger objects and callback functions
        logger_component = ui_components.get('logger') if ui_components else None
        if callable(logger_component):
            # If it's a function, wrap it in a simple logger interface
            self.logger = SimpleCallbackLogger(logger_component)
        else:
            self.logger = logger_component
    
    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration dari file"""
        config_path = Path('smartcash/configs/training_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Fallback config
        return {
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'mixed_precision': True,
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'metric': 'val_map50',
                    'mode': 'max'
                }
            },
            'device': {
                'auto_detect': True,
                'preferred': 'cuda',
                'mixed_precision': True
            },
            'data': {
                'dataset_dir': 'data/preprocessed',
                'batch_size': 16,
                'num_workers': 2
            }
        }
    
    def start_training(self, epochs: Optional[int] = None, resume_from: Optional[str] = None,
                      custom_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Start training process dengan comprehensive tracking
        
        Args:
            epochs: Override epochs dari config
            resume_from: Path ke checkpoint untuk resume
            custom_model: Custom model override
            
        Returns:
            Training results dan metrics
        """
        try:
            # Setup training
            training_config = self.config.get('training', {})
            total_epochs = epochs or training_config.get('epochs', 100)
            
            self._log_info(f"🚀 Memulai training untuk {total_epochs} epochs")
            
            # Prepare model
            if custom_model:
                self.model = custom_model
            elif resume_from:
                self.model = self._load_checkpoint_model(resume_from)
            else:
                # Build model menggunakan model API dari Fase 1
                result = self.model_api.build_model()
                if not result.get('success', False):
                    raise RuntimeError(f"❌ Gagal build model: {result.get('message', 'Unknown error')}")
                self.model = result['model']
            
            # Move model ke device
            self.model = model_to_device(self.model, self.device)
            
            # Create data loaders
            train_loader = self.data_factory.create_train_loader()
            val_loader = self.data_factory.create_val_loader()
            
            self._log_info(f"📊 Dataset loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
            
            # Setup optimizer dan scheduler
            self.optimizer = self.optimizer_factory.create_optimizer(self.model)
            self.scheduler = self.optimizer_factory.create_scheduler(self.optimizer, total_epochs)
            self.scaler = self.optimizer_factory.setup_mixed_precision()
            
            # Initialize progress tracking
            self.progress_bridge.start_training(total_epochs, len(train_loader))
            
            # Training loop
            best_checkpoint_path = None
            
            for epoch in range(self.current_epoch, total_epochs):
                if self.training_stopped:
                    break
                
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, epoch)
                
                # Validation phase  
                val_metrics = self._validate_epoch(val_loader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Update metrics tracker
                final_metrics = self.metrics_tracker.compute_epoch_metrics(epoch)
                
                # Check untuk best model
                is_best = self.metrics_tracker.is_best_model()
                if is_best:
                    best_checkpoint_path = self._save_checkpoint(epoch, final_metrics, is_best=True)
                    self.best_metrics = final_metrics.copy()
                
                # Scheduler step
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            monitor_metric = final_metrics.get('val_map50', 0)
                            self.scheduler.step(monitor_metric)
                        else:
                            self.scheduler.step()
                
                # Early stopping check
                monitor_metric = final_metrics.get(
                    self.config.get('training', {}).get('early_stopping', {}).get('metric', 'val_map50'),
                    0
                )
                should_stop = self.early_stopping(monitor_metric, self.model, epoch)
                
                if should_stop:
                    self._log_info(f"🛑 Early stopping triggered di epoch {epoch}")
                    break
                
                # Complete epoch
                self.progress_bridge.complete_epoch(final_metrics)
                
                # Log epoch summary
                self._log_epoch_summary(epoch, final_metrics)
            
            # Training completed
            return self._complete_training(best_checkpoint_path)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self._log_error(error_msg)
            self.progress_bridge.training_error(error_msg, e)
            return {'success': False, 'error': error_msg, 'exception': str(e)}
    
    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Training phase untuk single epoch"""
        self.model.train()
        self.metrics_tracker.start_epoch()
        self.progress_bridge.start_epoch(epoch, "training")
        
        total_batches = len(train_loader)
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            self.metrics_tracker.start_batch()
            
            # Move data ke device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.scaler is not None):
                # Model prediction - sesuaikan dengan SmartCashYOLO dari Fase 1
                predictions = self.model(images)
                
                # Ensure predictions dalam format yang benar untuk loss calculation
                if not isinstance(predictions, dict):
                    # Convert single layer output ke dict format
                    predictions = {'banknote': predictions}
                
                # Calculate loss
                loss, loss_breakdown = self.loss_manager.compute_loss(
                    predictions, targets, images.shape[-1]
                )
            
            # Backward pass
            step_info = self.optimizer_factory.step_optimizer(
                self.optimizer, loss, retain_graph=False
            )
            
            # Update metrics
            batch_metrics = {**loss_breakdown}
            if step_info.get('grad_norm'):
                batch_metrics['grad_norm'] = step_info['grad_norm']
            
            # Get learning rate
            current_lr = self.optimizer_factory.get_current_lr(self.optimizer)
            batch_metrics['learning_rate'] = list(current_lr.values())[0]
            
            self.metrics_tracker.update_train_metrics(batch_metrics, batch_metrics['learning_rate'])
            
            # Update progress
            self.progress_bridge.update_batch(
                batch_idx, total_batches, 
                loss=loss.item(), 
                metrics=batch_metrics
            )
            
            epoch_losses.append(loss.item())
            
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        return {'train_loss': sum(epoch_losses) / len(epoch_losses)}
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validation phase untuk single epoch"""
        self.model.eval()
        self.progress_bridge.start_epoch(epoch, "validation")
        
        total_batches = len(val_loader)
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Move data ke device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    predictions = self.model(images)
                    
                    # Ensure predictions format
                    if not isinstance(predictions, dict):
                        predictions = {'banknote': predictions}
                    
                    # Calculate loss
                    loss, loss_breakdown = self.loss_manager.compute_loss(
                        predictions, targets, images.shape[-1]
                    )
                
                # Update validation metrics
                self.metrics_tracker.update_val_metrics(loss_breakdown)
                
                # Update progress
                self.progress_bridge.update_batch(
                    batch_idx, total_batches,
                    loss=loss.item()
                )
                
                val_losses.append(loss.item())
        
        return {'val_loss': sum(val_losses) / len(val_losses)}
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                        is_best: bool = False) -> str:
        """Save training checkpoint"""
        try:
            checkpoint_dir = Path(self.config.get('data', {}).get('checkpoints_dir', 'data/checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate checkpoint filename menggunakan model API
            checkpoint_info = {
                'epoch': epoch,
                'metrics': metrics,
                'is_best': is_best
            }
            
            checkpoint_path = self.model_api.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                **checkpoint_info
            )
            
            if checkpoint_path and is_best:
                self._log_info(f"💾 Best checkpoint disimpan: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            self._log_error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def _load_checkpoint_model(self, checkpoint_path: str) -> nn.Module:
        """Load model dari checkpoint untuk resume training"""
        try:
            result = self.model_api.load_checkpoint(checkpoint_path)
            if result.get('success', False):
                return result['model']
            else:
                raise RuntimeError(f"Failed to load checkpoint: {result.get('message', 'Unknown error')}")
        except Exception as e:
            self._log_error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def _complete_training(self, best_checkpoint_path: Optional[str]) -> Dict[str, Any]:
        """Complete training dan return results"""
        final_metrics = self.metrics_tracker.get_current_metrics()
        best_metrics = self.metrics_tracker.get_best_metrics()
        timing_summary = self.progress_bridge.get_timing_summary()
        
        # Complete progress tracking
        completion_metrics = {
            **final_metrics,
            **best_metrics,
            'total_epochs_trained': self.current_epoch + 1,
            'best_checkpoint': best_checkpoint_path
        }
        
        self.progress_bridge.complete_training(completion_metrics, best_checkpoint_path)
        
        # Save metrics history
        metrics_path = Path('data/logs/training') / f'training_metrics_{int(time.time())}.json'
        self.metrics_tracker.save_metrics(str(metrics_path))
        
        self._log_info("✅ Training selesai!")
        self._log_info(f"📊 Best mAP@0.5: {best_metrics.get('val_map50', 0):.3f}")
        
        return {
            'success': True,
            'message': 'Training completed successfully',
            'final_metrics': final_metrics,
            'best_metrics': best_metrics,
            'best_checkpoint': best_checkpoint_path,
            'timing': timing_summary,
            'epochs_trained': self.current_epoch + 1,
            'metrics_file': str(metrics_path)
        }
    
    def stop_training(self) -> None:
        """Gracefully stop training"""
        self.training_stopped = True
        self._log_info("🛑 Training stop requested")
    
    def resume_training(self, checkpoint_path: str, additional_epochs: int = None) -> Dict[str, Any]:
        """Resume training dari checkpoint"""
        try:
            # Load checkpoint
            result = self.model_api.load_checkpoint(checkpoint_path)
            if not result.get('success', False):
                raise RuntimeError(f"Failed to load checkpoint: {result.get('message')}")
            
            # Extract checkpoint info
            checkpoint_data = result.get('checkpoint_data', {})
            self.current_epoch = checkpoint_data.get('epoch', 0) + 1
            
            # Load state
            self.model = result['model']
            if 'optimizer_state_dict' in checkpoint_data and self.optimizer:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Resume training
            total_epochs = self.current_epoch + (additional_epochs or 50)
            self._log_info(f"🔄 Resuming training dari epoch {self.current_epoch}")
            
            return self.start_training(epochs=total_epochs)
            
        except Exception as e:
            error_msg = f"Resume training error: {str(e)}"
            self._log_error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _log_info(self, message: str) -> None:
        """Log info message"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"ℹ️ {message}")
    
    def _log_error(self, message: str) -> None:
        """Log error message"""
        if self.logger:
            self.logger.error(message)
        else:
            print(f"❌ {message}")
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch summary"""
        summary = self.metrics_tracker.get_metrics_summary()
        self._log_info(f"📈 Epoch {epoch + 1} | {summary}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'current_epoch': self.current_epoch,
            'training_stopped': self.training_stopped,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'current_metrics': self.metrics_tracker.get_current_metrics(),
            'best_metrics': self.metrics_tracker.get_best_metrics(),
            'early_stopping_info': self.early_stopping.get_best_info() if hasattr(self.early_stopping, 'get_best_info') else {},
            'progress_state': self.progress_bridge.get_current_state()
        }

# Convenience functions
def create_training_service(model_api, config: Dict[str, Any] = None, 
                           ui_components: Dict[str, Any] = None,
                           progress_callback: Callable = None,
                           metrics_callback: Callable = None) -> TrainingService:
    """Factory function untuk create training service"""
    return TrainingService(model_api, config, ui_components, progress_callback, metrics_callback)

def quick_train_model(model_api, epochs: int = 10, ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """One-liner untuk quick training"""
    service = TrainingService(model_api, ui_components=ui_components)
    return service.start_training(epochs=epochs)