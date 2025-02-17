# File: models/trainers/base_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Abstract base class untuk training pipeline dengan 
# dukungan callback dan experiment tracking

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.logger import SmartCashLogger
from utils.early_stopping import EarlyStopping
from utils.model_checkpoint import ModelCheckpoint

class BaseTrainer(ABC):
    """Base class untuk semua trainer"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        self.model = model
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        self.training_history = []
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Detect hardware
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
    def _setup_callbacks(self) -> List:
        """Setup default callbacks"""
        callbacks = []
        
        # Early stopping
        if self.config['training'].get('early_stopping', True):
            callbacks.append(EarlyStopping(
                patience=self.config['training'].get('patience', 5),
                min_delta=self.config['training'].get('min_delta', 0.001)
            ))
            
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            dirpath=self.config['training'].get('checkpoint_dir', 'checkpoints'),
            filename='model_{epoch:02d}_{val_loss:.3f}',
            save_top_k=2,
            mode='min',
            monitor='val_loss'
        ))
        
        return callbacks
        
    @abstractmethod
    def train_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Single training step"""
        pass
        
    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        pass
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Training untuk satu epoch"""
        self.model.train()
        epoch_metrics = {}
        
        self.logger.info(f"🏃 Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        start_time = time.time()
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            step_metrics = self.train_step(batch, batch_idx)
            
            # Update metrics
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v.item())
                
            # Log progress
            if batch_idx % self.config['training'].get('log_interval', 10) == 0:
                self._log_progress(epoch_metrics, batch_idx, total_batches)
                
            self.global_step += 1
            
        # Calculate epoch metrics
        epoch_metrics = {
            k: sum(v) / len(v)
            for k, v in epoch_metrics.items()
        }
        
        epoch_time = time.time() - start_time
        self.logger.metric(
            f"⏱️ Epoch selesai dalam {epoch_time:.2f}s\n"
            f"📊 Metrics: " + ", ".join(
                [f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]
            )
        )
        
        return epoch_metrics
        
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validasi model"""
        self.model.eval()
        val_metrics = {}
        
        self.logger.info("🔍 Memulai validasi...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Validation step
                step_metrics = self.validation_step(batch, batch_idx)
                
                # Update metrics
                for k, v in step_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v.item())
                    
        # Calculate validation metrics
        val_metrics = {
            k: sum(v) / len(v)
            for k, v in val_metrics.items()
        }
        
        self.logger.metric(
            "📊 Validation metrics:\n" + "\n".join(
                [f"   {k}: {v:.4f}" for k, v in val_metrics.items()]
            )
        )
        
        return val_metrics
        
    def _log_progress(
        self,
        metrics: Dict[str, List[float]],
        batch_idx: int,
        total_batches: int
    ) -> None:
        """Log training progress"""
        avg_metrics = {
            k: sum(v) / len(v)
            for k, v in metrics.items()
        }
        
        self.logger.info(
            f"💫 Progress: {batch_idx}/{total_batches} "
            f"({(batch_idx/total_batches)*100:.1f}%) | " +
            " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        )
        
    def save_checkpoint(self, path: str) -> None:
        """Simpan model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.success(f"💾 Model checkpoint disimpan ke {path}")
        
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = checkpoint['training_history']
        
        self.logger.success(f"📂 Model checkpoint dimuat dari {path}")
        
    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Setup optimizer dan scheduler"""
        pass