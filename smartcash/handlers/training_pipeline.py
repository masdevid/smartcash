# File: handlers/training_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pipeline training model dengan progress tracking

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import time
from datetime import datetime
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.model_handler import ModelHandler
from smartcash.handlers.data_handler import DataHandler
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.utils.model_checkpoint import ModelCheckpoint
from smartcash.utils.metrics import MetricsCalculator

class TrainingPipeline:
    """Handler untuk pipeline training model SmartCash."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Initialize handlers
        # Setup model handler
        num_classes = len(config.get('layers', ['banknote']))  # Default ke single layer
        self.model_handler = ModelHandler(
            config_path=config.get('config_path', 'configs/base_config.yaml'),
            num_classes=num_classes,
            logger=self.logger
        )
        self.data_handler = DataHandler(
            config_path=config.get('config_path'),
            data_dir=config.get('data_dir', 'data'),
            logger=self.logger
        )
        self.metrics = MetricsCalculator()
        
        # Setup training directory
        self.train_dir = Path(config.get('output_dir', 'runs')) / 'train' / self._get_run_name()
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint handler
        self.checkpoint = ModelCheckpoint(
            save_dir=str(self.train_dir / 'weights'),
            logger=self.logger
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            logger=self.logger
        )
        
    def _get_run_name(self) -> str:
        """Generate nama unik untuk training run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backbone = self.config['backbone']
        mode = self.config['detection_mode']
        return f"{backbone}_{mode}_{timestamp}"
        
    def train(self) -> Dict:
        """
        Jalankan pipeline training lengkap.
        
        Returns:
            Dict berisi hasil training & best metrics
        """
        self.logger.info("🚀 Memulai pipeline training...")
        
        try:
            # Setup data loaders
            train_loader = self.data_handler.get_train_loader(
                batch_size=self.config['training']['batch_size']
            )
            val_loader = self.data_handler.get_val_loader(
                batch_size=self.config['training']['batch_size']
            )
            
            # Initialize model & optimizer
            model = self.model_handler.get_model()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            # Training loop dengan progress tracking
            best_metrics = {}
            n_epochs = self.config['training']['epochs']
            
            for epoch in range(n_epochs):
                # Training phase
                train_metrics = self._train_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    epoch=epoch,
                    total_epochs=n_epochs
                )
                
                # Validation phase
                val_metrics = self._validate_epoch(
                    model=model,
                    loader=val_loader,
                    epoch=epoch,
                    total_epochs=n_epochs
                )
                
                # Update early stopping
                if self.early_stopping(val_metrics):
                    self.logger.info("🛑 Early stopping triggered")
                    break
                    
                # Save checkpoint
                is_best = val_metrics['val_loss'] < best_metrics.get('val_loss', float('inf'))
                if is_best:
                    best_metrics = val_metrics
                    
                self.checkpoint.save(
                    model=model,
                    config=self.config,
                    epoch=epoch,
                    loss=val_metrics['val_loss'],
                    is_best=is_best
                )
                
            return {
                'train_dir': str(self.train_dir),
                'best_metrics': best_metrics,
                'config': self.config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Training gagal: {str(e)}")
            raise
            
    def _train_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int
    ) -> Dict:
        """Training satu epoch dengan progress tracking."""
        model.train()
        self.metrics.reset()
        epoch_loss = 0
        
        # Setup progress bar
        desc = f"Epoch {epoch+1}/{total_epochs} [Train]"
        pbar = tqdm(loader, desc=desc)
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Forward pass
            predictions = model(images)
            loss = model.compute_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss['total_loss'].backward()
            optimizer.step()
            
            # Update metrics
            self.metrics.update(predictions, targets)
            epoch_loss += loss['total_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss['total_loss'].item():.4f}"
            })
            
        # Calculate epoch metrics
        metrics = self.metrics.compute()
        metrics['train_loss'] = epoch_loss / len(loader)
        
        return metrics
        
    def _validate_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict:
        """Validasi satu epoch dengan progress tracking."""
        model.eval()
        self.metrics.reset()
        epoch_loss = 0
        
        # Setup progress bar
        desc = f"Epoch {epoch+1}/{total_epochs} [Val]"
        pbar = tqdm(loader, desc=desc)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(pbar):
                # Forward pass
                predictions = model(images)
                loss = model.compute_loss(predictions, targets)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                epoch_loss += loss['total_loss'].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss['total_loss'].item():.4f}"
                })
                
        # Calculate epoch metrics
        metrics = self.metrics.compute()
        metrics['val_loss'] = epoch_loss / len(loader)
        
        return metrics