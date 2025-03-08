"""
File: smartcash/utils/training/training_epoch.py
Author: Alfrida Sabar
Deskripsi: Handler untuk proses training pada satu epoch dengan dukungan berbagai model
"""

import torch
from typing import Dict, Any, Optional

from smartcash.utils.training.training_callbacks import TrainingCallbacks

class TrainingEpoch:
    """
    Handler untuk proses training pada satu epoch dengan dukungan
    untuk berbagai format data dan model.
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi handler epoch training
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger
    
    def run(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        callbacks: Optional[TrainingCallbacks] = None
    ) -> float:
        """
        Jalankan satu epoch training.
        
        Args:
            epoch: Nomor epoch
            model: Model yang akan ditraining
            optimizer: Optimizer
            train_loader: Dataloader untuk training
            device: Device (cuda/cpu)
            callbacks: Handler callback
            
        Returns:
            Rata-rata loss untuk epoch ini
        """
        total_loss = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Handle berbagai format data
            images, targets = self._process_batch_data(data, device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss = self._compute_loss(model, predictions, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss tracking
            total_loss += loss.item()
            batch_count += 1
            
            # Trigger batch end callback
            if callbacks:
                callbacks.trigger(
                    event='batch_end',
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                    batch_size=images.size(0)
                )
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / max(1, batch_count)  # Hindari division by zero
    
    def _process_batch_data(self, data: Any, device: torch.device) -> tuple:
        """Proses data batch ke format yang konsisten"""
        # Handle berbagai format data
        if isinstance(data, dict):
            # Format multilayer dataset
            images = data['image'].to(device)
            targets = {k: v.to(device) for k, v in data['targets'].items()}
        elif isinstance(data, tuple) and len(data) == 2:
            # Format (images, targets)
            images, targets = data
            images = images.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, dict):
                targets = {k: v.to(device) for k, v in targets.items()}
        else:
            if self.logger:
                self.logger.warning(f"⚠️ Format data tidak didukung: {type(data)}")
            # Set default values to prevent failure
            images = torch.tensor([], device=device)
            targets = torch.tensor([], device=device)
            
        return images, targets
    
    def _compute_loss(
        self, 
        model: torch.nn.Module, 
        predictions: Any, 
        targets: Any
    ) -> torch.Tensor:
        """Hitung loss berdasarkan format prediction dan target"""
        # Jika model memiliki metode compute_loss sendiri
        if hasattr(model, 'compute_loss'):
            loss_dict = model.compute_loss(predictions, targets)
            return loss_dict['total_loss']
        
        # Jika model multi-layer dan targets adalah dict
        if isinstance(predictions, dict) and isinstance(targets, dict):
            loss = 0
            for layer_name in predictions:
                if layer_name in targets:
                    layer_pred = predictions[layer_name]
                    layer_target = targets[layer_name]
                    loss += torch.nn.functional.mse_loss(layer_pred, layer_target)
            return loss
        
        # Format standar
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(predictions, targets)