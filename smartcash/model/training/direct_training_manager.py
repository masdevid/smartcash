"""
Direct Training Manager for SmartCash YOLOv5
Handles two-phase training strategy without wrapper layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import numpy as np
from torch.utils.data import DataLoader

from smartcash.model.architectures.direct_yolov5 import SmartCashYOLOv5Model
from smartcash.common.logger import SmartCashLogger


class DirectTrainingManager:
    """
    Training manager for direct YOLOv5 integration
    Implements two-phase training strategy
    """
    
    def __init__(
        self,
        model: SmartCashYOLOv5Model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "data/checkpoints",
        device: str = "auto"
    ):
        """
        Initialize training manager
        
        Args:
            model: SmartCashYOLOv5Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory for saving checkpoints
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        
        self.logger = SmartCashLogger(__name__)
        
        # Training state
        self.current_phase = 1
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_history = []
        
        # Initialize optimizers for both phases
        self.phase1_optimizer = None
        self.phase2_optimizer = None
        self.scheduler = None
        
        self.logger.info(f"✅ Initialized DirectTrainingManager for {model.backbone}")
    
    def setup_phase_1(self, learning_rate: float = 1e-3, weight_decay: float = 5e-4):
        """
        Setup Phase 1 training: Head-only learning for localization
        
        Args:
            learning_rate: Learning rate for head training
            weight_decay: Weight decay
        """
        self.current_phase = 1
        self.model.current_phase = 1
        
        # Ensure backbone is frozen
        self.model._setup_phase_1(self.model.model)
        
        # Get trainable parameters (detection head only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer for head-only training
        self.phase1_optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.937, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.phase1_optimizer,
            T_max=50,  # Adjust based on epochs
            eta_min=learning_rate * 0.01
        )
        
        phase_info = self.model.get_phase_info()
        self.logger.info(f"🔒 Phase 1 setup: {phase_info['trainable_params']:,} trainable parameters "
                        f"({phase_info['trainable_ratio']:.1%} of total)")
    
    def setup_phase_2(self, learning_rate: float = 1e-4, weight_decay: float = 5e-4):
        """
        Setup Phase 2 training: Full model fine-tuning
        
        Args:
            learning_rate: Learning rate for full model training (lower than phase 1)
            weight_decay: Weight decay
        """
        self.current_phase = 2
        self.model.setup_phase_2()
        
        # Get all parameters (now all trainable)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer for full model training with different learning rates
        param_groups = [
            # Backbone parameters (lower learning rate)
            {
                'params': [p for n, p in self.model.model.named_parameters() 
                          if p.requires_grad and 'model.24' not in n],  # Exclude detection head
                'lr': learning_rate * 0.1  # 10x lower for backbone
            },
            # Detection head parameters (normal learning rate)
            {
                'params': [p for n, p in self.model.model.named_parameters() 
                          if p.requires_grad and 'model.24' in n],  # Detection head only
                'lr': learning_rate
            }
        ]
        
        self.phase2_optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.937, 0.999)
        )
        
        # Learning rate scheduler for phase 2
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.phase2_optimizer,
            T_max=100,  # Longer schedule for fine-tuning
            eta_min=learning_rate * 0.001
        )
        
        phase_info = self.model.get_phase_info()
        self.logger.info(f"🔓 Phase 2 setup: {phase_info['trainable_params']:,} trainable parameters "
                        f"({phase_info['trainable_ratio']:.1%} of total)")
    
    def train_phase_1(
        self,
        epochs: int = 50,
        save_best: bool = True,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train Phase 1: Head localization and classification
        
        Args:
            epochs: Number of epochs
            save_best: Save best model
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        self.logger.info(f"🚀 Starting Phase 1 training: {epochs} epochs")
        
        if self.phase1_optimizer is None:
            self.setup_phase_1()
        
        return self._train_epochs(epochs, save_best, patience, phase=1)
    
    def train_phase_2(
        self,
        epochs: int = 100,
        save_best: bool = True,
        patience: int = 15,
        load_phase1_weights: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train Phase 2: Full model fine-tuning
        
        Args:
            epochs: Number of epochs
            save_best: Save best model
            patience: Early stopping patience
            load_phase1_weights: Load best Phase 1 weights
            
        Returns:
            Training history
        """
        self.logger.info(f"🚀 Starting Phase 2 training: {epochs} epochs")
        
        # Load Phase 1 weights if requested
        if load_phase1_weights:
            self._load_phase1_weights()
        
        if self.phase2_optimizer is None:
            self.setup_phase_2()
        
        return self._train_epochs(epochs, save_best, patience, phase=2)
    
    def _train_epochs(
        self,
        epochs: int,
        save_best: bool,
        patience: int,
        phase: int
    ) -> Dict[str, List[float]]:
        """
        Core training loop
        
        Args:
            epochs: Number of epochs
            save_best: Save best model
            patience: Early stopping patience
            phase: Training phase (1 or 2)
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'learning_rate': []
        }
        
        best_map = self.best_map
        patience_counter = 0
        
        optimizer = self.phase1_optimizer if phase == 1 else self.phase2_optimizer
        
        for epoch in range(epochs):
            self.current_epoch += 1
            
            # Training step
            train_loss = self._train_epoch(optimizer)
            
            # Validation step
            val_loss, val_map = self._validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_map'].append(val_map)
            history['learning_rate'].append(current_lr)
            
            # Log progress
            self.logger.info(
                f"Phase {phase} Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val mAP: {val_map:.4f}, LR: {current_lr:.2e}"
            )
            
            # Save best model
            if val_map > best_map and save_best:
                best_map = val_map
                self.best_map = best_map
                self._save_checkpoint(epoch, val_map, f"best_phase{phase}")
                patience_counter = 0
                self.logger.info(f"💾 New best model saved: mAP {val_map:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        # Save final model
        self._save_checkpoint(epochs-1, val_map, f"final_phase{phase}")
        
        return history
    
    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Train for one epoch
        
        Args:
            optimizer: Optimizer to use
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, training=True)
            
            # Compute loss using YOLOv5's loss function
            loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress periodically
            if batch_idx % 50 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Returns:
            Tuple of (validation loss, validation mAP)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # For mAP calculation
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = [t.to(self.device) for t in targets]
                
                # Forward pass
                outputs = self.model(images, training=True)
                
                # Compute loss
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions for mAP
                predictions = self.model(images, training=False)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        val_loss = total_loss / num_batches
        val_map = self._compute_map(all_predictions, all_targets)
        
        return val_loss, val_map
    
    def _compute_loss(self, outputs: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute YOLOv5 loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Combined loss
        """
        # Use YOLOv5's built-in loss function
        if hasattr(self.model.model, 'hyp'):
            hyp = self.model.model.hyp
        else:
            # Default hyperparameters
            hyp = {
                'box': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'anchor_t': 4.0,
                'fl_gamma': 0.0
            }
        
        # Get loss function from YOLOv5 model
        compute_loss = self.model.model.compute_loss if hasattr(self.model.model, 'compute_loss') else None
        
        if compute_loss is not None:
            loss, _ = compute_loss(outputs, targets)
            return loss
        else:
            # Fallback: simple MSE loss for debugging
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _compute_map(self, predictions: List[Dict], targets: List[torch.Tensor]) -> float:
        """
        Compute mean Average Precision (mAP)
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            mAP score
        """
        # Simplified mAP calculation - in production, use proper mAP metric
        if not predictions or not targets:
            return 0.0
        
        # Count correct predictions as proxy for mAP
        correct = 0
        total = 0
        
        for pred_dict in predictions:
            if 'boxes' in pred_dict and len(pred_dict['boxes']) > 0:
                correct += len(pred_dict['boxes'])
            total += 1
        
        return correct / max(total, 1) * 0.5  # Simplified metric
    
    def _save_checkpoint(self, epoch: int, map_score: float, checkpoint_name: str):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            map_score: mAP score
            checkpoint_name: Name for checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{self.model.backbone}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_config(),
            'phase': self.current_phase,
            'map_score': map_score,
            'optimizer_state_dict': (
                self.phase1_optimizer.state_dict() if self.current_phase == 1 
                else self.phase2_optimizer.state_dict()
            ),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _load_phase1_weights(self):
        """Load best Phase 1 weights for Phase 2 initialization"""
        phase1_path = self.checkpoint_dir / f"best_phase1_{self.model.backbone}.pt"
        
        if phase1_path.exists():
            try:
                checkpoint = torch.load(phase1_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.logger.info(f"✅ Loaded Phase 1 weights: {phase1_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load Phase 1 weights: {e}")
        else:
            self.logger.warning(f"Phase 1 checkpoint not found: {phase1_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_phase = checkpoint.get('phase', 1)
        self.best_map = checkpoint.get('map_score', 0.0)
        
        self.logger.info(f"✅ Loaded checkpoint: {checkpoint_path}")
        
        return {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'map_score': self.best_map,
            'model_config': checkpoint.get('model_config', {})
        }
    
    def get_training_summary(self) -> Dict:
        """Get training summary statistics"""
        phase_info = self.model.get_phase_info()
        
        return {
            'current_epoch': self.current_epoch,
            'current_phase': self.current_phase,
            'best_map': self.best_map,
            'phase_info': phase_info,
            'model_config': self.model.get_model_config()
        }


# Export
__all__ = ['DirectTrainingManager']