# File: smartcash/handlers/multilayer_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk mengelola dataset dengan multiple layer anotasi

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.handlers.data_manager import DataManager
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.utils.model_checkpoint import ModelCheckpoint

class MultilayerHandler:
    """Handler for multilayer detection training and inference."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Initialize layer-specific configurations berdasarkan nama kelas yang aktual
        self.layers = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'weight': 1.0
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'weight': 0.8
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'weight': 0.6
            }
        }
        
        # Calculate total number of classes
        self.num_classes = sum(len(layer['classes']) for layer in self.layers.values())
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize data manager
        self.data_manager = DataManager(
            config_path=config.get('config_path', 'config.yaml'),
            logger=self.logger
        )
        
    def _create_model(self) -> YOLOv5Model:
        """Create and initialize the YOLOv5 model for multilayer detection."""
        model = YOLOv5Model(
            backbone_type=self.config['backbone'],
            num_classes=self.num_classes,
            layers=list(self.layers.keys())
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            self.logger.info("🚀 Using GPU for training")
        
        return model
    
    def train(self):
        """Train the multilayer detection model."""
        try:
            # Create data loaders
            train_loader = self.data_manager.get_train_loader(
                batch_size=self.config['training']['batch_size']
            )
            val_loader = self.data_manager.get_val_loader(
                batch_size=self.config['training']['batch_size']
            )
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            # Initialize learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
            
            # Initialize early stopping
            early_stopping = EarlyStopping(
                patience=self.config['training']['early_stopping_patience'],
                verbose=True,
                logger=self.logger
            )
            
            # Initialize model checkpoint
            checkpoint = ModelCheckpoint(
                save_dir=self.config['training']['checkpoint_dir'],
                logger=self.logger
            )
            
            # Training loop with progress bars
            for epoch in range(self.config['training']['epochs']):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                train_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Train]",
                    unit='batch'
                )
                
                for images, targets in train_pbar:
                    if torch.cuda.is_available():
                        images = images.cuda()
                        targets = targets.cuda()
                    
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Calculate loss for each layer
                    total_loss = 0
                    start_idx = 0
                    
                    for layer_name, layer_config in self.layers.items():
                        num_classes = len(layer_config['classes'])
                        layer_predictions = predictions[:, start_idx:start_idx + num_classes]
                        layer_targets = targets[:, start_idx:start_idx + num_classes]
                        
                        # Calculate layer loss and apply weight
                        layer_loss = nn.BCEWithLogitsLoss()(layer_predictions, layer_targets)
                        total_loss += layer_config['weight'] * layer_loss
                        
                        start_idx += num_classes
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                    
                    # Update progress bar
                    train_pbar.set_postfix({'loss': total_loss.item()})
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Val]",
                    unit='batch'
                )
                
                with torch.no_grad():
                    for images, targets in val_pbar:
                        if torch.cuda.is_available():
                            images = images.cuda()
                            targets = targets.cuda()
                        
                        predictions = self.model(images)
                        
                        # Calculate validation loss for each layer
                        total_val_loss = 0
                        start_idx = 0
                        
                        for layer_name, layer_config in self.layers.items():
                            num_classes = len(layer_config['classes'])
                            layer_predictions = predictions[:, start_idx:start_idx + num_classes]
                            layer_targets = targets[:, start_idx:start_idx + num_classes]
                            
                            layer_loss = nn.BCEWithLogitsLoss()(layer_predictions, layer_targets)
                            total_val_loss += layer_config['weight'] * layer_loss
                            
                            start_idx += num_classes
                        
                        val_loss += total_val_loss.item()
                        
                        # Update progress bar
                        val_pbar.set_postfix({'loss': total_val_loss.item()})
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Log epoch results
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.config['training']['epochs']}] "
                    f"Train Loss: {avg_train_loss:.4f} "
                    f"Val Loss: {avg_val_loss:.4f}"
                )
                
                # Save checkpoint
                checkpoint.save(
                    model=self.model,
                    config=self.config,
                    epoch=epoch,
                    loss=avg_val_loss
                )
                
                # Early stopping check
                if early_stopping.step(avg_val_loss):
                    self.logger.info("Early stopping triggered")
                    break
            
            self.logger.info("✅ Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform prediction on an image.
        
        Args:
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Dictionary containing predictions for each layer
        """
        self.model.eval()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            
            predictions = self.model(image)
            
            # Split predictions by layer
            results = {}
            start_idx = 0
            
            for layer_name, layer_config in self.layers.items():
                num_classes = len(layer_config['classes'])
                layer_predictions = predictions[:, start_idx:start_idx + num_classes]
                results[layer_name] = {
                    'predictions': torch.sigmoid(layer_predictions),
                    'classes': layer_config['classes']
                }
                start_idx += num_classes
            
            return results