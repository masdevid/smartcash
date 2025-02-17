# File: handlers/efficient_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk model YOLOv5 dengan EfficientNet backbone

from typing import Dict, Optional
import torch
import yaml
from pathlib import Path

from models.yolo5_efficient import YOLOv5Efficient
from utils.logger import SmartCashLogger
from utils.early_stopping import EarlyStopping
from utils.model_checkpoint import ModelCheckpoint

class EfficientNetHandler:
    """Handler untuk mengelola model YOLOv5 dengan EfficientNet backbone"""
    
    def __init__(
        self,
        config_path: str,
        num_classes: int = 7,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.num_classes = num_classes
        
        # Initialize components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup callbacks
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training'].get('patience', 5)
        )
        
        self.checkpoint = ModelCheckpoint(
            dirpath=self.config['training'].get('checkpoint_dir', 'checkpoints'),
            monitor='val_loss'
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi model"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def build(self) -> None:
        """Inisialisasi model dengan EfficientNet backbone"""
        self.logger.info("🔄 Membangun model YOLOv5 dengan EfficientNet backbone...")
        
        try:
            self.model = YOLOv5Efficient(
                num_classes=self.num_classes,
                pretrained=self.config['model'].get('pretrained', True)
            )
            
            # Pindahkan model ke device yang sesuai
            self.model = self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['lr0'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            # Setup scheduler
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['lr0'],
                epochs=self.config['training']['epochs'],
                steps_per_epoch=1  # Update per epoch
            )
            
            self.logger.success("✨ Model berhasil diinisialisasi!")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membangun model: {str(e)}")
            raise e
            
    def load_weights(self, weights_path: str) -> None:
        """Load model weights"""
        try:
            self.logger.info(f"📂 Loading weights dari {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.success("✅ Weights berhasil dimuat!")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat weights: {str(e)}")
            raise e
            
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Training untuk satu epoch"""
        self.model.train()
        epoch_metrics = {}
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Pindahkan data ke device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.model.compute_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            if batch_idx % 10 == 0:
                self.logger.metric(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )
                
        # Update learning rate
        self.scheduler.step()
        
        return epoch_metrics
        
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Validasi model"""
        self.model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate metrics
                batch_metrics = self.model.compute_metrics(predictions, targets)
                
                # Update metrics
                for k, v in batch_metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
                    
        # Average metrics
        val_metrics = {
            k: sum(v) / len(v)
            for k, v in val_metrics.items()
        }
        
        return val_metrics
        
    def save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """Simpan model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        save_path = Path(self.config['training']['checkpoint_dir'])
        save_path.mkdir(exist_ok=True)
        
        torch.save(
            checkpoint,
            save_path / f"model_checkpoint_latest.pt"
        )