"""
File: smartcash/ui/training/adapters/training_service_adapter.py
Deskripsi: Adapter untuk training service dengan UI-friendly interface
"""

from typing import Dict, Any, Optional
from smartcash.model.service.training_service import TrainingService

class TrainingServiceAdapter:
    """Adapter untuk training service dengan interface yang UI-friendly"""
    
    def __init__(self, training_service: TrainingService, logger):
        self.training_service = training_service
        self.logger = logger
        
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training dengan config yang disederhanakan untuk UI"""
        try:
            # Extract training parameters dari config
            epochs = config.get('epochs', 100)
            learning_rate = config.get('learning_rate', 0.001)
            batch_size = config.get('batch_size', 16)
            weight_decay = config.get('weight_decay', 0.0005)
            early_stopping = config.get('early_stopping', True)
            patience = config.get('patience', 10)
            save_best = config.get('save_best', True)
            save_interval = config.get('save_interval', 10)
            
            self.logger.info(f"🚀 Starting training dengan:")
            self.logger.info(f"   • Epochs: {epochs}")
            self.logger.info(f"   • Learning rate: {learning_rate}")
            self.logger.info(f"   • Batch size: {batch_size}")
            
            # TODO: Setup actual data loaders dari dataset
            # Untuk sementara gunakan dummy data
            train_loader, val_loader = self._create_dummy_data_loaders(batch_size)
            
            # Delegate ke training service
            result = self.training_service.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stopping=early_stopping,
                patience=patience,
                save_best=save_best,
                save_interval=save_interval,
                config=config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Training adapter error: {str(e)}")
            raise
    
    def stop_training(self) -> None:
        """Stop training"""
        self.training_service.stop_training()
        
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress"""
        return self.training_service.get_training_progress()
        
    def is_training_running(self) -> bool:
        """Check apakah training sedang berjalan"""
        return self.training_service.is_training_running()
    
    def _create_dummy_data_loaders(self, batch_size: int) -> tuple:
        """Create dummy data loaders untuk testing"""
        import torch
        
        # Dummy train data
        train_images = torch.randn(100, 3, 640, 640)
        train_targets = torch.randint(0, 7, (100, 6)).float()  # [img_idx, class, x, y, w, h]
        
        train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        # Dummy val data
        val_images = torch.randn(20, 3, 640, 640)
        val_targets = torch.randint(0, 7, (20, 6)).float()
        
        val_dataset = torch.utils.data.TensorDataset(val_images, val_targets)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        self.logger.info(f"📊 Dummy data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        return train_loader, val_loader