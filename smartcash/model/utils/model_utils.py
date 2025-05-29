"""
File: smartcash/model/utils/model_utils.py
Deskripsi: Helper utilities untuk model operations dan training support
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from smartcash.common.logger import get_logger


class ModelLoaderUtils:
    """Utilities untuk loading model dan pre-trained weights"""
    
    def __init__(self):
        self.logger = get_logger('model_utils')
        self.drive_models_path = Path('/content/drive/MyDrive/SmartCash/models')
    
    def load_pretrained_backbone(self, model: torch.nn.Module, pretrained_file: str) -> bool:
        """Load pre-trained weights untuk backbone"""
        try:
            pretrained_path = self.drive_models_path / pretrained_file
            
            if not pretrained_path.exists():
                self.logger.warning(f"⚠️ Pre-trained file tidak ditemukan: {pretrained_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract state dict
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load hanya backbone weights
            backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
            
            if hasattr(model, 'backbone') and backbone_state:
                model.backbone.load_state_dict(backbone_state, strict=False)
                self.logger.info(f"✅ Backbone weights loaded dari {pretrained_file}")
                return True
            else:
                self.logger.warning(f"⚠️ Tidak ada backbone weights ditemukan dalam {pretrained_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading pretrained weights: {str(e)}")
            return False
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata models yang tersedia"""
        try:
            metadata_path = self.drive_models_path / 'model_metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def validate_model_compatibility(self, model_type: str, pretrained_file: str) -> bool:
        """Validate compatibility antara model type dan pretrained file"""
        compatibility_map = {
            'efficientnet_b4.pt': ['efficient_optimized', 'efficient_advanced', 'efficient_basic'],
            'yolov5s.pt': ['yolov5s', 'cspdarknet_s']
        }
        
        compatible_types = compatibility_map.get(pretrained_file, [])
        return model_type in compatible_types


class TrainingProgressTracker:
    """Helper untuk tracking training progress dan metrics"""
    
    def __init__(self):
        self.metrics_history = {'train_loss': [], 'val_loss': [], 'map': [], 'precision': [], 'recall': [], 'f1': []}
        self.best_map = 0.0
        self.epochs_without_improvement = 0
    
    def update_metrics(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update dan track metrics"""
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Check improvement
        current_map = metrics.get('map', 0.0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.epochs_without_improvement = 0
            improvement = True
        else:
            self.epochs_without_improvement += 1
            improvement = False
        
        return {
            'improved': improvement,
            'best_map': self.best_map,
            'epochs_without_improvement': self.epochs_without_improvement,
            'metrics_history': self.metrics_history
        }
    
    def should_early_stop(self, patience: int = 20) -> bool:
        """Check apakah training harus dihentikan early"""
        return self.epochs_without_improvement >= patience
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics dari training"""
        if not self.metrics_history['train_loss']:
            return {}
        
        import numpy as np
        
        return {
            'total_epochs': len(self.metrics_history['train_loss']),
            'best_map': self.best_map,
            'final_train_loss': self.metrics_history['train_loss'][-1],
            'final_val_loss': self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            'avg_improvement_rate': np.mean(np.diff(self.metrics_history['map'])) if len(self.metrics_history['map']) > 1 else 0
        }


class ModelConfigValidator:
    """Validator untuk model configuration"""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> List[str]:
        """Validate training configuration dan return list errors"""
        errors = []
        
        training_config = config.get('training', {})
        
        # Validate required fields
        required_fields = ['epochs', 'learning_rate', 'batch_size']
        for field in required_fields:
            if field not in training_config:
                errors.append(f"Missing required field: training.{field}")
        
        # Validate ranges
        if 'epochs' in training_config and training_config['epochs'] <= 0:
            errors.append("Epochs harus > 0")
        
        if 'learning_rate' in training_config and (training_config['learning_rate'] <= 0 or training_config['learning_rate'] > 1):
            errors.append("Learning rate harus antara 0 dan 1")
        
        if 'batch_size' in training_config and training_config['batch_size'] <= 0:
            errors.append("Batch size harus > 0")
        
        return errors
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        model_config = config.get('model', {})
        
        # Validate model type
        valid_types = ['efficient_basic', 'efficient_optimized', 'efficient_advanced', 'yolov5s']
        model_type = model_config.get('model_type', '')
        if model_type and model_type not in valid_types:
            errors.append(f"Invalid model_type: {model_type}. Valid: {valid_types}")
        
        # Validate backbone
        valid_backbones = ['efficientnet_b4', 'cspdarknet_s']
        backbone = model_config.get('backbone', '')
        if backbone and backbone not in valid_backbones:
            errors.append(f"Invalid backbone: {backbone}. Valid: {valid_backbones}")
        
        return errors


# One-liner utilities untuk quick access
load_pretrained_weights = lambda model, file: ModelLoaderUtils().load_pretrained_backbone(model, file)
validate_training_config = lambda config: ModelConfigValidator.validate_training_config(config)
validate_model_config = lambda config: ModelConfigValidator.validate_model_config(config)
create_progress_tracker = lambda: TrainingProgressTracker()