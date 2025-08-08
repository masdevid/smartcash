"""
SmartCash YOLOv5 Simplified Architecture
Direct integration with Ultralytics YOLOv5 without wrapper layers

This replaces the complex multi-layer wrapper system with a clean, direct approach:
- Training: 17 classes (0-6: denominations, 7-13: features, 14-16: authenticity)
- Inference: Maps to 7 denominations with confidence adjustment
- Two-phase training: Head → Full fine-tuning
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from smartcash.model.architectures.direct_yolov5 import SmartCashYOLOv5Model, SmartCashDirectYOLO
from smartcash.model.training.direct_training_manager import DirectTrainingManager
from smartcash.model.inference.post_prediction_mapper import PostPredictionMapper
from smartcash.common.logger import SmartCashLogger


class SmartCashYOLOv5:
    """
    Simplified SmartCash YOLOv5 integration
    Replaces complex wrapper system with direct Ultralytics integration
    """
    
    def __init__(
        self,
        backbone: str = "yolov5s",
        pretrained: bool = True,
        device: str = "auto",
        checkpoint_dir: str = "data/checkpoints"
    ):
        """
        Initialize SmartCash YOLOv5
        
        Args:
            backbone: YOLOv5 variant (yolov5s, yolov5m, yolov5l, yolov5x)
            pretrained: Use pretrained weights
            device: Target device
            checkpoint_dir: Directory for checkpoints
        """
        self.logger = SmartCashLogger(__name__)
        self.backbone = backbone
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create model
        self.model = SmartCashDirectYOLO.create_model(
            backbone=backbone,
            pretrained=pretrained,
            device=device
        )
        
        # Initialize post-prediction mapper
        self.mapper = PostPredictionMapper()
        
        # Training manager (initialized when needed)
        self.training_manager = None
        
        self.logger.info(f"✅ SmartCashYOLOv5 initialized: {backbone}")
    
    def train(
        self,
        train_loader,
        val_loader,
        phase1_epochs: int = 50,
        phase2_epochs: int = 100,
        phase1_lr: float = 1e-3,
        phase2_lr: float = 1e-4
    ) -> Dict:
        """
        Train the model with two-phase strategy
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            phase1_epochs: Phase 1 epochs (head only)
            phase2_epochs: Phase 2 epochs (full model)
            phase1_lr: Phase 1 learning rate
            phase2_lr: Phase 2 learning rate
            
        Returns:
            Training history
        """
        # Initialize training manager
        if self.training_manager is None:
            self.training_manager = DirectTrainingManager(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=str(self.checkpoint_dir)
            )
        
        # Phase 1: Head-only training
        self.logger.info("🚀 Starting Phase 1: Head localization training")
        self.training_manager.setup_phase_1(learning_rate=phase1_lr)
        phase1_history = self.training_manager.train_phase_1(epochs=phase1_epochs)
        
        # Phase 2: Full model fine-tuning
        self.logger.info("🚀 Starting Phase 2: Full model fine-tuning")
        self.training_manager.setup_phase_2(learning_rate=phase2_lr)
        phase2_history = self.training_manager.train_phase_2(
            epochs=phase2_epochs,
            load_phase1_weights=True
        )
        
        # Combine histories
        combined_history = {
            'phase_1': phase1_history,
            'phase_2': phase2_history,
            'summary': self.training_manager.get_training_summary()
        }
        
        return combined_history
    
    def predict(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        return_raw: bool = False
    ) -> Union[List[Dict], torch.Tensor]:
        """
        Make predictions on images
        
        Args:
            images: Input images
            return_raw: Return raw YOLOv5 output (for training) or processed results
            
        Returns:
            Processed denomination predictions or raw YOLOv5 output
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get raw predictions
            raw_output = self.model(images, training=False)
            
            if return_raw:
                return raw_output
            
            # Process with post-prediction mapper
            if isinstance(raw_output, list):
                # Multiple images
                processed_results = []
                for output in raw_output:
                    if output is not None and 'boxes' in output:
                        # Convert to expected format for mapper
                        if len(output['boxes']) > 0:
                            pred_tensor = torch.cat([
                                output['boxes'],
                                output['scores'].unsqueeze(1),
                                output['labels'].unsqueeze(1).float()
                            ], dim=1)
                        else:
                            pred_tensor = torch.empty(0, 6)
                    else:
                        pred_tensor = torch.empty(0, 6)
                    
                    result = self.mapper._map_single_image(pred_tensor)
                    processed_results.append(result)
                
                return processed_results
            else:
                # Single image - use raw tensor directly
                return self.mapper._map_single_image(raw_output)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint metadata
        """
        if self.training_manager is None:
            # Create training manager for checkpoint loading
            self.training_manager = DirectTrainingManager(
                model=self.model,
                train_loader=None,
                val_loader=None,
                checkpoint_dir=str(self.checkpoint_dir)
            )
        
        return self.training_manager.load_checkpoint(checkpoint_path)
    
    def save_checkpoint(self, checkpoint_name: str = "manual_save") -> str:
        """
        Save current model state
        
        Args:
            checkpoint_name: Name for checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if self.training_manager is None:
            raise RuntimeError("Training manager not initialized. Train model first or load from checkpoint.")
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{self.backbone}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_model_config(),
            'phase': self.model.current_phase,
            'backbone': self.backbone
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            'backbone': self.backbone,
            'model_config': self.model.get_model_config(),
            'phase_info': self.model.get_phase_info(),
            'mapper_info': self.mapper.get_mapping_info(),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def set_phase(self, phase: int):
        """
        Set training phase
        
        Args:
            phase: Training phase (1 or 2)
        """
        if phase == 1:
            self.model._setup_phase_1(self.model.model)
        elif phase == 2:
            self.model.setup_phase_2()
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")
        
        self.logger.info(f"📍 Set to Phase {phase}")


def create_smartcash_yolov5(
    backbone: str = "yolov5s",
    pretrained: bool = True,
    device: str = "auto"
) -> SmartCashYOLOv5:
    """
    Factory function for creating SmartCash YOLOv5
    
    Args:
        backbone: YOLOv5 variant
        pretrained: Use pretrained weights
        device: Target device
        
    Returns:
        SmartCashYOLOv5 instance
    """
    return SmartCashYOLOv5(
        backbone=backbone,
        pretrained=pretrained,
        device=device
    )


# Backward compatibility aliases
SmartCashYOLOv5Integration = SmartCashYOLOv5
create_smartcash_yolov5_model = create_smartcash_yolov5

# Export key components
__all__ = [
    'SmartCashYOLOv5',
    'create_smartcash_yolov5',
    'SmartCashYOLOv5Integration',  # Backward compatibility
    'create_smartcash_yolov5_model'  # Backward compatibility
]