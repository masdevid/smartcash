import os
import torch
from typing import Optional, Dict
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger

class ModelCheckpoint:
    """Model checkpoint handler that saves models with different configurations."""
    
    def __init__(
        self,
        save_dir: str = 'checkpoints',
        logger: Optional[SmartCashLogger] = None
    ):
        self.save_dir = save_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.best_loss = float('inf')
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_model_name(self, config: Dict) -> str:
        """Generate a unique model name based on configuration."""
        # Get detection mode
        if isinstance(config['layers'], list):
            mode = 'multilayer' if len(config['layers']) > 1 else 'single'
        else:
            mode = 'single'
            
        # Get backbone type
        backbone = config['backbone']
        
        # Get data source
        data_source = config['data_source']
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct name
        name = f"smartcash_{mode}_{backbone}_{data_source}_{timestamp}"
        
        return name
    
    def save(
        self,
        model: torch.nn.Module,
        config: Dict,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: The model to save
            config: Training configuration
            epoch: Current epoch
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        try:
            # Generate model name
            model_name = self._get_model_name(config)
            
            # Save model state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'config': config
            }
            
            # Save latest checkpoint
            latest_path = os.path.join(self.save_dir, f"{model_name}_latest.pth")
            torch.save(checkpoint, latest_path)
            self.logger.info(f"💾 Saved latest checkpoint: {latest_path}")
            
            # Save epoch checkpoint
            epoch_path = os.path.join(self.save_dir, f"{model_name}_epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_path)
            self.logger.info(f"💾 Saved epoch checkpoint: {epoch_path}")
            
            # Update and save best model if needed
            if loss < self.best_loss:
                self.best_loss = loss
                best_path = os.path.join(self.save_dir, f"{model_name}_best.pth")
                torch.save(checkpoint, best_path)
                self.logger.info(f"🏆 Saved best model: {best_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {str(e)}")
            raise
    
    def load(self, model_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            model_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing model state and metadata
        """
        try:
            checkpoint = torch.load(model_path)
            self.logger.info(f"📂 Loaded checkpoint: {model_path}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {str(e)}")
            raise