"""
Service for handling model inference.
"""
from typing import Dict, Any, List
import torch
from smartcash.common.logger import get_logger
from smartcash.model.architectures.smartcash_yolov5 import create_smartcash_yolov5
from smartcash.model.inference.post_prediction_mapper import PostPredictionMapper

class InferenceService:
    """
    A service for loading a model and running inference.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.model = None
        self.mapper = PostPredictionMapper()

    def load_model(self, checkpoint_path: str, backbone: str = "yolov5s", pretrained: bool = True):
        """
        Loads a model from a checkpoint.

        Args:
            checkpoint_path (str): The path to the model checkpoint.
            backbone (str): The model backbone.
            pretrained (bool): Whether to use a pretrained model.
        """
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            self.model = create_smartcash_yolov5(backbone=backbone, pretrained=pretrained)
            self.model.load_state_dict(torch.load(checkpoint_path)['model'])
            self.model.eval()
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def predict(self, images: List[Any]) -> List[Dict[str, Any]]:
        """
        Runs inference on a list of images.

        Args:
            images (List[Any]): A list of images to run inference on.

        Returns:
            List[Dict[str, Any]]: A list of prediction results.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.logger.info(f"Running prediction on {len(images)} images.")
        try:
            predictions = self.model.predict(images)
            return self.mapper.map_predictions(predictions)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def cleanup(self):
        """Cleans up model resources."""
        if self.model:
            try:
                self.model.cpu()
                self.logger.debug("Model moved to CPU")
            except Exception as e:
                self.logger.warning(f"Failed to move model to CPU: {e}")
            
            del self.model
            self.model = None
            self.logger.info("Model resources cleaned up.")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear CUDA cache: {e}")
