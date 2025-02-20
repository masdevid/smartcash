"""Base evaluation handler for model evaluation."""

import os
import torch
import numpy as np
from typing import Dict, Optional, List
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import time

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.handlers.data_handler import DataHandler
from smartcash.utils.metrics import MetricsCalculator

class BaseEvaluationHandler:
    """Base handler for model evaluation."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize base evaluation handler.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        self.data_handler = DataHandler(config=config)
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict:
        """
        Evaluate a single model on a dataset.
        
        Args:
            model_path: Path to model checkpoint
            dataset_path: Path to evaluation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating model: {os.path.basename(model_path)}")
        
        # Load model
        model = self._load_model(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create data loader
        eval_loader = self.data_handler.get_eval_loader(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        # Initialize metrics tracking
        inference_times = []
        
        # Evaluation loop
        with torch.no_grad():
            for images, targets in tqdm(eval_loader, desc="Evaluating"):
                images = images.to(device)
                targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Update metrics
                self.metrics_calculator.update(predictions, targets)
        
        # Calculate final metrics
        metrics = self.metrics_calculator.compute()
        metrics['inference_time'] = np.mean(inference_times)
        
        # Log results
        self._log_results(metrics)
        
        return metrics
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                if config.get('backbone') == 'efficientnet':
                    model = YOLOv5Model(
                        backbone_type='efficientnet',
                        num_classes=config['num_classes']
                    )
                else:
                    model = YOLOv5Model(
                        backbone_type='cspdarknet',
                        num_classes=config['num_classes']
                    )
            else:
                # Default to CSPDarknet if no config
                model = YOLOv5Model(
                    backbone_type='cspdarknet',
                    num_classes=self.config['model']['num_classes']
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.success(f"Model loaded from {model_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _log_results(self, metrics: Dict) -> None:
        """
        Log evaluation results.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        self.logger.info(
            f"\nEvaluation Results:\n"
            f"  Akurasi: {metrics['accuracy']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}\n"
            f"  F1-Score: {metrics['f1']:.4f}\n"
            f"  mAP: {metrics['mAP']:.4f}\n"
            f"  Waktu Inferensi: {metrics['inference_time']*1000:.2f}ms"
        )
