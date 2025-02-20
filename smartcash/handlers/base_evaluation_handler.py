"""Base evaluation handler for model evaluation."""

import os
import torch
import numpy as np
from typing import Dict, Optional, List, Union
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.output_dir = Path("outputs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
    
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
        true_labels = []
        pred_scores = []
        
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
                true_labels.extend(targets.cpu().numpy())
                pred_scores.extend(predictions.cpu().numpy())
        
        # Calculate final metrics
        metrics = self.metrics_calculator.compute()
        metrics['inference_time'] = np.mean(inference_times)
        
        # Plot confusion matrix
        confusion_mat = confusion_matrix(true_labels, np.argmax(pred_scores, axis=1))
        self.plot_confusion_matrix(confusion_mat, list(range(self.config['model']['num_classes'])))
        
        # Plot ROC curves
        self.plot_roc_curves(true_labels, pred_scores, list(range(self.config['model']['num_classes'])))
        
        # Plot precision-recall curves
        self.plot_pr_curves(true_labels, pred_scores, list(range(self.config['model']['num_classes'])))
        
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

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True
    ) -> str:
        """Plot confusion matrix and save to file.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize values
            
        Returns:
            Path to saved plot
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
        else:
            fmt = "d"
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={"shrink": .8}
        )
        plt.title(title, size=16)
        plt.ylabel("True Label", size=14)
        plt.xlabel("Predicted Label", size=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        plot_path = self.plot_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return str(plot_path)
        
    def plot_roc_curves(
        self,
        true_labels: Union[List[int], np.ndarray],
        pred_scores: Union[List[float], np.ndarray],
        class_names: List[str],
        title: str = "ROC Curves"
    ) -> str:
        """Plot ROC curves for each class.
        
        Args:
            true_labels: True class labels
            pred_scores: Predicted class scores
            class_names: List of class names
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((true_labels == i).astype(int), pred_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")
            
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title, size=16)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        plot_path = self.plot_dir / "roc_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return str(plot_path)
        
    def plot_pr_curves(
        self,
        true_labels: Union[List[int], np.ndarray],
        pred_scores: Union[List[float], np.ndarray],
        class_names: List[str],
        title: str = "Precision-Recall Curves"
    ) -> str:
        """Plot precision-recall curves for each class.
        
        Args:
            true_labels: True class labels
            pred_scores: Predicted class scores
            class_names: List of class names
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve((true_labels == i).astype(int), pred_scores[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f"{class_name} (AUC = {pr_auc:.2f})")
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title, size=16)
        plt.legend(loc="lower left")
        plt.grid(True)
        
        # Save plot
        plot_path = self.plot_dir / "pr_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return str(plot_path)
