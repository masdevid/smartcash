# File: handlers/evaluation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk proses evaluasi dan perbandingan model deteksi nilai mata uang

import torch
import yaml
import json
import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.metrics import MetricsCalculator
from smartcash.utils.visualization import ResultVisualizer
from smartcash.models.baseline import BaselineModel
from smartcash.models.yolov5_model import YOLOv5Model
from .data_handler import DataHandler

class EvaluationHandler:
    """Handler untuk evaluasi dan perbandingan model deteksi nilai mata uang."""
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = "results",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize evaluation handler.
        
        Args:
            config_path: Path ke file konfigurasi
            output_dir: Path untuk menyimpan hasil evaluasi
            logger: Optional logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.metrics_calc = MetricsCalculator()
        self.visualizer = ResultVisualizer(output_dir)
        self.data_handler = DataHandler(config_path=config_path)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> None:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _prepare_scenario(
        self,
        scenario_name: str,
        test_data_path: str
    ) -> torch.utils.data.DataLoader:
        """
        Prepare evaluation scenario.
        
        Args:
            scenario_name: Nama skenario evaluasi
            test_data_path: Path ke dataset pengujian
            
        Returns:
            DataLoader untuk evaluasi
        """
        self.logger.info(f"Menyiapkan skenario: {scenario_name}")
        
        # Get evaluation dataloader
        return self.data_handler.get_eval_loader(
            dataset_path=test_data_path,
            batch_size=self.config['evaluation']['batch_size'],
            num_workers=self.config['evaluation']['num_workers']
        )
    
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        scenario_name: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate
            test_loader: DataLoader untuk testing
            scenario_name: Optional nama skenario
            
        Returns:
            Dictionary berisi hasil evaluasi
        """
        self.logger.info(f"Mengevaluasi model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Reset metrics calculator
        self.metrics_calc.reset()
        
        # Start evaluation
        inference_times = []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Update metrics
                self.metrics_calc.update(predictions, targets)
                
                # Save visualization if needed
                if scenario_name and batch_idx < self.config['evaluation']['vis_batches']:
                    self.visualizer.save_batch_predictions(
                        images, predictions, targets,
                        scenario_name, batch_idx
                    )
        
        # Calculate metrics
        metrics = self.metrics_calc.compute()
        metrics['inference_time'] = np.mean(inference_times)
        
        # Log results
        self._log_results(metrics, scenario_name)
        
        return metrics
    
    def _log_results(self, metrics: Dict, scenario_name: Optional[str] = None) -> None:
        """Log evaluation results."""
        prefix = f"[{scenario_name}] " if scenario_name else ""
        
        self.logger.info(
            f"{prefix}Hasil Evaluasi:\n"
            f"  Akurasi: {metrics['accuracy']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}\n"
            f"  F1-Score: {metrics['f1']:.4f}\n"
            f"  mAP: {metrics['mAP']:.4f}\n"
            f"  Waktu Inferensi: {metrics['inference_time']*1000:.2f}ms"
        )
    
    def save_results(self, results: Dict, scenario_name: str) -> None:
        """Save evaluation results."""
        # Create results directory
        results_dir = self.output_dir / scenario_name
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = results_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.success(f"Hasil evaluasi disimpan di {metrics_file}")
    
    def compare_scenarios(self, scenario_results: Dict[str, Dict]) -> None:
        """Compare results from different scenarios."""
        # Create comparison plots
        self.visualizer.create_comparison_plots(scenario_results)
        
        # Create comparison table
        comparison_df = pd.DataFrame.from_dict(scenario_results, orient='index')
        comparison_file = self.output_dir / 'scenario_comparison.csv'
        comparison_df.to_csv(comparison_file)
        
        self.logger.success(f"Perbandingan skenario disimpan di {comparison_file}")


class ResearchScenarioEvaluator:
    """Handler for research scenario evaluation."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize research scenario evaluator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        self.data_handler = DataHandler(config=config)
        self.results_df = pd.DataFrame(
            columns=['Skenario', 'Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Waktu Inferensi']
        )
    
    def evaluate_scenario(
        self,
        scenario_name: str,
        model_path: str,
        test_data_path: str
    ) -> Dict:
        """
        Evaluate a specific research scenario.
        
        Args:
            scenario_name: Name of the scenario being tested
            model_path: Path to the model checkpoint
            test_data_path: Path to the test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating scenario: {scenario_name}")
        
        # Load model
        model = BaselineModel()  # You might want to make this configurable
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Get evaluation dataloader
        test_loader = self.data_handler.get_eval_loader(
            dataset_path=test_data_path,
            batch_size=32,  # You might want to make this configurable
            num_workers=4
        )
        
        # Evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        metrics = {}
        inference_times = []
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Evaluating {scenario_name}"):
                images = images.to(device)
                targets = targets.to(device)
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Store predictions and targets for later analysis
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # Convert lists to arrays
        all_predictions = np.concatenate(predictions_list)
        all_targets = np.concatenate(targets_list)
        
        # Calculate metrics
        metrics['mAP'] = self._calculate_map(all_predictions, all_targets)
        metrics['Waktu Inferensi'] = np.mean(inference_times)
        
        # Calculate other metrics
        metrics.update(self._calculate_detection_metrics(all_predictions, all_targets))
        
        # Update results DataFrame
        self.results_df = pd.concat([
            self.results_df,
            pd.DataFrame([{
                'Skenario': scenario_name,
                'Akurasi': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'mAP': metrics['mAP'],
                'Waktu Inferensi': metrics['Waktu Inferensi']
            }])
        ], ignore_index=True)
        
        return metrics
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all research scenarios and return results.
        
        Returns:
            DataFrame containing results for all scenarios
        """
        for scenario in self.config['research_scenarios']:
            self.evaluate_scenario(
                scenario_name=scenario['name'],
                model_path=scenario['model_path'],
                test_data_path=scenario['test_data']
            )
        
        return self.results_df
    
    def _calculate_map(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate mean Average Precision."""
        aps = []
        for class_idx in range(predictions.shape[1]):
            ap = self._calculate_ap(
                predictions[:, class_idx],
                targets[:, class_idx]
            )
            if not np.isnan(ap):
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def _calculate_ap(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Average Precision for a single class."""
        # Sort by confidence
        sorted_indices = np.argsort(predictions)[::-1]
        predictions = predictions[sorted_indices]
        targets = targets[sorted_indices]
        
        # Calculate precision and recall
        true_positives = np.cumsum(targets == 1)
        false_positives = np.cumsum(targets == 0)
        recall = true_positives / np.sum(targets == 1)
        precision = true_positives / (true_positives + false_positives)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        return ap


"""Evaluation handler for SmartCash models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import torch
import time
from pathlib import Path

from .base_evaluation_handler import BaseEvaluationHandler
from smartcash.utils.logger import SmartCashLogger

logger = SmartCashLogger("evaluation-handler")

class EvaluationHandler(BaseEvaluationHandler):
    """Handler for evaluating SmartCash models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation handler.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def evaluate(self, eval_type: str = 'regular') -> Dict[str, Any]:
        """Evaluate model based on specified type.
        
        Args:
            eval_type: Type of evaluation ('regular' or 'research')
            
        Returns:
            Dictionary containing evaluation results
        """
        if eval_type == 'regular':
            return self._evaluate_regular()
        elif eval_type == 'research':
            return self._evaluate_research()
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")
            
    def _evaluate_regular(self) -> Dict[str, Any]:
        """Regular evaluation on test dataset.
        
        Returns:
            Dictionary containing metrics for each model
        """
        try:
            # Load model
            model = self._load_model()
            model.to(self.device)
            model.eval()
            
            # Load test dataset
            test_loader = self._load_test_data()
            
            # Initialize metrics
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mAP': 0.0,
                'inference_time': 0.0
            }
            
            # Evaluate
            total_time = 0
            num_batches = 0
            
            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Measure inference time
                    start_time = time.time()
                    predictions = model(images)
                    total_time += (time.time() - start_time)
                    
                    # Update metrics
                    batch_metrics = self._calculate_metrics(predictions, targets)
                    for k, v in batch_metrics.items():
                        metrics[k] += v
                        
                    num_batches += 1
            
            # Average metrics
            for k in metrics:
                metrics[k] /= num_batches
                
            metrics['inference_time'] = total_time / num_batches
            
            return {'model': metrics}
            
        except Exception as e:
            logger.error(f"Error during regular evaluation: {str(e)}")
            raise
            
    def _evaluate_research(self) -> Dict[str, Any]:
        """Research-focused evaluation with different scenarios.
        
        Returns:
            Dictionary containing research evaluation results
        """
        try:
            scenarios = [
                {'name': 'Kondisi Cahaya Normal', 'augment': None},
                {'name': 'Pencahayaan Rendah', 'augment': 'low_light'},
                {'name': 'Pencahayaan Tinggi', 'augment': 'high_light'},
                {'name': 'Rotasi', 'augment': 'rotation'},
                {'name': 'Oklusi Parsial', 'augment': 'occlusion'}
            ]
            
            results = []
            
            for scenario in scenarios:
                # Load model
                model = self._load_model()
                model.to(self.device)
                model.eval()
                
                # Load augmented test data
                test_loader = self._load_test_data(augment=scenario['augment'])
                
                # Evaluate scenario
                metrics = self._evaluate_scenario(model, test_loader)
                
                # Add to results
                results.append({
                    'Skenario': scenario['name'],
                    'Akurasi': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'mAP': metrics['mAP'],
                    'Waktu Inferensi': metrics['inference_time']
                })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Save results
            output_dir = Path(self.config.get('output_dir', 'outputs'))
            output_dir.mkdir(exist_ok=True)
            df.to_csv(output_dir / 'research_results.csv', index=False)
            
            return {'research_results': df}
            
        except Exception as e:
            logger.error(f"Error during research evaluation: {str(e)}")
            raise
            
    def _load_model(self):
        """Load model based on configuration."""
        try:
            from smartcash.models import get_model
            return get_model(self.config)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _load_test_data(self, augment: Optional[str] = None):
        """Load test dataset with optional augmentation."""
        try:
            from smartcash.data import get_test_loader
            return get_test_loader(self.config, augment=augment)
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
            
    def _calculate_metrics(self, predictions: List[Dict[str, torch.Tensor]], 
                         targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Calculate evaluation metrics for a batch."""
        try:
            from smartcash.utils.metrics import calculate_detection_metrics
            return calculate_detection_metrics(predictions, targets)
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def _evaluate_scenario(self, model: torch.nn.Module, 
                         test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on a specific scenario."""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mAP': 0.0,
            'inference_time': 0.0
        }
        
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Measure inference time
                start_time = time.time()
                predictions = model(images)
                total_time += (time.time() - start_time)
                
                # Update metrics
                batch_metrics = self._calculate_metrics(predictions, targets)
                for k, v in batch_metrics.items():
                    metrics[k] += v
                    
                num_batches += 1
        
        # Average metrics
        for k in metrics:
            metrics[k] /= num_batches
            
        metrics['inference_time'] = total_time / num_batches
        
        return metrics