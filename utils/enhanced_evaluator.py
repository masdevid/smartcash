# File: src/evaluation/enhanced_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Modul evaluasi lanjutan untuk SmartCash Detector dengan analisis komprehensif

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_curve,
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging import ColoredLogger
from data.dataset import RupiahDataset

class AdvancedDetectorEvaluator:
    def __init__(self, model, test_dir, device='cuda'):
        self.logger = ColoredLogger('AdvancedEvaluator')
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Scenarios for comprehensive testing
        self.scenarios = {
            'normal': RupiahDataset(test_dir / 'normal'),
            'occlusion': RupiahDataset(test_dir / 'occlusion'),
            'varied_light': RupiahDataset(test_dir / 'varied_light'),
            'tilted': RupiahDataset(test_dir / 'tilted')
        }
        
        self.class_names = ['1000', '2000', '5000', '10000', '20000', '50000', '100000']

    def run_comprehensive_evaluation(self, output_dir='evaluation_results'):
        """
        Conduct multi-dimensional model evaluation
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        comprehensive_results = {}
        
        for scenario, dataset in self.scenarios.items():
            self.logger.info(f'🔍 Evaluating scenario: {scenario}')
            
            # Detailed scenario evaluation
            scenario_results = self._evaluate_scenario(dataset)
            comprehensive_results[scenario] = scenario_results
            
            # Visualization
            self._plot_scenario_metrics(scenario_results, output_path, scenario)
        
        # Aggregated report
        self._generate_aggregated_report(comprehensive_results, output_path)
        
        return comprehensive_results

    def _evaluate_scenario(self, dataset, conf_thresh=0.25):
        """
        Detailed evaluation for a specific scenario
        """
        all_preds, all_targets = [], []
        pred_confidences = []
        
        for img, targets in dataset:
            img = img.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(img)
            
            # Process predictions
            for pred in preds:
                high_conf_pred = pred[pred[:, 4] > conf_thresh]
                
                if len(high_conf_pred) > 0:
                    best_pred = high_conf_pred[high_conf_pred[:, 4].argmax()]
                    all_preds.append(int(best_pred[5]))
                    pred_confidences.append(float(best_pred[4]))
                    
                    # Match with ground truth
                    matching_target = targets[
                        (targets[:, 4] == best_pred[5]) & 
                        (self._calculate_iou(best_pred[:4], targets[:, :4]) > 0.5)
                    ]
                    
                    all_targets.append(int(matching_target[0][4]) if len(matching_target) > 0 else -1)
        
        return {
            'predictions': all_preds,
            'targets': all_targets,
            'confidences': pred_confidences
        }

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)

    def _plot_scenario_metrics(self, scenario_results, output_path, scenario):
        """
        Generate diagnostic plots for each scenario
        """
        # Confusion Matrix
        cm = confusion_matrix(
            scenario_results['targets'], 
            scenario_results['predictions'], 
            labels=range(len(self.class_names))
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {scenario.capitalize()} Scenario')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(output_path / f'{scenario}_confusion_matrix.png')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(
            scenario_results['targets'], 
            scenario_results['confidences']
        )
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.')
        plt.title(f'Precision-Recall Curve - {scenario.capitalize()} Scenario')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(output_path / f'{scenario}_pr_curve.png')
        plt.close()

    def _generate_aggregated_report(self, comprehensive_results, output_path):
        """
        Create a comprehensive markdown report
        """
        report_path = output_path / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write("# SmartCash Detector - Comprehensive Evaluation Report\n\n")
            
            for scenario, results in comprehensive_results.items():
                f.write(f"## {scenario.capitalize()} Scenario\n\n")
                
                report = classification_report(
                    results['targets'], 
                    results['predictions'], 
                    target_names=self.class_names
                )
                f.write("### Performance Metrics\n")
                f.write(f"```\n{report}\n```\n\n")
        
        self.logger.info(f'📄 Comprehensive report generated: {report_path}')

# Optional: Experimental Setup
def run_comparative_evaluation():
    # Placeholder for comparative model evaluation
    pass