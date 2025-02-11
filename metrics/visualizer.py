# File: src/metrics/visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Modul visualisasi metrik untuk SmartCash Detector

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from .detection_metrics import DetectionMetrics

class MetricsVisualizer:
    def __init__(self, output_dir: str = 'evaluation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn')
        
    def plot_metrics_comparison(self, 
                              baseline_metrics: Dict[str, DetectionMetrics],
                              proposed_metrics: Dict[str, DetectionMetrics],
                              scenario: str):
        """Generate comparative visualization of metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        thresholds = list(baseline_metrics.keys())
        metrics_data = {
            'Baseline': [m.map for m in baseline_metrics.values()],
            'Proposed': [m.map for m in proposed_metrics.values()]
        }
        
        # 1. mAP comparison
        df_map = pd.DataFrame(metrics_data, index=thresholds)
        df_map.plot(kind='bar', ax=ax1)
        ax1.set_title('mAP Comparison')
        ax1.set_xlabel('IoU Threshold')
        ax1.set_ylabel('mAP')
        
        # 2. Precision-Recall curves
        for thresh in thresholds:
            ax2.plot(baseline_metrics[thresh].recall,
                    baseline_metrics[thresh].precision,
                    '--', label=f'Baseline IoU={thresh}')
            ax2.plot(proposed_metrics[thresh].recall,
                    proposed_metrics[thresh].precision,
                    '-', label=f'Proposed IoU={thresh}')
        ax2.set_title('Precision-Recall Curves')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend()
        
        # 3. Inference Time Distribution
        inference_data = {
            'Baseline': [m.inference_time for m in baseline_metrics.values()],
            'Proposed': [m.inference_time for m in proposed_metrics.values()]
        }
        df_time = pd.DataFrame(inference_data)
        sns.boxplot(data=df_time, ax=ax3)
        ax3.set_title('Inference Time Distribution')
        ax3.set_ylabel('Time (ms)')
        
        # 4. Feature Quality Comparison
        feature_data = {
            'Baseline': [m.feature_quality for m in baseline_metrics.values()],
            'Proposed': [m.feature_quality for m in proposed_metrics.values()]
        }
        df_quality = pd.DataFrame(feature_data, index=thresholds)
        df_quality.plot(kind='bar', ax=ax4)
        ax4.set_title('Feature Quality Comparison')
        ax4.set_xlabel('IoU Threshold')
        ax4.set_ylabel('Feature Quality Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{scenario}_metrics_comparison.png')
        plt.close()
        
    def plot_scenarios_comparison(self, 
                                scenarios_metrics: Dict[str, Dict[str, DetectionMetrics]]):
        """Generate comparison plots across different testing scenarios"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        metrics_names = ['map', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics_names):
            data = []
            for scenario, metrics in scenarios_metrics.items():
                for thresh, m in metrics.items():
                    data.append({
                        'Scenario': scenario,
                        'IoU': thresh,
                        'Value': getattr(m, metric)
                    })
            
            df = pd.DataFrame(data)
            sns.barplot(x='Scenario', y='Value', hue='IoU', data=df, ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()} by Scenario')
            axes[idx].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenarios_comparison.png')
        plt.close()
        
    def generate_report(self, 
                       scenarios_metrics: Dict[str, Dict[str, DetectionMetrics]],
                       comparison_results: Dict):
        """Generate comprehensive markdown report"""
        report_path = self.output_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# SmartCash Detector - Evaluation Report\n\n")
            
            # Overall Performance
            f.write("## Overall Performance\n\n")
            for scenario, metrics in scenarios_metrics.items():
                f.write(f"### {scenario} Scenario\n\n")
                for thresh, m in metrics.items():
                    f.write(f"#### IoU Threshold: {thresh}\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| mAP | {m.map:.4f} |\n")
                    f.write(f"| Precision | {m.precision:.4f} |\n")
                    f.write(f"| Recall | {m.recall:.4f} |\n")
                    f.write(f"| F1 Score | {m.f1_score:.4f} |\n")
                    f.write(f"| Inference Time | {m.inference_time:.2f}ms |\n")
                    f.write("\n")
            
            # Model Comparison Results
            f.write("## Model Comparison Analysis\n\n")
            for thresh, results in comparison_results.items():
                f.write(f"### IoU Threshold: {thresh}\n\n")
                
                # Relative Improvements
                f.write("#### Performance Improvements\n")
                f.write("| Metric | Improvement (%) |\n")
                f.write("|--------|----------------|\n")
                for metric, improvement in results['improvements'].items():
                    color = '🟢' if improvement > 0 else '🔴'
                    f.write(f"| {metric} | {color} {improvement:.2f}% |\n")
                
                # Statistical Significance
                f.write("\n#### Statistical Analysis\n")
                sig = results['statistical_tests']
                f.write(f"- T-Statistic: {sig['t_statistic']:.4f}\n")
                f.write(f"- P-Value: {sig['p_value']:.4f}\n")
                f.write(f"- Significant: {'✅' if sig['significant'] else '❌'}\n\n")
                
    def plot_feature_analysis(self, baseline_features: np.ndarray, 
                            proposed_features: np.ndarray,
                            scenario: str):
        """Visualize feature quality analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Feature activation strength
        sns.heatmap(baseline_features.mean(axis=0), 
                   ax=ax1, cmap='viridis')
        ax1.set_title('Baseline Feature Activation')
        
        sns.heatmap(proposed_features.mean(axis=0), 
                   ax=ax2, cmap='viridis')
        ax2.set_title('Proposed Feature Activation')
        
        plt.savefig(self.output_dir / f'{scenario}_feature_analysis.png')
        plt.close()
        
    def plot_error_analysis(self, error_types: Dict[str, int], 
                          scenario: str):
        """Visualize error distribution"""
        plt.figure(figsize=(10, 6))
        
        error_df = pd.DataFrame(list(error_types.items()), 
                              columns=['Error Type', 'Count'])
        sns.barplot(x='Error Type', y='Count', data=error_df)
        
        plt.title('Error Distribution Analysis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'{scenario}_error_analysis.png')
        plt.close()