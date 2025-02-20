# File: utils/visualization.py
# Author: Alfrida Sabar
# Deskripsi: Visualisasi hasil evaluasi dan perbandingan model

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
from datetime import datetime

class ResultVisualizer:
    """Visualisasi hasil evaluasi model deteksi mata uang"""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style plot
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def save_batch_predictions(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        scenario_name: str,
        batch_idx: int
    ) -> None:
        """
        Simpan visualisasi hasil deteksi untuk satu batch
        Args:
            images: Batch gambar [B, C, H, W]
            predictions: Prediksi model [B, num_pred, 6]
            targets: Ground truth [B, num_targets, 6]
            scenario_name: Nama skenario evaluasi
            batch_idx: Index batch
        """
        # Buat subfolder untuk skenario
        save_dir = self.output_dir / scenario_name / f"batch_{batch_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for img_idx, (image, preds, tgts) in enumerate(zip(images, predictions, targets)):
            # Convert ke numpy untuk plotting
            img_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Plot original dengan ground truth
            plt.figure(figsize=(12, 4))
            
            # Original + Ground Truth
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            self._plot_boxes(tgts, 'green', 'Ground Truth')
            plt.title('Ground Truth')
            
            # Original + Predictions
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            self._plot_boxes(preds, 'red', 'Prediction')
            plt.title('Prediksi Model')
            
            # Save plot
            save_path = save_dir / f"sample_{img_idx}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
    def _plot_boxes(
        self,
        boxes: torch.Tensor,
        color: str,
        label: str
    ) -> None:
        """Plot bounding boxes pada gambar"""
        for box in boxes:
            x, y, w, h = box[:4]
            conf = box[4]
            cls = box[5]
            
            rect = plt.Rectangle(
                (x - w/2, y - h/2),
                w, h,
                fill=False,
                edgecolor=color,
                linewidth=2,
                label=f"{label} ({conf:.2f})"
            )
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(
                x - w/2,
                y - h/2 - 5,
                f"Class {int(cls)}",
                color=color,
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none',
                    boxstyle='round'
                )
            )
            
    def create_comparison_plots(self, results: Dict) -> None:
        """
        Buat visualisasi perbandingan hasil antar skenario
        Args:
            results: Dict berisi hasil evaluasi semua skenario
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Metric Comparison Bar Plot
        self._create_metric_comparison(results, timestamp)
        
        # 2. Inference Time Comparison
        self._create_inference_comparison(results, timestamp)
        
        # 3. Per-Class Performance
        self._create_class_performance(results, timestamp)
        
    def _create_metric_comparison(
        self,
        results: Dict,
        timestamp: str
    ) -> None:
        """Buat plot perbandingan metrik antar skenario"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'mAP']
        scenarios = list(results.keys())
        
        # Prepare data
        metric_values = {
            metric: [results[s]['metrics'][metric] for s in scenarios]
            for metric in metrics
        }
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(scenarios))
        width = 0.15
        
        for i, (metric, values) in enumerate(metric_values.items()):
            plt.bar(
                x + i * width,
                values,
                width,
                label=metric.capitalize()
            )
            
        plt.xlabel('Skenario')
        plt.ylabel('Nilai')
        plt.title('Perbandingan Metrik Evaluasi')
        plt.xticks(x + width * 2, scenarios, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        save_path = self.output_dir / f"metric_comparison_{timestamp}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def _create_inference_comparison(
        self,
        results: Dict,
        timestamp: str
    ) -> None:
        """Buat plot perbandingan waktu inferensi"""
        scenarios = list(results.keys())
        inference_times = [
            results[s]['metrics']['inference_time']
            for s in scenarios
        ]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(scenarios, inference_times)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}ms',
                ha='center',
                va='bottom'
            )
            
        plt.xlabel('Skenario')
        plt.ylabel('Waktu Inferensi (ms)')
        plt.title('Perbandingan Waktu Inferensi')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        save_path = self.output_dir / f"inference_comparison_{timestamp}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    def _create_class_performance(
        self,
        results: Dict,
        timestamp: str
    ) -> None:
        """Buat heatmap performa per kelas"""
        scenarios = list(results.keys())
        classes = range(7)  # 7 kelas denominasi
        
        # Prepare matrices untuk precision dan recall
        precision_matrix = np.zeros((len(scenarios), len(classes)))
        recall_matrix = np.zeros((len(scenarios), len(classes)))
        
        for i, scenario in enumerate(scenarios):
            metrics = results[scenario]['metrics']
            for j in classes:
                precision_matrix[i, j] = metrics.get(f'precision_cls_{j}', 0)
                recall_matrix[i, j] = metrics.get(f'recall_cls_{j}', 0)
                
        # Plot heatmaps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Precision heatmap
        sns.heatmap(
            precision_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'Class {i}' for i in classes],
            yticklabels=scenarios,
            ax=ax1
        )
        ax1.set_title('Precision per Kelas')
        
        # Recall heatmap
        sns.heatmap(
            recall_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'Class {i}' for i in classes],
            yticklabels=scenarios,
            ax=ax2
        )
        ax2.set_title('Recall per Kelas')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"class_performance_{timestamp}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()