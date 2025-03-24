"""
File: smartcash/dataset/visualization/helpers/quality_visualizer.py
Deskripsi: Helper untuk visualisasi metrik kualitas dataset dan skor perbandingan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any


class QualityVisualizer:
    """Helper untuk visualisasi metrik kualitas dataset dan skor perbandingan."""
    
    def __init__(self):
        """Inisialisasi QualityVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
        
        # Color scheme untuk level kualitas
        self.quality_colors = {
            'low': '#f44336',     # Merah
            'medium': '#ff9800',  # Oranye
            'good': '#ffeb3b',    # Kuning
            'high': '#4caf50'     # Hijau
        }
    
    def get_quality_color(self, score: float) -> str:
        """
        Dapatkan warna berdasarkan skor kualitas.
        
        Args:
            score: Skor kualitas dataset (0-100)
            
        Returns:
            Kode warna hex
        """
        if score < 50:
            return self.quality_colors['low']
        elif score < 70:
            return self.quality_colors['medium']
        elif score < 85:
            return self.quality_colors['good']
        else:
            return self.quality_colors['high']
    
    def get_quality_category(self, score: float) -> str:
        """
        Dapatkan kategori berdasarkan skor kualitas.
        
        Args:
            score: Skor kualitas dataset (0-100)
            
        Returns:
            Nama kategori kualitas
        """
        if score < 50:
            return "Perlu Perbaikan"
        elif score < 70:
            return "Cukup"
        elif score < 85:
            return "Baik"
        else:
            return "Sangat Baik"
    
    def plot_quality_gauge(
        self, 
        ax, 
        quality_score: float,
        title: str = "Skor Kualitas Dataset"
    ) -> None:
        """
        Plot gauge untuk skor kualitas.
        
        Args:
            ax: Axes untuk plot
            quality_score: Skor kualitas dataset (0-100)
            title: Judul visualisasi
        """
        # Tentukan warna dan kategori berdasarkan skor
        color = self.get_quality_color(quality_score)
        category = self.get_quality_category(quality_score)
        
        # Create a circular gauge
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#f5f5f5'))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color='white'))
        
        # Add score and label
        ax.text(0.5, 0.5, f"{quality_score:.1f}", ha='center', va='center', fontsize=24, 
               fontweight='bold', color=color)
        ax.text(0.5, 0.3, "dari 100", ha='center', va='center', fontsize=10)
        
        # Add title and category
        ax.text(0.5, 0.85, title, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.5, 0.2, category, ha='center', va='center', fontsize=12, color=color)
        
        # Remove axis
        ax.axis('off')
    
    def plot_quality_scores_comparison(self, ax, reports: List[Dict[str, Any]], labels: List[str]) -> None:
        """
        Plot perbandingan skor kualitas dari beberapa laporan.
        
        Args:
            ax: Axes untuk plot
            reports: List laporan dataset
            labels: Label untuk setiap laporan
        """
        # Extract quality scores
        scores = [report.get('quality_score', 0) for report in reports]
        
        # Determine colors based on scores
        colors = [self.get_quality_color(score) for score in scores]
        
        # Plot
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, scores, color=colors)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, i, f"{score:.1f}", va='center')
        
        # Styling
        ax.set_title('Skor Kualitas Dataset')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Skor (0-100)')
        ax.set_xlim(0, 105)  # Sedikit lebih dari 100 untuk label
    
    def plot_imbalance_comparison(self, ax, reports: List[Dict[str, Any]], labels: List[str]) -> None:
        """
        Plot perbandingan skor imbalance dari beberapa laporan.
        
        Args:
            ax: Axes untuk plot
            reports: List laporan dataset
            labels: Label untuk setiap laporan
        """
        # Extract imbalance scores
        class_imbalance = []
        layer_imbalance = []
        
        for report in reports:
            class_metrics = report.get('class_metrics', {})
            layer_metrics = report.get('layer_metrics', {})
            
            class_imbalance.append(class_metrics.get('imbalance_score', 0))
            layer_imbalance.append(layer_metrics.get('imbalance_score', 0))
        
        # Setup data
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot
        ax.bar(x - width/2, class_imbalance, width, label='Kelas', color=self.palette[0])
        ax.bar(x + width/2, layer_imbalance, width, label='Layer', color=self.palette[2])
        
        # Add labels
        for i, v in enumerate(class_imbalance):
            ax.text(i - width/2, v + 0.1, f"{v:.1f}", ha='center')
            
        for i, v in enumerate(layer_imbalance):
            ax.text(i + width/2, v + 0.1, f"{v:.1f}", ha='center')
        
        # Styling
        ax.set_title('Skor Ketidakseimbangan')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Skor (0-10)')
        ax.set_ylim(0, 10.5)  # Sedikit lebih tinggi untuk label
        ax.legend()
    
    def plot_quality_metrics(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot metrik kualitas dataset.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
        # Extract quality metrics
        quality_score = report.get('quality_score', 0)
        
        # Create a gauge-like visualization
        color = self.get_quality_color(quality_score)
        category = self.get_quality_category(quality_score)
            
        # Draw gauge
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#f5f5f5'))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color='white'))
        
        # Add score and label
        ax.text(0.5, 0.5, f"{quality_score:.1f}", ha='center', va='center', fontsize=24, 
               fontweight='bold', color=color)
        ax.text(0.5, 0.3, "dari 100", ha='center', va='center', fontsize=10)
        
        # Add title and category
        ax.text(0.5, 0.85, "Skor Kualitas Dataset", ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.5, 0.2, category, ha='center', va='center', fontsize=12, color=color)
        
        # Remove axis
        ax.axis('off')