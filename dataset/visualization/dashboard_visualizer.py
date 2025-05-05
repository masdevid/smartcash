"""
File: smartcash/dataset/visualization/helpers/dashboard_visualizer.py
Deskripsi: Helper untuk membuat dashboard visualisasi komprehensif dataset
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.dataset.visualization.dashboard.class_visualizer import ClassVisualizer
from smartcash.dataset.visualization.dashboard.layer_visualizer import LayerVisualizer
from smartcash.dataset.visualization.dashboard.bbox_visualizer import BBoxVisualizer
from smartcash.dataset.visualization.dashboard.quality_visualizer import QualityVisualizer
from smartcash.dataset.visualization.dashboard.split_visualizer import SplitVisualizer
from smartcash.dataset.visualization.dashboard.recommendation_visualizer import RecommendationVisualizer


class DashboardVisualizer:
    """Helper untuk membuat dashboard visualisasi komprehensif dataset."""
    
    def __init__(self):
        """Inisialisasi DashboardVisualizer."""
        # Inisialisasi helper visualizer
        self.class_visualizer = ClassVisualizer()
        self.layer_visualizer = LayerVisualizer()
        self.bbox_visualizer = BBoxVisualizer()
        self.quality_visualizer = QualityVisualizer()
        self.split_visualizer = SplitVisualizer()
        self.recommendation_visualizer = RecommendationVisualizer()
    
    def create_dashboard(
        self, 
        report: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 12),
        title: str = "Dashboard Dataset SmartCash"
    ) -> plt.Figure:
        """
        Buat dashboard visualisasi dari laporan dataset.
        
        Args:
            report: Laporan dataset
            figsize: Ukuran figure
            title: Judul dashboard
            
        Returns:
            Figure matplotlib
        """
        # Setup figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Distribusi split
        ax_split = fig.add_subplot(gs[0, 0])
        self.split_visualizer.plot_split_summary(ax_split, report)
        
        # 2. Distribusi kelas (top 5)
        ax_class = fig.add_subplot(gs[0, 1:])
        self.class_visualizer.plot_class_summary(ax_class, report)
        
        # 3. Distribusi layer
        ax_layer = fig.add_subplot(gs[1, 0])
        self.layer_visualizer.plot_layer_summary(ax_layer, report)
        
        # 4. Distribusi ukuran bbox
        ax_bbox = fig.add_subplot(gs[1, 1])
        self.bbox_visualizer.plot_bbox_summary(ax_bbox, report)
        
        # 5. Metrik kualitas
        ax_quality = fig.add_subplot(gs[1, 2])
        self.quality_visualizer.plot_quality_metrics(ax_quality, report)
        
        # 6. Ringkasan rekomendasi
        ax_rec = fig.add_subplot(gs[2, :])
        self.recommendation_visualizer.plot_recommendations_summary(ax_rec, report)
        
        # Title dan layout
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def create_comparison_dashboard(
        self, 
        reports: List[Dict[str, Any]],
        labels: List[str],
        figsize: Tuple[int, int] = (15, 12),
        title: str = "Perbandingan Dataset"
    ) -> plt.Figure:
        """
        Buat dashboard perbandingan dari beberapa laporan dataset.
        
        Args:
            reports: List laporan dataset
            labels: Label untuk setiap laporan
            figsize: Ukuran figure
            title: Judul dashboard
            
        Returns:
            Figure matplotlib
        """
        # Validasi input
        if len(reports) != len(labels) or not reports:
            raise ValueError("Jumlah laporan dan label harus sama dan tidak boleh kosong")
            
        # Setup figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Perbandingan skor kualitas
        ax_quality = fig.add_subplot(gs[0, 0])
        self.quality_visualizer.plot_quality_scores_comparison(ax_quality, reports, labels)
        
        # 2. Perbandingan distribusi kelas
        ax_class = fig.add_subplot(gs[0, 1])
        self.class_visualizer.plot_class_distribution_comparison(ax_class, reports, labels)
        
        # 3. Perbandingan distribusi layer
        ax_layer = fig.add_subplot(gs[1, 0])
        self.layer_visualizer.plot_layer_distribution_comparison(ax_layer, reports, labels)
        
        # 4. Perbandingan imbalance scores
        ax_imbalance = fig.add_subplot(gs[1, 1])
        self.quality_visualizer.plot_imbalance_comparison(ax_imbalance, reports, labels)
        
        # Title dan layout
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig