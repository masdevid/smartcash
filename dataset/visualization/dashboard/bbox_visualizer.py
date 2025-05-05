"""
File: smartcash/dataset/visualization/helpers/bbox_visualizer.py
Deskripsi: Helper untuk visualisasi statistik dan distribusi bounding box dalam dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any


class BBoxVisualizer:
    """Helper untuk visualisasi statistik dan distribusi bounding box dalam dataset."""
    
    def __init__(self):
        """Inisialisasi BBoxVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
    
    def plot_bbox_size_distribution(
        self, 
        ax, 
        size_distribution: Dict[str, Any],
        title: str = "Distribusi Ukuran Bounding Box"
    ) -> None:
        """
        Plot visualisasi distribusi ukuran bounding box.
        
        Args:
            ax: Axes untuk plot
            size_distribution: Dictionary berisi distribusi ukuran bbox
            title: Judul visualisasi
        """
        # Data
        categories = ['Kecil', 'Sedang', 'Besar']
        counts = [
            size_distribution.get('small', 0),
            size_distribution.get('medium', 0),
            size_distribution.get('large', 0)
        ]
        
        if sum(counts) == 0:
            ax.text(0.5, 0.5, "Tidak ada data bounding box", ha='center', va='center')
            ax.set_title(title)
            return
        
        # Buat plot
        bars = ax.bar(categories, counts, color=self.palette[3:6])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Jumlah')
        
        # Tambahkan nilai di bar
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                  str(count), ha='center', va='bottom')
    
    def plot_bbox_pie_distribution(
        self, 
        ax, 
        size_distribution: Dict[str, Any],
        title: str = "Persentase Ukuran Bounding Box"
    ) -> None:
        """
        Plot visualisasi distribusi ukuran bounding box dengan pie chart.
        
        Args:
            ax: Axes untuk plot
            size_distribution: Dictionary berisi distribusi ukuran bbox
            title: Judul visualisasi
        """
        # Data
        categories = ['Kecil', 'Sedang', 'Besar']
        percentages = [
            size_distribution.get('small_pct', 0),
            size_distribution.get('medium_pct', 0),
            size_distribution.get('large_pct', 0)
        ]
        
        if sum(percentages) == 0:
            ax.text(0.5, 0.5, "Tidak ada data bounding box", ha='center', va='center')
            ax.set_title(title)
            return
        
        # Buat plot
        wedges, texts, autotexts = ax.pie(
            percentages, 
            labels=categories, 
            autopct='%1.1f%%',
            colors=self.palette[3:6],
            textprops={'fontsize': 10}
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('equal')
    
    def plot_bbox_distribution(
        self, 
        fig, 
        size_distribution: Dict[str, Any],
        title: str = "Distribusi Ukuran Bounding Box"
    ) -> None:
        """
        Plot visualisasi distribusi ukuran bounding box dengan dual chart.
        
        Args:
            fig: Figure untuk plot
            size_distribution: Dictionary berisi distribusi ukuran bbox
            title: Judul visualisasi
        """
        # Data
        categories = ['Kecil', 'Sedang', 'Besar']
        counts = [
            size_distribution.get('small', 0),
            size_distribution.get('medium', 0),
            size_distribution.get('large', 0)
        ]
        
        percentages = [
            size_distribution.get('small_pct', 0),
            size_distribution.get('medium_pct', 0),
            size_distribution.get('large_pct', 0)
        ]
        
        # Buat plot
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Bar chart untuk count
        bars = ax1.bar(categories, counts, color=self.palette[3:6])
        ax1.set_title('Jumlah per Kategori', fontsize=12)
        ax1.set_ylabel('Jumlah')
        
        # Tambahkan nilai di bar
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom')
        
        # Pie chart untuk persentase
        wedges, texts, autotexts = ax2.pie(
            percentages, 
            labels=categories, 
            autopct='%1.1f%%',
            colors=self.palette[3:6],
            textprops={'fontsize': 10}
        )
        ax2.set_title('Persentase per Kategori', fontsize=12)
        ax2.axis('equal')
        
        # Title untuk keseluruhan
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    def plot_bbox_stats(self, ax, bbox_stats: Dict[str, Any]) -> None:
        """
        Plot statistik bounding box seperti width, height, dan aspect ratio.
        
        Args:
            ax: Axes untuk plot
            bbox_stats: Statistik bounding box
        """
        if not bbox_stats:
            ax.text(0.5, 0.5, "Tidak ada data statistik bbox", ha='center', va='center')
            return
            
        # Extract stats
        width = bbox_stats.get('width', {})
        height = bbox_stats.get('height', {})
        aspect_ratio = bbox_stats.get('aspect_ratio', {})
        
        if not width or not height:
            ax.text(0.5, 0.5, "Data statistik bbox tidak lengkap", ha='center', va='center')
            return
            
        # Data untuk plot
        stats = ['Min', 'Max', 'Mean', 'Median']
        width_data = [
            width.get('min', 0),
            width.get('max', 0),
            width.get('mean', 0),
            width.get('median', 0)
        ]
        
        height_data = [
            height.get('min', 0),
            height.get('max', 0),
            height.get('mean', 0),
            height.get('median', 0)
        ]
        
        # Create a grouped bar chart
        x = np.arange(len(stats))
        width_bar = 0.35
        
        ax.bar(x - width_bar/2, width_data, width_bar, label='Lebar', color=self.palette[0])
        ax.bar(x + width_bar/2, height_data, width_bar, label='Tinggi', color=self.palette[2])
        
        # Styling
        ax.set_title('Statistik Dimensi BBox')
        ax.set_xticks(x)
        ax.set_xticklabels(stats)
        ax.set_ylabel('Nilai')
        ax.legend()
    
    def plot_bbox_summary(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot ringkasan statistik bounding box.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
        # Extract bbox size distribution
        bbox_metrics = report.get('bbox_metrics', {})
        size_dist = bbox_metrics.get('size_distribution', {})
        
        if not size_dist:
            ax.text(0.5, 0.5, "Tidak ada data ukuran bbox", ha='center', va='center')
            return
            
        # Get size categories and counts
        categories = ['Kecil', 'Sedang', 'Besar']
        counts = [size_dist.get('small', 0), size_dist.get('medium', 0), size_dist.get('large', 0)]
        
        # Create bar chart
        x_pos = np.arange(len(categories))
        ax.bar(x_pos, counts, color=self.palette[3:6])
        
        # Add count labels
        for i, v in enumerate(counts):
            if v > 0:
                ax.text(i, v + max(counts) * 0.02, str(v), ha='center')
        
        # Styling
        ax.set_title('Distribusi Ukuran Bounding Box')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)