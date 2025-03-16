"""
File: smartcash/dataset/visualization/report.py
Deskripsi: Utilitas untuk membuat visualisasi laporan dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from matplotlib.gridspec import GridSpec

from smartcash.common.logger import get_logger


class ReportVisualizer:
    """Komponen untuk membuat visualisasi komprehensif untuk laporan dataset."""
    
    def __init__(self, output_dir: str, logger=None):
        """
        Inisialisasi ReportVisualizer.
        
        Args:
            output_dir: Direktori untuk menyimpan visualisasi
            logger: Logger kustom (opsional)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("report_visualizer")
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
        
        self.logger.info(f"📊 ReportVisualizer diinisialisasi dengan output di: {self.output_dir}")
    
    def create_class_distribution_summary(
        self, 
        class_stats: Dict[str, Dict[str, int]], 
        title: str = "Distribusi Kelas per Split",
        save_path: Optional[str] = None,
        top_n: int = 8
    ) -> str:
        """
        Buat visualisasi distribusi kelas per split.
        
        Args:
            class_stats: Dictionary {class_name: {split: count, ...}}
            title: Judul visualisasi
            save_path: Path untuk menyimpan visualisasi (opsional)
            top_n: Jumlah kelas teratas yang ditampilkan
            
        Returns:
            Path ke file visualisasi
        """
        # Transformasi data untuk plotting
        if not class_stats:
            return ""
            
        # Hitung total sampel per kelas dari semua split
        class_totals = {}
        for cls, split_counts in class_stats.items():
            class_totals[cls] = sum(split_counts.values())
            
        # Ambil kelas teratas berdasarkan total
        top_classes = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_class_names = [cls for cls, _ in top_classes]
        
        # Dapatkan splits yang tersedia
        all_splits = set()
        for cls_data in class_stats.values():
            all_splits.update(cls_data.keys())
        splits = sorted(all_splits)
        
        if not splits:
            return ""
            
        # Siapkan data untuk bar chart
        x = np.arange(len(top_class_names))
        width = 0.8 / len(splits)  # Lebar bar
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, split in enumerate(splits):
            split_data = []
            for cls in top_class_names:
                split_data.append(class_stats.get(cls, {}).get(split, 0))
            
            ax.bar(x + i*width - width*len(splits)/2 + width/2, 
                 split_data, 
                 width, 
                 label=split,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Labels dan title
        ax.set_xlabel('Kelas')
        ax.set_ylabel('Jumlah Sampel')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_class_names, rotation=45, ha='right')
        ax.legend(title='Split')
        
        plt.tight_layout()
        
        # Simpan visualisasi
        if save_path:
            output_path = save_path
        else:
            output_path = str(self.output_dir / "class_distribution_summary.png")
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_dataset_dashboard(
        self, 
        report: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Buat dashboard visualisasi dari laporan dataset.
        
        Args:
            report: Laporan dataset
            save_path: Path untuk menyimpan visualisasi (opsional)
            
        Returns:
            Path ke file dashboard
        """
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Distribusi split
        ax_split = fig.add_subplot(gs[0, 0])
        self._plot_split_distribution(ax_split, report)
        
        # 2. Distribusi kelas (top 5)
        ax_class = fig.add_subplot(gs[0, 1:])
        self._plot_class_summary(ax_class, report)
        
        # 3. Distribusi layer
        ax_layer = fig.add_subplot(gs[1, 0])
        self._plot_layer_distribution(ax_layer, report)
        
        # 4. Distribusi ukuran bbox
        ax_bbox = fig.add_subplot(gs[1, 1])
        self._plot_bbox_distribution(ax_bbox, report)
        
        # 5. Metrik kualitas
        ax_quality = fig.add_subplot(gs[1, 2])
        self._plot_quality_metrics(ax_quality, report)
        
        # 6. Ringkasan rekomendasi
        ax_rec = fig.add_subplot(gs[2, :])
        self._plot_recommendations(ax_rec, report)
        
        # Title dan layout
        fig.suptitle("Dashboard Dataset SmartCash", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Simpan dashboard
        if save_path:
            output_path = save_path
        else:
            output_path = str(self.output_dir / "dataset_dashboard.png")
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Dashboard dataset disimpan ke: {output_path}")
        return output_path
    
    def _plot_split_distribution(self, ax, report: Dict[str, Any]) -> None:
        """Plot distribusi data per split."""
        # Extract data
        split_data = report.get('split_stats', {})
        
        if not split_data:
            ax.text(0.5, 0.5, "Tidak ada data split", ha='center', va='center')
            return
        
        splits = []
        images = []
        labels = []
        
        for split, stats in split_data.items():
            splits.append(split)
            images.append(stats.get('images', 0))
            labels.append(stats.get('labels', 0))
        
        # Create grouped bar chart
        x = np.arange(len(splits))
        width = 0.35
        
        ax.bar(x - width/2, images, width, label='Gambar', color=self.palette[0])
        ax.bar(x + width/2, labels, width, label='Label', color=self.palette[2])
        
        # Add text labels
        for i, v in enumerate(images):
            ax.text(i - width/2, v + max(images) * 0.02, str(v), ha='center', fontsize=9)
        
        for i, v in enumerate(labels):
            ax.text(i + width/2, v + max(labels) * 0.02, str(v), ha='center', fontsize=9)
        
        # Styling
        ax.set_title('Distribusi Split')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend()
        
    def _plot_class_summary(self, ax, report: Dict[str, Any]) -> None:
        """Plot ringkasan distribusi kelas."""
        # Extract class distribution from report
        class_metrics = report.get('class_metrics', {})
        class_percentages = class_metrics.get('class_percentages', {})
        
        if not class_percentages:
            ax.text(0.5, 0.5, "Tidak ada data distribusi kelas", ha='center', va='center')
            return
        
        # Get top 5 classes
        top_classes = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
        classes, percentages = zip(*top_classes)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, percentages, color=self.palette[:len(classes)])
        
        # Add percentage labels
        for i, v in enumerate(percentages):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
        
        # Styling
        ax.set_title('Top 5 Kelas')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Persentase')
        
    def _plot_layer_distribution(self, ax, report: Dict[str, Any]) -> None:
        """Plot distribusi layer."""
        # Extract layer distribution
        layer_metrics = report.get('layer_metrics', {})
        layer_percentages = layer_metrics.get('layer_percentages', {})
        
        if not layer_percentages:
            ax.text(0.5, 0.5, "Tidak ada data distribusi layer", ha='center', va='center')
            return
            
        # Create pie chart
        layers = list(layer_percentages.keys())
        values = list(layer_percentages.values())
        
        # Use % as labels
        labels = [f'{layer}: {val:.1f}%' for layer, val in zip(layers, values)]
        
        ax.pie(values, labels=None, autopct='%1.1f%%', startangle=90, 
              colors=self.accent_palette[:len(values)])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Add legend
        ax.legend(labels, loc="upper right", bbox_to_anchor=(1.1, 1))
        ax.set_title('Distribusi Layer')
        
    def _plot_bbox_distribution(self, ax, report: Dict[str, Any]) -> None:
        """Plot distribusi ukuran bounding box."""
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
            ax.text(i, v + max(counts) * 0.02, str(v), ha='center')
        
        # Styling
        ax.set_title('Distribusi Ukuran Bounding Box')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        
    def _plot_quality_metrics(self, ax, report: Dict[str, Any]) -> None:
        """Plot metrik kualitas dataset."""
        # Extract quality metrics
        quality_score = report.get('quality_score', 0)
        
        # Create a gauge-like visualization
        gauge_colors = ['#f44336', '#f57c00', '#ffeb3b', '#4caf50']
        
        # Determine color based on score
        if quality_score < 50:
            color_idx = 0
        elif quality_score < 70:
            color_idx = 1
        elif quality_score < 85:
            color_idx = 2
        else:
            color_idx = 3
            
        # Draw gauge
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='#f5f5f5'))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color='white'))
        
        # Add score and label
        ax.text(0.5, 0.5, f"{quality_score:.1f}", ha='center', va='center', fontsize=24, 
               fontweight='bold', color=gauge_colors[color_idx])
        ax.text(0.5, 0.3, "dari 100", ha='center', va='center', fontsize=10)
        
        # Add title
        ax.text(0.5, 0.85, "Skor Kualitas Dataset", ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # Remove axis
        ax.axis('off')
        
    def _plot_recommendations(self, ax, report: Dict[str, Any]) -> None:
        """Plot ringkasan rekomendasi."""
        # Extract recommendations
        recommendations = report.get('recommendations', [])
        
        if not recommendations:
            ax.text(0.5, 0.5, "Tidak ada rekomendasi", ha='center', va='center')
            return
            
        # Create a text box with recommendations
        recommendations_text = "Rekomendasi:\n\n"
        for i, rec in enumerate(recommendations[:3], 1):  # Limit to top 3
            recommendations_text += f"{i}. {rec}\n"
            
        if len(recommendations) > 3:
            recommendations_text += f"\n...dan {len(recommendations) - 3} rekomendasi lainnya."
        
        # Plot text
        ax.text(0.5, 0.5, recommendations_text, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='#f5f5f5', alpha=0.5))
        
        # Remove axis
        ax.axis('off')
        ax.set_title('Rekomendasi Utama')