"""
File: smartcash/dataset/visualization/helpers/class_visualizer.py
Deskripsi: Helper untuk visualisasi distribusi dan metrik kelas dalam dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any


class ClassVisualizer:
    """Helper untuk visualisasi distribusi dan metrik kelas dalam dataset."""
    
    def __init__(self):
        """Inisialisasi ClassVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
    
    def plot_class_distribution(
        self, 
        ax, 
        class_percentages: Dict[str, float],
        title: str = "Distribusi Kelas",
        top_n: int = 10
    ) -> None:
        """
        Plot visualisasi distribusi kelas.
        
        Args:
            ax: Axes untuk plot
            class_percentages: Dictionary berisi persentase per kelas
            title: Judul visualisasi
            top_n: Jumlah kelas teratas yang ditampilkan
        """
        # Ambil kelas teratas
        top_classes = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not top_classes:
            ax.text(0.5, 0.5, "Tidak ada data distribusi kelas", ha='center', va='center')
            ax.set_title(title)
            return
            
        classes, percentages = zip(*top_classes)
        
        # Buat plot
        bars = ax.barh(range(len(classes)), percentages, color=self.palette[:len(classes)])
        
        # Tambahkan label
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel('Persentase (%)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Urutan top-to-bottom
        
        # Tambahkan nilai di bar
        for i, (bar, percentage) in enumerate(zip(bars, percentages)):
            ax.text(percentage + 0.5, i, f"{percentage:.1f}%", va='center')
    
    def plot_class_summary(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot ringkasan distribusi kelas.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
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
    
    def plot_class_comparison(self, ax, comparison: Dict[str, Any]) -> None:
        """
        Plot perbandingan distribusi kelas.
        
        Args:
            ax: Axes untuk plot
            comparison: Data perbandingan
        """
        # Extract class comparison
        class_comparison = comparison.get('class_comparison', {})
        
        if not class_comparison:
            ax.text(0.5, 0.5, "Tidak ada data perbandingan kelas", ha='center', va='center')
            return
            
        # Get common classes across all datasets
        all_classes = set()
        for dataset, percentages in class_comparison.items():
            all_classes.update(percentages.keys())
        
        # Get top 5 classes across all datasets
        class_total_percentages = {}
        for cls in all_classes:
            total = 0
            for dataset, percentages in class_comparison.items():
                total += percentages.get(cls, 0)
            class_total_percentages[cls] = total
            
        top_classes = sorted(class_total_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
        top_class_names = [cls for cls, _ in top_classes]
        
        # Create grouped bar chart
        datasets = list(class_comparison.keys())
        x = np.arange(len(top_class_names))
        width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            percentages = class_comparison[dataset]
            values = [percentages.get(cls, 0) for cls in top_class_names]
            
            ax.bar(x + i*width - width*len(datasets)/2 + width/2, 
                 values, 
                 width, 
                 label=dataset,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Styling
        ax.set_title('Perbandingan Top 5 Kelas')
        ax.set_xticks(x)
        ax.set_xticklabels(top_class_names, rotation=45, ha='right')
        ax.set_ylabel('Persentase')
        ax.legend(title='Dataset')
    
    def plot_class_distribution_comparison(self, ax, reports: List[Dict[str, Any]], labels: List[str]) -> None:
        """
        Plot perbandingan distribusi kelas.
        
        Args:
            ax: Axes untuk plot
            reports: List laporan dataset
            labels: Label untuk setiap laporan
        """
        # Extract class distributions
        class_percentages_list = []
        
        for report in reports:
            class_metrics = report.get('class_metrics', {})
            class_percentages = class_metrics.get('class_percentages', {})
            class_percentages_list.append(class_percentages)
        
        # Get common classes
        all_classes = set()
        for class_percentages in class_percentages_list:
            all_classes.update(class_percentages.keys())
        
        # Get top classes across all reports
        class_totals = {}
        for cls in all_classes:
            total = 0
            for class_percentages in class_percentages_list:
                total += class_percentages.get(cls, 0)
            class_totals[cls] = total
            
        top_classes = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        top_class_names = [cls for cls, _ in top_classes]
        
        if not top_class_names:
            ax.text(0.5, 0.5, "Tidak ada data kelas", ha='center', va='center')
            ax.set_title('Perbandingan Distribusi Kelas')
            return
        
        # Create bar chart
        x = np.arange(len(top_class_names))
        width = 0.8 / len(labels)
        
        for i, (label, class_percentages) in enumerate(zip(labels, class_percentages_list)):
            values = [class_percentages.get(cls, 0) for cls in top_class_names]
            
            ax.bar(x + i*width - width*len(labels)/2 + width/2, 
                 values, 
                 width, 
                 label=label,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Styling
        ax.set_title('Distribusi Top 5 Kelas')
        ax.set_xticks(x)
        ax.set_xticklabels(top_class_names, rotation=45, ha='right')
        ax.set_ylabel('Persentase')
        ax.legend(title='Dataset')