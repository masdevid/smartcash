"""
File: smartcash/dataset/visualization/helpers/layer_visualizer.py
Deskripsi: Helper untuk visualisasi distribusi dan metrik layer dalam dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any


class LayerVisualizer:
    """Helper untuk visualisasi distribusi dan metrik layer dalam dataset."""
    
    def __init__(self):
        """Inisialisasi LayerVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
    
    def plot_layer_distribution(
        self, 
        ax, 
        layer_percentages: Dict[str, float],
        title: str = "Distribusi Layer"
    ) -> None:
        """
        Plot visualisasi distribusi layer.
        
        Args:
            ax: Axes untuk plot
            layer_percentages: Dictionary berisi persentase per layer
            title: Judul visualisasi
        """
        # Data
        layers = list(layer_percentages.keys())
        percentages = list(layer_percentages.values())
        
        if not layers:
            ax.text(0.5, 0.5, "Tidak ada data distribusi layer", ha='center', va='center')
            ax.set_title(title)
            return
        
        # Buat plot
        colors = self.accent_palette[:len(layers)]
        wedges, texts, autotexts = ax.pie(
            percentages, 
            labels=layers, 
            autopct='%1.1f%%',
            colors=colors,
            textprops={'fontsize': 10}
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('equal')  # Equal aspect ratio
    
    def plot_layer_bar_distribution(
        self,
        ax,
        layer_percentages: Dict[str, float],
        title: str = "Distribusi Layer"
    ) -> None:
        """
        Plot visualisasi distribusi layer dengan bar chart.
        
        Args:
            ax: Axes untuk plot
            layer_percentages: Dictionary berisi persentase per layer
            title: Judul visualisasi
        """
        # Data
        layers = list(layer_percentages.keys())
        percentages = list(layer_percentages.values())
        
        if not layers:
            ax.text(0.5, 0.5, "Tidak ada data distribusi layer", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Buat plot
        y_pos = np.arange(len(layers))
        ax.barh(y_pos, percentages, color=self.palette[:len(layers)])
        
        # Tambahkan label
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers)
        ax.set_xlabel('Persentase (%)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Tambahkan nilai di bar
        for i, percentage in enumerate(percentages):
            ax.text(percentage + 0.5, i, f"{percentage:.1f}%", va='center')
    
    def plot_layer_summary(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot ringkasan distribusi layer.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
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
    
    def plot_layer_comparison(self, ax, comparison: Dict[str, Any]) -> None:
        """
        Plot perbandingan distribusi layer.
        
        Args:
            ax: Axes untuk plot
            comparison: Data perbandingan
        """
        # Extract layer comparison
        layer_comparison = comparison.get('layer_comparison', {})
        
        if not layer_comparison:
            ax.text(0.5, 0.5, "Tidak ada data perbandingan layer", ha='center', va='center')
            return
            
        # Get all layers
        all_layers = set()
        for dataset, percentages in layer_comparison.items():
            all_layers.update(percentages.keys())
            
        layers = sorted(all_layers)
        datasets = list(layer_comparison.keys())
        
        # Create grouped bar chart
        x = np.arange(len(layers))
        width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            percentages = layer_comparison[dataset]
            values = [percentages.get(layer, 0) for layer in layers]
            
            ax.bar(x + i*width - width*len(datasets)/2 + width/2, 
                 values, 
                 width, 
                 label=dataset,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Styling
        ax.set_title('Perbandingan Distribusi Layer')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylabel('Persentase')
        ax.legend(title='Dataset')
    
    def plot_layer_distribution_comparison(self, ax, reports: List[Dict[str, Any]], labels: List[str]) -> None:
        """
        Plot perbandingan distribusi layer.
        
        Args:
            ax: Axes untuk plot
            reports: List laporan dataset
            labels: Label untuk setiap laporan
        """
        # Extract layer distributions
        layer_percentages_list = []
        
        for report in reports:
            layer_metrics = report.get('layer_metrics', {})
            layer_percentages = layer_metrics.get('layer_percentages', {})
            layer_percentages_list.append(layer_percentages)
        
        # Get all layers
        all_layers = set()
        for layer_percentages in layer_percentages_list:
            all_layers.update(layer_percentages.keys())
            
        layers = sorted(all_layers)
        
        if not layers:
            ax.text(0.5, 0.5, "Tidak ada data layer", ha='center', va='center')
            ax.set_title('Perbandingan Distribusi Layer')
            return
        
        # Create bar chart
        x = np.arange(len(layers))
        width = 0.8 / len(labels)
        
        for i, (label, layer_percentages) in enumerate(zip(labels, layer_percentages_list)):
            values = [layer_percentages.get(layer, 0) for layer in layers]
            
            ax.bar(x + i*width - width*len(labels)/2 + width/2, 
                 values, 
                 width, 
                 label=label,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Styling
        ax.set_title('Distribusi Layer')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylabel('Persentase')
        ax.legend(title='Dataset')