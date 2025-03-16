"""
File: smartcash/dataset/visualization/helpers/split_visualizer.py
Deskripsi: Helper untuk visualisasi distribusi dan perbandingan split dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any


class SplitVisualizer:
    """Helper untuk visualisasi distribusi dan perbandingan split dataset."""
    
    def __init__(self):
        """Inisialisasi SplitVisualizer."""
        # Setup style
        self.palette = sns.color_palette("viridis", 12)
        self.accent_palette = sns.color_palette("Set2", 8)
    
    def plot_split_distribution(
        self, 
        ax, 
        split_stats: Dict[str, Dict[str, int]],
        title: str = "Distribusi Split"
    ) -> None:
        """
        Plot visualisasi distribusi split.
        
        Args:
            ax: Axes untuk plot
            split_stats: Dictionary berisi statistik per split
            title: Judul visualisasi
        """
        # Ekstrak data
        splits = []
        images = []
        labels = []
        
        for split, stats in split_stats.items():
            splits.append(split)
            images.append(stats.get('images', 0))
            labels.append(stats.get('labels', 0))
        
        if not splits:
            ax.text(0.5, 0.5, "Tidak ada data split", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Buat plot
        x = np.arange(len(splits))
        width = 0.35
        
        image_bars = ax.bar(x - width/2, images, width, label='Gambar', color=self.palette[0])
        label_bars = ax.bar(x + width/2, labels, width, label='Label', color=self.palette[2])
        
        # Tambahkan nilai di bar
        for bar, count in zip(image_bars, images):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom')
                   
        for bar, count in zip(label_bars, labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom')
        
        # Labels dan legends
        ax.set_xlabel('Split')
        ax.set_ylabel('Jumlah')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend()
    
    def plot_split_summary(self, ax, report: Dict[str, Any]) -> None:
        """
        Plot ringkasan statistik split.
        
        Args:
            ax: Axes untuk plot
            report: Data laporan dataset
        """
        # Extract split data
        split_data = report.get('split_stats', {})
        
        if not split_data:
            ax.text(0.5, 0.5, "Tidak ada data split", ha='center', va='center')
            ax.set_title('Statistik Split')
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
            if v > 0:
                ax.text(i - width/2, v + max(images) * 0.02, str(v), ha='center', fontsize=9)
        
        for i, v in enumerate(labels):
            if v > 0:
                ax.text(i + width/2, v + max(labels) * 0.02, str(v), ha='center', fontsize=9)
        
        # Styling
        ax.set_title('Distribusi Split')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel('Jumlah')
        ax.legend()
    
    def plot_split_comparison(self, ax, comparison: Dict[str, Any]) -> None:
        """
        Plot perbandingan statistik split.
        
        Args:
            ax: Axes untuk plot
            comparison: Data perbandingan
        """
        # Extract split comparison
        split_comparison = comparison.get('split_comparison', {})
        
        if not split_comparison:
            ax.text(0.5, 0.5, "Tidak ada data perbandingan split", ha='center', va='center')
            ax.set_title('Perbandingan Split')
            return
            
        # Get all splits
        all_splits = set()
        for dataset, split_stats in split_comparison.items():
            all_splits.update(split_stats.keys())
            
        splits = sorted(all_splits)
        datasets = list(split_comparison.keys())
        
        # Create grouped bar chart for image counts
        x = np.arange(len(splits))
        width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            split_stats = split_comparison[dataset]
            values = [split_stats.get(split, {}).get('images', 0) for split in splits]
            
            ax.bar(x + i*width - width*len(datasets)/2 + width/2, 
                 values, 
                 width, 
                 label=dataset,
                 color=self.accent_palette[i % len(self.accent_palette)])
        
        # Styling
        ax.set_title('Perbandingan Jumlah Gambar per Split')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel('Jumlah Gambar')
        ax.legend(title='Dataset')
    
    def plot_split_ratio_comparison(self, ax, reports: List[Dict[str, Any]], labels: List[str]) -> None:
        """
        Plot perbandingan rasio split dari beberapa laporan.
        
        Args:
            ax: Axes untuk plot
            reports: List laporan dataset
            labels: Label untuk setiap laporan
        """
        # Collect split ratios
        ratios = []
        splits = []
        
        for report in reports:
            split_stats = report.get('split_stats', {})
            
            if not split_stats:
                continue
                
            # Identify splits
            for split in split_stats.keys():
                if split not in splits:
                    splits.append(split)
        
        if not splits:
            ax.text(0.5, 0.5, "Tidak ada data split", ha='center', va='center')
            ax.set_title('Rasio Split')
            return
        
        # Calculate ratios
        for report in reports:
            split_stats = report.get('split_stats', {})
            
            if not split_stats:
                continue
                
            total_images = sum(stats.get('images', 0) for stats in split_stats.values())
            
            if total_images > 0:
                report_ratios = []
                for split in splits:
                    if split in split_stats:
                        ratio = (split_stats[split].get('images', 0) / total_images) * 100
                    else:
                        ratio = 0
                    report_ratios.append(ratio)
                
                ratios.append(report_ratios)
            
        # Create stacked bar chart
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        
        for i, split in enumerate(splits):
            split_values = [ratio_list[i] if i < len(ratio_list) else 0 for ratio_list in ratios]
            ax.bar(x, split_values, bottom=bottom, label=split, 
                  color=self.accent_palette[i % len(self.accent_palette)])
            bottom += split_values
        
        # Styling
        ax.set_title('Rasio Split')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Persentase (%)')
        ax.set_ylim(0, 100)
        ax.legend()