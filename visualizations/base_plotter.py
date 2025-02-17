# File: src/visualizations/base_plotter.py
# Author: Alfrida Sabar
# Deskripsi: Base plotter untuk visualisasi data dengan dukungan tema standar

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class BasePlotter(ABC):
    """Base class untuk semua visualizer dengan konfigurasi standar"""
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self._setup_plot_style()
        
    def _setup_plot_style(self):
        """Setup style plot standar"""
        try:
            # Use a built-in Matplotlib style instead of relying on Seaborn
            plt.style.use('ggplot')  # A widely supported style
        except Exception:
            # Fallback to default style if 'ggplot' fails
            plt.style.use('default')
        
        # Configure plot parameters
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['font.size'] = 10
        
        # Set color palette if Seaborn is available
        try:
            sns.set_palette("Set2")  # Alternative color palette
        except Exception:
            pass  # Ignore if Seaborn is not fully configured
        
    def _ensure_dir(self, subdir: Optional[str] = None) -> Path:
        """Pastikan direktori tujuan ada"""
        save_path = self.save_dir
        if subdir:
            save_path = save_path / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
        
    def _save_plot(self, filename: str, subdir: Optional[str] = None):
        """Simpan plot dengan format standar"""
        save_path = self._ensure_dir(subdir)
        plt.savefig(save_path / f"{filename}.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    @abstractmethod
    def plot(self, data: Dict, **kwargs):
        """Plot data dengan konfigurasi yang diberikan"""
        pass

class DatasetPlotter(BasePlotter):
    """Visualizer untuk statistik dataset"""
    def __init__(self):
        super().__init__('src/visualizations/dataset_stats')
        
    def plot(self, data: Dict, **kwargs):
        """Implementasi plot untuk dataset"""
        splits = list(data.keys())
        classes = list(data[splits[0]].keys())
        
        plt.figure(figsize=(12, 6))
        
        width = 0.8 / len(splits)
        for i, split in enumerate(splits):
            x = [j + width*i for j in range(len(classes))]
            plt.bar(x, [data[split][cls] for cls in classes], 
                   width=width, label=split)
        
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah Sampel')
        plt.title('Distribusi Kelas per Split')
        plt.xticks([i + width*(len(splits)-1)/2 for i in range(len(classes))], 
                  classes, rotation=45)
        plt.legend()
        
        self._save_plot('class_distribution', 'distributions')

class QualityPlotter(BasePlotter):
    """Visualizer untuk metrik kualitas"""
    def __init__(self):
        super().__init__('src/visualizations/quality_metrics')
        
    def plot(self, data: Dict, **kwargs):
        """Plot metrik kualitas gambar"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Blur scores
        sns.histplot(data=data['blur_scores'], ax=ax1)
        ax1.set_title('Distribusi Blur Score')
        ax1.set_xlabel('Blur Score')
        
        # Brightness
        sns.histplot(data=data['brightness'], ax=ax2)
        ax2.set_title('Distribusi Brightness')
        ax2.set_xlabel('Mean Brightness')
        
        # Contrast
        sns.histplot(data=data['contrast'], ax=ax3)
        ax3.set_title('Distribusi Contrast')
        ax3.set_xlabel('Contrast Score')
        
        plt.tight_layout()
        self._save_plot('quality_metrics', 'quality')
        
class BoxPlotter(BasePlotter):
    """Visualizer untuk statistik bounding box"""
    def __init__(self):
        super().__init__('src/visualizations/box_stats')
        
    def plot(self, data: Dict, **kwargs):
        """Plot statistik bounding box"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box sizes
        sns.histplot(data=data['sizes'], ax=ax1)
        ax1.set_title('Distribusi Ukuran Box')
        ax1.set_xlabel('Normalized Area')
        
        # Box aspects
        sns.histplot(data=data['aspects'], ax=ax2)
        ax2.set_title('Distribusi Aspect Ratio')
        ax2.set_xlabel('Width/Height Ratio')
        
        plt.tight_layout()
        self._save_plot('box_statistics', 'boxes')