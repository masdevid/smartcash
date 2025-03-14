"""
File: smartcash/model/visualization/base_research_visualizer.py
Deskripsi: Modul dasar untuk visualisasi penelitian model deteksi objek
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.model.visualization.base_visualizer import VisualizationHelper
from smartcash.common.logger import get_logger

class BaseResearchVisualizer:
    """Kelas dasar untuk visualisasi hasil penelitian dengan fungsionalitas umum."""
    
    def __init__(
        self, 
        output_dir: str = "results/research",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi base visualizer penelitian.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            logger: Logger untuk logging
        """
        self.output_dir = VisualizationHelper.create_output_directory(output_dir)
        self.logger = logger or get_logger("research_visualizer")
        
        # Setup plot style
        VisualizationHelper.set_plot_style()
    
    def _create_styled_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Buat DataFrame dengan styling untuk highlight nilai terbaik.
        
        Args:
            df: DataFrame input
            
        Returns:
            Styled DataFrame
        """
        # Identifikasi kolom metrik dan performansi
        metric_cols = [col for col in df.columns if col in 
                     ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']]
        time_col = next((col for col in df.columns if 'Time' in col or 'Waktu' in col), None)
        
        # Buat salinan untuk styling
        styled_df = df.copy()
        
        # Highlight nilai terbaik untuk metrik (nilai tertinggi)
        for col in metric_cols:
            max_val = df[col].max()
            styled_df.loc[df[col] == max_val, col] = f"**{df.loc[df[col] == max_val, col].values[0]:.2f}**"
        
        # Highlight nilai terbaik untuk waktu (nilai terendah)
        if time_col:
            min_val = df[time_col].min()
            styled_df.loc[df[time_col] == min_val, time_col] = f"**{df.loc[df[time_col] == min_val, time_col].values[0]:.2f}**"
        
        return styled_df
    
    def _add_tradeoff_regions(self, ax: plt.Axes) -> None:
        """
        Tambahkan regions untuk visualisasi trade-off.
        
        Args:
            ax: Axes matplotlib
        """
        # Get current axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Define regions
        x_mid1 = x_min + (x_max - x_min) * 0.33
        x_mid2 = x_min + (x_max - x_min) * 0.66
        y_mid1 = y_min + (y_max - y_min) * 0.33
        y_mid2 = y_min + (y_max - y_min) * 0.66
        
        # Add regions with low alpha
        # High accuracy, slow speed (top right)
        ax.fill_betweenx([y_mid2, y_max], x_mid2, x_max, alpha=0.1, color='orange')
        ax.text(x_mid2 + (x_max - x_mid2)/2, y_mid2 + (y_max - y_mid2)/2, 
               "Akurasi Tinggi\nKecepatan Rendah", ha='center', va='center', alpha=0.7)
        
        # Balanced (middle)
        ax.fill_betweenx([y_mid1, y_mid2], x_mid1, x_mid2, alpha=0.1, color='green')
        ax.text(x_mid1 + (x_mid2 - x_mid1)/2, y_mid1 + (y_mid2 - y_mid1)/2, 
               "Seimbang", ha='center', va='center', alpha=0.7)
        
        # High speed, low accuracy (bottom left)
        ax.fill_betweenx([y_min, y_mid1], x_min, x_mid1, alpha=0.1, color='red')
        ax.text(x_min + (x_mid1 - x_min)/2, y_min + (y_mid1 - y_min)/2, 
               "Kecepatan Tinggi\nAkurasi Rendah", ha='center', va='center', alpha=0.7)
        
        # Best region (top left): high accuracy, high speed
        ax.fill_betweenx([y_mid2, y_max], x_min, x_mid1, alpha=0.1, color='green')
        ax.text(x_min + (x_mid1 - x_min)/2, y_mid2 + (y_max - y_mid2)/2, 
               "Optimal\n(Akurasi & Kecepatan Tinggi)", ha='center', va='center', 
               fontweight='bold', alpha=0.7)
    
    def save_visualization(self, fig: plt.Figure, filename: Optional[str]) -> bool:
        """
        Simpan visualisasi ke file.
        
        Args:
            fig: Figure matplotlib
            filename: Nama file (opsional)
            
        Returns:
            Boolean sukses/gagal
        """
        if not filename:
            return False
            
        output_path = self.output_dir / filename
        return VisualizationHelper.save_figure(fig, output_path, logger=self.logger)