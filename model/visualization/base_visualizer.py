"""
File: smartcash/model/visualization/base_visualizer.py
Deskripsi: Modul dasar untuk visualisasi model deteksi objek
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Any

from smartcash.common.visualization.core.visualization_base import VisualizationBase
from smartcash.common.logger import get_logger

class ModelVisualizationBase(VisualizationBase):
    """
    Kelas dasar untuk visualisasi model dengan fungsionalitas umum yang digunakan oleh visualizer lain
    """
    
    def __init__(
        self, 
        output_dir: Union[str, Path] = "results/visualizations",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi base visualizer.
        
        Args:
            output_dir: Direktori output untuk visualisasi
            logger: Logger untuk logging
        """
        super().__init__()
        self.output_dir = self.create_output_directory(output_dir)
        self.logger = logger or get_logger(self.__class__.__name__.lower())
        
        # Setup default style
        self.set_plot_style()
        
        self.logger.info(f"🎨 {self.__class__.__name__} diinisialisasi")
    
    @staticmethod
    def set_plot_style(style: str = "whitegrid") -> None:
        """
        Set style untuk matplotlib plots.
        
        Args:
            style: Gaya plot seaborn
        """
        try:
            sns.set_style(style)
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 12
        except Exception as e:
            print(f"⚠️ Gagal mengatur plot style: {str(e)}")
    
    def save_figure(
        self,
        fig: plt.Figure, 
        filepath: Union[str, Path], 
        dpi: int = 300, 
        bbox_inches: str = 'tight'
    ) -> bool:
        """
        Simpan figure matplotlib dengan error handling.
        
        Args:
            fig: Figure matplotlib
            filepath: Path untuk menyimpan gambar
            dpi: DPI untuk output
            bbox_inches: Pengaturan area simpan
            
        Returns:
            Boolean sukses/gagal
        """
        try:
            # Pastikan direktori ada
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simpan gambar
            fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
            
            self.logger.info(f"📊 Plot disimpan ke {output_path}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyimpan plot: {str(e)}")
            return False
    
    @staticmethod
    def create_output_directory(output_dir: Union[str, Path]) -> Path:
        """
        Buat direktori output jika belum ada.
        
        Args:
            output_dir: Direktori output
            
        Returns:
            Path direktori yang dibuat
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path