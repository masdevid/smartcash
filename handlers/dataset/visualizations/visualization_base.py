# File: smartcash/handlers/dataset/visualizations/visualization_base.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk semua visualisasi dataset

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config


class VisualizationBase:
    """
    Kelas dasar untuk semua visualizer dataset.
    Menyediakan fungsi dan konfigurasi umum yang digunakan berbagai visualisasi.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None,
        config: Optional[Dict] = None
    ):
        """
        Inisialisasi visualizer dasar.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output untuk visualisasi
            logger: Logger kustom (opsional)
            config: Konfigurasi dataset (opsional)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'visualizations'
        self.logger = logger or get_logger("visualization_base")
        self.config = config or {}
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi layer config untuk info kelas
        self.layer_config = get_layer_config()
        
        # Setup style plot
        self._setup_plot_style()
        
        self.logger.info(f"🎨 Visualizer diinisialisasi: {self.data_dir}")
    
    def _setup_plot_style(self) -> None:
        """Setup style untuk plot yang konsisten."""
        # Set tema Seaborn
        sns.set_theme(style="whitegrid")
        
        # Palet warna untuk berbagai kebutuhan
        self.color_palette = sns.color_palette("viridis", 15)
        self.color_palette_alt = sns.color_palette("mako", 15)
        self.color_palette_cat = sns.color_palette("Set1", 9)
        
        # Tambahkan colormap untuk heatmap
        self.heatmap_cmap = "viridis"
        
        # Setup warna untuk layer yang umum digunakan
        self.layer_colors = {
            'banknote': '#FF5555',  # Merah muda
            'nominal': '#5555FF',   # Biru
            'security': '#55AA55',  # Hijau
            'watermark': '#AA55AA', # Ungu
            'default': '#AAAAAA'    # Abu-abu untuk layer yang tidak dikenal
        }
        
        # Font yang konsisten
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
    
    def _get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset tertentu.
        
        Args:
            split: Nama split ('train', 'valid', 'test')
            
        Returns:
            Path ke direktori split
        """
        # Normalisasi nama split
        if split in ('val', 'validation'):
            split = 'valid'
        
        return self.data_dir / split
    
    def _get_timestamp(self) -> str:
        """
        Dapatkan timestamp untuk penamaan file.
        
        Returns:
            String timestamp dengan format 'YYYYMMDDHHmmss'
        """
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def _get_class_name(self, cls_id: int) -> str:
        """
        Dapatkan nama kelas berdasarkan ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama kelas atau string ID jika tidak ditemukan
        """
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            class_ids = layer_config['class_ids']
            classes = layer_config['classes']
            
            if cls_id in class_ids:
                idx = class_ids.index(cls_id)
                if idx < len(classes):
                    return classes[idx]
        
        return f"Class-{cls_id}"
    
    def _get_layer_for_class(self, cls_id: int) -> str:
        """
        Dapatkan layer berdasarkan ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama layer atau string default jika tidak ditemukan
        """
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                return layer_name
        
        return "default"
    
    def save_plot(
        self, 
        fig, 
        save_path: Optional[str] = None, 
        default_name: str = "visualization.png",
        dpi: int = 300
    ) -> str:
        """
        Simpan plot ke file dengan penanganan error yang baik.
        
        Args:
            fig: Figure matplotlib yang akan disimpan
            save_path: Path untuk menyimpan visualisasi
            default_name: Nama file default jika save_path tidak disediakan
            dpi: Resolusi gambar output
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        try:
            # Tentukan path simpan
            if save_path is None:
                save_path = str(self.output_dir / default_name)
            
            # Buat direktori jika belum ada
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Simpan plot
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"✅ Visualisasi tersimpan di: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan plot: {str(e)}")
            return ""