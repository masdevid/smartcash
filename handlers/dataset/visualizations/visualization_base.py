# File: smartcash/handlers/dataset/visualizations/visualization_base.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk semua visualisasi dataset

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.handlers.dataset.core.dataset_explorer import DatasetExplorer


class VisualizationBase:
    """
    Kelas dasar untuk semua visualizer dataset.
    Menyediakan konfigurasi dan fungsi dasar yang umum digunakan.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi visualizer dasar.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset
            output_dir: Direktori output untuk visualisasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'visualizations'
        self.logger = logger or get_logger("visualization_base")
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi layer config untuk info kelas
        self.layer_config = get_layer_config()
        
        # Inisialisasi explorer untuk analisis data
        self.explorer = DatasetExplorer(config, data_dir, logger=self.logger)
        
        # Setup style plot
        self._setup_plot_style()
        
        self.logger.info(f"🎨 Visualizer diinisialisasi: {self.data_dir}")
    
    def _setup_plot_style(self) -> None:
        """Setup style untuk plot yang konsisten."""
        # Set tema Seaborn
        sns.set_theme(style="whitegrid")
        
        # Style warna
        self.color_palette = sns.color_palette("viridis", 15)
        
        # Tambah warna alternatif
        self.color_palette_alt = sns.color_palette("mako", 15)
        
        # Font yang konsisten
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
    
    def get_class_name(self, cls_id: int) -> str:
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
    
    def get_layer_for_class(self, cls_id: int) -> str:
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
        default_name: str = "visualization.png"
    ) -> str:
        """
        Simpan plot ke file dengan penanganan error yang baik.
        
        Args:
            fig: Figure matplotlib yang akan disimpan
            save_path: Path untuk menyimpan visualisasi
            default_name: Nama file default jika save_path tidak disediakan
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        try:
            # Simpan plot
            if save_path is None:
                save_path = str(self.output_dir / default_name)
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan plot: {str(e)}")
            return ""