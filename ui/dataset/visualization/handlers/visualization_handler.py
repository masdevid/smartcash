"""
File: smartcash/ui/dataset/visualization/handlers/visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset dan hasil preprocessing/augmentasi
"""

import logging
from typing import Any, Dict, Optional
import os
import json
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartcash.common.logger import get_logger
from smartcash.common.utils import load_yaml
from smartcash.dataset.preprocessor import get_preprocessing_stats
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

class DatasetVisualizationHandler:
    """Handler untuk visualisasi dataset dan hasil preprocessing/augmentasi"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Inisialisasi handler visualisasi
        
        Args:
            config_path: Path ke file konfigurasi (opsional)
        """
        self.config = self._load_config(config_path)
        self.current_dataset = None
        self.preprocessing_stats = {}
        self.augmentation_stats = {}
        self.ui_components = {}
        self.augmentation_data = {
            'total_generated': 0,
            'total_normalized': 0,
            'processing_time': 0.0
        }
        self.preprocessing_data = {
            'success': False,
            'processing_time': 0.0,
            'stats': {'valid_images': 0}
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Memuat konfigurasi visualisasi
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Dict konfigurasi
        """
        default_config = {
            "visualization": {
                "max_images_to_show": 10,
                "image_size": (300, 300),
                "theme": "plotly_white",
                "colors": {
                    "train": "#1f77b4",
                    "val": "#ff7f0e",
                    "test": "#2ca02c"
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                custom_config = load_yaml(config_path)
                # Merge dengan konfigurasi default
                default_config.update(custom_config)
                logger.info(f"Konfigurasi visualisasi dimuat dari {config_path}")
            except Exception as e:
                logger.warning(f"Gagal memuat konfigurasi dari {config_path}: {e}")
        
        return default_config
    
    def load_dataset(self, dataset_name: str) -> bool:
        """Memuat dataset untuk divisualisasikan
        
        Args:
            dataset_name: Nama dataset yang akan dimuat
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Cek ketersediaan dataset
            available_datasets = list_available_datasets()
            if dataset_name not in available_datasets:
                logger.error(f"Dataset '{dataset_name}' tidak ditemukan")
                return False
                
            self.current_dataset = dataset_name
            
            # Muat statistik preprocessing dengan error handling
            try:
                self.preprocessing_stats = get_preprocessing_stats(dataset_name)
            except Exception as e:
                logger.warning(f"Gagal memuat statistik preprocessing: {e}")
                self.preprocessing_stats = {}
            
            # Muat metadata dataset
            self.dataset_metadata = get_dataset_metadata(dataset_name)
            
            logger.info(f"Dataset '{dataset_name}' berhasil dimuat")
            return True
            
        except Exception as e:
            logger.error(f"Gagal memuat dataset '{dataset_name}': {e}")
            return False
    
    def visualize_class_distribution(self) -> go.Figure:
        """Visualisasi distribusi kelas dalam dataset
        
        Returns:
            go.Figure: Plot distribusi kelas
        """
        if not self.preprocessing_stats:
            logger.warning("Tidak ada data preprocessing yang tersedia")
            return None
            
        try:
            # Siapkan data untuk plotting
            class_data = self.preprocessing_stats.get("class_distribution", {})
            splits = ["train", "val", "test"]
            
            # Buat figure
            fig = make_subplots(rows=1, cols=len(splits), 
                              subplot_titles=[f"Distribusi Kelas - {split.upper()}" 
                                            for split in splits])
            
            for i, split in enumerate(splits, 1):
                if split in class_data:
                    classes = list(class_data[split].keys())
                    counts = list(class_data[split].values())
                    
                    fig.add_trace(
                        go.Bar(
                            x=classes,
                            y=counts,
                            name=f"{split.upper()}",
                            marker_color=self.config["visualization"]["colors"].get(split, "gray")
                        ),
                        row=1, col=i
                    )
            
            # Update layout
            fig.update_layout(
                title_text="Distribusi Kelas per Split Dataset",
                showlegend=False,
                height=500,
                template=self.config["visualization"].get("theme", "plotly_white")
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Gagal membuat visualisasi distribusi kelas: {e}")
            return None
    
    def visualize_image_sizes(self) -> go.Figure:
        """Visualisasi distribusi ukuran gambar dalam dataset
        
        Returns:
            go.Figure: Plot distribusi ukuran gambar
        """
        if not self.preprocessing_stats:
            logger.warning("Tidak ada data preprocessing yang tersedia")
            return None
            
        try:
            # Siapkan data untuk plotting
            size_data = self.preprocessing_stats.get("image_sizes", {})
            
            # Buat figure
            fig = go.Figure()
            
            for split, sizes in size_data.items():
                if sizes:
                    widths = [s[0] for s in sizes]
                    heights = [s[1] for s in sizes]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=widths,
                            y=heights,
                            mode='markers',
                            name=split.upper(),
                            marker=dict(
                                color=self.config["visualization"]["colors"].get(split, "gray"),
                                size=8,
                                opacity=0.7
                            )
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title="Distribusi Ukuran Gambar",
                xaxis_title="Lebar (px)",
                yaxis_title="Tinggi (px)",
                legend_title="Split",
                template=self.config["visualization"].get("theme", "plotly_white"),
                height=600
            )
            
            # Tambahkan garis aspek rasio umum
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper",
                         line=dict(color="gray", dash="dash"))
            
            return fig
            
        except Exception as e:
            logger.error(f"Gagal membuat visualisasi ukuran gambar: {e}")
            return None
    
    def visualize_augmentation_examples(self, num_examples: int = 5) -> widgets.Widget:
        """Visualisasi contoh hasil augmentasi
        
        Args:
            num_examples: Jumlah contoh yang akan ditampilkan
            
        Returns:
            widgets.Widget: Widget berisi contoh hasil augmentasi
        """
        if not self.current_dataset:
            logger.warning("Tidak ada dataset yang dimuat")
            return None
            
        try:
            # Dapatkan path ke direktori dataset
            dataset_path = os.path.join("data", "processed", self.current_dataset)
            
            # Dapatkan daftar gambar dari direktori train
            train_img_dir = os.path.join(dataset_path, "images", "train")
            if not os.path.exists(train_img_dir):
                logger.error(f"Direktori train tidak ditemukan: {train_img_dir}")
                return None
                
            image_files = [f for f in os.listdir(train_img_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.warning("Tidak ada gambar yang ditemukan di direktori train")
                return None
                
            # Batasi jumlah contoh
            image_files = image_files[:min(num_examples, len(image_files))]
            
            # Buat widget untuk menampilkan contoh
            output = widgets.Output()
            
            with output:
                display(widgets.HTML("<h3>Contoh Hasil Augmentasi</h3>"))
                
                for img_file in image_files:
                    img_path = os.path.join(train_img_dir, img_file)
                    
                    # Tampilkan gambar asli
                    display(widgets.HTML(f"<h4>Original: {img_file}</h4>"))
                    display(widgets.Image(value=open(img_path, 'rb').read(), 
                                        format='jpg',
                                        width=300))
                    
                    # Tambahkan pembatas
                    display(widgets.HTML("<hr>"))
            
            return output
            
        except Exception as e:
            logger.error(f"Gagal membuat visualisasi contoh augmentasi: {e}")
            return None
    
    def fetch_augmentation_status(self):
        """Ambil status augmentasi (dummy jika API tidak tersedia)"""
        try:
            # Gunakan data dummy jika API tidak tersedia
            self.augmentation_data = {
                'service_ready': False,
                'train_augmented': 0,
                'train_preprocessed': 0
            }
        except ImportError:
            pass

    def fetch_preprocessing_status(self):
        """Ambil status preprocessing (dummy jika API tidak tersedia)"""
        try:
            from smartcash.dataset.preprocessor import get_preprocessing_status
            self.preprocessing_data = get_preprocessing_status(self.config)
        except ImportError:
            # Gunakan data dummy jika API tidak tersedia
            self.preprocessing_data = {
                'success': False,
                'stats': {'valid_images': 0}
            }

    def get_stats(self):
        """Kumpulkan statistik untuk ditampilkan"""
        self.fetch_augmentation_status()
        self.fetch_preprocessing_status()
        
        # Hitung total gambar dari data preprocessing
        total_images = self.preprocessing_data.get('stats', {}).get('total_images', 0)
        
        return {
            'total_images': total_images,
            'augmented': self.augmentation_data.get('total_generated', 0),
            'preprocessed': self.augmentation_data.get('total_normalized', 0),
            'validation_rate': f"{self.preprocessing_data['stats'].get('validation_rate', 0)}%"
        }
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan komponen UI untuk visualisasi
        
        Returns:
            Dict berisi komponen UI
        """
        if not self.ui_components:
            self._create_ui_components()
        return self.ui_components
    
    def _create_ui_components(self):
        """Buat komponen UI untuk visualisasi"""
        # Buat tab untuk berbagai jenis visualisasi
        self.ui_components = {}
        
        # Tab untuk distribusi kelas
        class_dist_tab = widgets.Output()
        with class_dist_tab:
            display(self.visualize_class_distribution())
        
        # Tab untuk ukuran gambar
        img_size_tab = widgets.Output()
        with img_size_tab:
            display(self.visualize_image_sizes())
        
        # Tab untuk contoh augmentasi
        aug_tab = widgets.Output()
        with aug_tab:
            display(self.visualize_augmentation_examples())
        
        # Buat tab container
        self.ui_components['tabs'] = widgets.Tab()
        self.ui_components['tabs'].children = [class_dist_tab, img_size_tab, aug_tab]
        self.ui_components['tabs'].set_title(0, 'Distribusi Kelas')
        self.ui_components['tabs'].set_title(1, 'Ukuran Gambar')
        self.ui_components['tabs'].set_title(2, 'Contoh Augmentasi')
        
        # Container utama
        self.ui_components['main_container'] = widgets.VBox([
            widgets.HTML("<h2>Visualisasi Dataset</h2>"),
            self.ui_components['tabs']
        ])
        
        return self.ui_components
