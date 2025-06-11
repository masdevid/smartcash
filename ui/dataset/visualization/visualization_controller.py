"""
File: smartcash/ui/dataset/visualization/visualization_controller.py
Deskripsi: Controller utama untuk visualisasi dataset
"""

from typing import Dict, Any, Optional, List
import os
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor import get_preprocessing_stats, list_available_datasets
from smartcash.ui.dataset.visualization.components import (
    DatasetStatsComponent,
    AugmentationVisualizer
)
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

class VisualizationController:
    """Controller utama untuk visualisasi dataset"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inisialisasi controller visualisasi
        
        Args:
            config: Konfigurasi visualisasi
        """
        self.config = config or {}
        self.ui_components = {}
        self.current_dataset = None
        self.dataset_stats = {}
        
        # Inisialisasi komponen
        self.stats_component = DatasetStatsComponent(
            config=self.config.get('stats', {})
        )
        
        # Inisialisasi visualizer augmentasi (akan diinisialisasi setelah memuat dataset)
        self.aug_visualizer = None
    
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
            
            # Muat statistik preprocessing
            self.dataset_stats = get_preprocessing_stats(dataset_name)
            
            # Perbarui komponen statistik
            self.stats_component.update_stats(self.dataset_stats)
            
            # Inisialisasi visualizer augmentasi
            dataset_path = os.path.join("data", "processed", dataset_name)
            self.aug_visualizer = AugmentationVisualizer(
                dataset_path=dataset_path,
                config=self.config.get('augmentation', {})
            )
            
            logger.info(f"Dataset '{dataset_name}' berhasil dimuat")
            return True
            
        except Exception as e:
            logger.error(f"Gagal memuat dataset '{dataset_name}': {e}")
            return False
    
    def _create_dataset_selector(self) -> widgets.Widget:
        """Buat komponen pemilih dataset"""
        # Dapatkan daftar dataset yang tersedia
        available_datasets = list_available_datasets()
        
        # Buat dropdown untuk memilih dataset
        self.dataset_dropdown = widgets.Dropdown(
            options=available_datasets,
            value=available_datasets[0] if available_datasets else None,
            description='Dataset:',
            disabled=not available_datasets
        )
        
        # Tombol untuk memuat dataset
        self.load_btn = widgets.Button(
            description='Muat Dataset',
            button_style='primary',
            icon='folder-open',
            disabled=not available_datasets
        )
        self.load_btn.on_click(self._on_load_dataset)
        
        # Tampilkan status
        self.status_output = widgets.Output()
        
        # Gabungkan komponen
        return widgets.VBox([
            widgets.HBox([self.dataset_dropdown, self.load_btn]),
            self.status_output
        ])
    
    def _create_main_tabs(self) -> widgets.Widget:
        """Buat tab utama untuk visualisasi"""
        # Buat tab untuk berbagai jenis visualisasi
        self.tabs = widgets.Tab()
        
        # Tab statistik dataset
        stats_tab = widgets.VBox([
            widgets.HTML("<h3>Statistik Dataset</h3>"),
            self.stats_component.get_ui_components()['main_container']
        ])
        
        # Tab augmentasi
        aug_tab = widgets.VBox([
            widgets.HTML("<h3>Visualisasi Augmentasi</h3>"),
            self.aug_visualizer.get_ui_components()['main_container'] if self.aug_visualizer else 
            widgets.HTML("<p>Muat dataset terlebih dahulu untuk melihat visualisasi augmentasi</p>")
        ])
        
        # Atur tab
        self.tabs.children = [stats_tab, aug_tab]
        self.tabs.set_title(0, '📊 Statistik')
        self.tabs.set_title(1, '🔄 Augmentasi')
        
        return self.tabs
    
    def _create_ui(self) -> widgets.Widget:
        """Buat UI lengkap"""
        # Buat header
        header = widgets.HTML("""
        <div style='background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
            <h1 style='margin: 0;'>Visualisasi Dataset SmartCash</h1>
            <p style='margin: 5px 0 0 0; color: #666;'>
                Visualisasi dataset dan hasil preprocessing/augmentasi
            </p>
        </div>
        """)
        
        # Buat selector dataset
        dataset_selector = self._create_dataset_selector()
        
        # Buat tab utama
        main_tabs = self._create_main_tabs()
        
        # Gabungkan semua komponen
        main_container = widgets.VBox([
            header,
            dataset_selector,
            main_tabs
        ])
        
        return main_container
    
    def _on_load_dataset(self, btn) -> None:
        """Handler untuk tombol muat dataset"""
        dataset_name = self.dataset_dropdown.value
        
        with self.status_output:
            clear_output(wait=True)
            print(f"Memuat dataset '{dataset_name}'...")
            
            if self.load_dataset(dataset_name):
                # Perbarui tab utama dengan data baru
                self._update_main_tabs()
                print(f"{ICONS.get('success', '✅')} Dataset berhasil dimuat")
            else:
                print(f"{ICONS.get('error', '❌')} Gagal memuat dataset")
    
    def _update_main_tabs(self) -> None:
        """Perbarui konten tab utama"""
        if not hasattr(self, 'tabs') or self.tabs is None:
            return
            
        # Perbarui tab statistik
        stats_tab = widgets.VBox([
            widgets.HTML("<h3>Statistik Dataset</h3>"),
            self.stats_component.get_ui_components()['main_container']
        ])
        
        # Perbarui tab augmentasi
        aug_tab = widgets.VBox([
            widgets.HTML("<h3>Visualisasi Augmentasi</h3>")
        ])
        
        if self.aug_visualizer:
            aug_tab.children += (self.aug_visualizer.get_ui_components()['main_container'],)
        else:
            aug_tab.children += (widgets.HTML("<p>Gagal memuat visualizer augmentasi</p>"),)
        
        # Perbarui tab
        self.tabs.children = [stats_tab, aug_tab]
    
    def display(self) -> None:
        """Tampilkan UI visualisasi"""
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
        display(self.main_container)
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan komponen UI
        
        Returns:
            Dict berisi komponen UI
        """
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
            
        return {
            'main_container': self.main_container,
            'stats_component': self.stats_component,
            'aug_visualizer': self.aug_visualizer
        }
