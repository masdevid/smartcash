"""
File: smartcash/ui/dataset/augmentation/visualization/components/visualization_components.py
Deskripsi: Komponen UI untuk visualisasi augmentasi dataset
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import COLORS, ICONS


class AugmentationVisualizationComponents:
    """Komponen UI untuk visualisasi augmentasi dataset"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi komponen UI visualisasi augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("augmentation_visualization_ui")
        
        # Dapatkan konfigurasi visualisasi
        self.vis_config = self.config.get('visualization', {})
        
        # Inisialisasi komponen UI
        self._init_components()
        
    def _init_components(self):
        """Inisialisasi komponen UI."""
        # Definisikan ikon jika tidak ada dalam ICONS
        visualization_icon = ICONS.get('visualization', '🔍')
        compare_icon = ICONS.get('compare', '⚖️')
        
        # Tab untuk visualisasi
        self.visualization_tabs = widgets.Tab()
        self.visualization_tabs.set_title(0, f"{visualization_icon} Sampel Augmentasi")
        self.visualization_tabs.set_title(1, f"{compare_icon} Perbandingan")
        
        # Output untuk visualisasi
        self.sample_output = widgets.Output()
        self.compare_output = widgets.Output()
        
        # Status output
        self.status_output = widgets.Output()
        
        # Tombol visualisasi sampel
        self.visualize_samples_button = widgets.Button(
            description='Visualisasi Sampel',
            icon='eye',
            button_style='info',
            tooltip='Visualisasikan sampel hasil augmentasi',
            layout=widgets.Layout(width='auto')
        )
        
        # Tombol visualisasi variasi
        self.visualize_variations_button = widgets.Button(
            description='Visualisasi Variasi',
            icon='random',
            button_style='info',
            tooltip='Visualisasikan variasi augmentasi untuk satu gambar',
            layout=widgets.Layout(width='auto')
        )
        
        # Tombol visualisasi perbandingan
        self.visualize_compare_button = widgets.Button(
            description='Bandingkan dengan Preprocess',
            icon='exchange-alt',
            button_style='info',
            tooltip='Visualisasikan perbandingan preprocess vs augmentasi',
            layout=widgets.Layout(width='auto')
        )
        
        # Tombol visualisasi dampak
        self.visualize_impact_button = widgets.Button(
            description='Dampak Augmentasi',
            icon='chart-line',
            button_style='info',
            tooltip='Visualisasikan dampak berbagai jenis augmentasi',
            layout=widgets.Layout(width='auto')
        )
        
        # Dropdown jenis augmentasi
        self.aug_type_dropdown = widgets.Dropdown(
            options=[
                ('Combined: Kombinasi posisi dan pencahayaan', 'combined'),
                ('Position: Variasi posisi', 'position'),
                ('Lighting: Variasi pencahayaan', 'lighting')
            ],
            value='combined',
            description='Jenis Augmentasi:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Dropdown split dataset
        self.split_dropdown = widgets.Dropdown(
            options=['train', 'valid', 'test'],
            value='train',
            description='Split Dataset:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Slider jumlah sampel
        self.sample_count_slider = widgets.IntSlider(
            value=self.vis_config.get('sample_count', 5),
            min=1,
            max=10,
            step=1,
            description='Jumlah Sampel:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='auto')
        )
        
        # Checkbox tampilkan bbox
        self.show_bbox_checkbox = widgets.Checkbox(
            value=self.vis_config.get('show_bboxes', True),
            description='Tampilkan Bounding Box',
            indent=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Input direktori dataset
        self.data_dir_text = widgets.Text(
            value='data',
            placeholder='Direktori dataset',
            description='Dataset Dir:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Input direktori preprocessed
        self.preprocessed_dir_text = widgets.Text(
            value='data/preprocessed',
            placeholder='Direktori preprocessed',
            description='Preprocessed Dir:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        
    def create_sample_tab(self) -> widgets.VBox:
        """
        Buat tab untuk visualisasi sampel.
        
        Returns:
            Widget VBox berisi komponen visualisasi sampel
        """
        # Layout kontrol
        controls = widgets.VBox([
            widgets.HBox([
                self.aug_type_dropdown,
                self.split_dropdown
            ], layout=widgets.Layout(justify_content='space-between')),
            widgets.HBox([
                self.sample_count_slider,
                self.show_bbox_checkbox
            ], layout=widgets.Layout(justify_content='space-between')),
            widgets.HBox([
                self.data_dir_text
            ], layout=widgets.Layout(justify_content='flex-start')),
            widgets.HBox([
                self.visualize_samples_button,
                self.visualize_variations_button
            ], layout=widgets.Layout(justify_content='space-between'))
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
        
        # Layout tab
        tab = widgets.VBox([
            controls,
            self.sample_output
        ], layout=widgets.Layout(width='100%'))
        
        return tab
        
    def create_compare_tab(self) -> widgets.VBox:
        """
        Buat tab untuk visualisasi perbandingan.
        
        Returns:
            Widget VBox berisi komponen visualisasi perbandingan
        """
        # Layout kontrol
        controls = widgets.VBox([
            widgets.HBox([
                self.aug_type_dropdown,
                self.split_dropdown
            ], layout=widgets.Layout(justify_content='space-between')),
            widgets.HBox([
                self.sample_count_slider,
                self.show_bbox_checkbox
            ], layout=widgets.Layout(justify_content='space-between')),
            widgets.HBox([
                self.data_dir_text
            ], layout=widgets.Layout(justify_content='flex-start')),
            widgets.HBox([
                self.preprocessed_dir_text
            ], layout=widgets.Layout(justify_content='flex-start')),
            widgets.HBox([
                self.visualize_compare_button,
                self.visualize_impact_button
            ], layout=widgets.Layout(justify_content='space-between'))
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
        
        # Layout tab
        tab = widgets.VBox([
            controls,
            self.compare_output
        ], layout=widgets.Layout(width='100%'))
        
        return tab
        
    def create_visualization_ui(self) -> widgets.VBox:
        """
        Buat UI visualisasi augmentasi.
        
        Returns:
            Widget VBox berisi UI visualisasi
        """
        # Buat tab
        sample_tab = self.create_sample_tab()
        compare_tab = self.create_compare_tab()
        
        # Set tab children
        self.visualization_tabs.children = [sample_tab, compare_tab]
        
        # Layout utama
        main_layout = widgets.VBox([
            self.visualization_tabs,
            self.status_output
        ], layout=widgets.Layout(width='100%'))
        
        return main_layout
        
    def show_status(self, message: str, status: str = 'info') -> None:
        """
        Tampilkan pesan status.
        
        Args:
            message: Pesan yang akan ditampilkan
            status: Status pesan (info, success, warning, error)
        """
        try:
            with self.status_output:
                clear_output(wait=True)
                
                # Tentukan warna berdasarkan status
                color = COLORS.get(status, COLORS['info'])
                icon = ICONS.get(status, ICONS['info'])
                
                # Tampilkan pesan
                display(widgets.HTML(f"<div style='padding: 10px; background-color: {color}; color: white; border-radius: 5px;'>{icon} {message}</div>"))
        except Exception as e:
            # Fallback untuk pengujian
            print(f"Status: {status} - {message}")
            
    def show_figure(self, fig, output_widget: widgets.Output) -> None:
        """
        Tampilkan figure di output widget.
        
        Args:
            fig: Figure matplotlib
            output_widget: Widget output
        """
        try:
            with output_widget:
                clear_output(wait=True)
                display(fig)
        except Exception as e:
            # Fallback untuk pengujian
            print(f"Menampilkan figure di output widget")
            
    def register_handlers(self, 
                         on_visualize_samples: Callable = None,
                         on_visualize_variations: Callable = None,
                         on_visualize_compare: Callable = None,
                         on_visualize_impact: Callable = None) -> None:
        """
        Daftarkan handler untuk tombol visualisasi.
        
        Args:
            on_visualize_samples: Handler untuk tombol visualisasi sampel
            on_visualize_variations: Handler untuk tombol visualisasi variasi
            on_visualize_compare: Handler untuk tombol visualisasi perbandingan
            on_visualize_impact: Handler untuk tombol visualisasi dampak
        """
        if on_visualize_samples:
            self.visualize_samples_button.on_click(on_visualize_samples)
            
        if on_visualize_variations:
            self.visualize_variations_button.on_click(on_visualize_variations)
            
        if on_visualize_compare:
            self.visualize_compare_button.on_click(on_visualize_compare)
            
        if on_visualize_impact:
            self.visualize_impact_button.on_click(on_visualize_impact)
