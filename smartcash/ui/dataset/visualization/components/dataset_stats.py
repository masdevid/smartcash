"""
File: smartcash/ui/dataset/visualization/components/dataset_stats.py
Deskripsi: Komponen untuk menampilkan statistik dataset
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.visualization.utils import (
    create_class_distribution_plot,
    create_image_size_plot
)

logger = get_logger(__name__)


class DatasetStatsComponent:
    """Komponen untuk menampilkan statistik dataset"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inisialisasi komponen statistik dataset
        
        Args:
            config: Konfigurasi komponen
        """
        self.config = config or {}
        self.ui_components = {}
        self.stats_data = {}
        
    def update_stats(self, stats_data: Dict[str, Any]) -> None:
        """Perbarui data statistik
        
        Args:
            stats_data: Data statistik
        """
        self.stats_data = stats_data
        self._update_ui()
    
    def _create_summary_cards(self) -> widgets.Widget:
        """Buat kartu ringkasan statistik"""
        if not self.stats_data:
            return widgets.HTML("<p>Tidak ada data statistik yang tersedia</p>")
        
        try:
            # Ambil data ringkasan
            summary = self.stats_data.get('summary', {})
            
            # Buat kartu untuk setiap metrik
            cards = []
            
            # Total Gambar
            total_imgs = sum(split.get('total_images', 0) for split in summary.values())
            img_card = widgets.VBox([
                widgets.HTML('<div style="font-size: 24px; font-weight: bold; color: #1f77b4;">{:,}</div>'.format(total_imgs)),
                widgets.HTML('<div>Total Gambar</div>')
            ], layout=widgets.Layout(width='150px', margin='5px'))
            
            # Total Anotasi
            total_anns = sum(split.get('total_annotations', 0) for split in summary.values())
            ann_card = widgets.VBox([
                widgets.HTML('<div style="font-size: 24px; font-weight: bold; color: #ff7f0e;">{:,}</div>'.format(total_anns)),
                widgets.HTML('<div>Total Anotasi</div>')
            ], layout=widgets.Layout(width='150px', margin='5px'))
            
            # Rata-rata Anotasi per Gambar
            avg_anns = total_anns / max(total_imgs, 1)
            avg_card = widgets.VBox([
                widgets.HTML('<div style="font-size: 24px; font-weight: bold; color: #2ca02c;">{:.2f}</div>'.format(avg_anns)),
                widgets.HTML('<div>Rata-rata Anotasi/Gambar</div>')
            ], layout=widgets.Layout(width='180px', margin='5px'))
            
            # Total Kelas
            class_counts = {}
            for split_data in summary.values():
                for cls, count in split_data.get('class_distribution', {}).items():
                    class_counts[cls] = class_counts.get(cls, 0) + count
            
            class_card = widgets.VBox([
                widgets.HTML('<div style="font-size: 24px; font-weight: bold; color: #d62728;">{}</div>'.format(len(class_counts))),
                widgets.HTML('<div>Total Kelas</div>')
            ], layout=widgets.Layout(width='120px', margin='5px'))
            
            # Gabungkan semua kartu
            cards_row = widgets.HBox([
                img_card, ann_card, avg_card, class_card
            ], layout=widgets.Layout(justify_content='space-around'))
            
            return cards_row
            
        except Exception as e:
            logger.error(f"Gagal membuat ringkasan statistik: {e}")
            return widgets.HTML(f"<p style='color: red;'>Error: {str(e)}</p>")
    
    def _create_split_stats(self) -> widgets.Widget:
        """Buat tabel statistik per split"""
        if not self.stats_data:
            return widgets.HTML("<p>Tidak ada data statistik yang tersedia</p>")
            
        try:
            # Buat header tabel
            header = [
                widgets.HTML('<b>Split</b>'),
                widgets.HTML('<b>Gambar</b>'),
                widgets.HTML('<b>Anotasi</b>'),
                widgets.HTML('<b>Anotasi/Gambar</b>'),
                widgets.HTML('<b>Kelas</b>'),
                widgets.HTML('<b>Ukuran Rata-rata</b>')
            ]
            
            # Buat baris untuk setiap split
            rows = []
            for split, data in self.stats_data.get('summary', {}).items():
                total_imgs = data.get('total_images', 0)
                total_anns = data.get('total_annotations', 0)
                avg_anns = total_anns / max(total_imgs, 1)
                num_classes = len(data.get('class_distribution', {}))
                
                # Hitung ukuran rata-rata
                sizes = self.stats_data.get('image_sizes', {}).get(split, [])
                if sizes:
                    avg_width = sum(w for w, _ in sizes) / len(sizes)
                    avg_height = sum(h for _, h in sizes) / len(sizes)
                    avg_size = f"{int(avg_width)}x{int(avg_height)}"
                else:
                    avg_size = "N/A"
                
                row = [
                    widgets.HTML(f"<b>{split.upper()}</b>"),
                    widgets.HTML(f"{total_imgs:,}"),
                    widgets.HTML(f"{total_anns:,}"),
                    widgets.HTML(f"{avg_anns:.2f}"),
                    widgets.HTML(f"{num_classes}"),
                    widgets.HTML(avg_size)
                ]
                rows.append(row)
            
            # Buat grid layout
            grid = widgets.GridBox(
                children=[item for row in [header] + rows for item in row],
                layout=widgets.Layout(
                    grid_template_columns='repeat(6, 1fr)',
                    grid_gap='10px',
                    width='100%',
                    padding='10px'
                )
            )
            
            # Tambahkan style untuk header
            for i in range(6):
                grid.children[i].add_class('header-cell')
            
            return grid
            
        except Exception as e:
            logger.error(f"Gagal membuat tabel statistik: {e}")
            return widgets.HTML(f"<p style='color: red;'>Error: {str(e)}</p>")
    
    def _create_class_distribution_plot(self) -> widgets.Widget:
        """Buat plot distribusi kelas"""
        if not self.stats_data:
            return widgets.HTML("<p>Tidak ada data distribusi kelas yang tersedia</p>")
            
        try:
            class_data = self.stats_data.get('class_distribution', {})
            colors = self.config.get('colors', {
                'train': '#1f77b4',
                'val': '#ff7f0e',
                'test': '#2ca02c'
            })
            
            fig = create_class_distribution_plot(class_data, colors)
            
            if fig is None:
                return widgets.HTML("<p>Gagal membuat plot distribusi kelas</p>")
                
            return go.FigureWidget(fig)
            
        except Exception as e:
            logger.error(f"Gagal membuat plot distribusi kelas: {e}")
            return widgets.HTML(f"<p style='color: red;'>Error: {str(e)}</p>")
    
    def _create_image_size_plot(self) -> widgets.Widget:
        """Buat plot distribusi ukuran gambar"""
        if not self.stats_data:
            return widgets.HTML("<p>Tidak ada data ukuran gambar yang tersedia</p>")
            
        try:
            size_data = self.stats_data.get('image_sizes', {})
            colors = self.config.get('colors', {
                'train': '#1f77b4',
                'val': '#ff7f0e',
                'test': '#2ca02c'
            })
            
            fig = create_image_size_plot(size_data, colors)
            
            if fig is None:
                return widgets.HTML("<p>Gagal membuat plot ukuran gambar</p>")
                
            return go.FigureWidget(fig)
            
        except Exception as e:
            logger.error(f"Gagal membuat plot ukuran gambar: {e}")
            return widgets.HTML(f"<p style='color: red;'>Error: {str(e)}</p>")
    
    def _create_ui(self) -> widgets.Widget:
        """Buat UI komponen"""
        # Buat tab untuk berbagai visualisasi
        tab_titles = ['Ringkasan', 'Distribusi Kelas', 'Ukuran Gambar']
        tab_contents = [
            widgets.VBox([
                self._create_summary_cards(),
                widgets.HTML("<h3>Statistik per Split</h3>"),
                self._create_split_stats()
            ]),
            self._create_class_distribution_plot(),
            self._create_image_size_plot()
        ]
        
        tabs = widgets.Tab()
        tabs.children = tab_contents
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
        
        # Container utama
        main_container = widgets.VBox([
            widgets.HTML("<h2>Statistik Dataset</h2>"),
            tabs
        ], layout=widgets.Layout(width='100%'))
        
        return main_container
    
    def _update_ui(self) -> None:
        """Perbarui UI dengan data terbaru"""
        if hasattr(self, 'main_container') and self.main_container is not None:
            self.main_container.close()
        self.main_container = self._create_ui()
    
    def display(self) -> None:
        """Tampilkan komponen"""
        if not hasattr(self, 'main_container') or self.main_container is None:
            self._update_ui()
        display(self.main_container)
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan komponen UI
        
        Returns:
            Dict berisi komponen UI
        """
        if not hasattr(self, 'main_container') or self.main_container is None:
            self._update_ui()
            
        return {
            'main_container': self.main_container,
            'summary_cards': self._create_summary_cards(),
            'split_stats': self._create_split_stats(),
            'class_dist_plot': self._create_class_distribution_plot(),
            'image_size_plot': self._create_image_size_plot()
        }
