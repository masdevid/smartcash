"""
File: smartcash/ui/dataset/visualization/components/dataset_comparator.py
Deskripsi: Komponen untuk membandingkan dua dataset secara side-by-side
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor import get_preprocessing_stats, get_dataset_metadata

logger = get_logger(__name__)


class DatasetComparator:
    """Komponen untuk membandingkan dua dataset secara side-by-side"""
    
    def __init__(self):
        """Inisialisasi komparator"""
        self.dataset1 = None
        self.dataset2 = None
        self.ui_components = {}
        
    def get_ui_components(self) -> dict:
        """Dapatkan komponen UI untuk perbandingan dataset"""
        # Dropdown untuk memilih dataset pertama
        self.dataset1_dropdown = widgets.Dropdown(
            options=list_available_datasets(),
            description='Dataset 1:',
            disabled=False
        )
        
        # Dropdown untuk memilih dataset kedua
        self.dataset2_dropdown = widgets.Dropdown(
            options=list_available_datasets(),
            description='Dataset 2:',
            disabled=False
        )
        
        # Tombol untuk membandingkan
        compare_btn = widgets.Button(
            description='Bandingkan',
            button_style='success'
        )
        compare_btn.on_click(self._on_compare_clicked)
        
        # Output area
        output_area = widgets.Output()
        
        # Layout
        ui = widgets.VBox([
            widgets.HBox([self.dataset1_dropdown, self.dataset2_dropdown]),
            compare_btn,
            output_area
        ])
        
        self.ui_components['main_container'] = ui
        self.ui_components['output'] = output_area
        
        return self.ui_components
    
    def _on_compare_clicked(self, btn) -> None:
        """Handler untuk tombol bandingkan"""
        self.dataset1 = self.dataset1_dropdown.value
        self.dataset2 = self.dataset2_dropdown.value
        
        with self.ui_components['output']:
            clear_output()
            print(f"Membandingkan {self.dataset1} vs {self.dataset2}...")
            
            # Dapatkan statistik untuk kedua dataset
            stats1 = get_preprocessing_stats(self.dataset1)
            stats2 = get_preprocessing_stats(self.dataset2)
            
            # Tampilkan perbandingan
            self._display_comparison(stats1, stats2)
    
    def _display_comparison(self, stats1: dict, stats2: dict) -> None:
        """Tampilkan visualisasi perbandingan"""
        # Tampilkan statistik dasar
        self._display_basic_stats_comparison(stats1, stats2)
        
        # Tampilkan perbandingan distribusi kelas
        self._compare_class_distribution()
        
        # Tampilkan perbandingan ukuran gambar
        self._compare_image_sizes(stats1, stats2)
        
        # Tampilkan perbandingan ukuran objek
        self._compare_object_sizes(stats1, stats2)
        
        # Tampilkan perbandingan kerapatan anotasi
        self._compare_annotation_density(stats1, stats2)
    
    def _display_basic_stats_comparison(self, stats1: dict, stats2: dict) -> None:
        """Tampilkan perbandingan statistik dasar"""
        # Buat tabel perbandingan
        table = "<table style='width:100%; border-collapse: collapse; margin-bottom: 20px;'>"
        table += """
        <tr>
            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Statistik</th>
            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{}</th>
            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{}</th>
        </tr>
        """.format(self.dataset1, self.dataset2)
        
        # Definisikan statistik yang akan ditampilkan
        stats_to_show = [
            ('Jumlah Gambar', 'total_images'),
            ('Jumlah Objek', 'total_objects'),
            ('Jumlah Kelas', 'num_classes'),
            ('Ukuran Gambar Rata-rata', 'avg_image_size'),
            ('Ukuran Objek Rata-rata', 'avg_object_size'),
            ('Kerapatan Anotasi Rata-rata', 'avg_annotation_density')
        ]
        
        for stat_name, stat_key in stats_to_show:
            val1 = stats1.get(stat_key, 'N/A')
            val2 = stats2.get(stat_key, 'N/A')
            
            # Format nilai jika berupa tuple (width, height)
            if isinstance(val1, tuple) and len(val1) == 2:
                val1 = f"{val1[0]}x{val1[1]}"
            if isinstance(val2, tuple) and len(val2) == 2:
                val2 = f"{val2[0]}x{val2[1]}"
            
            table += f"""
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>{stat_name}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{val1}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{val2}</td>
            </tr>
            """
        
        table += "</table>"
        
        # Tampilkan tabel
        display(widgets.HTML(table))
    
    def _compare_class_distribution(self) -> None:
        """Bandingkan distribusi kelas antara dua dataset"""
        # Muat metadata dataset
        meta1 = get_dataset_metadata(self.dataset1)
        meta2 = get_dataset_metadata(self.dataset2)
        
        # Siapkan data untuk plot
        labels = sorted(set(meta1['classes'].keys()) | set(meta2['classes'].keys()))
        counts1 = [meta1['classes'].get(cls, 0) for cls in labels]
        counts2 = [meta2['classes'].get(cls, 0) for cls in labels]
        
        # Buat figure
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=(f'Distribusi {self.dataset1}', f'Distribusi {self.dataset2}'))
        
        # Plot dataset 1
        fig.add_trace(
            go.Bar(x=labels, y=counts1, name=self.dataset1, marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # Plot dataset 2
        fig.add_trace(
            go.Bar(x=labels, y=counts2, name=self.dataset2, marker_color='#ff7f0e'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Perbandingan Distribusi Kelas: {self.dataset1} vs {self.dataset2}',
            height=500,
            showlegend=False
        )
        
        display(fig)
    
    def _compare_image_sizes(self, stats1: dict, stats2: dict) -> None:
        """Bandingkan ukuran gambar antara dua dataset"""
        # Tampilkan perbandingan ukuran gambar
        widths1 = [size[0] for size in stats1.get('image_sizes', [])]
        heights1 = [size[1] for size in stats1.get('image_sizes', [])]
        widths2 = [size[0] for size in stats2.get('image_sizes', [])]
        heights2 = [size[1] for size in stats2.get('image_sizes', [])]
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Distribusi Ukuran', 'Perbandingan Aspect Ratio'),
                           specs=[[{'type': 'xy'}, {'type': 'xy'}]])
        
        fig.add_trace(
            go.Scatter(x=widths1, y=heights1, mode='markers', 
                      name=self.dataset1, marker=dict(color='#1f77b4', opacity=0.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=widths2, y=heights2, mode='markers', 
                      name=self.dataset2, marker=dict(color='#ff7f0e', opacity=0.5)),
            row=1, col=1
        )
        
        ar1 = [w/h for w, h in zip(widths1, heights1)]
        ar2 = [w/h for w, h in zip(widths2, heights2)]
        
        fig.add_trace(
            go.Histogram(x=ar1, name=self.dataset1, marker_color='#1f77b4', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=ar2, name=self.dataset2, marker_color='#ff7f0e', opacity=0.7),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Perbandingan Ukuran Gambar: {self.dataset1} vs {self.dataset2}',
            height=500,
            barmode='overlay'
        )
        fig.update_xaxes(title_text='Lebar', row=1, col=1)
        fig.update_yaxes(title_text='Tinggi', row=1, col=1)
        fig.update_xaxes(title_text='Aspect Ratio', row=1, col=2)
        fig.update_yaxes(title_text='Jumlah Gambar', row=1, col=2)
        
        display(fig)
    
    def _compare_object_sizes(self, stats1: dict, stats2: dict) -> None:
        """Bandingkan ukuran objek antara dua dataset"""
        # Tampilkan perbandingan ukuran objek
        widths1 = [size[0] for size in stats1.get('object_sizes', [])]
        heights1 = [size[1] for size in stats1.get('object_sizes', [])]
        widths2 = [size[0] for size in stats2.get('object_sizes', [])]
        heights2 = [size[1] for size in stats2.get('object_sizes', [])]
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Distribusi Ukuran Objek', 'Perbandingan Area Objek'),
                           specs=[[{'type': 'xy'}, {'type': 'xy'}]])
        
        fig.add_trace(
            go.Scatter(x=widths1, y=heights1, mode='markers', 
                      name=self.dataset1, marker=dict(color='#1f77b4', opacity=0.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=widths2, y=heights2, mode='markers', 
                      name=self.dataset2, marker=dict(color='#ff7f0e', opacity=0.5)),
            row=1, col=1
        )
        
        areas1 = [w*h for w, h in zip(widths1, heights1)]
        areas2 = [w*h for w, h in zip(widths2, heights2)]
        
        fig.add_trace(
            go.Histogram(x=areas1, name=self.dataset1, marker_color='#1f77b4', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=areas2, name=self.dataset2, marker_color='#ff7f0e', opacity=0.7),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Perbandingan Ukuran Objek: {self.dataset1} vs {self.dataset2}',
            height=500,
            barmode='overlay'
        )
        fig.update_xaxes(title_text='Lebar Objek', row=1, col=1)
        fig.update_yaxes(title_text='Tinggi Objek', row=1, col=1)
        fig.update_xaxes(title_text='Area Objek', row=1, col=2)
        fig.update_yaxes(title_text='Jumlah Objek', row=1, col=2)
        
        display(fig)
    
    def _compare_annotation_density(self, stats1: dict, stats2: dict) -> None:
        """Bandingkan kerapatan anotasi antara dua dataset"""
        # Tampilkan perbandingan kerapatan anotasi
        stats1 = stats1.get('annotation_density', [])
        stats2 = stats2.get('annotation_density', [])
        
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(
            go.Histogram(x=stats1, name=self.dataset1, marker_color='#1f77b4', opacity=0.7)
        )
        fig.add_trace(
            go.Histogram(x=stats2, name=self.dataset2, marker_color='#ff7f0e', opacity=0.7)
        )
        
        fig.update_layout(
            title=f'Perbandingan Kerapatan Anotasi: {self.dataset1} vs {self.dataset2}',
            height=400,
            barmode='overlay',
            xaxis_title='Kerapatan Anotasi (objek per gambar)',
            yaxis_title='Jumlah Gambar'
        )
        
        display(fig)
