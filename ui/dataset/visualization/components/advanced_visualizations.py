"""
File: smartcash/ui/dataset/visualization/components/advanced_visualizations.py
Deskripsi: Komponen visualisasi lanjutan untuk dataset
"""

import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from IPython.display import display, clear_output
from plotly.subplots import make_subplots

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Fungsi plotting yang dipindahkan dari dataset_stats_cards.py
def create_class_distribution_plot(summary: Dict[str, Any]) -> go.FigureWidget:
    """Buat plot distribusi kelas"""
    try:
        # Siapkan data
        splits = list(summary.keys())
        class_dist = {}
        
        # Kumpulkan distribusi kelas per split
        for split in splits:
            for cls, count in summary[split].get('class_distribution', {}).items():
                if cls not in class_dist:
                    class_dist[cls] = {s: 0 for s in splits}
                class_dist[cls][split] = count
        
        # Buat figure
        fig = go.Figure()
        
        # Warna untuk setiap kelas
        colors = px.colors.qualitative.Plotly
        
        # Tambahkan trace untuk setiap kelas
        for i, (cls, dist) in enumerate(class_dist.items()):
            fig.add_trace(go.Bar(
                x=splits,
                y=[dist[split] for split in splits],
                name=cls,
                marker_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        # Update layout
        fig.update_layout(
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation='h', y=1.1),
            xaxis_title='Split',
            yaxis_title='Jumlah',
            height=400
        )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat plot distribusi kelas: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: {str(e)}</div>")

def create_image_size_plot(summary: Dict[str, Any]) -> go.FigureWidget:
    """Buat plot distribusi ukuran gambar"""
    try:
        # Siapkan data
        splits = list(summary.keys())
        
        # Buat subplot
        fig = make_subplots(rows=1, cols=len(splits), 
                          subplot_titles=[f"{split.upper()}" for split in splits],
                          shared_yaxes=True)
        
        # Warna untuk setiap split
        colors = ['#4e79a7', '#f28e2b', '#e15759']
        
        # Tambahkan scatter plot untuk setiap split
        for i, split in enumerate(splits):
            sizes = summary[split].get('image_sizes', [])
            if not sizes:
                continue
                
            widths = [w for w, h in sizes]
            heights = [h for w, h in sizes]
            
            fig.add_trace(
                go.Scatter(
                    x=widths,
                    y=heights,
                    mode='markers',
                    name=split,
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    )
                ),
                row=1, col=i+1
            )
            
            # Update sumbu
            fig.update_xaxes(title_text='Lebar (px)', row=1, col=i+1)
            if i == 0:
                fig.update_yaxes(title_text='Tinggi (px)', row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title_text='Distribusi Ukuran Gambar per Split',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat plot ukuran gambar: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: {str(e)}</div>")

def create_heatmap_visualization(stats: Dict[str, Any]) -> widgets.Widget:
    """
    Buat visualisasi heatmap untuk korelasi antar kelas
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        Widget berisi heatmap interaktif
    """
    try:
        # Ambil data distribusi kelas
        summary = stats.get('summary', {})
        if not summary:
            return widgets.HTML("<div>Tidak ada data yang tersedia untuk heatmap</div>")
        
        # Siapkan data untuk heatmap
        all_classes = set()
        class_data = {}
        
        # Kumpulkan distribusi kelas per split
        for split, data in summary.items():
            class_dist = data.get('class_distribution', {})
            for cls, count in class_dist.items():
                all_classes.add(cls)
                if cls not in class_data:
                    class_data[cls] = {}
                class_data[cls][split] = count
        
        # Buat DataFrame
        df = pd.DataFrame.from_dict(class_data, orient='index')
        
        # Hitung matriks korelasi
        corr_matrix = df.corr()
        
        # Buat heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}"
        ))
        
        # Update layout
        fig.update_layout(
            title='Korelasi Antar Kelas',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat heatmap: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: {str(e)}</div>")

def create_outlier_detection(stats: Dict[str, Any]) -> widgets.Widget:
    """
    Buat visualisasi untuk deteksi outlier dalam jumlah anotasi per gambar
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        Widget berisi box plot interaktif
    """
    try:
        # Ambil data jumlah anotasi per gambar
        summary = stats.get('summary', {})
        if not summary:
            return widgets.HTML("<div>Tidak ada data yang tersedia untuk deteksi outlier</div>")
        
        # Siapkan data
        annotations_per_image = {}
        for split, data in summary.items():
            annotations = data.get('annotations_per_image', [])
            if annotations:
                annotations_per_image[split] = annotations
        
        # Buat box plot
        fig = go.Figure()
        
        for split, annotations in annotations_per_image.items():
            fig.add_trace(go.Box(
                y=annotations,
                name=split,
                boxpoints='outliers',
                marker_color='#3D9970',
                line_color='#3D9970'
            ))
        
        # Update layout
        fig.update_layout(
            title='Distribusi Jumlah Anotasi per Gambar',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
            yaxis_title="Jumlah Anotasi",
            xaxis_title="Split Dataset"
        )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat visualisasi outlier: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: {str(e)}</div>")

def create_annotation_distribution(stats: Dict[str, Any]) -> widgets.Widget:
    """
    Buat histogram distribusi jumlah anotasi per gambar
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        Widget berisi histogram interaktif
    """
    try:
        # Ambil data jumlah anotasi per gambar
        summary = stats.get('summary', {})
        if not summary:
            return widgets.HTML("<div>Tidak ada data yang tersedia untuk distribusi anotasi</div>")
        
        # Siapkan data
        all_annotations = []
        for split, data in summary.items():
            annotations = data.get('annotations_per_image', [])
            all_annotations.extend(annotations)
        
        # Buat histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=all_annotations,
            nbinsx=50,
            marker_color='#FF851B',
            opacity=0.7
        ))
        
        # Update layout
        fig.update_layout(
            title='Distribusi Jumlah Anotasi per Gambar',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
            xaxis_title="Jumlah Anotasi",
            yaxis_title="Frekuensi"
        )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat distribusi anotasi: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: {str(e)}</div>")

def create_outlier_detection_old(stats: Dict[str, Any]) -> widgets.Widget:
    """
    Buat visualisasi untuk mendeteksi outlier dalam dataset
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        Widget berisi visualisasi outlier
    """
    try:
        # Ambil data ukuran gambar
        summary = stats.get('summary', {})
        if not summary:
            return widgets.HTML("<div>Tidak ada data yang tersedia untuk deteksi outlier</div>")
        
        # Siapkan data
        data = []
        for split, split_data in summary.items():
            sizes = split_data.get('image_sizes', [])
            for w, h in sizes:
                data.append({
                    'split': split,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w/h if h > 0 else 0,
                    'area': w * h
                })
        
        if not data:
            return widgets.HTML("<div>Tidak ada data ukuran gambar yang tersedia</div>")
        
        df = pd.DataFrame(data)
        
        # Buat scatter plot untuk outlier detection
        fig = px.scatter(
            df, 
            x='width', 
            y='height',
            color='split',
            title='Deteksi Outlier Ukuran Gambar',
            labels={'width': 'Lebar (px)', 'height': 'Tinggi (px)'},
            hover_data=['aspect_ratio', 'area']
        )
        
        # Hitung IQR untuk outlier detection
        q1 = df['area'].quantile(0.25)
        q3 = df['area'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Tandai outlier
        outliers = df[(df['area'] < lower_bound) | (df['area'] > upper_bound)]
        
        if not outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=outliers['width'],
                    y=outliers['height'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        line=dict(color='black', width=2)
                    ),
                    name='Outlier',
                    text='Outlier',
                    hoverinfo='text+x+y'
                )
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Tambahkan informasi outlier
        total_images = len(df)
        num_outliers = len(outliers)
        outlier_percentage = (num_outliers / total_images * 100) if total_images > 0 else 0
        
        outlier_info = f"""
        <div style='margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
            <h4>Informasi Outlier</h4>
            <p>Jumlah total gambar: {total_images:,}</p>
            <p>Jumlah outlier terdeteksi: {num_outliers:,}</p>
            <p>Persentase outlier: {outlier_percentage:.2f}%</p>
        </div>
        """
        
        return widgets.VBox([
            go.FigureWidget(fig),
            widgets.HTML(outlier_info)
        ])
        
    except Exception as e:
        logger.error(f"Gagal membuat visualisasi outlier: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: Gagal membuat visualisasi outlier: {str(e)}</div>")

def create_sample_preview(stats: Dict[str, Any], num_samples: int = 5) -> widgets.Widget:
    """
    Buat preview sampel gambar dari dataset
    
    Args:
        stats: Dictionary berisi statistik dataset
        num_samples: Jumlah sampel yang akan ditampilkan per split
        
    Returns:
        Widget berisi preview gambar
    """
    try:
        # Ambil data sampel gambar
        summary = stats.get('summary', {})
        if not summary:
            return widgets.HTML("<div>Tidak ada data sampel yang tersedia</div>")
        
        # Buat tab untuk setiap split
        tabs = []
        
        for split, split_data in summary.items():
            samples = split_data.get('sample_images', [])
            if not samples:
                continue
                
            # Ambil beberapa sampel acak
            samples = samples[:num_samples]
            
            # Buat grid untuk menampilkan gambar
            grid = widgets.GridBox(
                children=[],
                layout=widgets.Layout(
                    width='100%',
                    grid_template_columns='repeat(auto-fill, minmax(150px, 1fr))',
                    grid_gap='10px',
                    padding='10px'
                )
            )
            
            for img_path in samples:
                try:
                    # Buat widget gambar
                    img_widget = widgets.Image(
                        value=open(img_path, 'rb').read() if isinstance(img_path, str) else img_path,
                        format='jpg',
                        width=150,
                        height=150,
                        layout=widgets.Layout(
                            object_fit='contain',
                            border='1px solid #ddd',
                            padding='5px',
                            background='white'
                        )
                    )
                    
                    # Tambahkan ke grid
                    grid.children += (img_widget,)
                except Exception as e:
                    logger.warning(f"Gagal memuat gambar {img_path}: {e}")
            
            # Tambahkan tab
            tabs.append(widgets.Box(
                [grid],
                layout=widgets.Layout(
                    display='flex',
                    flex_flow='column',
                    align_items='center',
                    padding='10px'
                )
            ))
        
        if not tabs:
            return widgets.HTML("<div>Tidak ada gambar sampel yang tersedia</div>")
        
        # Buat tab panel
        tab_titles = [f"{split.upper()} ({len(tab.children[0].children)} gambar)" for split, tab in zip(summary.keys(), tabs)]
        tab = widgets.Tab(children=tabs)
        for i, title in enumerate(tab_titles):
            tab.set_title(i, title)
        
        return tab
        
    except Exception as e:
        logger.error(f"Gagal membuat preview sampel: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: Gagal membuat preview sampel: {str(e)}</div>")
