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

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


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
        split_data = {}
        
        for split, data in summary.items():
            class_dist = data.get('class_distribution', {})
            split_data[split] = class_dist
            all_classes.update(class_dist.keys())
        
        # Buat DataFrame untuk heatmap
        df = pd.DataFrame(index=sorted(all_classes), columns=sorted(summary.keys()))
        
        for split, classes in split_data.items():
            for cls, count in classes.items():
                df.at[cls, split] = count
        
        df = df.fillna(0)
        
        # Buat heatmap
        fig = px.imshow(
            df.values,
            labels=dict(x="Split", y="Kelas", color="Jumlah"),
            x=df.columns,
            y=df.index,
            aspect="auto",
            color_continuous_scale='Viridis',
            title="Distribusi Kelas per Split"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Split",
            yaxis_title="Kelas",
            coloraxis_colorbar=dict(title="Jumlah"),
            height=400 + len(df.index) * 15,  # Sesuaikan tinggi berdasarkan jumlah kelas
            margin=dict(l=100, r=20, t=50, b=50)
        )
        
        # Tambahkan anotasi
        for i, row in enumerate(df.values):
            for j, value in enumerate(row):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(int(value)),
                    showarrow=False,
                    font=dict(color='white' if value > df.values.max()/2 else 'black')
                )
        
        return go.FigureWidget(fig)
        
    except Exception as e:
        logger.error(f"Gagal membuat heatmap: {e}")
        return widgets.HTML(f"<div style='color:red;'>Error: Gagal membuat heatmap: {str(e)}</div>")

def create_outlier_detection(stats: Dict[str, Any]) -> widgets.Widget:
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
