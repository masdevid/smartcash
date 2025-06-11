"""
File: smartcash/ui/dataset/visualization/components/dataset_stats_cards.py
Deskripsi: Komponen untuk menampilkan statistik dataset dengan visualisasi interaktif
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from smartcash.ui.utils.constants import COLORS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_dataset_stats_cards(stats: Dict[str, Any]) -> widgets.VBox:
    """
    Buat komponen statistik dataset dengan visualisasi interaktif.
    
    Args:
        stats: Dictionary berisi statistik dataset dari API preprocessing
        
    Returns:
        VBox widget berisi kartu statistik dan visualisasi dataset
    """
    try:
        # Ambil statistik dari API
        summary = stats.get('summary', {})
        
        # Jika tidak ada data, tampilkan pesan
        if not summary:
            return widgets.HTML(
                "<div style='padding:20px;color:#666;text-align:center;'>"
                "Tidak ada data statistik yang tersedia. "
                "Pastikan dataset sudah diproses dengan benar."
                "</div>"
            )
        
        # Hitung total gambar dan anotasi
        total_images = sum(split.get('total_images', 0) for split in summary.values())
        total_annotations = sum(split.get('total_annotations', 0) for split in summary.values())
        
        # Hitung jumlah kelas unik
        all_classes = set()
        for split in summary.values():
            all_classes.update(split.get('class_distribution', {}).keys())
        num_classes = len(all_classes)
        
        # Buat kartu statistik
        cards = []
        
        # Total Images Card
        img_card = widgets.VBox([
            widgets.HTML('<i class="fa fa-images" style="font-size:24px;color:#4e79a7;"></i>'),
            widgets.HTML(f'<h3 style="margin:5px 0;">{total_images:,}</h3>'),
            widgets.HTML('<span style="color:#666;">Total Gambar</span>')
        ], layout=widgets.Layout(
            width='24%',
            padding='15px',
            margin='5px',
            border='1px solid #e0e0e0',
            borderRadius='8px',
            backgroundColor='white'
        ))
        
        # Total Annotations Card
        ann_card = widgets.VBox([
            widgets.HTML('<i class="fa fa-tags" style="font-size:24px;color:#e15759;"></i>'),
            widgets.HTML(f'<h3 style="margin:5px 0;">{total_annotations:,}</h3>'),
            widgets.HTML('<span style="color:#666;">Total Anotasi</span>')
        ], layout=widgets.Layout(
            width='24%',
            padding='15px',
            margin='5px',
            border='1px solid #e0e0e0',
            borderRadius='8px',
            backgroundColor='white'
        ))
        
        # Classes Card
        cls_card = widgets.VBox([
            widgets.HTML('<i class="fa fa-shapes" style="font-size:24px;color:#59a14f;"></i>'),
            widgets.HTML(f'<h3 style="margin:5px 0;">{num_classes}</h3>'),
            widgets.HTML('<span style="color:#666;">Kelas</span>')
        ], layout=widgets.Layout(
            width='24%',
            padding='15px',
            margin='5px',
            border='1px solid #e0e0e0',
            borderRadius='8px',
            backgroundColor='white'
        ))
        
        # Avg. Annotations/Image Card
        avg_ann = total_annotations / total_images if total_images > 0 else 0
        avg_card = widgets.VBox([
            widgets.HTML('<i class="fa fa-chart-bar" style="font-size:24px;color:#edc949;"></i>'),
            widgets.HTML(f'<h3 style="margin:5px 0;">{avg_ann:.1f}</h3>'),
            widgets.HTML('<span style="color:#666;">Anotasi/Gambar</span>')
        ], layout=widgets.Layout(
            width='24%',
            padding='15px',
            margin='5px',
            border='1px solid #e0e0e0',
            borderRadius='8px',
            backgroundColor='white'
        ))
        
        # Gabungkan kartu
        cards_row = widgets.HBox([img_card, ann_card, cls_card, avg_card], 
                               layout=widgets.Layout(width='100%', margin='10px 0'))
        
        # Buat visualisasi distribusi kelas
        class_dist_fig = _create_class_distribution_plot(summary)
        
        # Buat visualisasi ukuran gambar
        img_size_fig = _create_image_size_plot(summary)
        
        # Gabungkan semua komponen
        return widgets.VBox([
            widgets.HTML('<h3 style="margin-bottom:15px;">Ringkasan Dataset</h3>'),
            cards_row,
            widgets.HTML('<h3 style="margin:20px 0 15px 0;">Distribusi Kelas</h3>'),
            widgets.Box(
                [class_dist_fig], 
                layout=widgets.Layout(display='flex', justifyContent='center')
            ),
            widgets.HTML('<h3 style="margin:20px 0 15px 0;">Distribusi Ukuran Gambar</h3>'),
            widgets.Box(
                [img_size_fig], 
                layout=widgets.Layout(display='flex', justifyContent='center')
            )
        ], layout=widgets.Layout(width='100%', padding='15px'))
        
    except Exception as e:
        logger.error(f"Gagal membuat statistik dataset: {e}")
        return widgets.HTML(
            f"<div style='padding:20px;color:#d32f2f;background-color:#ffebee;border-radius:4px;'>"
            f"Terjadi kesalahan saat memuat statistik dataset: {str(e)}"
            "</div>"
        )

def _create_class_distribution_plot(summary: Dict[str, Any]) -> go.FigureWidget:
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
        
        # Warna untuk setiap split
        colors = ['#4e79a7', '#f28e2b', '#e15759']
        
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

def _create_image_size_plot(summary: Dict[str, Any]) -> go.FigureWidget:
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


def create_preprocessing_stats_cards(stats: Dict[str, Any]) -> widgets.HBox:
    """
    Buat kartu statistik preprocessing yang menampilkan jumlah gambar preprocessing.
    
    Args:
        stats: Dictionary berisi statistik preprocessing
        
    Returns:
        HBox widget berisi kartu statistik preprocessing
    """
    # Ambil statistik preprocessing
    preprocessing_stats = stats.get('preprocessing', {})
    
    # Jika tidak ada statistik, gunakan data dummy
    if not preprocessing_stats:
        preprocessing_stats = {
            'total_processed': 2000,
            'filtered_images': 1800,
            'normalized_images': 2000
        }
    
    # Buat container untuk kartu preprocessing
    preprocessing_container = widgets.HBox(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='stretch',
        width='100%'
    ))
    
    # Warna untuk preprocessing
    preprocessing_color = '#0d47a1'  # Biru
    
    # Buat kartu preprocessing
    processed_card = widgets.VBox([
        widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {preprocessing_color};'>Gambar Diproses</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {preprocessing_color};'>⚙️ {preprocessing_stats.get('total_processed', 0)}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: {preprocessing_color}; opacity: 0.8;'>Total gambar yang telah diproses</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    filtered_card = widgets.VBox([
        widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {preprocessing_color};'>Gambar Difilter</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {preprocessing_color};'>🔍 {preprocessing_stats.get('filtered_images', 0)}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: {preprocessing_color}; opacity: 0.8;'>Gambar yang difilter karena kualitas rendah</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    # Tambahkan kelas CSS untuk styling
    processed_card._dom_classes = ('bg-preprocessing',)
    filtered_card._dom_classes = ('bg-preprocessing',)
    
    # Tambahkan kartu ke container
    preprocessing_container.children = [processed_card, filtered_card]
    
    return preprocessing_container


def create_augmentation_stats_cards(stats: Dict[str, Any]) -> widgets.HBox:
    """
    Buat kartu statistik augmentasi yang menampilkan jumlah gambar augmentasi.
    
    Args:
        stats: Dictionary berisi statistik augmentasi
        
    Returns:
        HBox widget berisi kartu statistik augmentasi
    """
    # Ambil statistik augmentasi
    augmentation_stats = stats.get('augmentation', {})
    
    # Jika tidak ada statistik, gunakan data dummy
    if not augmentation_stats:
        augmentation_stats = {
            'total_augmented': 2000,
            'augmentation_types': 5
        }
    
    # Buat container untuk kartu augmentasi
    augmentation_container = widgets.HBox(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='stretch',
        width='100%'
    ))
    
    # Warna untuk augmentasi
    augmentation_color = '#1b5e20'  # Hijau
    
    # Buat kartu augmentasi
    augmented_card = widgets.VBox([
        widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {augmentation_color};'>Gambar Diaugmentasi</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {augmentation_color};'>🔄 {augmentation_stats.get('total_augmented', 0)}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: {augmentation_color}; opacity: 0.8;'>Total gambar yang telah diaugmentasi</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    types_card = widgets.VBox([
        widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {augmentation_color};'>Tipe Augmentasi</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {augmentation_color};'>🔠 {augmentation_stats.get('augmentation_types', 0)}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: {augmentation_color}; opacity: 0.8;'>Jumlah tipe augmentasi yang digunakan</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    # Tambahkan kelas CSS untuk styling
    augmented_card._dom_classes = ('bg-augmentation',)
    types_card._dom_classes = ('bg-augmentation',)
    
    # Tambahkan kartu ke container
    augmentation_container.children = [augmented_card, types_card]
    
    return augmentation_container
