"""
File: smartcash/ui/dataset/visualization/components/split_stats_cards.py
Deskripsi: Komponen untuk menampilkan statistik split dataset (train, test, val)
"""

import ipywidgets as widgets
from typing import Dict, Any
import math

def create_split_stats_cards(stats: Dict[str, Any]) -> widgets.Box:
    """
    Buat kartu statistik untuk split dataset (train, test, val).
    
    Args:
        stats: Dictionary berisi statistik split dataset
        
    Returns:
        Box widget berisi kartu statistik split
    """
    # Ambil statistik split
    split_stats = stats.get('split', {})
    
    # Jika tidak ada statistik split, gunakan data dummy
    if not split_stats:
        split_stats = {
            'train': {'images': 1400, 'labels': 1400},
            'val': {'images': 300, 'labels': 300},
            'test': {'images': 300, 'labels': 300}
        }
    
    # Hitung total gambar
    total_images = sum(split['images'] for split in split_stats.values())
    
    # Buat container untuk kartu split
    split_container = widgets.Box(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='stretch',
        width='100%'
    ))
    
    # Buat kartu untuk setiap split
    split_cards = []
    
    # Warna untuk setiap split
    split_colors = {
        'train': '#4285F4',  # Biru
        'val': '#FBBC05',    # Kuning
        'test': '#34A853'    # Hijau
    }
    
    # Ikon untuk setiap split
    split_icons = {
        'train': '🧠',
        'val': '⚖️',
        'test': '🧪'
    }
    
    for split_name, split_data in split_stats.items():
        # Hitung persentase dari total
        percentage = (split_data['images'] / total_images * 100) if total_images > 0 else 0
        
        # Buat kartu untuk split
        card = widgets.VBox([
            widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{split_name.capitalize()} Split</div>"),
            widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{split_icons.get(split_name, '📊')} {split_data['images']}</div>"),
            widgets.HTML(f"<div style='font-size: 12px; color: {split_colors.get(split_name, '#0d47a1')}; opacity: 0.8;'>{percentage:.1f}% dari total gambar</div>")
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            margin='5px',
            padding='10px',
            min_width='150px',
            flex='1 1 auto'
        ))
        
        # Tambahkan kelas CSS untuk styling
        card._dom_classes = (f'bg-{split_name}',)
        
        # Tambahkan kartu ke list
        split_cards.append(card)
    
    # Tambahkan kartu total
    total_card = widgets.VBox([
        widgets.HTML("<div style='font-size: 14px; font-weight: bold; color: #673AB7;'>Total Gambar</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: #673AB7;'>📊 {total_images}</div>"),
        widgets.HTML("<div style='font-size: 12px; color: #673AB7; opacity: 0.8;'>Jumlah seluruh gambar dataset</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    # Tambahkan kelas CSS untuk styling
    total_card._dom_classes = ('bg-total',)
    
    # Tambahkan semua kartu ke container
    split_container.children = split_cards + [total_card]
    
    return split_container
