"""
File: smartcash/ui/dataset/visualization/components/dataset_stats_cards.py
Deskripsi: Komponen untuk menampilkan statistik dataset dengan pendekatan minimalis
"""

import ipywidgets as widgets
from typing import Dict, Any
import math

def create_dataset_stats_cards(stats: Dict[str, Any]) -> widgets.Box:
    """
    Buat kartu statistik dataset yang menampilkan jumlah gambar per split (train, val, test).
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        Box widget berisi kartu statistik dataset
    """
    # Ambil statistik split
    split_stats = stats.get('split', {})
    
    # Jika tidak ada statistik split, gunakan data dummy
    if not split_stats:
        split_stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
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
    
    # Warna untuk setiap split
    split_colors = {
        'train': '#4285F4',  # Biru
        'val': '#FBBC05',    # Kuning
        'test': '#34A853',   # Hijau
        'total': '#673AB7'   # Ungu
    }
    
    # Ikon untuk setiap split
    split_icons = {
        'train': '🧠',
        'val': '⚖️',
        'test': '🧪',
        'total': '📊'
    }
    
    # Buat kartu untuk setiap split
    split_cards = []
    
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


def create_preprocessing_stats_cards(stats: Dict[str, Any]) -> widgets.Box:
    """
    Buat kartu statistik preprocessing yang menampilkan jumlah gambar preprocessing per split.
    
    Args:
        stats: Dictionary berisi statistik preprocessing
        
    Returns:
        Box widget berisi kartu statistik preprocessing
    """
    # Ambil statistik preprocessing dan split
    preprocessing_stats = stats.get('preprocessing', {})
    split_stats = stats.get('split', {})
    
    # Jika tidak ada statistik, gunakan data dummy
    if not preprocessing_stats:
        preprocessing_stats = {
            'train_processed': 0,
            'val_processed': 0,
            'test_processed': 0,
            'total_processed': 0
        }
    
    # Jika tidak ada statistik split, gunakan data dummy
    if not split_stats:
        split_stats = {
            'train': {'images': 0},
            'val': {'images': 0},
            'test': {'images': 0}
        }
    
    # Hitung total gambar
    total_images = sum(split['images'] for split in split_stats.values())
    
    # Buat container untuk kartu preprocessing
    preprocessing_container = widgets.Box(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='stretch',
        width='100%'
    ))
    
    # Warna untuk setiap split
    split_colors = {
        'train': '#4285F4',  # Biru
        'val': '#FBBC05',    # Kuning
        'test': '#34A853',   # Hijau
        'total': '#673AB7'   # Ungu
    }
    
    # Ikon untuk preprocessing
    preprocessing_icon = '⚙️'
    
    # Buat kartu untuk setiap split preprocessing
    preprocessing_cards = []
    
    for split_name in ['train', 'val', 'test']:
        # Ambil jumlah gambar yang diproses untuk split ini
        processed_count = preprocessing_stats.get(f'{split_name}_processed', 0)
        
        # Hitung persentase dari total split
        split_count = split_stats.get(split_name, {}).get('images', 0)
        percentage = (processed_count / split_count * 100) if split_count > 0 else 0
        
        # Buat kartu untuk split preprocessing
        card = widgets.VBox([
            widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{split_name.capitalize()} Preprocessing</div>"),
            widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{preprocessing_icon} {processed_count}</div>"),
            widgets.HTML(f"<div style='font-size: 12px; color: {split_colors.get(split_name, '#0d47a1')}; opacity: 0.8;'>{percentage:.1f}% dari {split_name}</div>")
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            margin='5px',
            padding='10px',
            min_width='150px',
            flex='1 1 auto'
        ))
        
        # Tambahkan kelas CSS untuk styling
        card._dom_classes = (f'bg-{split_name}-preprocessing',)
        
        # Tambahkan kartu ke list
        preprocessing_cards.append(card)
    
    # Tambahkan kartu total preprocessing
    total_processed = preprocessing_stats.get('total_processed', 0)
    percentage = (total_processed / total_images * 100) if total_images > 0 else 0
    
    total_card = widgets.VBox([
        widgets.HTML("<div style='font-size: 14px; font-weight: bold; color: #673AB7;'>Total Preprocessing</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: #673AB7;'>{preprocessing_icon} {total_processed}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: #673AB7; opacity: 0.8;'>{percentage:.1f}% dari total gambar</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    # Tambahkan kelas CSS untuk styling
    total_card._dom_classes = ('bg-total-preprocessing',)
    
    # Tambahkan semua kartu ke container
    preprocessing_container.children = preprocessing_cards + [total_card]
    
    return preprocessing_container


def create_augmentation_stats_cards(stats: Dict[str, Any]) -> widgets.Box:
    """
    Buat kartu statistik augmentasi yang menampilkan jumlah gambar augmentasi per split.
    
    Args:
        stats: Dictionary berisi statistik augmentasi
        
    Returns:
        Box widget berisi kartu statistik augmentasi
    """
    # Ambil statistik augmentasi dan split
    augmentation_stats = stats.get('augmentation', {})
    split_stats = stats.get('split', {})
    
    # Jika tidak ada statistik, gunakan data dummy
    if not augmentation_stats:
        augmentation_stats = {
            'train_augmented': 0,
            'val_augmented': 0,
            'test_augmented': 0,
            'total_augmented': 0
        }
    
    # Jika tidak ada statistik split, gunakan data dummy
    if not split_stats:
        split_stats = {
            'train': {'images': 0},
            'val': {'images': 0},
            'test': {'images': 0}
        }
    
    # Hitung total gambar
    total_images = sum(split['images'] for split in split_stats.values())
    
    # Buat container untuk kartu augmentasi
    augmentation_container = widgets.Box(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='stretch',
        width='100%'
    ))
    
    # Warna untuk setiap split
    split_colors = {
        'train': '#4285F4',  # Biru
        'val': '#FBBC05',    # Kuning
        'test': '#34A853',   # Hijau
        'total': '#673AB7'   # Ungu
    }
    
    # Ikon untuk augmentasi
    augmentation_icon = '🔄'
    
    # Buat kartu untuk setiap split augmentasi
    augmentation_cards = []
    
    for split_name in ['train', 'val', 'test']:
        # Ambil jumlah gambar yang diaugmentasi untuk split ini
        augmented_count = augmentation_stats.get(f'{split_name}_augmented', 0)
        
        # Hitung persentase dari total split
        split_count = split_stats.get(split_name, {}).get('images', 0)
        percentage = (augmented_count / split_count * 100) if split_count > 0 else 0
        
        # Buat kartu untuk split augmentasi
        card = widgets.VBox([
            widgets.HTML(f"<div style='font-size: 14px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{split_name.capitalize()} Augmentasi</div>"),
            widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: {split_colors.get(split_name, '#0d47a1')};'>{augmentation_icon} {augmented_count}</div>"),
            widgets.HTML(f"<div style='font-size: 12px; color: {split_colors.get(split_name, '#0d47a1')}; opacity: 0.8;'>{percentage:.1f}% dari {split_name}</div>")
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            margin='5px',
            padding='10px',
            min_width='150px',
            flex='1 1 auto'
        ))
        
        # Tambahkan kelas CSS untuk styling
        card._dom_classes = (f'bg-{split_name}-augmentation',)
        
        # Tambahkan kartu ke list
        augmentation_cards.append(card)
    
    # Tambahkan kartu total augmentasi
    total_augmented = augmentation_stats.get('total_augmented', 0)
    percentage = (total_augmented / total_images * 100) if total_images > 0 else 0
    
    total_card = widgets.VBox([
        widgets.HTML("<div style='font-size: 14px; font-weight: bold; color: #673AB7;'>Total Augmentasi</div>"),
        widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; color: #673AB7;'>{augmentation_icon} {total_augmented}</div>"),
        widgets.HTML(f"<div style='font-size: 12px; color: #673AB7; opacity: 0.8;'>{percentage:.1f}% dari total gambar</div>")
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        margin='5px',
        padding='10px',
        min_width='150px',
        flex='1 1 auto'
    ))
    
    # Tambahkan kelas CSS untuk styling
    total_card._dom_classes = ('bg-total-augmentation',)
    
    # Tambahkan semua kartu ke container
    augmentation_container.children = augmentation_cards + [total_card]
    
    return augmentation_container
