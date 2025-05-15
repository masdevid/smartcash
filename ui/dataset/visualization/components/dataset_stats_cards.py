"""
File: smartcash/ui/dataset/visualization/components/dataset_stats_cards.py
Deskripsi: Komponen untuk menampilkan statistik dataset dengan pendekatan minimalis
"""

import ipywidgets as widgets
from typing import Dict, Any
import math

def create_dataset_stats_cards(stats: Dict[str, Any]) -> widgets.HBox:
    """
    Buat kartu statistik dataset yang menampilkan jumlah gambar per split (train, val, test).
    
    Args:
        stats: Dictionary berisi statistik dataset
        
    Returns:
        HBox widget berisi kartu statistik dataset
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
    split_container = widgets.HBox(layout=widgets.Layout(
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
