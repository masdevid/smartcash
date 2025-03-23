"""
File: smartcash/ui/dataset/augmentation_visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset hasil augmentasi dengan perbaikan algoritma pencocokan gambar
"""

import os
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
import re
from typing import Dict, Any, Optional, List, Tuple, Set
from smartcash.ui.utils.file_utils import shorten_filename
from smartcash.ui.utils.constants import COLORS, ICONS

def setup_visualization_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk visualisasi dataset augmentasi dengan dukungan lokasi baru."""
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi dengan fungsi konsolidasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='single')

    # Handler untuk tombol komparasi dengan fungsi konsolidasi
    def on_compare_click(b):
        """Handler untuk komparasi sampel dataset asli dengan yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='compare')
    
    # Setup handlers untuk tombol visualisasi
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'children') and len(visualization_buttons.children) >= 2:
        visualization_buttons.children[0].on_click(on_visualize_click)
        visualization_buttons.children[1].on_click(on_compare_click)
    
    # Tambahkan handlers ke komponen UI
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'visualize_dataset': visualize_dataset
    })
    
    return ui_components

def visualize_dataset(ui_components: Dict[str, Any], mode: str = 'single', num_samples: int = 5):
    """
    Fungsi visualisasi dataset yang mendukung lokasi baru augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        mode: Mode visualisasi ('single' untuk augmented saja, 'compare' untuk komparasi)
        num_samples: Jumlah sampel yang akan ditampilkan
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
    # Use the visualization container if available
    output_widget = ui_components.get('visualization_container', ui_components.get('status'))
    if not output_widget: return
    
    # Dapatkan lokasi dari konfigurasi
    config = ui_components.get('config', {})
    
    # Dapatkan lokasi preprocessed (untuk mencari hasil augmentasi)
    preprocessed_dir = config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
    
    # Dapatkan lokasi temp augmented (untuk backup)
    augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
    
    # Siapkan lokasi yang akan dicari (prioritaskan preprocessed/train)
    primary_path = Path(preprocessed_dir) / 'train' / 'images'
    secondary_path = Path(augmented_dir) / 'images'
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan {'komparasi' if mode == 'compare' else 'visualisasi'}..."))
        
        # Periksa lokasi mana yang memiliki file hasil augmentasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Cek apakah ada file augmentasi di primary path
        primary_aug_files = list(primary_path.glob(f"{aug_prefix}_*.jpg")) if primary_path.exists() else []
        
        # Jika tidak ada di primary path, cek di secondary path
        active_path = primary_path
        if not primary_aug_files and secondary_path.exists():
            secondary_aug_files = list(secondary_path.glob(f"{aug_prefix}_*.jpg"))
            if secondary_aug_files:
                active_path = secondary_path
                display(create_info_alert(f"Menggunakan lokasi augmentasi temporary: {active_path}", "info"))
        
        # Jika mode compare, dapatkan lokasi data asli
        if mode == 'compare':
            # Cek apakah ada file original di lokasi preprocessed
            original_path = Path(preprocessed_dir) / 'train' / 'images'
            if not original_path.exists():
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset untuk komparasi tidak ditemukan"))
                return
            from smartcash.ui.visualization.compare_original_vs_augmented import compare_original_vs_augmented
            # Gunakan visualisasi komparasi
            compare_original_vs_augmented(original_path, active_path, output_widget, ui_components, num_samples)
        else:
            # Visualisasi sampel augmented
            from smartcash.ui.visualization.visualize_augmented_samples import visualize_augmented_samples
            visualize_augmented_samples(active_path, output_widget, ui_components, num_samples)
        
        # Show visualization container
        if 'visualization_container' in ui_components:
            ui_components['visualization_container'].layout.display = 'block'


def get_original_from_augmented(aug_filename: str, orig_prefix: str = "rp") -> Optional[str]:
    """
    Ekstrak nama file original dari nama file augmentasi.
    Format nama file augmentasi: [augmented_prefix]_[source_prefix]_[class_name]_[uuid]_var[n]
    
    Args:
        aug_filename: Nama file augmentasi
        orig_prefix: Prefix untuk file original
        
    Returns:
        Nama file original yang sesuai atau None jika tidak ditemukan
    """
    # Pattern untuk mendeteksi bagian unik dari nama file
    # Format: aug_rp_class_uuid_var1.jpg -> rp_class_uuid.jpg
    pattern = r'(?:[^_]+)_(' + re.escape(orig_prefix) + r'_[^_]+_[^_]+)_var\d+'
    match = re.search(pattern, aug_filename)
    
    if match:
        # Ekstrak bagian yang cocok dengan original
        original_part = match.group(1)
        return f"{original_part}.jpg"
    
    return None


def find_matching_pairs(
    augmented_files: List[Path], 
    original_files: List[Path], 
    orig_prefix: str = "rp",
    aug_prefix: str = "aug"
) -> List[Tuple[Path, Path]]:
    """
    Temukan pasangan file augmentasi dan original berdasarkan uuid dalam nama file.
    
    Args:
        augmented_files: List path file augmentasi
        original_files: List path file original
        orig_prefix: Prefix untuk file original
        aug_prefix: Prefix untuk file augmentasi
        
    Returns:
        List tuple (original_path, augmented_path)
    """
    # Siapkan mapping original filename -> path
    original_map = {f.name: f for f in original_files}
    
    # Cari pasangan yang cocok
    matched_pairs = []
    
    for aug_file in augmented_files:
        # Dapatkan nama file original yang sesuai
        orig_filename = get_original_from_augmented(aug_file.name, orig_prefix)
        
        if orig_filename and orig_filename in original_map:
            orig_file = original_map[orig_filename]
            matched_pairs.append((orig_file, aug_file))
    
    return matched_pairs

