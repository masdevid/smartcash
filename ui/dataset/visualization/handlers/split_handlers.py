"""
File: smartcash/ui/dataset/visualization/handlers/split_handlers.py
Deskripsi: Handler untuk tab distribusi split di visualisasi dataset
"""

from typing import Dict, Any, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def setup_split_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab distribusi split.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Dapatkan komponen tab split
    split_tab = ui_components.get('visualization_components', {}).get('split_tab', {})
    
    # Setup handler untuk tombol distribusi split
    if split_tab and 'button' in split_tab:
        split_tab['button'].on_click(
            lambda b: on_split_button_click(b, ui_components)
        )
    
    return ui_components

def get_split_distribution(dataset_path: str) -> Tuple[Dict[str, Dict[str, int]], bool]:
    """
    Dapatkan distribusi split dari dataset.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Tuple berisi dictionary distribusi split dan flag data dummy
    """
    try:
        # Cek path dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Path dataset tidak valid: {dataset_path}")
            return {}, True
        
        # Cek direktori images dan labels
        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Direktori images atau labels tidak ditemukan")
            return {}, True
        
        # Hitung jumlah file per split
        split_counts = {}
        for split in ['train', 'val', 'test']:
            split_images_path = os.path.join(images_path, split)
            split_labels_path = os.path.join(labels_path, split)
            
            images_count = 0
            labels_count = 0
            
            if os.path.exists(split_images_path):
                images_count = len([f for f in os.listdir(split_images_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if os.path.exists(split_labels_path):
                labels_count = len([f for f in os.listdir(split_labels_path) 
                                  if f.endswith('.txt')])
            
            split_counts[split] = {
                'images': images_count,
                'labels': labels_count
            }
        
        # Jika tidak ada data, gunakan data dummy
        if not any(split_counts.values()):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak ada data split ditemukan")
            return {}, True
            
        return split_counts, False
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil distribusi split: {str(e)}")
        return {}, True

def get_dummy_split_distribution() -> Dict[str, Dict[str, int]]:
    """
    Dapatkan distribusi split dummy untuk visualisasi.
    
    Returns:
        Dictionary berisi distribusi split dummy
    """
    return {
        'train': {'images': 1400, 'labels': 1400},
        'val': {'images': 300, 'labels': 300},
        'test': {'images': 300, 'labels': 300}
    }

def plot_split_distribution(split_counts: Dict[str, Dict[str, int]], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi split.
    
    Args:
        split_counts: Dictionary berisi jumlah file per split
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menggunakan data dummy untuk visualisasi"))
        
        # Ekstrak data untuk plot
        splits = list(split_counts.keys())
        images_counts = [split_counts[split]['images'] for split in splits]
        
        # Kapitalisasi nama split untuk tampilan
        display_splits = [split.capitalize() for split in splits]
        
        # Plot distribusi split
        plt.figure(figsize=(10, 6))
        colors = ['#4285F4', '#FBBC05', '#34A853'][:len(splits)]
        bars = plt.bar(display_splits, images_counts, color=colors)
        
        # Tambahkan nilai di atas bar
        for bar, count in zip(bars, images_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom')
        
        plt.title('Distribusi Split Dataset')
        plt.xlabel('Split')
        plt.ylabel('Jumlah Gambar')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Tampilkan tabel distribusi
        total_images = sum(images_counts)
        
        data = []
        for i, split in enumerate(splits):
            images = split_counts[split]['images']
            labels = split_counts[split]['labels']
            percentage = (images / total_images * 100) if total_images > 0 else 0
            
            data.append({
                'Split': display_splits[i],
                'Jumlah Gambar': images,
                'Jumlah Label': labels,
                'Persentase': f"{percentage:.1f}%"
            })
        
        df = pd.DataFrame(data)
        display(df)
        
        # Plot pie chart untuk persentase split
        plt.figure(figsize=(8, 8))
        plt.pie(images_counts, labels=display_splits, autopct='%1.1f%%', 
                startangle=90, colors=colors)
        plt.axis('equal')
        plt.title('Persentase Split Dataset')
        plt.tight_layout()
        
        display(plt.gcf())

def on_split_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi split.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        split_tab = ui_components.get('visualization_components', {}).get('split_tab', {})
        output = split_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        # Tampilkan loading status
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi split..."))
        
        # Dapatkan path dataset
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', '')
        
        # Jika path tidak valid, tampilkan error
        if not dataset_path or not os.path.exists(dataset_path):
            with output:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
            return
        
        # Dapatkan distribusi split
        split_counts, is_dummy = get_split_distribution(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            split_counts = get_dummy_split_distribution()
        
        # Plot distribusi split
        plot_split_distribution(split_counts, output, is_dummy)
        
    except Exception as e:
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi split: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}")) 