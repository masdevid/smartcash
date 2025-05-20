"""
File: smartcash/ui/dataset/visualization/handlers/bbox_handlers.py
Deskripsi: Handler untuk visualisasi bounding box pada dataset
"""

from typing import Dict, Any, List, Tuple, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pathlib import Path

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_error_status, show_warning_status, show_dummy_data_warning
)
from smartcash.dataset.visualization.dashboard.bbox_visualizer import BBoxVisualizer

logger = get_logger(__name__)

def setup_bbox_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab visualisasi bounding box.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Dapatkan komponen tab bbox
    bbox_tab = ui_components.get('visualization_components', {}).get('bbox_tab', {})
    
    # Setup handler untuk tombol bbox
    if bbox_tab and 'button' in bbox_tab:
        bbox_tab['button'].on_click(
            lambda b: on_bbox_button_click(b, ui_components)
        )
    
    return ui_components

def get_bbox_data(dataset_path: str) -> Tuple[Dict[str, Any], bool]:
    """
    Dapatkan data bounding box dari dataset.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Tuple berisi dictionary data bbox dan flag data dummy
    """
    try:
        # Cek path dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Path dataset tidak valid: {dataset_path}")
            return {}, True
        
        # Cek direktori labels
        labels_path = os.path.join(dataset_path, 'labels')
        if not os.path.exists(labels_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Direktori labels tidak ditemukan: {labels_path}")
            return {}, True
        
        # Inisialisasi visualizer
        bbox_visualizer = BBoxVisualizer(dataset_path)
        
        # Dapatkan data bbox
        bbox_data = bbox_visualizer.get_bbox_data()
        
        # Jika tidak ada data, gunakan data dummy
        if not bbox_data:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak ada data bbox ditemukan")
            return {}, True
            
        return bbox_data, False
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil data bbox: {str(e)}")
        return {}, True

def get_dummy_bbox_data() -> Dict[str, Any]:
    """
    Dapatkan data bbox dummy untuk visualisasi.
    
    Returns:
        Dictionary berisi data bbox dummy
    """
    # Data dummy untuk posisi bbox
    x_centers = np.random.uniform(0.2, 0.8, 500)
    y_centers = np.random.uniform(0.2, 0.8, 500)
    widths = np.random.uniform(0.1, 0.3, 500)
    heights = np.random.uniform(0.1, 0.3, 500)
    classes = np.random.randint(0, 7, 500)  # 7 kelas (Rp1000-Rp100000)
    
    # Data untuk distribusi ukuran
    size_data = widths * heights
    
    # Data untuk distribusi posisi
    position_data = {
        'x_center': x_centers,
        'y_center': y_centers
    }
    
    # Data untuk distribusi aspek rasio
    aspect_ratios = widths / heights
    
    return {
        'positions': position_data,
        'sizes': size_data,
        'aspect_ratios': aspect_ratios,
        'classes': classes
    }

def plot_bbox_position_distribution(bbox_data: Dict[str, Any], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi posisi bounding box.
    
    Args:
        bbox_data: Dictionary berisi data bbox
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            show_dummy_data_warning(output)
        
        # Dapatkan data posisi
        x_centers = bbox_data['positions']['x_center']
        y_centers = bbox_data['positions']['y_center']
        
        # Plot distribusi posisi (heatmap)
        plt.figure(figsize=(10, 8))
        plt.hist2d(x_centers, y_centers, bins=30, cmap='viridis')
        plt.colorbar(label='Jumlah Objek')
        plt.title('Distribusi Posisi Objek dalam Gambar')
        plt.xlabel('Posisi X (relatif)')
        plt.ylabel('Posisi Y (relatif)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Plot distribusi posisi X dan Y (histogram)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(x_centers, bins=20, color='skyblue', edgecolor='black')
        ax1.set_title('Distribusi Posisi X')
        ax1.set_xlabel('Posisi X (relatif)')
        ax1.set_ylabel('Frekuensi')
        ax1.grid(alpha=0.3)
        
        ax2.hist(y_centers, bins=20, color='lightgreen', edgecolor='black')
        ax2.set_title('Distribusi Posisi Y')
        ax2.set_xlabel('Posisi Y (relatif)')
        ax2.set_ylabel('Frekuensi')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        display(plt.gcf())

def plot_bbox_size_distribution(bbox_data: Dict[str, Any], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi ukuran bounding box.
    
    Args:
        bbox_data: Dictionary berisi data bbox
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        # Dapatkan data ukuran
        sizes = bbox_data['sizes']
        
        # Plot distribusi ukuran
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=30, color='coral', edgecolor='black')
        plt.title('Distribusi Ukuran Objek')
        plt.xlabel('Ukuran (area relatif)')
        plt.ylabel('Frekuensi')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Statistik ukuran
        size_stats = {
            'Min': np.min(sizes),
            'Max': np.max(sizes),
            'Mean': np.mean(sizes),
            'Median': np.median(sizes),
            'Std Dev': np.std(sizes)
        }
        
        # Tampilkan statistik ukuran
        stats_df = pd.DataFrame(list(size_stats.items()), columns=['Statistik', 'Nilai'])
        stats_df['Nilai'] = stats_df['Nilai'].round(4)
        display(stats_df)
        
        # Plot distribusi aspek rasio
        aspect_ratios = bbox_data['aspect_ratios']
        
        plt.figure(figsize=(10, 6))
        plt.hist(aspect_ratios, bins=30, color='lightblue', edgecolor='black')
        plt.title('Distribusi Aspek Rasio Objek')
        plt.xlabel('Aspek Rasio (width/height)')
        plt.ylabel('Frekuensi')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        display(plt.gcf())

def on_bbox_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol visualisasi bbox.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        bbox_tab = ui_components.get('visualization_components', {}).get('bbox_tab', {})
        output = bbox_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        # Tampilkan loading status
        show_loading_status(output, "Memuat data bounding box...")
        
        # Dapatkan path dataset
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', '')
        
        # Jika path tidak valid, tampilkan error
        if not dataset_path or not os.path.exists(dataset_path):
            show_error_status(output, "Path dataset tidak valid")
            return
        
        # Dapatkan data bbox
        bbox_data, is_dummy = get_bbox_data(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            bbox_data = get_dummy_bbox_data()
        
        # Plot distribusi posisi dan ukuran bbox
        plot_bbox_position_distribution(bbox_data, output, is_dummy)
        plot_bbox_size_distribution(bbox_data, output, is_dummy)
        
    except Exception as e:
        if output:
            show_error_status(output, f"Error: {str(e)}")
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan visualisasi bbox: {str(e)}") 