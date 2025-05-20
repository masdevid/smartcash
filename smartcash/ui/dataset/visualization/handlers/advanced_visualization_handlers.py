"""
File: smartcash/ui/dataset/visualization/handlers/advanced_visualization_handlers.py
Deskripsi: Handler untuk visualisasi lanjutan seperti distribusi layer dan heatmap
"""

from typing import Dict, Any, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import random
from PIL import Image
import seaborn as sns

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def setup_advanced_visualization_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab visualisasi lanjutan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol distribusi layer
    layer_tab = ui_components.get('visualization_components', {}).get('layer_tab', {})
    if layer_tab and 'button' in layer_tab:
        layer_tab['button'].on_click(
            lambda b: on_layer_button_click(b, ui_components)
        )
    
    # Setup handler untuk tombol heatmap
    heatmap_tab = ui_components.get('visualization_components', {}).get('heatmap_tab', {})
    if heatmap_tab and 'button' in heatmap_tab:
        heatmap_tab['button'].on_click(
            lambda b: on_heatmap_button_click(b, ui_components)
        )
    
    return ui_components

def get_label_data(dataset_path: str) -> Tuple[List[List[float]], bool]:
    """
    Dapatkan data label dari dataset untuk visualisasi layer dan heatmap.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Tuple berisi list data label dan flag data dummy
    """
    try:
        # Cek path dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Path dataset tidak valid: {dataset_path}")
            return [], True
        
        # Cek direktori labels
        labels_path = os.path.join(dataset_path, 'labels')
        if not os.path.exists(labels_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Direktori labels tidak ditemukan")
            return [], True
        
        # Kumpulkan data label
        label_data = []
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(labels_path, split)
            if os.path.exists(split_path):
                for label_file in os.listdir(split_path):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(split_path, label_file), 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:  # Format YOLO: class x y w h
                                    try:
                                        class_id = int(parts[0])
                                        x = float(parts[1])
                                        y = float(parts[2])
                                        w = float(parts[3])
                                        h = float(parts[4])
                                        label_data.append([class_id, x, y, w, h])
                                    except (ValueError, IndexError):
                                        continue
        
        # Jika tidak ada data, gunakan data dummy
        if not label_data:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak ada data label ditemukan")
            return [], True
            
        return label_data, False
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil data label: {str(e)}")
        return [], True

def get_dummy_label_data(num_samples: int = 1000) -> List[List[float]]:
    """
    Dapatkan data label dummy untuk visualisasi layer dan heatmap.
    
    Args:
        num_samples: Jumlah sampel data dummy
        
    Returns:
        List data label dummy
    """
    # Generate random data untuk simulasi
    label_data = []
    
    # Distribusi kelas yang tidak seragam
    class_weights = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
    classes = list(range(len(class_weights)))
    
    for _ in range(num_samples):
        # Pilih kelas berdasarkan distribusi
        class_id = np.random.choice(classes, p=class_weights)
        
        # Koordinat pusat objek (x, y) dengan distribusi tertentu
        if class_id in [0, 1]:  # Pecahan kecil lebih ke tengah
            x = np.random.normal(0.5, 0.15)
            y = np.random.normal(0.5, 0.15)
        elif class_id in [2, 3]:  # Pecahan menengah lebih tersebar
            x = np.random.normal(0.5, 0.2)
            y = np.random.normal(0.5, 0.2)
        else:  # Pecahan besar lebih tersebar lagi
            x = np.random.normal(0.5, 0.25)
            y = np.random.normal(0.5, 0.25)
        
        # Pastikan x dan y dalam range [0, 1]
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Ukuran objek (w, h) berdasarkan kelas
        if class_id in [0, 1]:  # Pecahan kecil
            w = np.random.uniform(0.05, 0.15)
            h = np.random.uniform(0.05, 0.15)
        elif class_id in [2, 3]:  # Pecahan menengah
            w = np.random.uniform(0.1, 0.2)
            h = np.random.uniform(0.1, 0.2)
        else:  # Pecahan besar
            w = np.random.uniform(0.15, 0.3)
            h = np.random.uniform(0.15, 0.3)
        
        label_data.append([class_id, x, y, w, h])
    
    return label_data

def plot_layer_distribution(label_data: List[List[float]], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi layer (ukuran objek).
    
    Args:
        label_data: List data label
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menggunakan data dummy untuk visualisasi"))
        
        # Ekstrak data untuk plot
        class_ids = [data[0] for data in label_data]
        widths = [data[3] for data in label_data]
        heights = [data[4] for data in label_data]
        areas = [w * h for w, h in zip(widths, heights)]
        
        # Konversi ke persentase dari gambar
        widths_percent = [w * 100 for w in widths]
        heights_percent = [h * 100 for h in heights]
        areas_percent = [a * 100 for a in areas]
        
        # Plot distribusi ukuran objek
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Distribusi lebar
        plt.subplot(1, 3, 1)
        plt.hist(widths_percent, bins=20, color='skyblue', alpha=0.7)
        plt.title('Distribusi Lebar Objek')
        plt.xlabel('Lebar (% dari gambar)')
        plt.ylabel('Frekuensi')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Distribusi tinggi
        plt.subplot(1, 3, 2)
        plt.hist(heights_percent, bins=20, color='lightgreen', alpha=0.7)
        plt.title('Distribusi Tinggi Objek')
        plt.xlabel('Tinggi (% dari gambar)')
        plt.ylabel('Frekuensi')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Distribusi area
        plt.subplot(1, 3, 3)
        plt.hist(areas_percent, bins=20, color='salmon', alpha=0.7)
        plt.title('Distribusi Area Objek')
        plt.xlabel('Area (% dari gambar)')
        plt.ylabel('Frekuensi')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        display(plt.gcf())
        
        # Analisis layer berdasarkan ukuran
        small = sum(1 for a in areas if a < 0.02)
        medium = sum(1 for a in areas if 0.02 <= a < 0.1)
        large = sum(1 for a in areas if a >= 0.1)
        
        total = len(areas)
        
        # Tampilkan tabel distribusi layer
        data = [
            {'Layer': 'Kecil (< 2%)', 'Jumlah Objek': small, 'Persentase': f"{small/total*100:.1f}%"},
            {'Layer': 'Sedang (2-10%)', 'Jumlah Objek': medium, 'Persentase': f"{medium/total*100:.1f}%"},
            {'Layer': 'Besar (> 10%)', 'Jumlah Objek': large, 'Persentase': f"{large/total*100:.1f}%"}
        ]
        
        df = pd.DataFrame(data)
        display(df)
        
        # Plot pie chart untuk persentase layer
        plt.figure(figsize=(8, 8))
        plt.pie([small, medium, large], 
                labels=['Kecil (< 2%)', 'Sedang (2-10%)', 'Besar (> 10%)'], 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['#FF9999', '#66B2FF', '#99FF99'])
        plt.axis('equal')
        plt.title('Persentase Layer Berdasarkan Ukuran')
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Plot scatter untuk visualisasi hubungan lebar dan tinggi
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(widths_percent, heights_percent, 
                             c=[int(c) for c in class_ids], 
                             alpha=0.6, 
                             cmap='viridis')
        plt.colorbar(scatter, label='Kelas')
        plt.title('Hubungan Lebar dan Tinggi Objek')
        plt.xlabel('Lebar (% dari gambar)')
        plt.ylabel('Tinggi (% dari gambar)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        display(plt.gcf())

def plot_heatmap(label_data: List[List[float]], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot heatmap posisi objek.
    
    Args:
        label_data: List data label
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menggunakan data dummy untuk visualisasi"))
        
        # Ekstrak data untuk plot
        x_centers = [data[1] for data in label_data]
        y_centers = [data[2] for data in label_data]
        class_ids = [int(data[0]) for data in label_data]
        
        # Buat heatmap untuk semua kelas
        plt.figure(figsize=(10, 8))
        
        # Gunakan histogram 2D untuk membuat heatmap
        heatmap, xedges, yedges = np.histogram2d(x_centers, y_centers, bins=50, range=[[0, 1], [0, 1]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='gaussian')
        plt.colorbar(label='Jumlah Objek')
        plt.title('Heatmap Posisi Objek (Semua Kelas)')
        plt.xlabel('Posisi X')
        plt.ylabel('Posisi Y')
        plt.grid(False)
        
        display(plt.gcf())
        
        # Buat scatter plot untuk posisi objek berdasarkan kelas
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_centers, y_centers, c=class_ids, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Kelas')
        plt.title('Posisi Objek Berdasarkan Kelas')
        plt.xlabel('Posisi X')
        plt.ylabel('Posisi Y')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Dapatkan kelas unik
        unique_classes = sorted(set(class_ids))
        
        # Buat grid untuk heatmap per kelas
        n_classes = len(unique_classes)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        if n_classes > 1:
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, class_id in enumerate(unique_classes):
                # Filter data untuk kelas tertentu
                x_class = [x for x, c in zip(x_centers, class_ids) if c == class_id]
                y_class = [y for y, c in zip(y_centers, class_ids) if c == class_id]
                
                plt.subplot(n_rows, n_cols, i + 1)
                
                # Gunakan kernel density estimation untuk heatmap yang lebih halus
                if len(x_class) > 10:  # Minimal data untuk KDE
                    try:
                        # Buat grid untuk KDE
                        xx, yy = np.mgrid[0:1:100j, 0:1:100j]
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        values = np.vstack([x_class, y_class])
                        
                        from scipy.stats import gaussian_kde
                        kernel = gaussian_kde(values)
                        f = np.reshape(kernel(positions).T, xx.shape)
                        
                        plt.imshow(f.T, extent=[0, 1, 0, 1], origin='lower', cmap='hot', interpolation='gaussian')
                        plt.colorbar(label='Densitas')
                    except Exception:
                        # Fallback jika KDE gagal
                        heatmap, _, _ = np.histogram2d(x_class, y_class, bins=20, range=[[0, 1], [0, 1]])
                        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='gaussian')
                        plt.colorbar(label='Jumlah Objek')
                else:
                    # Jika data terlalu sedikit untuk KDE, gunakan scatter
                    plt.scatter(x_class, y_class, alpha=0.6, c='red')
                
                plt.title(f'Heatmap Kelas {class_id}')
                plt.xlabel('Posisi X')
                plt.ylabel('Posisi Y')
                plt.grid(False)
            
            plt.tight_layout()
            display(plt.gcf())

def on_layer_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi layer.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        layer_tab = ui_components.get('visualization_components', {}).get('layer_tab', {})
        output = layer_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        # Tampilkan loading status
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi layer..."))
        
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
        
        # Dapatkan data label
        label_data, is_dummy = get_label_data(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            label_data = get_dummy_label_data(1000)
        
        # Plot distribusi layer
        plot_layer_distribution(label_data, output, is_dummy)
        
    except Exception as e:
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi layer: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_heatmap_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol heatmap.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        heatmap_tab = ui_components.get('visualization_components', {}).get('heatmap_tab', {})
        output = heatmap_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        # Tampilkan loading status
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat heatmap deteksi..."))
        
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
        
        # Dapatkan data label
        label_data, is_dummy = get_label_data(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            label_data = get_dummy_label_data(1000)
        
        # Plot heatmap
        plot_heatmap(label_data, output, is_dummy)
        
    except Exception as e:
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan heatmap: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}")) 