"""
File: smartcash/ui/dataset/visualization/handlers/layer_handlers.py
Deskripsi: Handler untuk visualisasi layer pada dataset
"""

from typing import Dict, Any, List, Tuple, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pathlib import Path
import random

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_error_status, show_warning_status, show_dummy_data_warning
)
from smartcash.dataset.visualization.dashboard.layer_visualizer import LayerVisualizer

logger = get_logger(__name__)

def setup_layer_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab visualisasi layer.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Dapatkan komponen tab layer
    layer_tab = ui_components.get('visualization_components', {}).get('layer_tab', {})
    
    # Setup handler untuk tombol layer
    if layer_tab and 'button' in layer_tab:
        layer_tab['button'].on_click(
            lambda b: on_layer_button_click(b, ui_components)
        )
    
    return ui_components

def get_layer_data(dataset_path: str) -> Tuple[Dict[str, Any], bool]:
    """
    Dapatkan data layer dari dataset.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Tuple berisi dictionary data layer dan flag data dummy
    """
    try:
        # Cek path dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"{ICONS.get('warning', '⚠️')} Path dataset tidak valid: {dataset_path}")
            return {}, True
        
        # Inisialisasi visualizer
        layer_visualizer = LayerVisualizer(dataset_path)
        
        # Dapatkan data layer
        layer_data = layer_visualizer.get_layer_data()
        
        # Jika tidak ada data, gunakan data dummy
        if not layer_data:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak ada data layer ditemukan")
            return {}, True
            
        return layer_data, False
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil data layer: {str(e)}")
        return {}, True

def get_dummy_layer_data() -> Dict[str, Any]:
    """
    Dapatkan data layer dummy untuk visualisasi.
    
    Returns:
        Dictionary berisi data layer dummy
    """
    # Data dummy untuk layer
    layers = ['Layer 1', 'Layer 2', 'Layer 3']
    classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
    
    # Buat data dummy untuk setiap layer
    class_counts = {}
    for layer in layers:
        class_counts[layer] = [random.randint(50, 200) for _ in range(len(classes))]
    
    # Buat data dummy untuk feature maps
    feature_maps = {}
    for layer in layers:
        # Simulasi feature maps dengan matriks acak
        feature_maps[layer] = np.random.rand(7, 7)
    
    # Buat data dummy untuk statistik layer
    layer_stats = {}
    for layer in layers:
        layer_stats[layer] = {
            'mean_activation': random.uniform(0.3, 0.7),
            'std_activation': random.uniform(0.1, 0.3),
            'max_activation': random.uniform(0.8, 1.0),
            'min_activation': random.uniform(0.0, 0.2)
        }
    
    return {
        'class_counts': class_counts,
        'feature_maps': feature_maps,
        'layer_stats': layer_stats,
        'layers': layers,
        'classes': classes
    }

def plot_layer_class_distribution(layer_data: Dict[str, Any], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi kelas per layer.
    
    Args:
        layer_data: Dictionary berisi data layer
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            show_dummy_data_warning(output)
        
        # Dapatkan data distribusi kelas per layer
        class_counts = layer_data['class_counts']
        layers = layer_data['layers']
        classes = layer_data['classes']
        
        # Plot distribusi kelas per layer
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, layer in enumerate(layers):
            offset = width * (i - 1)
            plt.bar(x + offset, class_counts[layer], width, label=layer)
        
        plt.title('Distribusi Kelas per Layer')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah Objek')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        display(plt.gcf())
        
        # Tampilkan tabel distribusi
        for layer in layers:
            print(f"\nDistribusi Kelas untuk {layer}:")
            layer_df = pd.DataFrame({
                'Kelas': classes,
                'Jumlah Objek': class_counts[layer]
            })
            display(layer_df)

def plot_feature_maps(layer_data: Dict[str, Any], output: widgets.Output) -> None:
    """
    Plot feature maps untuk setiap layer.
    
    Args:
        layer_data: Dictionary berisi data layer
        output: Widget output untuk menampilkan plot
    """
    with output:
        # Dapatkan data feature maps
        feature_maps = layer_data['feature_maps']
        layers = layer_data['layers']
        
        # Plot feature maps untuk setiap layer
        plt.figure(figsize=(15, 5))
        
        for i, layer in enumerate(layers):
            plt.subplot(1, len(layers), i+1)
            plt.imshow(feature_maps[layer], cmap='viridis')
            plt.title(f'Feature Map {layer}')
            plt.colorbar()
            plt.tight_layout()
        
        display(plt.gcf())

def plot_layer_statistics(layer_data: Dict[str, Any], output: widgets.Output) -> None:
    """
    Plot statistik untuk setiap layer.
    
    Args:
        layer_data: Dictionary berisi data layer
        output: Widget output untuk menampilkan plot
    """
    with output:
        # Dapatkan data statistik layer
        layer_stats = layer_data['layer_stats']
        layers = layer_data['layers']
        
        # Buat dataframe untuk statistik layer
        stats_data = []
        for layer in layers:
            stats = layer_stats[layer]
            stats_data.append({
                'Layer': layer,
                'Mean Activation': stats['mean_activation'],
                'Std Activation': stats['std_activation'],
                'Max Activation': stats['max_activation'],
                'Min Activation': stats['min_activation']
            })
        
        stats_df = pd.DataFrame(stats_data)
        display(stats_df)
        
        # Plot statistik layer
        metrics = ['mean_activation', 'std_activation', 'max_activation', 'min_activation']
        metric_labels = ['Mean Activation', 'Std Activation', 'Max Activation', 'Min Activation']
        
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            values = [layer_stats[layer][metric] for layer in layers]
            plt.bar(layers, values, color='skyblue')
            plt.title(metric_labels[i])
            plt.ylabel('Nilai')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        display(plt.gcf())

def on_layer_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol visualisasi layer.
    
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
        show_loading_status(output, "Memuat data layer...")
        
        # Dapatkan path dataset
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', '')
        
        # Jika path tidak valid, tampilkan error
        if not dataset_path or not os.path.exists(dataset_path):
            show_error_status(output, "Path dataset tidak valid")
            return
        
        # Dapatkan data layer
        layer_data, is_dummy = get_layer_data(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            layer_data = get_dummy_layer_data()
        
        # Plot visualisasi layer
        plot_layer_class_distribution(layer_data, output, is_dummy)
        plot_feature_maps(layer_data, output)
        plot_layer_statistics(layer_data, output)
        
    except Exception as e:
        if output:
            show_error_status(output, f"Error: {str(e)}")
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan visualisasi layer: {str(e)}") 