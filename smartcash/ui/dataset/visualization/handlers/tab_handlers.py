"""
File: smartcash/ui/dataset/visualization/handlers/tab_handlers.py
Deskripsi: Handler untuk tab visualisasi dataset
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from PIL import Image
import random

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import ConfigManager

logger = get_logger(__name__)

def setup_tab_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab visualisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol distribusi kelas
    distribution_tab = ui_components.get('distribution_tab', {})
    if distribution_tab and 'button' in distribution_tab:
        distribution_tab['button'].on_click(
            lambda b: on_distribution_tab_click(b, ui_components)
        )
    
    # Setup handler untuk tombol distribusi split
    split_tab = ui_components.get('split_tab', {})
    if split_tab and 'button' in split_tab:
        split_tab['button'].on_click(
            lambda b: on_split_tab_click(b, ui_components)
        )
    
    # Setup handler untuk tombol distribusi layer
    layer_tab = ui_components.get('layer_tab', {})
    if layer_tab and 'button' in layer_tab:
        layer_tab['button'].on_click(
            lambda b: on_layer_tab_click(b, ui_components)
        )
    
    # Setup handler untuk tombol heatmap
    heatmap_tab = ui_components.get('heatmap_tab', {})
    if heatmap_tab and 'button' in heatmap_tab:
        heatmap_tab['button'].on_click(
            lambda b: on_heatmap_tab_click(b, ui_components)
        )
    
    return ui_components

def on_distribution_tab_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol distribusi kelas.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        distribution_tab = ui_components.get('distribution_tab', {})
        output = distribution_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi kelas..."))
            
            # Dapatkan dataset path
            config_manager = ConfigManager()
            dataset_config = config_manager.get_module_config('dataset_config')
            dataset_path = dataset_config.get('dataset_path', '')
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Visualisasi distribusi kelas (dummy data untuk contoh)
            clear_output(wait=True)
            
            # Buat data dummy untuk visualisasi
            classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            counts = [120, 150, 180, 200, 170, 160, 140]
            
            # Plot distribusi kelas
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classes, counts, color='skyblue')
            
            # Tambahkan nilai di atas bar
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(count), ha='center', va='bottom')
            
            plt.title('Distribusi Kelas Dataset')
            plt.xlabel('Kelas')
            plt.ylabel('Jumlah Objek')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            display(plt.gcf())
            
            # Tampilkan tabel distribusi
            df = pd.DataFrame({
                'Kelas': classes,
                'Jumlah Objek': counts,
                'Persentase': [f"{count/sum(counts)*100:.1f}%" for count in counts]
            })
            
            display(df)
            
    except Exception as e:
        distribution_tab = ui_components.get('distribution_tab', {})
        output = distribution_tab.get('output')
        
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_split_tab_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol distribusi split.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        split_tab = ui_components.get('split_tab', {})
        output = split_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi split..."))
            
            # Dapatkan dataset path
            config_manager = ConfigManager()
            dataset_config = config_manager.get_module_config('dataset_config')
            dataset_path = dataset_config.get('dataset_path', '')
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Visualisasi distribusi split (dummy data untuk contoh)
            clear_output(wait=True)
            
            # Buat data dummy untuk visualisasi
            splits = ['Train', 'Validation', 'Test']
            counts = [1400, 300, 300]
            
            # Plot distribusi split
            plt.figure(figsize=(10, 6))
            bars = plt.bar(splits, counts, color=['#4285F4', '#FBBC05', '#34A853'])
            
            # Tambahkan nilai di atas bar
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(count), ha='center', va='bottom')
            
            plt.title('Distribusi Split Dataset')
            plt.xlabel('Split')
            plt.ylabel('Jumlah Gambar')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            display(plt.gcf())
            
            # Tampilkan tabel distribusi
            total = sum(counts)
            df = pd.DataFrame({
                'Split': splits,
                'Jumlah Gambar': counts,
                'Persentase': [f"{count/total*100:.1f}%" for count in counts]
            })
            
            display(df)
            
            # Plot pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(counts, labels=splits, autopct='%1.1f%%', startangle=90, 
                   colors=['#4285F4', '#FBBC05', '#34A853'])
            plt.axis('equal')
            plt.title('Distribusi Split Dataset (Persentase)')
            
            display(plt.gcf())
            
    except Exception as e:
        split_tab = ui_components.get('split_tab', {})
        output = split_tab.get('output')
        
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_layer_tab_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol distribusi layer.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        layer_tab = ui_components.get('layer_tab', {})
        output = layer_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi layer..."))
            
            # Visualisasi distribusi layer (dummy data untuk contoh)
            clear_output(wait=True)
            
            # Buat data dummy untuk visualisasi
            layers = ['Layer 1', 'Layer 2', 'Layer 3']
            classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            
            # Buat data dummy untuk setiap layer
            data = {}
            for layer in layers:
                data[layer] = [random.randint(50, 200) for _ in range(len(classes))]
            
            # Plot distribusi layer
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(classes))
            width = 0.25
            
            for i, layer in enumerate(layers):
                offset = width * (i - 1)
                plt.bar(x + offset, data[layer], width, label=layer)
            
            plt.title('Distribusi Kelas per Layer')
            plt.xlabel('Kelas')
            plt.ylabel('Jumlah Objek')
            plt.xticks(x, classes, rotation=45)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            display(plt.gcf())
            
            # Tampilkan tabel distribusi
            df = pd.DataFrame({
                'Kelas': classes,
                **{layer: data[layer] for layer in layers},
                'Total': [sum(data[layer][i] for layer in layers) for i in range(len(classes))]
            })
            
            display(df)
            
    except Exception as e:
        layer_tab = ui_components.get('layer_tab', {})
        output = layer_tab.get('output')
        
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_heatmap_tab_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol heatmap.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        heatmap_tab = ui_components.get('heatmap_tab', {})
        output = heatmap_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat heatmap deteksi..."))
            
            # Visualisasi heatmap (dummy data untuk contoh)
            clear_output(wait=True)
            
            # Buat data dummy untuk heatmap
            img_width, img_height = 640, 640
            heatmap_data = np.zeros((img_height, img_width))
            
            # Simulasi 1000 deteksi dengan distribusi normal
            num_detections = 1000
            x_mean, x_std = img_width / 2, img_width / 4
            y_mean, y_std = img_height / 2, img_height / 4
            
            x_coords = np.random.normal(x_mean, x_std, num_detections).astype(int)
            y_coords = np.random.normal(y_mean, y_std, num_detections).astype(int)
            
            # Filter koordinat yang valid
            valid_idx = (x_coords >= 0) & (x_coords < img_width) & (y_coords >= 0) & (y_coords < img_height)
            x_coords, y_coords = x_coords[valid_idx], y_coords[valid_idx]
            
            # Buat heatmap
            for x, y in zip(x_coords, y_coords):
                heatmap_data[y, x] += 1
            
            # Smooth heatmap
            heatmap_data = np.log1p(heatmap_data)  # Log scale untuk visualisasi yang lebih baik
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap_data, cmap='hot', interpolation='gaussian')
            plt.colorbar(label='Log(Jumlah Deteksi)')
            plt.title('Heatmap Posisi Objek Deteksi')
            plt.xlabel('Posisi X')
            plt.ylabel('Posisi Y')
            plt.tight_layout()
            
            display(plt.gcf())
            
            # Tambahkan informasi statistik
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(x_coords, bins=30, color='skyblue', alpha=0.7)
            plt.title('Distribusi Posisi X')
            plt.xlabel('Posisi X')
            plt.ylabel('Frekuensi')
            
            plt.subplot(1, 2, 2)
            plt.hist(y_coords, bins=30, color='salmon', alpha=0.7)
            plt.title('Distribusi Posisi Y')
            plt.xlabel('Posisi Y')
            plt.ylabel('Frekuensi')
            
            plt.tight_layout()
            display(plt.gcf())
            
    except Exception as e:
        heatmap_tab = ui_components.get('heatmap_tab', {})
        output = heatmap_tab.get('output')
        
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
