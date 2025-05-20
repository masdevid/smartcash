"""
File: smartcash/ui/dataset/visualization/handlers/tabs/layer_tab.py
Deskripsi: Handler untuk tab distribusi layer
"""

import os
from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.common.config.manager import get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_success_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def on_layer_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi layer.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat visualisasi distribusi layer...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    layer_tab = visualization_components.get('layer_tab', {})
    output = layer_tab.get('output')
    
    if output is None:
        show_error_status(ui_components, "Komponen output tidak ditemukan")
        return
    
    try:
        # Dapatkan dataset path dari konfigurasi
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset_config')
        dataset_path = dataset_config.get('dataset_path', '/content/data')
        
        if not dataset_path or not os.path.exists(dataset_path):
            show_warning_status(ui_components, "Dataset path tidak valid. Menampilkan data dummy.")
            _show_dummy_layer_distribution(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Analisis distribusi layer untuk semua split
            train_distribution = explorer_service.analyze_layer_distribution('train')
            val_distribution = explorer_service.analyze_layer_distribution('val')
            test_distribution = explorer_service.analyze_layer_distribution('test')
            
            # Tampilkan visualisasi
            _show_layer_distribution(output, train_distribution, val_distribution, test_distribution)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Visualisasi distribusi layer berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan distribusi layer: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_layer_distribution(output: widgets.Output) -> None:
    """
    Tampilkan distribusi layer dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat data dummy
        layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
        train_counts = [500, 450, 400, 350]
        val_counts = [120, 110, 100, 90]
        test_counts = [150, 140, 130, 120]
        
        # Buat DataFrame untuk visualisasi
        df = pd.DataFrame({
            'Layer': layers,
            'Train': train_counts,
            'Validation': val_counts,
            'Test': test_counts,
            'Total': [sum(x) for x in zip(train_counts, val_counts, test_counts)]
        })
        
        # Plot distribusi layer
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(layers))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Layer Dataset (Data Dummy)', fontsize=16)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Jumlah Objek', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi layer
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df['Total'], labels=layers, autopct='%1.1f%%', 
               colors=['#4285F4', '#FBBC05', '#34A853', '#EA4335'])
        ax.set_title('Persentase Objek per Layer', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df)
        
        # Tampilkan informasi tambahan
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #4285F4;">
            <h3 style="margin-top: 0;">Informasi Layer</h3>
            <p>Distribusi layer menunjukkan bagaimana objek dideteksi pada berbagai skala dalam model YOLOv5:</p>
            <ul>
                <li><b>Layer 1:</b> Deteksi objek besar (large objects)</li>
                <li><b>Layer 2:</b> Deteksi objek medium-besar (medium-large objects)</li>
                <li><b>Layer 3:</b> Deteksi objek medium-kecil (medium-small objects)</li>
                <li><b>Layer 4:</b> Deteksi objek kecil (small objects)</li>
            </ul>
            <p>Distribusi yang seimbang antar layer menunjukkan dataset yang memiliki variasi ukuran objek yang baik.</p>
        </div>
        """))

def _show_layer_distribution(
    output: widgets.Output, 
    train_distribution: Dict[str, Any], 
    val_distribution: Dict[str, Any], 
    test_distribution: Dict[str, Any]
) -> None:
    """
    Tampilkan distribusi layer dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        train_distribution: Hasil analisis distribusi layer untuk train split
        val_distribution: Hasil analisis distribusi layer untuk validation split
        test_distribution: Hasil analisis distribusi layer untuk test split
    """
    with output:
        clear_output(wait=True)
        
        # Dapatkan data distribusi layer
        train_layers = train_distribution.get('layer_distribution', {})
        val_layers = val_distribution.get('layer_distribution', {})
        test_layers = test_distribution.get('layer_distribution', {})
        
        # Gabungkan semua layer unik
        all_layers = sorted(set(list(train_layers.keys()) + 
                              list(val_layers.keys()) + 
                              list(test_layers.keys())))
        
        # Siapkan data untuk DataFrame
        data = []
        for layer in all_layers:
            data.append({
                'Layer': layer,
                'Train': train_layers.get(layer, 0),
                'Validation': val_layers.get(layer, 0),
                'Test': test_layers.get(layer, 0),
                'Total': train_layers.get(layer, 0) + val_layers.get(layer, 0) + test_layers.get(layer, 0)
            })
        
        # Buat DataFrame
        df = pd.DataFrame(data)
        
        # Plot distribusi layer
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(all_layers))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Layer Dataset', fontsize=16)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Jumlah Objek', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(all_layers)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi layer
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df['Total'], labels=all_layers, autopct='%1.1f%%', 
               colors=['#4285F4', '#FBBC05', '#34A853', '#EA4335'])
        ax.set_title('Persentase Objek per Layer', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df)
        
        # Tampilkan informasi tambahan
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #4285F4;">
            <h3 style="margin-top: 0;">Informasi Layer</h3>
            <p>Distribusi layer menunjukkan bagaimana objek dideteksi pada berbagai skala dalam model YOLOv5:</p>
            <ul>
                <li><b>Layer 0:</b> Deteksi objek besar (large objects)</li>
                <li><b>Layer 1:</b> Deteksi objek medium-besar (medium-large objects)</li>
                <li><b>Layer 2:</b> Deteksi objek medium-kecil (medium-small objects)</li>
                <li><b>Layer 3:</b> Deteksi objek kecil (small objects)</li>
            </ul>
            <p>Distribusi yang seimbang antar layer menunjukkan dataset yang memiliki variasi ukuran objek yang baik.</p>
        </div>
        """)) 