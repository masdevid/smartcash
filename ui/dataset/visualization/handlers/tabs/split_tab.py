"""
File: smartcash/ui/dataset/visualization/handlers/tabs/split_tab.py
Deskripsi: Handler untuk tab distribusi split
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

def on_split_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi split.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat visualisasi distribusi split...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    split_tab = visualization_components.get('split_tab', {})
    output = split_tab.get('output')
    
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
            _show_dummy_split_distribution(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Dapatkan statistik untuk setiap split
            train_stats = explorer_service.analyze_class_distribution('train')
            val_stats = explorer_service.analyze_class_distribution('val')
            test_stats = explorer_service.analyze_class_distribution('test')
            
            # Hitung total gambar dan objek per split
            train_images = sum(train_stats.get('class_distribution', {}).values())
            val_images = sum(val_stats.get('class_distribution', {}).values())
            test_images = sum(test_stats.get('class_distribution', {}).values())
            
            # Tampilkan visualisasi
            _show_split_distribution(output, train_images, val_images, test_images)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Visualisasi distribusi split berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan distribusi split: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_split_distribution(output: widgets.Output) -> None:
    """
    Tampilkan distribusi split dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat data dummy
        splits = ['Train', 'Validation', 'Test']
        images = [1170, 290, 360]
        objects = [2340, 580, 720]
        
        # Buat DataFrame untuk visualisasi
        df = pd.DataFrame({
            'Split': splits,
            'Jumlah Gambar': images,
            'Jumlah Objek': objects,
            'Objek/Gambar': [o/i for o, i in zip(objects, images)]
        })
        
        # Plot distribusi split
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot distribusi gambar
        ax1.bar(splits, images, color=['#4285F4', '#FBBC05', '#34A853'])
        ax1.set_title('Distribusi Gambar per Split (Data Dummy)', fontsize=14)
        ax1.set_ylabel('Jumlah Gambar', fontsize=12)
        for i, v in enumerate(images):
            ax1.text(i, v + 20, str(v), ha='center')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot distribusi objek
        ax2.bar(splits, objects, color=['#4285F4', '#FBBC05', '#34A853'])
        ax2.set_title('Distribusi Objek per Split (Data Dummy)', fontsize=14)
        ax2.set_ylabel('Jumlah Objek', fontsize=12)
        for i, v in enumerate(objects):
            ax2.text(i, v + 40, str(v), ha='center')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi split
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart untuk gambar
        ax1.pie(images, labels=splits, autopct='%1.1f%%', colors=['#4285F4', '#FBBC05', '#34A853'])
        ax1.set_title('Persentase Gambar per Split', fontsize=14)
        
        # Pie chart untuk objek
        ax2.pie(objects, labels=splits, autopct='%1.1f%%', colors=['#4285F4', '#FBBC05', '#34A853'])
        ax2.set_title('Persentase Objek per Split', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df)

def _show_split_distribution(
    output: widgets.Output, 
    train_images: int, 
    val_images: int, 
    test_images: int
) -> None:
    """
    Tampilkan distribusi split dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        train_images: Jumlah gambar di train split
        val_images: Jumlah gambar di validation split
        test_images: Jumlah gambar di test split
    """
    with output:
        clear_output(wait=True)
        
        # Siapkan data
        splits = ['Train', 'Validation', 'Test']
        images = [train_images, val_images, test_images]
        
        # Estimasi jumlah objek (biasanya 2x jumlah gambar)
        objects = [count * 2 for count in images]
        
        # Buat DataFrame untuk visualisasi
        df = pd.DataFrame({
            'Split': splits,
            'Jumlah Gambar': images,
            'Jumlah Objek': objects,
            'Objek/Gambar': [o/i if i > 0 else 0 for o, i in zip(objects, images)]
        })
        
        # Plot distribusi split
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot distribusi gambar
        ax1.bar(splits, images, color=['#4285F4', '#FBBC05', '#34A853'])
        ax1.set_title('Distribusi Gambar per Split', fontsize=14)
        ax1.set_ylabel('Jumlah Gambar', fontsize=12)
        for i, v in enumerate(images):
            ax1.text(i, v + (max(images) * 0.02), str(v), ha='center')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot distribusi objek
        ax2.bar(splits, objects, color=['#4285F4', '#FBBC05', '#34A853'])
        ax2.set_title('Distribusi Objek per Split (Estimasi)', fontsize=14)
        ax2.set_ylabel('Jumlah Objek', fontsize=12)
        for i, v in enumerate(objects):
            ax2.text(i, v + (max(objects) * 0.02), str(v), ha='center')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi split
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart untuk gambar
        ax1.pie(images, labels=splits, autopct='%1.1f%%', colors=['#4285F4', '#FBBC05', '#34A853'])
        ax1.set_title('Persentase Gambar per Split', fontsize=14)
        
        # Pie chart untuk objek
        ax2.pie(objects, labels=splits, autopct='%1.1f%%', colors=['#4285F4', '#FBBC05', '#34A853'])
        ax2.set_title('Persentase Objek per Split (Estimasi)', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df)
        
        # Tampilkan rekomendasi split ratio
        total_images = sum(images)
        train_pct = train_images / total_images * 100 if total_images > 0 else 0
        val_pct = val_images / total_images * 100 if total_images > 0 else 0
        test_pct = test_images / total_images * 100 if total_images > 0 else 0
        
        display(widgets.HTML(f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #4285F4;">
            <h3 style="margin-top: 0;">Analisis Ratio Split Dataset</h3>
            <p>Ratio split saat ini: <b>Train {train_pct:.1f}% : Val {val_pct:.1f}% : Test {test_pct:.1f}%</b></p>
            <p>Rekomendasi ratio split yang umum digunakan:</p>
            <ul>
                <li><b>70% : 15% : 15%</b> - Untuk dataset berukuran kecil hingga sedang</li>
                <li><b>80% : 10% : 10%</b> - Untuk dataset berukuran besar</li>
                <li><b>60% : 20% : 20%</b> - Untuk dataset yang kompleks dengan banyak variasi</li>
            </ul>
        </div>
        """)) 