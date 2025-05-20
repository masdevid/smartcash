"""
File: smartcash/ui/dataset/visualization/handlers/tabs/bbox_tab.py
Deskripsi: Handler untuk tab analisis bounding box
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

def on_bbox_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol analisis bounding box.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat analisis bounding box...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    bbox_tab = visualization_components.get('bbox_tab', {})
    output = bbox_tab.get('output')
    
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
            _show_dummy_bbox_analysis(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Analisis statistik bbox untuk semua split
            train_stats = explorer_service.analyze_bbox_statistics('train')
            val_stats = explorer_service.analyze_bbox_statistics('val')
            test_stats = explorer_service.analyze_bbox_statistics('test')
            
            # Tampilkan visualisasi
            _show_bbox_analysis(output, train_stats, val_stats, test_stats)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Analisis bounding box berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan analisis bounding box: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_bbox_analysis(output: widgets.Output) -> None:
    """
    Tampilkan analisis bounding box dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat data dummy untuk ukuran bbox
        size_categories = ['Small', 'Medium', 'Large']
        train_sizes = [300, 500, 200]
        val_sizes = [80, 120, 50]
        test_sizes = [100, 150, 60]
        
        # Buat DataFrame untuk visualisasi ukuran
        df_size = pd.DataFrame({
            'Ukuran': size_categories,
            'Train': train_sizes,
            'Validation': val_sizes,
            'Test': test_sizes,
            'Total': [sum(x) for x in zip(train_sizes, val_sizes, test_sizes)]
        })
        
        # Plot distribusi ukuran bbox
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(size_categories))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df_size['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df_size['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df_size['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Ukuran Bounding Box (Data Dummy)', fontsize=16)
        ax.set_xlabel('Ukuran', fontsize=12)
        ax.set_ylabel('Jumlah Bounding Box', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(size_categories)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi ukuran bbox
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df_size['Total'], labels=size_categories, autopct='%1.1f%%', 
               colors=['#4285F4', '#FBBC05', '#34A853'])
        ax.set_title('Persentase Bounding Box per Ukuran', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Buat data dummy untuk aspect ratio bbox
        aspect_categories = ['Tall', 'Square', 'Wide']
        train_aspects = [250, 400, 350]
        val_aspects = [60, 100, 90]
        test_aspects = [80, 120, 110]
        
        # Buat DataFrame untuk visualisasi aspect ratio
        df_aspect = pd.DataFrame({
            'Aspect Ratio': aspect_categories,
            'Train': train_aspects,
            'Validation': val_aspects,
            'Test': test_aspects,
            'Total': [sum(x) for x in zip(train_aspects, val_aspects, test_aspects)]
        })
        
        # Plot distribusi aspect ratio bbox
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df_aspect['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df_aspect['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df_aspect['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Aspect Ratio Bounding Box (Data Dummy)', fontsize=16)
        ax.set_xlabel('Aspect Ratio', fontsize=12)
        ax.set_ylabel('Jumlah Bounding Box', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(aspect_categories)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(widgets.HTML("<h3>Distribusi Ukuran Bounding Box</h3>"))
        display(df_size)
        display(widgets.HTML("<h3>Distribusi Aspect Ratio Bounding Box</h3>"))
        display(df_aspect)
        
        # Tampilkan informasi tambahan
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #4285F4;">
            <h3 style="margin-top: 0;">Informasi Bounding Box</h3>
            <p>Analisis bounding box membantu memahami karakteristik objek dalam dataset:</p>
            <ul>
                <li><b>Ukuran:</b>
                    <ul>
                        <li><b>Small:</b> Area < 32x32 piksel (relatif terhadap gambar 640x640)</li>
                        <li><b>Medium:</b> Area antara 32x32 dan 96x96 piksel</li>
                        <li><b>Large:</b> Area > 96x96 piksel</li>
                    </ul>
                </li>
                <li><b>Aspect Ratio:</b>
                    <ul>
                        <li><b>Tall:</b> Tinggi > Lebar (rasio > 1.5)</li>
                        <li><b>Square:</b> Tinggi ≈ Lebar (rasio antara 0.67 dan 1.5)</li>
                        <li><b>Wide:</b> Lebar > Tinggi (rasio < 0.67)</li>
                    </ul>
                </li>
            </ul>
            <p>Dataset yang baik memiliki variasi ukuran dan aspect ratio yang seimbang.</p>
        </div>
        """))

def _show_bbox_analysis(
    output: widgets.Output, 
    train_stats: Dict[str, Any], 
    val_stats: Dict[str, Any], 
    test_stats: Dict[str, Any]
) -> None:
    """
    Tampilkan analisis bounding box dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        train_stats: Hasil analisis statistik bbox untuk train split
        val_stats: Hasil analisis statistik bbox untuk validation split
        test_stats: Hasil analisis statistik bbox untuk test split
    """
    with output:
        clear_output(wait=True)
        
        # Dapatkan data statistik bbox
        train_bbox_stats = train_stats.get('bbox_statistics', {})
        val_bbox_stats = val_stats.get('bbox_statistics', {})
        test_bbox_stats = test_stats.get('bbox_statistics', {})
        
        # Siapkan data untuk ukuran bbox
        size_categories = ['Small', 'Medium', 'Large']
        train_sizes = [
            train_bbox_stats.get('small', 0),
            train_bbox_stats.get('medium', 0),
            train_bbox_stats.get('large', 0)
        ]
        val_sizes = [
            val_bbox_stats.get('small', 0),
            val_bbox_stats.get('medium', 0),
            val_bbox_stats.get('large', 0)
        ]
        test_sizes = [
            test_bbox_stats.get('small', 0),
            test_bbox_stats.get('medium', 0),
            test_bbox_stats.get('large', 0)
        ]
        
        # Buat DataFrame untuk visualisasi ukuran
        df_size = pd.DataFrame({
            'Ukuran': size_categories,
            'Train': train_sizes,
            'Validation': val_sizes,
            'Test': test_sizes,
            'Total': [sum(x) for x in zip(train_sizes, val_sizes, test_sizes)]
        })
        
        # Plot distribusi ukuran bbox
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(size_categories))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df_size['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df_size['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df_size['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Ukuran Bounding Box', fontsize=16)
        ax.set_xlabel('Ukuran', fontsize=12)
        ax.set_ylabel('Jumlah Bounding Box', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(size_categories)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Plot pie chart untuk distribusi ukuran bbox
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df_size['Total'], labels=size_categories, autopct='%1.1f%%', 
               colors=['#4285F4', '#FBBC05', '#34A853'])
        ax.set_title('Persentase Bounding Box per Ukuran', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Siapkan data untuk aspect ratio bbox
        aspect_categories = ['Tall', 'Square', 'Wide']
        train_aspects = [
            train_bbox_stats.get('tall', 0),
            train_bbox_stats.get('square', 0),
            train_bbox_stats.get('wide', 0)
        ]
        val_aspects = [
            val_bbox_stats.get('tall', 0),
            val_bbox_stats.get('square', 0),
            val_bbox_stats.get('wide', 0)
        ]
        test_aspects = [
            test_bbox_stats.get('tall', 0),
            test_bbox_stats.get('square', 0),
            test_bbox_stats.get('wide', 0)
        ]
        
        # Buat DataFrame untuk visualisasi aspect ratio
        df_aspect = pd.DataFrame({
            'Aspect Ratio': aspect_categories,
            'Train': train_aspects,
            'Validation': val_aspects,
            'Test': test_aspects,
            'Total': [sum(x) for x in zip(train_aspects, val_aspects, test_aspects)]
        })
        
        # Plot distribusi aspect ratio bbox
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df_aspect['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df_aspect['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df_aspect['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Aspect Ratio Bounding Box', fontsize=16)
        ax.set_xlabel('Aspect Ratio', fontsize=12)
        ax.set_ylabel('Jumlah Bounding Box', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(aspect_categories)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(widgets.HTML("<h3>Distribusi Ukuran Bounding Box</h3>"))
        display(df_size)
        display(widgets.HTML("<h3>Distribusi Aspect Ratio Bounding Box</h3>"))
        display(df_aspect)
        
        # Tampilkan informasi tambahan
        display(widgets.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid #4285F4;">
            <h3 style="margin-top: 0;">Informasi Bounding Box</h3>
            <p>Analisis bounding box membantu memahami karakteristik objek dalam dataset:</p>
            <ul>
                <li><b>Ukuran:</b>
                    <ul>
                        <li><b>Small:</b> Area < 32x32 piksel (relatif terhadap gambar 640x640)</li>
                        <li><b>Medium:</b> Area antara 32x32 dan 96x96 piksel</li>
                        <li><b>Large:</b> Area > 96x96 piksel</li>
                    </ul>
                </li>
                <li><b>Aspect Ratio:</b>
                    <ul>
                        <li><b>Tall:</b> Tinggi > Lebar (rasio > 1.5)</li>
                        <li><b>Square:</b> Tinggi ≈ Lebar (rasio antara 0.67 dan 1.5)</li>
                        <li><b>Wide:</b> Lebar > Tinggi (rasio < 0.67)</li>
                    </ul>
                </li>
            </ul>
            <p>Dataset yang baik memiliki variasi ukuran dan aspect ratio yang seimbang.</p>
        </div>
        """)) 