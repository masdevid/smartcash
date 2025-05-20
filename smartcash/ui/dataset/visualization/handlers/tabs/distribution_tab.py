"""
File: smartcash/ui/dataset/visualization/handlers/tabs/distribution_tab.py
Deskripsi: Handler untuk tab distribusi kelas
"""

import os
from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.common.config.manager import get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.handlers.status_handlers import (
    show_loading_status, show_success_status, show_error_status, show_warning_status
)

logger = get_logger(__name__)

def on_distribution_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi kelas.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Tampilkan loading status
    show_loading_status(ui_components, "Memuat visualisasi distribusi kelas...")
    
    # Dapatkan output widget
    visualization_components = ui_components.get('visualization_components', {})
    distribution_tab = visualization_components.get('distribution_tab', {})
    output = distribution_tab.get('output')
    
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
            _show_dummy_class_distribution(output)
        else:
            # Dapatkan explorer service
            explorer_service = get_dataset_service(service_name='explorer')
            
            # Analisis distribusi kelas untuk semua split
            train_distribution = explorer_service.analyze_class_distribution('train')
            val_distribution = explorer_service.analyze_class_distribution('val')
            test_distribution = explorer_service.analyze_class_distribution('test')
            
            # Tampilkan visualisasi
            _show_class_distribution(output, train_distribution, val_distribution, test_distribution)
        
        # Tampilkan pesan sukses
        show_success_status(ui_components, "Visualisasi distribusi kelas berhasil ditampilkan")
    
    except Exception as e:
        error_message = f"Error saat menampilkan distribusi kelas: {str(e)}"
        logger.error(f"{ICONS.get('error', '❌')} {error_message}")
        show_error_status(ui_components, error_message)

def _show_dummy_class_distribution(output: widgets.Output) -> None:
    """
    Tampilkan distribusi kelas dummy.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
    """
    with output:
        clear_output(wait=True)
        
        # Buat data dummy
        classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
        train_counts = [120, 150, 180, 200, 170, 190, 160]
        val_counts = [30, 40, 45, 50, 40, 45, 40]
        test_counts = [40, 50, 55, 60, 50, 55, 50]
        
        # Buat DataFrame untuk visualisasi
        df = pd.DataFrame({
            'Kelas': classes,
            'Train': train_counts,
            'Validation': val_counts,
            'Test': test_counts
        })
        
        # Plot distribusi kelas
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(classes))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Kelas Dataset (Data Dummy)', fontsize=16)
        ax.set_xlabel('Kelas', fontsize=12)
        ax.set_ylabel('Jumlah Gambar', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df)

def _show_class_distribution(
    output: widgets.Output, 
    train_distribution: Dict[str, Any], 
    val_distribution: Dict[str, Any], 
    test_distribution: Dict[str, Any]
) -> None:
    """
    Tampilkan distribusi kelas dari data aktual.
    
    Args:
        output: Widget output untuk menampilkan visualisasi
        train_distribution: Hasil analisis distribusi kelas untuk train split
        val_distribution: Hasil analisis distribusi kelas untuk validation split
        test_distribution: Hasil analisis distribusi kelas untuk test split
    """
    with output:
        clear_output(wait=True)
        
        # Dapatkan data distribusi kelas
        train_classes = train_distribution.get('class_distribution', {})
        val_classes = val_distribution.get('class_distribution', {})
        test_classes = test_distribution.get('class_distribution', {})
        
        # Gabungkan semua kelas unik
        all_classes = sorted(set(list(train_classes.keys()) + 
                               list(val_classes.keys()) + 
                               list(test_classes.keys())))
        
        # Siapkan data untuk DataFrame
        data = []
        for cls in all_classes:
            data.append({
                'Kelas': cls,
                'Train': train_classes.get(cls, 0),
                'Validation': val_classes.get(cls, 0),
                'Test': test_classes.get(cls, 0),
                'Total': train_classes.get(cls, 0) + val_classes.get(cls, 0) + test_classes.get(cls, 0)
            })
        
        # Buat DataFrame
        df = pd.DataFrame(data)
        
        # Plot distribusi kelas
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.25
        index = range(len(all_classes))
        
        # Plot batang untuk setiap split
        ax.bar([i - bar_width for i in index], df['Train'], bar_width, label='Train', color='#4285F4')
        ax.bar(index, df['Validation'], bar_width, label='Validation', color='#FBBC05')
        ax.bar([i + bar_width for i in index], df['Test'], bar_width, label='Test', color='#34A853')
        
        # Konfigurasi plot
        ax.set_title('Distribusi Kelas Dataset', fontsize=16)
        ax.set_xlabel('Kelas', fontsize=12)
        ax.set_ylabel('Jumlah Gambar', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(all_classes, rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan tabel
        display(df) 