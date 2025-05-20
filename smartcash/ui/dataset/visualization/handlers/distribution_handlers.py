"""
File: smartcash/ui/dataset/visualization/handlers/distribution_handlers.py
Deskripsi: Handler untuk tab distribusi kelas di visualisasi dataset
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

def setup_distribution_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab distribusi kelas.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Dapatkan komponen tab distribusi
    distribution_tab = ui_components.get('visualization_components', {}).get('distribution_tab', {})
    
    # Setup handler untuk tombol distribusi kelas
    if distribution_tab and 'button' in distribution_tab:
        distribution_tab['button'].on_click(
            lambda b: on_distribution_button_click(b, ui_components)
        )
    
    return ui_components

def get_class_distribution(dataset_path: str) -> Tuple[Dict[str, int], bool]:
    """
    Dapatkan distribusi kelas dari dataset.
    
    Args:
        dataset_path: Path ke dataset
        
    Returns:
        Tuple berisi dictionary distribusi kelas dan flag data dummy
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
        
        # Hitung jumlah objek per kelas
        class_counts = {}
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(labels_path, split)
            if os.path.exists(split_path):
                for label_file in os.listdir(split_path):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(split_path, label_file), 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    class_name = f"Kelas {class_id}"
                                    if class_name not in class_counts:
                                        class_counts[class_name] = 0
                                    class_counts[class_name] += 1
        
        # Jika tidak ada data, gunakan data dummy
        if not class_counts:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak ada data kelas ditemukan")
            return {}, True
            
        return class_counts, False
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil distribusi kelas: {str(e)}")
        return {}, True

def get_dummy_class_distribution() -> Dict[str, int]:
    """
    Dapatkan distribusi kelas dummy untuk visualisasi.
    
    Returns:
        Dictionary berisi distribusi kelas dummy
    """
    classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
    counts = [120, 150, 180, 200, 170, 160, 140]
    
    class_counts = {}
    for i, cls in enumerate(classes):
        class_counts[cls] = counts[i]
        
    return class_counts

def plot_class_distribution(class_counts: Dict[str, int], output: widgets.Output, is_dummy: bool = False) -> None:
    """
    Plot distribusi kelas.
    
    Args:
        class_counts: Dictionary berisi jumlah objek per kelas
        output: Widget output untuk menampilkan plot
        is_dummy: Flag apakah menggunakan data dummy
    """
    with output:
        clear_output(wait=True)
        
        # Tampilkan peringatan jika menggunakan data dummy
        if is_dummy:
            display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menggunakan data dummy untuk visualisasi"))
        
        # Plot distribusi kelas
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
        
        # Tambahkan nilai di atas bar
        for bar, count in zip(bars, class_counts.values()):
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
        df = pd.DataFrame(list(class_counts.items()), columns=['Kelas', 'Jumlah Objek'])
        df['Persentase'] = df['Jumlah Objek'] / df['Jumlah Objek'].sum() * 100
        df['Persentase'] = df['Persentase'].round(2).astype(str) + '%'
        display(df)

def on_distribution_button_click(button: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol distribusi kelas.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        distribution_tab = ui_components.get('visualization_components', {}).get('distribution_tab', {})
        output = distribution_tab.get('output')
        
        if not output:
            logger.error(f"{ICONS.get('error', '❌')} Output widget tidak ditemukan")
            return
        
        # Tampilkan loading status
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi kelas..."))
        
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
        
        # Dapatkan distribusi kelas
        class_counts, is_dummy = get_class_distribution(dataset_path)
        
        # Jika data dummy, gunakan data dummy
        if is_dummy:
            class_counts = get_dummy_class_distribution()
        
        # Plot distribusi kelas
        plot_class_distribution(class_counts, output, is_dummy)
        
    except Exception as e:
        if output:
            with output:
                clear_output(wait=True)
                logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi kelas: {str(e)}")
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}")) 