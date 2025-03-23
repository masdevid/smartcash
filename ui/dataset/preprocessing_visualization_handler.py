"""
File: smartcash/ui/dataset/preprocessing_visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset preprocessing dengan integrasi standar dan perbaikan warna teks
"""

from typing import Dict, Any, Optional, List, Tuple
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_visualization_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi..."))
        
        # Dapatkan direktori dataset preprocessing
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        data_dir = ui_components.get('data_dir', 'data')
        
        # Cek apakah preprocessed dataset tersedia
        if not os.path.exists(preprocessed_dir):
            with ui_components['status']:
                display(create_status_indicator('warning', f"{ICONS['warning']} Dataset preprocessed tidak ditemukan di: {preprocessed_dir}"))
            return
            
        # Coba tampilkan visualisasi dengan error handling
        try:
            from smartcash.ui.visualization.visualize_preprocessed_samples import visualize_preprocessed_samples
            visualize_preprocessed_samples(ui_components, preprocessed_dir, data_dir)
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi dataset: {str(e)}")
    
    # Handler untuk tombol komparasi
    def on_compare_click(b):
        """Handler untuk komparasi sampel dataset mentah dengan yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi..."))
        
        # Dapatkan direktori dataset
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        data_dir = ui_components.get('data_dir', 'data')
        
        # Cek ketersediaan data
        if not os.path.exists(preprocessed_dir) or not os.path.exists(data_dir):
            with ui_components['status']:
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset tidak lengkap untuk komparasi"))
            return
            
        # Coba tampilkan visualisasi komparasi dengan error handling
        try:
            from smartcash.ui.visualization.compare_original_vs_preprocessed import compare_original_vs_preprocessed
            compare_original_vs_preprocessed(ui_components, data_dir, preprocessed_dir)
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Tambahkan handlers ke tombol jika tersedia
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_click)
    
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_click)
    
    from smartcash.ui.visualization.visualize_preprocessed_samples import visualize_preprocessed_samples
    from smartcash.ui.visualization.compare_original_vs_preprocessed import compare_original_vs_preprocessed
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'visualize_dataset': visualize_preprocessed_samples,
        'compare_datasets': compare_original_vs_preprocessed,
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click
    })

    return ui_components

