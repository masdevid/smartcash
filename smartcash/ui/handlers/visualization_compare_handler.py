"""
File: smartcash/ui/handlers/visualization_compare_handler.py
Deskripsi: Handler visualisasi perbandingan gambar asli dan hasil augmentasi
"""

from typing import Dict, Any, Optional, List
from IPython.display import display, clear_output
from pathlib import Path
import ipywidgets as widgets
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_box

# Import fungsi helper dari visualization_sample_handler
from smartcash.ui.handlers.visualization_sample_handler import (
    display_no_data_message, display_visualization_status, find_valid_image_directory
)

def compare_original_vs_processed(data_dir: str, processed_dir: str, output_widget: widgets.Output, 
                                 num_samples: int = 3, aug_prefix: str = None, logger=None):
    """
    Komparasi gambar original vs processed dengan pendekatan yang lebih efisien.
    
    Args:
        data_dir: Direktori data original
        processed_dir: Direktori data processed
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang akan ditampilkan
        aug_prefix: Prefix untuk gambar augmentasi (jika ada)
        logger: Logger untuk logging
    """
    try:
        import cv2
        
        # Validasi direktori
        original_dir = Path(data_dir)
        processed_dir = Path(processed_dir)
        
        # Cari direktori gambar yang valid
        original_img_dirs = [
            original_dir / 'images',
            original_dir / 'train' / 'images',
            original_dir
        ]
        processed_img_dirs = [
            processed_dir / 'images',
            processed_dir / 'train' / 'images',
            processed_dir
        ]
        
        # Temukan direktori yang valid
        valid_original_dir = find_valid_image_directory(original_img_dirs)
        valid_processed_dir = find_valid_image_directory(processed_img_dirs)
        
        # Validasi direktori
        if not valid_original_dir:
            with output_widget:
                display_no_data_message(output_widget, message="Direktori gambar original tidak ditemukan.")
            return
        
        if not valid_processed_dir:
            with output_widget:
                display_no_data_message(output_widget, message="Direktori gambar processed tidak ditemukan.")
            return
        
        # Cari file gambar di direktori original
        original_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            original_files.extend(list(valid_original_dir.glob(f"*{ext}")))
        
        # Validasi jumlah gambar
        if not original_files:
            with output_widget:
                display_no_data_message(output_widget, message=f"Tidak ada gambar ditemukan di {valid_original_dir}")
            return
        
        # Pilih sampel secara acak
        num_samples = min(num_samples, len(original_files))
        sample_files = random.sample(original_files, num_samples)
        
        # Tampilkan status
        with output_widget:
            display_visualization_status(output_widget, status_type="info", 
                                      title="Memuat gambar untuk perbandingan...", 
                                      messages=[f"Direktori original: {valid_original_dir}", 
                                               f"Direktori processed: {valid_processed_dir}",
                                               f"Jumlah sampel: {num_samples}"])
        
        # Buat figure dan axes
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, 2)
        
        # Temukan file yang sesuai di direktori processed
        processed_files = []
        for orig_file in sample_files:
            # Cari file dengan nama yang sama di direktori processed
            if aug_prefix:
                # Jika ada aug_prefix, cari file dengan prefix tersebut
                processed_file = list(valid_processed_dir.glob(f"{aug_prefix}*{orig_file.name}"))
                if not processed_file:
                    # Jika tidak ditemukan, cari file dengan nama yang sama
                    processed_file = list(valid_processed_dir.glob(f"*{orig_file.name}"))
            else:
                # Jika tidak ada aug_prefix, cari file dengan nama yang sama
                processed_file = list(valid_processed_dir.glob(f"*{orig_file.name}"))
            
            # Jika tidak ditemukan, gunakan None
            if processed_file:
                processed_files.append(processed_file[0])
            else:
                processed_files.append(None)
        
        # Fungsi untuk load dan display gambar
        def load_and_display_row(row_idx, img_paths):
            orig_path, proc_path = img_paths
            
            # Load dan tampilkan gambar original
            try:
                if orig_path:
                    img = cv2.imread(str(orig_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[row_idx, 0].imshow(img)
                        axes[row_idx, 0].set_title(f"Original: {orig_path.name}")
                    else:
                        axes[row_idx, 0].text(0.5, 0.5, f"Error loading {orig_path.name}", ha='center', va='center')
                else:
                    axes[row_idx, 0].text(0.5, 0.5, "No image found", ha='center', va='center')
                axes[row_idx, 0].axis('off')
            except Exception as e:
                axes[row_idx, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[row_idx, 0].axis('off')
            
            # Load dan tampilkan gambar processed
            try:
                if proc_path:
                    img = cv2.imread(str(proc_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[row_idx, 1].imshow(img)
                        axes[row_idx, 1].set_title(f"Processed: {proc_path.name}")
                    else:
                        axes[row_idx, 1].text(0.5, 0.5, f"Error loading {proc_path.name}", ha='center', va='center')
                else:
                    axes[row_idx, 1].text(0.5, 0.5, "No processed image found", ha='center', va='center')
                axes[row_idx, 1].axis('off')
            except Exception as e:
                axes[row_idx, 1].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[row_idx, 1].axis('off')
        
        # Load dan tampilkan gambar secara paralel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            list(executor.map(lambda x: load_and_display_row(x[0], (x[1], processed_files[x[0]])), 
                            enumerate(sample_files)))
        
        # Tampilkan plot
        plt.tight_layout()
        with output_widget:
            clear_output(wait=True)
            plt.show()
            
            # Tampilkan status
            display_visualization_status(output_widget, status_type="success", 
                                      title="Perbandingan Gambar Original vs Processed", 
                                      messages=[f"Direktori original: {valid_original_dir}", 
                                               f"Direktori processed: {valid_processed_dir}",
                                               f"Jumlah sampel: {num_samples}"])
    except Exception as e:
        with output_widget:
            display_visualization_status(output_widget, status_type="error", 
                                      title="Error saat perbandingan gambar", 
                                      messages=[str(e)])
