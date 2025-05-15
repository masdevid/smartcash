"""
File: smartcash/ui/dataset/preprocessing/handlers/visualization_sample_handler.py
Deskripsi: Handler untuk visualisasi sampel gambar pada preprocessing
"""

import os
import random
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("preprocessing_visualization")

def visualize_sample_images(ui_components: Dict[str, Any], 
                           num_samples: int = 4, 
                           split: Optional[str] = None,
                           figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Visualisasikan sampel gambar dari direktori preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        num_samples: Jumlah sampel yang akan ditampilkan
        split: Split dataset yang akan divisualisasikan (opsional)
        figsize: Ukuran gambar (lebar, tinggi)
    """
    try:
        # Dapatkan direktori output dari UI components
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Tentukan direktori gambar berdasarkan split
        if split:
            images_dir = Path(preprocessed_dir) / split / 'images'
        else:
            images_dir = Path(preprocessed_dir) / 'images'
        
        # Pastikan direktori ada
        if not images_dir.exists():
            logger.warning(f"{ICONS['warning']} Direktori gambar tidak ditemukan: {images_dir}")
            with ui_components.get('visualization_container', widgets.Output()):
                clear_output(wait=True)
                display(widgets.HTML(f"""
                <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;">
                    <h4>{ICONS['warning']} Direktori tidak ditemukan</h4>
                    <p>Direktori gambar tidak ditemukan: {images_dir}</p>
                    <p>Pastikan Anda telah menjalankan preprocessing terlebih dahulu.</p>
                </div>
                """))
            return
        
        # Dapatkan daftar gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
        
        if not image_files:
            logger.warning(f"{ICONS['warning']} Tidak ada gambar di direktori: {images_dir}")
            with ui_components.get('visualization_container', widgets.Output()):
                clear_output(wait=True)
                display(widgets.HTML(f"""
                <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;">
                    <h4>{ICONS['warning']} Tidak ada gambar</h4>
                    <p>Tidak ada gambar di direktori: {images_dir}</p>
                    <p>Pastikan Anda telah menjalankan preprocessing terlebih dahulu.</p>
                </div>
                """))
            return
        
        # Pilih sampel acak
        if len(image_files) < num_samples:
            num_samples = len(image_files)
            logger.info(f"{ICONS['info']} Hanya ada {num_samples} gambar tersedia")
        
        sample_files = random.sample(image_files, num_samples)
        
        # Tampilkan sampel
        with ui_components.get('visualization_container', widgets.Output()):
            clear_output(wait=True)
            
            # Tampilkan header
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #d4edda; color: #155724; border-radius: 4px; margin-bottom: 15px;">
                <h3>{ICONS['chart']} Sampel Gambar Hasil Preprocessing</h3>
                <p>Menampilkan {num_samples} sampel acak dari {len(image_files)} gambar di {images_dir}</p>
            </div>
            """))
            
            # Plot gambar
            fig, axes = plt.subplots(1, num_samples, figsize=figsize)
            if num_samples == 1:
                axes = [axes]
            
            for i, img_file in enumerate(sample_files):
                img = plt.imread(img_file)
                axes[i].imshow(img)
                axes[i].set_title(f"Sampel {i+1}\n{img_file.name}", fontsize=10)
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Tampilkan informasi tambahan
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-top: 15px;">
                <h4>{ICONS['info']} Informasi Gambar</h4>
                <ul>
                    <li><b>Direktori:</b> {images_dir}</li>
                    <li><b>Total gambar:</b> {len(image_files)}</li>
                    <li><b>Split:</b> {split if split else 'Semua'}</li>
                </ul>
            </div>
            """))
            
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat visualisasi sampel: {str(e)}")
        with ui_components.get('visualization_container', widgets.Output()):
            clear_output(wait=True)
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;">
                <h4>{ICONS['error']} Error</h4>
                <p>Error saat visualisasi sampel: {str(e)}</p>
            </div>
            """))
            
def setup_sample_visualization_button(ui_components: Dict[str, Any]) -> None:
    """
    Setup tombol untuk visualisasi sampel gambar.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Dapatkan tombol visualisasi sampel
    sample_button = ui_components.get('visualize_sample_button')
    if not sample_button:
        logger.warning(f"{ICONS['warning']} Tombol visualisasi sampel tidak ditemukan")
        return
    
    # Dapatkan split selector jika ada
    split_selector = ui_components.get('split_selector')
    
    # Setup handler
    def on_visualize_sample_click(b):
        # Dapatkan split dari UI jika ada
        split = None
        if split_selector:
            split_option = split_selector.value
            split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
            split = split_map.get(split_option)
        
        # Visualisasikan sampel
        visualize_sample_images(ui_components, split=split)
    
    # Tambahkan handler ke tombol
    sample_button.on_click(on_visualize_sample_click)
