"""
File: smartcash/ui/dataset/preprocessing/handlers/visualization_compare_handler.py
Deskripsi: Handler untuk visualisasi perbandingan gambar asli dan hasil preprocessing
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
from tqdm.auto import tqdm

logger = get_logger("preprocessing_compare_visualization")

def find_matching_images(input_dir: Path, output_dir: Path, num_samples: int = 4) -> List[Tuple[Path, Path]]:
    """
    Temukan pasangan gambar yang cocok antara direktori input dan output.
    
    Args:
        input_dir: Direktori gambar asli
        output_dir: Direktori gambar hasil preprocessing
        num_samples: Jumlah sampel yang akan dikembalikan
        
    Returns:
        List pasangan (gambar_asli, gambar_preprocessed)
    """
    # Dapatkan daftar gambar di kedua direktori
    input_images = []
    output_images = []
    
    for ext in ['.jpg', '.jpeg', '.png']:
        input_images.extend(list(input_dir.glob(f"*{ext}")))
        output_images.extend(list(output_dir.glob(f"*{ext}")))
    
    # Buat dictionary untuk gambar output berdasarkan nama file
    output_dict = {img.stem: img for img in output_images}
    
    # Temukan pasangan yang cocok
    matching_pairs = []
    for input_img in input_images:
        if input_img.stem in output_dict:
            matching_pairs.append((input_img, output_dict[input_img.stem]))
    
    # Pilih sampel acak
    if len(matching_pairs) < num_samples:
        num_samples = len(matching_pairs)
        logger.info(f"{ICONS['info']} Hanya ada {num_samples} pasangan gambar yang cocok")
    
    if matching_pairs:
        return random.sample(matching_pairs, num_samples)
    else:
        return []

def visualize_comparison(ui_components: Dict[str, Any], 
                        num_samples: int = 4, 
                        split: Optional[str] = None,
                        figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Visualisasikan perbandingan gambar asli dan hasil preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        num_samples: Jumlah sampel yang akan ditampilkan
        split: Split dataset yang akan divisualisasikan (opsional)
        figsize: Ukuran gambar (lebar, tinggi)
    """
    try:
        # Dapatkan direktori dari UI components
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Tentukan direktori input dan output berdasarkan split
        input_dir = Path(data_dir) / 'images'
        if not input_dir.exists():
            # Coba cari di data_dir langsung
            input_dir = Path(data_dir)
        
        if split:
            output_dir = Path(preprocessed_dir) / split / 'images'
        else:
            output_dir = Path(preprocessed_dir) / 'images'
        
        # Pastikan kedua direktori ada
        if not input_dir.exists() or not output_dir.exists():
            missing_dir = input_dir if not input_dir.exists() else output_dir
            logger.warning(f"{ICONS['warning']} Direktori tidak ditemukan: {missing_dir}")
            with ui_components.get('visualization_container', widgets.Output()):
                clear_output(wait=True)
                display(widgets.HTML(f"""
                <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;">
                    <h4>{ICONS['warning']} Direktori tidak ditemukan</h4>
                    <p>Direktori tidak ditemukan: {missing_dir}</p>
                    <p>Pastikan Anda telah menjalankan preprocessing terlebih dahulu.</p>
                </div>
                """))
            return
        
        # Temukan pasangan gambar yang cocok
        matching_pairs = find_matching_images(input_dir, output_dir, num_samples)
        
        if not matching_pairs:
            logger.warning(f"{ICONS['warning']} Tidak ada pasangan gambar yang cocok")
            with ui_components.get('visualization_container', widgets.Output()):
                clear_output(wait=True)
                display(widgets.HTML(f"""
                <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;">
                    <h4>{ICONS['warning']} Tidak ada pasangan gambar yang cocok</h4>
                    <p>Tidak ditemukan pasangan gambar yang cocok antara direktori input dan output.</p>
                    <p>Input: {input_dir}</p>
                    <p>Output: {output_dir}</p>
                </div>
                """))
            return
        
        # Tampilkan perbandingan
        with ui_components.get('visualization_container', widgets.Output()):
            clear_output(wait=True)
            
            # Tampilkan header
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #d4edda; color: #155724; border-radius: 4px; margin-bottom: 15px;">
                <h3>{ICONS['chart']} Perbandingan Gambar Asli dan Hasil Preprocessing</h3>
                <p>Menampilkan {len(matching_pairs)} pasangan gambar dari total {len(list(input_dir.glob('*.*')))} gambar</p>
                <p><b>Input:</b> {input_dir}</p>
                <p><b>Output:</b> {output_dir}</p>
            </div>
            """))
            
            # Plot gambar
            fig, axes = plt.subplots(2, len(matching_pairs), figsize=figsize)
            
            # Jika hanya satu sampel, pastikan axes adalah array 2D
            if len(matching_pairs) == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            
            # Tampilkan gambar asli di baris pertama dan hasil preprocessing di baris kedua
            for i, (input_img, output_img) in enumerate(matching_pairs):
                # Gambar asli
                img_original = plt.imread(input_img)
                axes[0, i].imshow(img_original)
                axes[0, i].set_title(f"Asli: {input_img.name}", fontsize=10)
                axes[0, i].axis('off')
                
                # Gambar hasil preprocessing
                img_preprocessed = plt.imread(output_img)
                axes[1, i].imshow(img_preprocessed)
                axes[1, i].set_title(f"Preprocessed: {output_img.name}", fontsize=10)
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Tampilkan informasi tambahan
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-top: 15px;">
                <h4>{ICONS['info']} Informasi Perbandingan</h4>
                <ul>
                    <li><b>Direktori input:</b> {input_dir}</li>
                    <li><b>Direktori output:</b> {output_dir}</li>
                    <li><b>Split:</b> {split if split else 'Semua'}</li>
                </ul>
            </div>
            """))
            
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat visualisasi perbandingan: {str(e)}")
        with ui_components.get('visualization_container', widgets.Output()):
            clear_output(wait=True)
            display(widgets.HTML(f"""
            <div style="padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;">
                <h4>{ICONS['error']} Error</h4>
                <p>Error saat visualisasi perbandingan: {str(e)}</p>
            </div>
            """))

def setup_comparison_visualization_button(ui_components: Dict[str, Any]) -> None:
    """
    Setup tombol untuk visualisasi perbandingan gambar.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Dapatkan tombol visualisasi perbandingan
    compare_button = ui_components.get('visualize_compare_button')
    if not compare_button:
        logger.warning(f"{ICONS['warning']} Tombol visualisasi perbandingan tidak ditemukan")
        return
    
    # Dapatkan split selector jika ada
    split_selector = ui_components.get('split_selector')
    
    # Setup handler
    def on_visualize_compare_click(b):
        # Dapatkan split dari UI jika ada
        split = None
        if split_selector:
            split_option = split_selector.value
            split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
            split = split_map.get(split_option)
        
        # Visualisasikan perbandingan
        visualize_comparison(ui_components, split=split)
    
    # Tambahkan handler ke tombol
    compare_button.on_click(on_visualize_compare_click)
