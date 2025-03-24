"""
File: smartcash/ui/charts/compare_original_vs_preprocessed.py
Deskripsi: Utilitas untuk menampilkan komparasi gambar original dengan gambar preprocessed dengan dukungan denominasi
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output, HTML
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import os

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.dataset.utils.denomination_utils import extract_info_from_filename
from smartcash.dataset.utils.data_utils import load_image

def compare_original_vs_preprocessed(ui_components: Dict[str, Any], raw_dir: str, preprocessed_dir: str, num_samples: int = 3):
    """
    Komparasi sampel dataset original dengan yang telah dipreprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        raw_dir: Direktori dataset original
        preprocessed_dir: Direktori dataset preprocessed
        num_samples: Jumlah sampel yang akan divisualisasikan
    """
    output_widget = ui_components.get('status')
    if not output_widget:
        return
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mencari pasangan gambar untuk komparasi..."))
        
        # Cari sampel dari train split (atau split pertama yang tersedia)
        preprocessed_train_dir = Path(preprocessed_dir) / 'train'
        raw_train_dir = Path(raw_dir) / 'train'
        
        # Coba split lain jika train tidak tersedia
        if not preprocessed_train_dir.exists() or not raw_train_dir.exists():
            for split in ['valid', 'test']:
                preprocessed_split = Path(preprocessed_dir) / split
                raw_split = Path(raw_dir) / split
                
                if preprocessed_split.exists() and raw_split.exists():
                    preprocessed_train_dir = preprocessed_split
                    raw_train_dir = raw_split
                    break
            else:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan split yang cocok untuk komparasi"))
                return
        
        # Dapatkan direktori gambar
        preprocessed_images_dir = preprocessed_train_dir / 'images'
        raw_images_dir = raw_train_dir / 'images'
        
        if not preprocessed_images_dir.exists() or not raw_images_dir.exists():
            display(create_status_indicator('warning', f"{ICONS['warning']} Direktori gambar tidak lengkap untuk komparasi"))
            return
        
        # Dapatkan file prefix dari ui_components
        file_prefix = "rp"
        if 'preprocess_options' in ui_components and len(ui_components['preprocess_options'].children) > 4:
            file_prefix = ui_components['preprocess_options'].children[4].value
        
        # Cari gambar preprocessed dengan format denominasi
        preprocessed_images = list(preprocessed_images_dir.glob(f'{file_prefix}_*.jpg')) + list(preprocessed_images_dir.glob(f'{file_prefix}_*.png'))
        
        if not preprocessed_images:
            display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada file gambar preprocessed dengan format denominasi"))
            return
        
        # Ambil sampel acak
        import random
        selected_samples = random.sample(preprocessed_images, min(len(preprocessed_images), num_samples))
        
        # Tampilkan komparasi
        display(create_info_alert(f"Komparasi {len(selected_samples)} sampel: mentah vs preprocessed", "info"))
        
        # Visualisasi komparasi
        fig, axes = plt.subplots(len(selected_samples), 2, figsize=(10, 4*len(selected_samples)))
        if len(selected_samples) == 1:
            axes = axes.reshape(1, 2)
            
        # Cari dan tampilkan pasangan gambar
        pairs_info = []
        for i, proc_path in enumerate(selected_samples):
            try:
                # Ekstrak info dari nama file preprocessed
                proc_info = extract_info_from_filename(proc_path.stem)
                
                # Tampilkan file preprocessed
                proc_img = load_image(proc_path)
                axes[i, 1].imshow(proc_img)
                axes[i, 1].set_title(f"Preprocessed: {proc_path.name}")
                axes[i, 1].axis('off')
                
                # Cari file mentah yang cocok berdasarkan denominasi
                raw_files = list(raw_images_dir.glob('*.jpg')) + list(raw_images_dir.glob('*.png'))
                raw_file = random.choice(raw_files) if raw_files else None
                
                if raw_file:
                    # Tampilkan file mentah
                    raw_img = load_image(raw_file)
                    axes[i, 0].imshow(raw_img)
                    axes[i, 0].set_title(f"Mentah: {raw_file.name}")
                    axes[i, 0].axis('off')
                    
                    # Simpan info untuk ditampilkan nanti
                    pairs_info.append({
                        'raw_path': raw_file,
                        'proc_path': proc_path,
                        'denomination': proc_info.get('denomination', 'unknown') if proc_info.get('is_valid') else 'unknown',
                        'raw_img': raw_img,
                        'proc_img': proc_img
                    })
                else:
                    axes[i, 0].text(0.5, 0.5, "Tidak ada file mentah yang cocok", ha='center', va='center')
                    axes[i, 0].axis('off')
            except Exception as e:
                axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i, 0].axis('off')
                axes[i, 1].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan informasi detail untuk setiap pasangan
        for pair in pairs_info:
            try:
                raw_file = pair['raw_path']
                proc_file = pair['proc_path']
                denomination = pair['denomination']
                raw_img = pair['raw_img']
                proc_img = pair['proc_img']
                
                raw_h, raw_w = raw_img.shape[:2]
                proc_h, proc_w = proc_img.shape[:2]
                
                # Tampilkan info dengan formatting yang bagus
                display(HTML(f"""
                <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                    <p style="color:{COLORS['dark']};"><strong>File:</strong> {proc_file.name}</p>
                    <p style="color:{COLORS['dark']};"><strong>Denominasi:</strong> {denomination}</p>
                    <p style="color:{COLORS['dark']};">Dimensi: {raw_w}x{raw_h} piksel → {proc_w}x{proc_h} piksel</p>
                    <p style="color:{COLORS['dark']};">Rasio: {(proc_w*proc_h)/(raw_w*raw_h):.2f}x</p>
                </div>
                """))
            except Exception:
                pass