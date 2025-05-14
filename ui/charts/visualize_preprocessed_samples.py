"""
File: smartcash/ui/charts/visualize_preprocessed_samples.py
Deskripsi: Utilitas untuk menampilkan sampel dataset yang telah dipreprocessing dengan dukungan format denominasi
"""
from smartcash.ui.utils.constants import COLORS, ICONS 
from IPython.display import display, clear_output, HTML
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

from smartcash.dataset.utils.denomination_utils import extract_info_from_filename, get_denomination_label
from smartcash.dataset.utils.data_utils import load_image

def visualize_preprocessed_samples(ui_components: Dict[str, Any], preprocessed_dir: str, original_dir: str, num_samples: int = 5):
    """
    Visualisasi sampel dataset yang telah dipreprocessing dengan dukungan format denominasi.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        original_dir: Direktori dataset mentah
        num_samples: Jumlah sampel yang akan divisualisasikan
    """
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
    output_widget = ui_components.get('status')
    if not output_widget:
        return
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mengambil sampel dari dataset..."))
        
        # Cari sampel dari train split
        train_dir = Path(preprocessed_dir) / 'train'
        if not train_dir.exists():
            # Coba split lain jika train tidak tersedia
            for split in ['valid', 'test']:
                split_dir = Path(preprocessed_dir) / split
                if split_dir.exists():
                    train_dir = split_dir
                    break
            else:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada split yang tersedia di {preprocessed_dir}"))
                return
        
        # Dapatkan sampel gambar
        images_dir = train_dir / 'images'
        if not images_dir.exists():
            display(create_status_indicator('warning', f"{ICONS['warning']} Direktori gambar tidak ditemukan di {train_dir}"))
            return
            
        # Dapatkan file prefix dari ui_components
        file_prefix = "rp"
        if 'preprocess_options' in ui_components and len(ui_components['preprocess_options'].children) > 4:
            file_prefix = ui_components['preprocess_options'].children[4].value
            
        # Ambil semua gambar dengan format denominasi
        image_files = list(images_dir.glob(f'{file_prefix}_*.jpg')) + list(images_dir.glob(f'{file_prefix}_*.png')) + list(images_dir.glob(f'{file_prefix}_*.npy'))
        if not image_files:
            display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada file gambar ditemukan di {images_dir}"))
            return
            
        # Batasi jumlah sampel
        import random
        image_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Tampilkan deskripsi dengan create_info_alert standar
        display(create_info_alert(
            f"Menampilkan {len(image_files)} sampel dataset yang telah dipreprocessing dari {train_dir.name}",
            "info"
        ))
        
        # Visualisasi sampel
        fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
        if len(image_files) == 1:
            axes = [axes]
            
        for i, img_path in enumerate(image_files):
            # Load gambar
            try:
                img = load_image(img_path)
                
                # Ekstrak info dari nama file
                file_info = extract_info_from_filename(img_path.stem)
                
                # Tampilkan gambar
                axes[i].imshow(img)
                
                # Tampilkan nama file yang pendek
                img_name = img_path.name
                
                # Tambahkan informasi denominasi jika tersedia
                title = f"{img_name[:10]}...{img_name[-7:]}"
                if file_info.get('is_valid') and 'denomination' in file_info:
                    title += f"\n({file_info['denomination']})"
                
                axes[i].set_title(title)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan informasi ukuran gambar
        for img_path in image_files:
            try:
                img = load_image(img_path)
                h, w = img.shape[:2]
                
                # Ekstrak info dari nama file
                file_info = extract_info_from_filename(img_path.stem)
                
                # Tampilkan info dengan denominasi
                denomination = file_info.get('denomination', 'unknown') if file_info.get('is_valid') else 'unknown'
                
                display(HTML(f"<p style='color:{COLORS['dark']}'><strong>{img_path.name}</strong>: {w}x{h} piksel | Denominasi: {denomination}</p>"))
            except Exception:
                pass