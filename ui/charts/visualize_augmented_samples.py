"""
File: smartcash/ui/charts/visualize_augmented_samples.py
Deskripsi: Utilitas untuk menampilkan sampel dataset yang telah diaugmentasi dengan dukungan format denominasi
"""
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import os

from smartcash.dataset.utils.denomination_utils import extract_info_from_filename
from smartcash.dataset.utils.data_utils import load_image
from smartcash.ui.utils.file_utils import shorten_filename, find_label_path
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.helpers.ui_helpers import display_label_info

def visualize_augmented_samples(images_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 5):
    """Visualisasi sampel dataset yang telah diaugmentasi dengan dukungan format denominasi."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Get augmentation prefix
    aug_prefix = "aug"
    if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
        aug_prefix = ui_components['aug_options'].children[2].value
    
    # Ambil gambar augmentasi dengan format denominasi
    image_files = list(images_dir.glob(f'{aug_prefix}_*.jpg'))
    if not image_files:
        display(create_info_alert(f"Tidak ada file gambar augmentasi ditemukan di {images_dir}", "warning"))
        return
    
    # Batasi jumlah sampel
    import random
    image_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Tampilkan deskripsi
    display(create_info_alert(f"Menampilkan {len(image_files)} sampel dataset yang telah diaugmentasi", "info"))
    
    # Visualisasi sampel
    fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
    if len(image_files) == 1: axes = [axes]
        
    for i, img_path in enumerate(image_files):
        try:
            img = load_image(img_path)
            
            # Ekstrak info dari nama file
            file_info = extract_info_from_filename(img_path.stem)
            
            # Tampilkan gambar
            axes[i].imshow(img)
            
            # Tampilkan nama file dan denominasi
            short_name = shorten_filename(img_path.name)
            title = short_name
            
            # Tambahkan informasi denominasi jika tersedia
            if file_info.get('is_valid') and 'denomination' in file_info:
                title += f"\n({file_info['denomination']})"
                
            axes[i].set_title(title)
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan informasi label jika tersedia
    labels = []
    for img_path in image_files:
        label_path = find_label_path(img_path)
        if label_path:
            labels.append((img_path, label_path))
    
    if labels:
        # Tampilkan informasi label dengan display_label_info standar
        display_label_info(labels, images_dir.parent.parent / 'labels')
        
    # Tampilkan informasi denominasi dari nama file
    display(HTML(f"<h4 style='color:{COLORS['dark']}'>Informasi Denominasi</h4>"))
    
    for img_path in image_files:
        # Ekstrak info dari nama file
        file_info = extract_info_from_filename(img_path.stem)
        
        if file_info.get('is_valid'):
            # Format informasi
            file_name = shorten_filename(img_path.name, 30)
            denomination = file_info.get('denomination', 'unknown')
            variation = file_info.get('variation', '')
            
            # Tampilkan informasi
            variation_info = f" (Variasi {variation})" if variation else ""
            
            display(HTML(f"""
            <div style="margin:5px 0; padding:5px; border-left:3px solid {COLORS['primary']};">
                <p style="margin:0; color:{COLORS['dark']};"><strong>{file_name}</strong>: Denominasi {denomination.upper()}{variation_info}</p>
            </div>
            """))
        else:
            # Format informasi untuk file yang tidak dikenali
            file_name = shorten_filename(img_path.name, 30)
            display(HTML(f"""
            <div style="margin:5px 0; padding:5px; border-left:3px solid {COLORS['warning']};">
                <p style="margin:0; color:{COLORS['dark']};"><strong>{file_name}</strong>: Format tidak dikenali</p>
            </div>
            """))