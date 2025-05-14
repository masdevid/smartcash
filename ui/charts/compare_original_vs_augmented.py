"""
File: smartcash/ui/charts/compare_original_vs_augmented.py
Deskripsi: Utilitas untuk menampilkan komparasi gambar original dengan gambar augmentasi dengan dukungan format denominasi
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import re
import os
import cv2
import numpy as np

from smartcash.dataset.utils.denomination_utils import extract_info_from_filename
from smartcash.dataset.utils.data_utils import load_image
from smartcash.ui.utils.file_utils import shorten_filename
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.io import format_size

def match_original_with_augmented(aug_filename: str, original_files: List[Path]) -> Optional[Path]:
    """
    Mencari file original yang cocok dengan file augmentasi berdasarkan format denominasi.
    
    Format file augmentasi: aug_rp_100k_uuid_var_1.jpg
    Format file original: rp_100k_uuid.jpg
    
    Args:
        aug_filename: Nama file augmentasi
        original_files: List file original
        
    Returns:
        Path file original yang cocok atau None jika tidak ditemukan
    """
    # Pattern untuk file augmentasi
    pattern = r'aug_(?P<prefix>\w+)_(?P<denomination>\w+)_(?P<uuid>[^_]+)_var_(?P<variation>\d+)'
    match = re.match(pattern, Path(aug_filename).stem)
    
    if not match:
        return None
    
    # Ekstrak informasi dari augmented file
    orig_prefix = match.group('prefix')
    denomination = match.group('denomination')
    uuid = match.group('uuid')
    
    # Cari file original yang cocok
    original_pattern = f"{orig_prefix}_{denomination}_{uuid}"
    
    for orig_file in original_files:
        if orig_file.stem.startswith(original_pattern):
            return orig_file
    
    return None

def compare_original_vs_augmented(original_dir: Path, augmented_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 3):
    """Komparasi sampel dataset asli dengan yang telah diaugmentasi dengan dukungan format denominasi."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Get augmentation prefix
    aug_prefix = "aug"
    if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
        aug_prefix = ui_components['aug_options'].children[2].value
    
    # Get original prefix
    orig_prefix = "rp"
    
    # Cari gambar augmentasi dan original
    augmented_images = list(augmented_dir.glob(f'{aug_prefix}_*.jpg'))
    original_images = list(original_dir.glob(f'{orig_prefix}_*.jpg'))
    
    if not augmented_images:
        display(create_info_alert(f"Tidak ada file gambar augmentasi ditemukan di {augmented_dir}", "warning"))
        return
        
    if not original_images:
        display(create_info_alert(f"Tidak ada file gambar original ditemukan di {original_dir}", "warning"))
        return
    
    # Cari pasangan yang cocok berdasarkan format denominasi
    matched_pairs = []
    
    for aug_file in augmented_images:
        # Cari file original yang cocok dengan format denominasi
        orig_file = match_original_with_augmented(aug_file.name, original_images)
        if orig_file:
            matched_pairs.append((orig_file, aug_file))
    
    # Jika tidak ada pasangan yang cocok, tampilkan pesan error
    if not matched_pairs:
        display(create_info_alert("Tidak ditemukan pasangan file yang cocok dengan format denominasi", "warning"))
        return
    
    # Batasi jumlah sampel
    if len(matched_pairs) > num_samples:
        import random
        matched_pairs = random.sample(matched_pairs, num_samples)
    
    # Tampilkan deskripsi
    display(create_info_alert(f"Komparasi {len(matched_pairs)} sampel: asli vs augmentasi", "info"))
    
    # Visualisasi komparasi
    fig, axes = plt.subplots(len(matched_pairs), 2, figsize=(10, 4*len(matched_pairs)))
    if len(matched_pairs) == 1: axes = axes.reshape(1, 2)
        
    for i, (orig_path, aug_path) in enumerate(matched_pairs):
        try:
            # Load gambar
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            # Tampilkan gambar dengan nama file yang dipersingkat
            orig_name = shorten_filename(orig_path.name)
            aug_name = shorten_filename(aug_path.name)
            
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {orig_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_img)
            axes[i, 1].set_title(f"Augmented: {aug_name}")
            axes[i, 1].axis('off')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan info perbandingan dan ekstrak denominasi
    for orig_path, aug_path in matched_pairs:
        try:
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            orig_h, orig_w = orig_img.shape[:2]
            aug_h, aug_w = aug_img.shape[:2]
            
            orig_size = os.path.getsize(orig_path)
            aug_size = os.path.getsize(aug_path)
            
            # Ekstrak denominasi dari nama file
            orig_info = extract_info_from_filename(orig_path.stem)
            aug_info = extract_info_from_filename(aug_path.stem)
            
            # Tentukan denominasi untuk ditampilkan
            denomination = "Unknown"
            if aug_info['is_valid'] and 'denomination' in aug_info:
                denomination = aug_info['denomination']
            elif orig_info['is_valid'] and 'denomination' in orig_info:
                denomination = orig_info['denomination']
            
            # Tampilkan nama file yang dipersingkat dan informasi detail
            orig_name = shorten_filename(orig_path.stem, 20)
            aug_name = shorten_filename(aug_path.stem, 20)
            
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['dark']};"><strong>Original:</strong> {orig_name} | <strong>Augmented:</strong> {aug_name}</p>
                <p style="color:{COLORS['dark']};"><strong>Denominasi:</strong> {denomination}</p>
                <p style="color:{COLORS['dark']};">Dimensi: {orig_w}×{orig_h} px → {aug_w}×{aug_h} px</p>
                <p style="color:{COLORS['dark']};">Ukuran: {format_size(orig_size)} → {format_size(aug_size)} ({aug_size/orig_size:.2f}×)</p>
            </div>
            """))
        except Exception as e:
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['danger']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['danger']};"><strong>Error saat memproses perbandingan:</strong> {str(e)}</p>
                <p style="color:{COLORS['dark']};"><strong>Original:</strong> {orig_path.name}</p>
                <p style="color:{COLORS['dark']};"><strong>Augmented:</strong> {aug_path.name}</p>
            </div>
            """))