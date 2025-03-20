"""
File: smartcash/ui/dataset/augmentation_visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset hasil augmentasi dengan konsolidasi fungsi
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
    """Setup handler untuk visualisasi dataset augmentasi."""
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi dengan fungsi konsolidasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='single')

    # Handler untuk tombol komparasi dengan fungsi konsolidasi
    def on_compare_click(b):
        """Handler untuk komparasi sampel dataset asli dengan yang telah diaugmentasi."""
        visualize_dataset(ui_components, mode='compare')
    
    # Setup handlers untuk tombol visualisasi
    visualization_buttons = ui_components.get('visualization_buttons')
    if visualization_buttons and hasattr(visualization_buttons, 'children') and len(visualization_buttons.children) >= 2:
        visualization_buttons.children[0].on_click(on_visualize_click)
        visualization_buttons.children[1].on_click(on_compare_click)
    
    # Tambahkan handlers ke komponen UI
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'visualize_dataset': visualize_dataset
    })
    
    return ui_components

def visualize_dataset(ui_components: Dict[str, Any], mode: str = 'single', num_samples: int = 5):
    """
    Fungsi konsolidasi untuk visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        mode: Mode visualisasi ('single' untuk augmented saja, 'compare' untuk komparasi)
        num_samples: Jumlah sampel yang akan ditampilkan
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
    # Use the visualization container if available
    output_widget = ui_components.get('visualization_container', ui_components.get('status'))
    if not output_widget: return
    
    data_dir = ui_components.get('data_dir', 'data')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan {'komparasi' if mode == 'compare' else 'visualisasi'}..."))
        
        # Cek ketersediaan data
        if mode == 'compare' and (not os.path.exists(data_dir) or not os.path.exists(augmented_dir)):
            display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset tidak lengkap untuk komparasi"))
            return
        elif not os.path.exists(augmented_dir):
            display(create_status_indicator('warning', f"{ICONS['warning']} Dataset augmentasi tidak ditemukan di: {augmented_dir}"))
            return
            
        # Check for images directory
        augmented_images_dir = Path(augmented_dir) / 'images'
        if not augmented_images_dir.exists():
            display(create_status_indicator('warning', f"{ICONS['warning']} Direktori gambar tidak ditemukan di {augmented_dir}"))
            return
        
        # Load images for visualization
        if mode == 'single':
            # Visualisasi sampel augmented
            visualize_augmented_samples(augmented_images_dir, output_widget, ui_components, num_samples)
        else:
            # Komparasi original vs augmented
            original_images_dir = Path(data_dir) / 'train' / 'images'
            if not original_images_dir.exists():
                for split in ['valid', 'test']:
                    split_dir = Path(data_dir) / split / 'images'
                    if split_dir.exists():
                        original_images_dir = split_dir
                        break
                else:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori gambar asli tidak ditemukan"))
                    return
                
            compare_original_vs_augmented(original_images_dir, augmented_images_dir, output_widget, ui_components, num_samples)
        
        # Show visualization container
        if 'visualization_container' in ui_components:
            ui_components['visualization_container'].layout.display = 'block'

def load_image(img_path: Path) -> np.ndarray:
    """Fungsi helper untuk loading gambar dengan berbagai format."""
    if str(img_path).endswith('.npy'):
        # Handle numpy array
        img = np.load(str(img_path))
        # Denormalisasi jika perlu
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        # Handle gambar biasa
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualize_augmented_samples(images_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 5):
    """Visualisasi sampel dataset yang telah diaugmentasi."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Ambil semua gambar
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.npy'))
    if not image_files:
        display(create_info_alert(f"Tidak ada file gambar ditemukan di {images_dir}", "warning"))
        return
    
    # Filter for only augmented images
    aug_images = [img for img in image_files if 'aug' in img.name]
    if aug_images:
        image_files = aug_images
        
    # Batasi jumlah sampel
    image_files = image_files[:min(num_samples, len(image_files))]
    
    # Tampilkan deskripsi
    display(create_info_alert(f"Menampilkan {len(image_files)} sampel dataset yang telah diaugmentasi", "info"))
    
    # Visualisasi sampel
    fig, axes = plt.subplots(1, len(image_files), figsize=(4*len(image_files), 4))
    if len(image_files) == 1: axes = [axes]
        
    for i, img_path in enumerate(image_files):
        try:
            img = load_image(img_path)
            axes[i].imshow(img)
            img_name = img_path.name
            if len(img_name) > 10: img_name = f"...{img_name[-10:]}"
            axes[i].set_title(f"{img_name}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan informasi label jika tersedia
    labels_dir = Path(images_dir).parent / 'labels'
    display_label_info(image_files, labels_dir)

def compare_original_vs_augmented(original_dir: Path, augmented_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 3):
    """Komparasi sampel dataset asli dengan yang telah diaugmentasi."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Cari gambar yang memiliki pasangan augmentasi
    original_images = list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.png'))
    augmented_images = list(augmented_dir.glob('*.jpg')) + list(augmented_dir.glob('*.png')) + list(augmented_dir.glob('*.npy'))
    
    aug_prefixes = set(img.stem.split('_aug_')[0] if '_aug_' in img.stem else '' for img in augmented_images)
    
    # Cari pasangan gambar
    matched_pairs = []
    for orig_img in original_images:
        if orig_img.stem in aug_prefixes:
            # Cari semua augmentasi untuk gambar ini
            aug_matches = [img for img in augmented_images if img.stem.startswith(f"{orig_img.stem}_aug_")]
            if aug_matches:
                matched_pairs.append((orig_img, aug_matches[0]))  # Ambil augmentasi pertama
                if len(matched_pairs) >= num_samples:
                    break
    
    # Fallback jika tidak menemukan pasangan yang cocok
    if not matched_pairs:
        display(create_info_alert("Tidak menemukan pasangan gambar yang cocok. Menggunakan sampel acak...", "warning"))
        for orig_img in original_images[:num_samples]:
            for aug_img in augmented_images:
                if 'aug' in aug_img.name:
                    matched_pairs.append((orig_img, aug_img))
                    break
    
    if not matched_pairs:
        display(create_info_alert("Tidak dapat menemukan pasangan gambar untuk komparasi", "error"))
        return
    
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
            
            # Tampilkan gambar
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"Original: {orig_path.name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_img)
            axes[i, 1].set_title(f"Augmented: {aug_path.name}")
            axes[i, 1].axis('off')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Tampilkan info perbandingan
    for orig_path, aug_path in matched_pairs:
        try:
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            orig_h, orig_w = orig_img.shape[:2]
            aug_h, aug_w = aug_img.shape[:2]
            
            orig_size = os.path.getsize(orig_path)
            aug_size = os.path.getsize(aug_path)
            
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['dark']};"><strong>{orig_path.stem}</strong></p>
                <p style="color:{COLORS['dark']};">Original: {orig_w}×{orig_h} px | Augmented: {aug_w}×{aug_h} px</p>
                <p style="color:{COLORS['dark']};">Size ratio: {aug_size/orig_size:.2f}× ({format_size(orig_size)} → {format_size(aug_size)})</p>
            </div>
            """))
        except Exception:
            pass

def display_label_info(image_files: List[Path], labels_dir: Path):
    """Tampilkan informasi label untuk gambar-gambar yang dipilih."""
    if not labels_dir.exists(): return
    
    matched_labels = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            matched_labels.append((img_file, label_file))
    
    if matched_labels:
        display(HTML(f"<h4 style='color:{COLORS['dark']}'>Informasi Label</h4>"))
        for img_file, label_file in matched_labels:
            try:
                with open(label_file, 'r') as f:
                    label_lines = f.read().strip().splitlines()
                
                num_boxes = len(label_lines)
                classes = set()
                for line in label_lines:
                    parts = line.split()
                    if parts: classes.add(parts[0])
                
                display(HTML(f"""
                <div style="margin:5px 0; padding:5px; border-left:3px solid {COLORS['primary']};">
                    <p style="margin:0; color:{COLORS['dark']};"><strong>{img_file.name}</strong>: {num_boxes} objek terdeteksi, {len(classes)} kelas</p>
                </div>
                """))
            except Exception:
                pass

def format_size(size_bytes: int) -> str:
    """Format ukuran file dalam bytes ke format yang mudah dibaca."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"