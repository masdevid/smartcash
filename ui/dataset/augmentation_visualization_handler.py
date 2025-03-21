"""
File: smartcash/ui/dataset/augmentation_visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset hasil augmentasi dengan perbaikan algoritma pencocokan gambar
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_visualization_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk visualisasi dataset augmentasi dengan dukungan lokasi baru."""
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
    Fungsi visualisasi dataset yang mendukung lokasi baru augmentasi.
    
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
    
    # Dapatkan lokasi dari konfigurasi
    config = ui_components.get('config', {})
    
    # Dapatkan lokasi preprocessed (untuk mencari hasil augmentasi)
    preprocessed_dir = config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
    
    # Dapatkan lokasi temp augmented (untuk backup)
    augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
    
    # Siapkan lokasi yang akan dicari (prioritaskan preprocessed/train)
    primary_path = Path(preprocessed_dir) / 'train' / 'images'
    secondary_path = Path(augmented_dir) / 'images'
    
    with output_widget:
        clear_output(wait=True)
        display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan {'komparasi' if mode == 'compare' else 'visualisasi'}..."))
        
        # Periksa lokasi mana yang memiliki file hasil augmentasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Cek apakah ada file augmentasi di primary path
        primary_aug_files = list(primary_path.glob(f"{aug_prefix}_*.jpg")) if primary_path.exists() else []
        
        # Jika tidak ada di primary path, cek di secondary path
        active_path = primary_path
        if not primary_aug_files and secondary_path.exists():
            secondary_aug_files = list(secondary_path.glob(f"{aug_prefix}_*.jpg"))
            if secondary_aug_files:
                active_path = secondary_path
                display(create_info_alert(f"Menggunakan lokasi augmentasi temporary: {active_path}", "info"))
        
        # Jika mode compare, dapatkan lokasi data asli
        if mode == 'compare':
            # Cek apakah ada file original di lokasi preprocessed
            original_path = Path(preprocessed_dir) / 'train' / 'images'
            if not original_path.exists():
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset untuk komparasi tidak ditemukan"))
                return
            
            # Gunakan visualisasi komparasi
            compare_original_vs_augmented(original_path, active_path, output_widget, ui_components, num_samples)
        else:
            # Visualisasi sampel augmented
            visualize_augmented_samples(active_path, output_widget, ui_components, num_samples)
        
        # Show visualization container
        if 'visualization_container' in ui_components:
            ui_components['visualization_container'].layout.display = 'block'

def find_label_path(img_path: Path) -> Optional[Path]:
    """
    Fungsi helper untuk mencari label path dari image path.
    
    Args:
        img_path: Path gambar
        
    Returns:
        Path file label atau None jika tidak ditemukan
    """
    # Cek apakah ada file label di folder paralel
    parent_dir = img_path.parent.parent
    label_path = parent_dir / 'labels' / f"{img_path.stem}.txt"
    
    if label_path.exists():
        return label_path
    
    # Cek apakah ada file label di folder sibling
    sibling_label_dir = img_path.parent.parent / 'labels'
    if sibling_label_dir.exists():
        sibling_label_path = sibling_label_dir / f"{img_path.stem}.txt"
        if sibling_label_path.exists():
            return sibling_label_path
    
    return None

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

def shorten_filename(filename: str, max_length: int = 15) -> str:
    """
    Persingkat nama file dengan ellipsis untuk tampilan yang lebih baik.
    
    Args:
        filename: Nama file yang akan dipersingkat
        max_length: Panjang maksimum nama file
        
    Returns:
        Nama file yang telah dipersingkat
    """
    if len(filename) <= max_length:
        return filename
    
    # Potong nama file dengan format "awal...akhir"
    prefix_len = max_length // 2 - 1
    suffix_len = max_length - prefix_len - 3  # 3 untuk "..."
    
    return f"{filename[:prefix_len]}...{filename[-suffix_len:]}"

def visualize_augmented_samples(images_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 5):
    """Visualisasi sampel dataset yang telah diaugmentasi dengan peningkatan tampilan nama file."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Get augmentation prefix
    aug_prefix = "aug"
    if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
        aug_prefix = ui_components['aug_options'].children[2].value
    
    # Ambil gambar augmentasi
    image_files = list(images_dir.glob(f'{aug_prefix}_*.jpg'))
    if not image_files:
        display(create_info_alert(f"Tidak ada file gambar augmentasi ditemukan di {images_dir}", "warning"))
        return
    
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
            # Tampilkan nama file yang dipersingkat
            shortened_name = shorten_filename(img_path.name)
            axes[i].set_title(f"{shortened_name}")
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
        # Perbaiki pemanggilan display_label_info dengan labels_dir yang benar
        display_label_info(labels, images_dir.parent.parent / 'labels')


def compare_original_vs_augmented(original_dir: Path, augmented_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 3):
    """Komparasi sampel dataset asli dengan yang telah diaugmentasi dengan algoritma pencocokan yang lebih baik."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.common.utils import format_size
    
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
    
    # Metode baru: Pencocokan berdasarkan kelas dan penggunaan random aug file untuk setiap kelas
    class_to_orig_images = {}  # Dictionary kelas -> list gambar original
    class_to_aug_images = {}   # Dictionary kelas -> list gambar augmented
    
    # Identifikasi kelas dari nama file
    def extract_class_from_filename(filename: str, prefix: str) -> Optional[str]:
        """Ekstrak nama kelas dari nama file dengan pola prefix_class_uuid."""
        parts = filename.split('_')
        if len(parts) < 3 or parts[0] != prefix:
            return None
        
        # Kelas ada di tengah (prefix_class_uuid)
        # Untuk format dengan multiple underscore di nama kelas, ambil semua kecuali prefix dan uuid terakhir
        return '_'.join(parts[1:-1])  # Semua bagian tengah adalah nama kelas
    
    # Kelompokkan file original berdasarkan kelas
    for img_path in original_images:
        class_name = extract_class_from_filename(img_path.stem, orig_prefix)
        if class_name:
            if class_name not in class_to_orig_images:
                class_to_orig_images[class_name] = []
            class_to_orig_images[class_name].append(img_path)
    
    # Kelompokkan file augmented berdasarkan kelas
    for img_path in augmented_images:
        class_name = extract_class_from_filename(img_path.stem, aug_prefix)
        if class_name:
            if class_name not in class_to_aug_images:
                class_to_aug_images[class_name] = []
            class_to_aug_images[class_name].append(img_path)
    
    # Buat pasangan orig-aug dari kelas yang sama dengan memastikan aug berbeda-beda
    matched_pairs = []
    common_classes = set(class_to_orig_images.keys()) & set(class_to_aug_images.keys())
    
    # Filter hanya kelas dengan aug dan orig sama-sama tersedia
    for cls in common_classes:
        orig_files = class_to_orig_images[cls]
        aug_files = class_to_aug_images[cls]
        
        # Hanya ambil maksimal 1 file per kelas untuk meningkatkan variasi
        if orig_files and aug_files:
            import random
            orig_file = random.choice(orig_files)
            aug_file = random.choice(aug_files)
            matched_pairs.append((orig_file, aug_file))
            
        # Batasi jumlah sampel
        if len(matched_pairs) >= num_samples:
            break
    
    # Jika masih kurang sampel, tambahkan kelas lain
    if len(matched_pairs) < num_samples and len(common_classes) < len(class_to_aug_images):
        remaining_classes = set(class_to_aug_images.keys()) - common_classes
        for cls in remaining_classes:
            if cls in class_to_orig_images and cls in class_to_aug_images:
                orig_files = class_to_orig_images[cls]
                aug_files = class_to_aug_images[cls]
                
                if orig_files and aug_files:
                    import random
                    orig_file = random.choice(orig_files)
                    aug_file = random.choice(aug_files)
                    matched_pairs.append((orig_file, aug_file))
                
                # Batasi jumlah sampel
                if len(matched_pairs) >= num_samples:
                    break
    
    # Fallback jika masih tidak cukup pasangan: gunakan random pairing
    if len(matched_pairs) < min(num_samples, len(augmented_images)):
        display(create_info_alert("Pencocokan berdasarkan kelas kurang optimal, menggunakan pencocokan random...", "warning"))
        import random
        
        while len(matched_pairs) < min(num_samples, len(augmented_images), len(original_images)):
            orig_file = random.choice(original_images)
            aug_file = random.choice([img for img in augmented_images 
                                    if not any(img == pair[1] for pair in matched_pairs)])
            matched_pairs.append((orig_file, aug_file))
    
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
    
    # Tampilkan info perbandingan
    for orig_path, aug_path in matched_pairs:
        try:
            orig_img = load_image(orig_path)
            aug_img = load_image(aug_path)
            
            orig_h, orig_w = orig_img.shape[:2]
            aug_h, aug_w = aug_img.shape[:2]
            
            orig_size = os.path.getsize(orig_path)
            aug_size = os.path.getsize(aug_path)
            
            # Ekstrak dan tampilkan class dengan jelas
            orig_class = extract_class_from_filename(orig_path.stem, orig_prefix) or "Unknown"
            aug_class = extract_class_from_filename(aug_path.stem, aug_prefix) or "Unknown"
            
            # Tampilkan nama file yang dipersingkat
            orig_name = shorten_filename(orig_path.stem, 20)
            aug_name = shorten_filename(aug_path.stem, 20)
            
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['dark']};"><strong>Original:</strong> {orig_name} | <strong>Augmented:</strong> {aug_name}</p>
                <p style="color:{COLORS['dark']};"><strong>Kelas:</strong> {orig_class} → {aug_class}</p>
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

def display_label_info(images_with_labels: List[Tuple[Path, Path]], labels_dir: Path):
    """
    Tampilkan informasi label untuk gambar-gambar yang dipilih.
    
    Args:
        images_with_labels: List of tuples (image_path, label_path)
        labels_dir: Path direktori label (untuk kompatibilitas)
    """
    if not images_with_labels: return
    
    display(HTML(f"<h4 style='color:{COLORS['dark']}'>Informasi Label</h4>"))
    for img_file, label_file in images_with_labels:
        try:
            with open(label_file, 'r') as f:
                label_lines = f.read().strip().splitlines()
            
            num_boxes = len(label_lines)
            classes = set()
            for line in label_lines:
                parts = line.split()
                if parts: classes.add(parts[0])
            
            # Tampilkan nama file yang dipersingkat
            shortened_name = shorten_filename(img_file.name, 20)
            
            display(HTML(f"""
            <div style="margin:5px 0; padding:5px; border-left:3px solid {COLORS['primary']};">
                <p style="margin:0; color:{COLORS['dark']};"><strong>{shortened_name}</strong>: {num_boxes} objek terdeteksi, {len(classes)} kelas</p>
            </div>
            """))
        except Exception:
            pass