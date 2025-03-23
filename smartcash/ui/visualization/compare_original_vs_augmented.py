"""
File: smartcash/ui/visualization/compare_original_vs_augmented.py
Deskripsi: Utilitas untuk menampilkan komparasi gambar original dengan gambar augmentasi
"""

from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML
import matplotlib.pyplot as plt

def compare_original_vs_augmented(original_dir: Path, augmented_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 3):
    """Komparasi sampel dataset asli dengan yang telah diaugmentasi dengan algoritma pencocokan yang lebih baik."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.common.utils import format_size
    from smartcash.ui.utils.file_utils import shorten_filename, find_matching_pairs, load_image
    from smartcash.ui.utils.constants import COLORS, ICONS 
    
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
    
    # Metode yang lebih baik: cari pasangan berdasarkan UUID
    matched_pairs = find_matching_pairs(augmented_images, original_images, orig_prefix, aug_prefix)
    
    # Jika tidak ada pasangan yang cocok, gunakan strategi pencocokan old style
    if not matched_pairs:
        display(create_info_alert("Tidak ditemukan pasangan file yang cocok, menggunakan pencocokan berdasarkan kelas...", "warning"))
        
        # Metode pencocokan berdasarkan kelas
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
            # Format baru: aug_rp_class_uuid_var1
            # Ekstrak class name dari augmentasi (parts[2])
            parts = img_path.stem.split('_')
            if len(parts) >= 5 and parts[0] == aug_prefix and parts[1] == orig_prefix:
                class_name = parts[2]
                if class_name not in class_to_aug_images:
                    class_to_aug_images[class_name] = []
                class_to_aug_images[class_name].append(img_path)
        
        # Buat pasangan orig-aug dari kelas yang sama
        matched_class_pairs = []
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
                matched_class_pairs.append((orig_file, aug_file))
            
            # Batasi jumlah sampel
            if len(matched_class_pairs) >= num_samples:
                break
        
        # Gunakan hasil dari pencocokan kelas jika ada
        if matched_class_pairs:
            matched_pairs = matched_class_pairs
    
    # Batasi jumlah sampel
    if len(matched_pairs) > num_samples:
        import random
        matched_pairs = random.sample(matched_pairs, num_samples)
    
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
            # Format baru: aug_rp_class_uuid_var1
            parts = aug_path.stem.split('_')
            if len(parts) >= 5 and parts[0] == aug_prefix and parts[1] == orig_prefix:
                class_name = parts[2]
            else:
                class_name = "Unknown"
            
            # Tampilkan nama file yang dipersingkat dan informasi detail
            orig_name = shorten_filename(orig_path.stem, 20)
            aug_name = shorten_filename(aug_path.stem, 20)
            
            display(HTML(f"""
            <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                <p style="color:{COLORS['dark']};"><strong>Original:</strong> {orig_name} | <strong>Augmented:</strong> {aug_name}</p>
                <p style="color:{COLORS['dark']};"><strong>Kelas:</strong> {class_name}</p>
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