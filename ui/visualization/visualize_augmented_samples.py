"""
File: smartcash/ui/visualization/visualize_augmented_samples.py
Deskripsi: Utilitas untuk menampilkan sampel dataset yang telah diaugmentasi
"""
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML

def visualize_augmented_samples(images_dir: Path, output_widget, ui_components: Dict[str, Any], num_samples: int = 5):
    """Visualisasi sampel dataset yang telah diaugmentasi dengan peningkatan tampilan nama file."""
    from smartcash.ui.utils.alert_utils import create_info_alert
    import matplotlib.pyplot as plt
    from smartcash.ui.utils.file_utils import shorten_filename
    from smartcash.ui.helpers.ui_helpers import display_label_info
    
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