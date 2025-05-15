"""
File: smartcash/ui/handlers/visualization_sample_handler.py
Deskripsi: Handler visualisasi sampel gambar dataset
"""

from typing import Dict, Any, Optional, List, Union
from IPython.display import display, clear_output
from pathlib import Path
import ipywidgets as widgets
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert, create_info_box
from smartcash.ui.handlers.status_handler import update_status_panel

def display_no_data_message(output_widget, title="Tidak Ada Data", message="", detail_message=""):
    """
    Menampilkan pesan informasi ketika tidak ada data yang tersedia dengan format yang konsisten.
    
    Args:
        output_widget: Widget output untuk menampilkan pesan
        title: Judul pesan
        message: Pesan utama
        detail_message: Pesan detail tambahan (opsional)
    """
    with output_widget:
        clear_output(wait=True)
        full_message = f"<strong>{title}</strong>"
        if message:
            full_message += f"<br>{message}"
        if detail_message:
            full_message += f"<br><small>{detail_message}</small>"
        
        # Menggunakan create_info_alert dari alert_utils
        alert = create_info_alert(full_message, alert_type="warning")
        display(alert)

def find_valid_image_directory(base_dirs, extensions=None):
    """
    Mencari direktori gambar yang valid dari daftar direktori potensial.
    
    Args:
        base_dirs: List direktori yang akan dicari
        extensions: List ekstensi file yang dicari (default: ['.jpg', '.jpeg', '.png'])
        
    Returns:
        Path ke direktori yang valid atau None jika tidak ditemukan
    """
    from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
    extensions = extensions or [ext[1:] for ext in IMG_EXTENSIONS]  # Hapus * dari '*.jpg'
    
    # Gunakan list comprehension dan any untuk memeriksa direktori
    valid_dirs = [Path(d) if not isinstance(d, Path) else d for d in base_dirs if isinstance(d, (str, Path))]
    valid_dirs = [d for d in valid_dirs if d.exists() and any(any(d.glob(f"*{ext}")) for ext in extensions)]
    
    # Return direktori pertama yang valid atau None jika tidak ada
    return valid_dirs[0] if valid_dirs else None

def find_valid_label_directory(base_dirs):
    """
    Mencari direktori label yang valid dari daftar direktori potensial.
    
    Args:
        base_dirs: List direktori yang akan dicari
        
    Returns:
        Path ke direktori yang valid atau None jika tidak ditemukan
    """
    # Konversi ke Path dan filter direktori yang valid dengan file .txt
    valid_dirs = [Path(d) if not isinstance(d, Path) else d for d in base_dirs if isinstance(d, (str, Path))]
    valid_dirs = [d for d in valid_dirs if d.exists() and any(d.glob("*.txt"))]
    
    # Return direktori pertama yang valid atau None jika tidak ada
    return valid_dirs[0] if valid_dirs else None

def display_visualization_status(output_widget, status_type="info", title="", messages: Optional[List[str]]=None):
    """
    Menampilkan status visualisasi dengan format yang konsisten.
    
    Args:
        output_widget: Widget output untuk menampilkan status
        status_type: Tipe status ("info", "success", "warning", "error")
        title: Judul status
        messages: List pesan yang akan ditampilkan
    """
    if messages is None:
        messages = []
    
    # Buat konten HTML untuk pesan
    content = ""
    if messages:
        content = "<ul style='margin-top: 8px; padding-left: 20px;'>"
        content += "".join([f"<li>{msg}</li>" for msg in messages])
        content += "</ul>"
    
    # Tampilkan info box dengan status yang sesuai
    with output_widget:
        clear_output(wait=True)
        display(create_info_box(title, content, box_type=status_type))

def visualize_images_with_annotations(output_widget, image_dir, label_dir=None, num_samples=5, show_labels=True, 
                                     fig_size=(15, 10), random_seed=None, class_names=None, logger=None, aug_prefix=None):
    """
    Menampilkan gambar dengan anotasi bounding box.
    
    Args:
        output_widget: Widget output untuk menampilkan gambar
        image_dir: Direktori gambar
        label_dir: Direktori label (opsional)
        num_samples: Jumlah sampel yang akan ditampilkan
        show_labels: Apakah menampilkan label
        fig_size: Ukuran figure
        random_seed: Seed untuk random sampling
        class_names: Nama kelas untuk label
        logger: Logger untuk logging
        aug_prefix: Prefix untuk gambar augmentasi
    """
    try:
        import cv2
        import matplotlib.patches as patches
        
        # Validasi direktori gambar
        image_dir = Path(image_dir)
        if not image_dir.exists():
            with output_widget:
                display_no_data_message(output_widget, message=f"Direktori gambar tidak ditemukan: {image_dir}")
            return
        
        # Cari file gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
        
        # Filter berdasarkan aug_prefix jika ada
        if aug_prefix:
            image_files = [f for f in image_files if aug_prefix in f.name]
        
        # Validasi jumlah gambar
        if not image_files:
            with output_widget:
                display_no_data_message(output_widget, message=f"Tidak ada gambar ditemukan di {image_dir}")
            return
        
        # Set random seed jika ada
        if random_seed is not None:
            random.seed(random_seed)
        
        # Pilih sampel secara acak
        num_samples = min(num_samples, len(image_files))
        sample_files = random.sample(image_files, num_samples)
        
        # Tampilkan status
        with output_widget:
            display_visualization_status(output_widget, status_type="info", 
                                      title="Memuat gambar...", 
                                      messages=[f"Direktori: {image_dir}", f"Jumlah sampel: {num_samples}"])
        
        # Buat figure dan axes
        fig, axes = plt.subplots(1, num_samples, figsize=fig_size)
        if num_samples == 1:
            axes = [axes]
        
        # Tampilkan gambar dan anotasi
        for i, (img_file, ax) in enumerate(zip(sample_files, axes)):
            # Load gambar
            img = cv2.imread(str(img_file))
            if img is None:
                ax.text(0.5, 0.5, f"Error loading {img_file.name}", ha='center', va='center')
                continue
            
            # Konversi BGR ke RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Tampilkan gambar
            ax.imshow(img)
            
            # Tampilkan anotasi jika ada
            if show_labels and label_dir:
                # Cari file label yang sesuai
                label_file = Path(label_dir) / f"{img_file.stem}.txt"
                if label_file.exists():
                    # Baca label
                    with open(label_file, 'r') as f:
                        lines = f.read().strip().split('\n')
                    
                    # Tampilkan bounding box
                    h, w = img.shape[:2]
                    for line in lines:
                        if not line.strip():
                            continue
                        
                        try:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            
                            # Parse YOLO format (class_id, x_center, y_center, width, height)
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * w
                            y_center = float(parts[2]) * h
                            box_width = float(parts[3]) * w
                            box_height = float(parts[4]) * h
                            
                            # Hitung koordinat untuk rectangle
                            x = x_center - box_width / 2
                            y = y_center - box_height / 2
                            
                            # Tambahkan rectangle
                            rect = patches.Rectangle((x, y), box_width, box_height, 
                                                   linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            
                            # Tambahkan label kelas jika tersedia
                            if class_names and class_id < len(class_names):
                                class_label = class_names[class_id]
                            else:
                                class_label = f"Class {class_id}"
                            
                            ax.text(x, y, class_label, color='white', 
                                   bbox=dict(facecolor='red', alpha=0.7))
                        except Exception as e:
                            if logger:
                                logger.warning(f"{ICONS['warning']} Error parsing label: {str(e)}")
            
            # Set judul
            ax.set_title(img_file.name)
            ax.axis('off')
        
        # Tampilkan plot
        plt.tight_layout()
        with output_widget:
            clear_output(wait=True)
            plt.show()
            
            # Tampilkan status
            display_visualization_status(output_widget, status_type="success", 
                                      title="Visualisasi Sampel", 
                                      messages=[f"Direktori gambar: {image_dir}", 
                                               f"Jumlah sampel: {num_samples}",
                                               f"Anotasi: {'Ditampilkan' if show_labels and label_dir else 'Tidak ditampilkan'}"])
    except Exception as e:
        with output_widget:
            display_visualization_status(output_widget, status_type="error", 
                                      title="Error saat visualisasi", 
                                      messages=[str(e)])

def visualize_samples(target_dir, output_widget: widgets.Output, num_samples: int = 4, aug_prefix: str = None):
    """
    Visualisasi sampel gambar dari direktori dengan pendekatan yang lebih efisien.
    
    Args:
        target_dir: Direktori target (Path atau str)
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang akan ditampilkan
        aug_prefix: Prefix untuk gambar augmentasi (jika ada)
    """
    try:
        import cv2
        
        # Validasi direktori
        target_dir = Path(target_dir)
        if not target_dir.exists():
            with output_widget:
                display_no_data_message(output_widget, message=f"Direktori tidak ditemukan: {target_dir}")
            return
        
        # Cari file gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(target_dir.glob(f"*{ext}")))
        
        # Filter berdasarkan aug_prefix jika ada
        if aug_prefix:
            image_files = [f for f in image_files if aug_prefix in f.name]
        
        # Validasi jumlah gambar
        if not image_files:
            with output_widget:
                display_no_data_message(output_widget, message=f"Tidak ada gambar ditemukan di {target_dir}")
            return
        
        # Pilih sampel secara acak
        num_samples = min(num_samples, len(image_files))
        sample_files = random.sample(image_files, num_samples)
        
        # Tampilkan status
        with output_widget:
            display_visualization_status(output_widget, status_type="info", 
                                      title="Memuat gambar...", 
                                      messages=[f"Direktori: {target_dir}", f"Jumlah sampel: {num_samples}"])
        
        # Buat figure dan axes
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 10))
        if num_samples == 1:
            axes = [axes]
        
        # Fungsi untuk load dan display gambar
        def load_and_display(idx, img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(img)
                    axes[idx].set_title(img_path.name)
                else:
                    axes[idx].text(0.5, 0.5, f"Error loading {img_path.name}", ha='center', va='center')
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[idx].axis('off')
        
        # Load dan tampilkan gambar secara paralel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            list(executor.map(lambda x: load_and_display(x[0], x[1]), enumerate(sample_files)))
        
        # Tampilkan plot
        plt.tight_layout()
        with output_widget:
            clear_output(wait=True)
            plt.show()
            
            # Tampilkan status
            display_visualization_status(output_widget, status_type="success", 
                                      title="Visualisasi Sampel", 
                                      messages=[f"Direktori: {target_dir}", 
                                               f"Jumlah sampel: {num_samples}",
                                               f"Total gambar: {len(image_files)}"])
    except Exception as e:
        with output_widget:
            display_visualization_status(output_widget, status_type="error", 
                                      title="Error saat visualisasi", 
                                      messages=[str(e)])
