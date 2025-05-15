"""
File: smartcash/ui/handlers/visualization_handler.py
Deskripsi: Handler visualisasi yang dapat digunakan bersama untuk berbagai modul dataset
"""

from typing import Dict, Any, Optional, Callable, List, Union
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
from smartcash.ui.handlers.status_handler import create_status_panel, update_status_panel

# Import handler terpisah untuk SRP
from smartcash.ui.handlers.visualization_sample_handler import (
    visualize_samples, visualize_images_with_annotations,
    display_no_data_message, display_visualization_status,
    find_valid_image_directory, find_valid_label_directory
)
from smartcash.ui.handlers.visualization_compare_handler import compare_original_vs_processed

# Fungsi helper untuk visualisasi
def display_no_data_message(output_widget, title="Tidak Ada Data", message="", detail_message=""):
    """
    Menampilkan pesan informasi ketika tidak ada data yang tersedia dengan format yang konsisten.
    Menggunakan create_info_alert dari alert_utils untuk konsistensi UI.
    
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
    Menggunakan one-liner style dan utilitas dari dataset_constants.
    
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
    Menggunakan one-liner style untuk kode yang lebih ringkas.
    
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
    Menggunakan create_info_box dari alert_utils untuk tampilan yang lebih informatif.
    
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
        content = "<ul style='margin-top: 5px; margin-bottom: 0;'>"
        for msg in messages:
            content += f"<li>{msg}</li>"
        content += "</ul>"
    
    # Gunakan create_info_box dari alert_utils untuk tampilan yang lebih baik
    with output_widget:
        clear_output(wait=True)
        # Jika tidak ada title tapi ada messages, gunakan message pertama sebagai title
        if not title and messages:
            title = messages[0]
            content = "" if len(messages) == 1 else "<ul style='margin-top: 5px; margin-bottom: 0;'>"
            for i, msg in enumerate(messages[1:], 1):
                content += f"<li>{msg}</li>"
            content += "</ul>" if len(messages) > 1 else ""
        
        # Jika konten kosong, gunakan create_info_alert yang lebih sederhana
        if not content:
            alert = create_info_alert(title, alert_type=status_type)
            display(alert)
        else:
            # Gunakan create_info_box untuk tampilan yang lebih informatif
            info_box = create_info_box(title, content, style=status_type)
            display(info_box)

def setup_visualization_handlers(ui_components: Dict[str, Any], module_name: str, env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset yang dapat digunakan bersama.
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul (augmentation, preprocessing, dll)
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    
    # Buat handler untuk visualisasi sampel
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset."""
        # Dapatkan widget output untuk visualisasi
        visualization_container = ui_components.get('visualization_container')
        if not visualization_container:
            if logger: logger.warning(f"{ICONS['warning']} Visualization container tidak ditemukan")
            return
        
        # Tampilkan status awal
        with visualization_container:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan visualisasi sampel..."))
        
        try:
            # Dapatkan direktori dataset
            data_dir = ui_components.get('data_dir', 'data')
            
            # Tentukan direktori berdasarkan modul
            if module_name == 'augmentation':
                # Untuk augmentasi, gunakan direktori augmented
                target_dir = ui_components.get('augmented_dir', 'data/augmented')
                aug_prefix = ui_components.get('aug_prefix', 'aug')
            elif module_name == 'preprocessing':
                # Untuk preprocessing, gunakan direktori preprocessed
                target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
                aug_prefix = None
            elif module_name == 'split':
                # Untuk split, gunakan direktori split
                target_dir = ui_components.get('split_dir', 'data/split')
                aug_prefix = None
            else:
                # Default ke data_dir
                target_dir = data_dir
                aug_prefix = None
            
            # Validasi direktori
            if not os.path.exists(target_dir):
                with visualization_container:
                    clear_output(wait=True)
                    display_no_data_message(visualization_container, 
                                          message=f"Direktori {target_dir} tidak ditemukan.",
                                          detail_message="Silakan proses dataset terlebih dahulu.")
                return
            
            # Cari direktori gambar yang valid
            img_dirs = [
                Path(target_dir) / 'images',
                Path(target_dir) / 'train' / 'images',
                Path(target_dir)
            ]
            
            # Cari direktori label yang valid
            label_dirs = [
                Path(target_dir) / 'labels',
                Path(target_dir) / 'train' / 'labels'
            ]
            
            # Temukan direktori yang valid
            valid_img_dir = find_valid_image_directory(img_dirs)
            valid_label_dir = find_valid_label_directory(label_dirs)
            
            # Validasi direktori gambar
            if not valid_img_dir:
                with visualization_container:
                    clear_output(wait=True)
                    display_no_data_message(visualization_container, 
                                          message=f"Tidak ditemukan gambar di {target_dir}",
                                          detail_message="Silakan proses dataset terlebih dahulu.")
                return
            
            # Tampilkan status
            with visualization_container:
                clear_output(wait=True)
                display_visualization_status(visualization_container, status_type="info", 
                                          title="Memuat gambar...", 
                                          messages=[f"Direktori: {valid_img_dir}", 
                                                   f"Anotasi: {'Tersedia' if valid_label_dir else 'Tidak tersedia'}"])
            
            # Dapatkan class names jika tersedia
            class_names = None
            if 'class_names' in ui_components:
                class_names = ui_components['class_names']
            
            # Tampilkan gambar dengan anotasi jika tersedia
            if valid_label_dir:
                # Gunakan visualize_images_with_annotations
                visualize_images_with_annotations(
                    visualization_container, 
                    valid_img_dir, 
                    valid_label_dir,
                    num_samples=5,
                    show_labels=True,
                    class_names=class_names,
                    logger=logger,
                    aug_prefix=aug_prefix
                )
            else:
                # Gunakan visualize_samples
                visualize_samples(
                    valid_img_dir, 
                    visualization_container,
                    num_samples=4,
                    aug_prefix=aug_prefix
                )
            
        except Exception as e:
            # Tampilkan error
            with visualization_container:
                clear_output(wait=True)
                display_visualization_status(visualization_container, status_type="error", 
                                          title="Error saat visualisasi", 
                                          messages=[str(e)])
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi: {str(e)}")
    
    # Buat handler untuk komparasi
    def on_compare_click(b):
        """Handler untuk komparasi dataset original vs processed."""
        # Dapatkan widget output untuk visualisasi
        visualization_container = ui_components.get('visualization_container')
        if not visualization_container:
            if logger: logger.warning(f"{ICONS['warning']} Visualization container tidak ditemukan")
            return
        
        # Tampilkan status awal
        with visualization_container:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mempersiapkan komparasi dataset..."))
        
        try:
            # Dapatkan direktori dataset
            data_dir = ui_components.get('data_dir', 'data')
            
            # Tentukan direktori berdasarkan modul
            if module_name == 'augmentation':
                # Untuk augmentasi, bandingkan original vs augmented
                processed_dir = ui_components.get('augmented_dir', 'data/augmented')
                aug_prefix = ui_components.get('aug_prefix', 'aug')
            elif module_name == 'preprocessing':
                # Untuk preprocessing, bandingkan original vs preprocessed
                processed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
                aug_prefix = None
            else:
                # Default ke data_dir
                processed_dir = data_dir
                aug_prefix = None
            
            # Validasi direktori
            if not os.path.exists(data_dir) or not os.path.exists(processed_dir):
                with visualization_container:
                    clear_output(wait=True)
                    display_no_data_message(visualization_container, 
                                          message=f"Direktori tidak ditemukan: {data_dir} atau {processed_dir}",
                                          detail_message="Silakan proses dataset terlebih dahulu.")
                return
            
            # Gunakan compare_original_vs_processed
            compare_original_vs_processed(
                data_dir, 
                processed_dir, 
                visualization_container,
                num_samples=3,
                aug_prefix=aug_prefix,
                logger=logger
            )
            
        except Exception as e:
            # Tampilkan error
            with visualization_container:
                clear_output(wait=True)
                display_visualization_status(visualization_container, status_type="error", 
                                          title="Error saat komparasi", 
                                          messages=[str(e)])
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi: {str(e)}")
    
    # Attach handler ke tombol jika tersedia
    if 'visualize_button' in ui_components and hasattr(ui_components['visualize_button'], 'on_click'):
        ui_components['visualize_button'].on_click(on_visualize_click)
    
    if 'compare_button' in ui_components and hasattr(ui_components['compare_button'], 'on_click'):
        ui_components['compare_button'].on_click(on_compare_click)
    
    # Tambahkan handler ke ui_components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click
    })
    
    return ui_components
