"""
File: smartcash/ui/dataset/preprocessing_visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset preprocessing dengan integrasi standar
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
    """
    Setup handler untuk visualisasi dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol visualisasi
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi..."))
        
        # Dapatkan direktori dataset preprocessing
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        data_dir = ui_components.get('data_dir', 'data')
        
        # Cek apakah preprocessed dataset tersedia
        if not os.path.exists(preprocessed_dir):
            with ui_components['status']:
                display(create_status_indicator('warning', f"{ICONS['warning']} Dataset preprocessed tidak ditemukan di: {preprocessed_dir}"))
            return
            
        # Coba tampilkan visualisasi dengan error handling
        try:
            visualize_preprocessed_samples(ui_components, preprocessed_dir, data_dir)
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi dataset: {str(e)}")
    
    # Handler untuk tombol komparasi
    def on_compare_click(b):
        """Handler untuk komparasi sampel dataset mentah dengan yang telah dipreprocessing."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi..."))
        
        # Dapatkan direktori dataset
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        data_dir = ui_components.get('data_dir', 'data')
        
        # Cek ketersediaan data
        if not os.path.exists(preprocessed_dir) or not os.path.exists(data_dir):
            with ui_components['status']:
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori dataset tidak lengkap untuk komparasi"))
            return
            
        # Coba tampilkan visualisasi komparasi dengan error handling
        try:
            compare_raw_vs_preprocessed(ui_components, data_dir, preprocessed_dir)
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Tambahkan handlers ke tombol jika tersedia
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].on_click(on_visualize_click)
        
    if 'compare_button' in ui_components:
        ui_components['compare_button'].on_click(on_compare_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'visualize_dataset': visualize_preprocessed_samples,
        'compare_datasets': compare_raw_vs_preprocessed,
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click
    })
    
    return ui_components

def visualize_preprocessed_samples(ui_components: Dict[str, Any], preprocessed_dir: str, original_dir: str, num_samples: int = 5):
    """
    Visualisasi sampel dataset yang telah dipreprocessing.
    
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
            
        # Ambil semua gambar
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.npy'))
        if not image_files:
            display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada file gambar ditemukan di {images_dir}"))
            return
            
        # Batasi jumlah sampel
        image_files = image_files[:min(num_samples, len(image_files))]
        
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
                if img_path.suffix == '.npy':
                    # Handle numpy array preprocessed
                    img = np.load(str(img_path))
                    # Denormalisasi jika perlu
                    if img.dtype == np.float32 and img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                else:
                    # Handle gambar biasa
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Tampilkan gambar
                axes[i].imshow(img)
                axes[i].set_title(f"{img_path.stem}")
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Tampilkan informasi ukuran gambar
        for img_path in image_files:
            try:
                if img_path.suffix == '.npy':
                    img = np.load(str(img_path))
                    h, w = img.shape[:2]
                else:
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                display(HTML(f"<p><strong>{img_path.name}</strong>: {w}x{h} piksel</p>"))
            except Exception:
                pass

def compare_raw_vs_preprocessed(ui_components: Dict[str, Any], raw_dir: str, preprocessed_dir: str, num_samples: int = 3):
    """
    Komparasi sampel dataset mentah dengan yang telah dipreprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        raw_dir: Direktori dataset mentah
        preprocessed_dir: Direktori dataset preprocessed
        num_samples: Jumlah sampel yang akan divisualisasikan
    """
    from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
    
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
        
        # Cari gambar yang ada di kedua direktori
        preprocessed_images = list(preprocessed_images_dir.glob('*.jpg')) + list(preprocessed_images_dir.glob('*.png')) + list(preprocessed_images_dir.glob('*.npy'))
        raw_images = {img.stem: img for img in (list(raw_images_dir.glob('*.jpg')) + list(raw_images_dir.glob('*.png')))}
        
        # Dapatkan pasangan gambar
        pairs = []
        for proc_img in preprocessed_images:
            if proc_img.stem in raw_images:
                pairs.append((raw_images[proc_img.stem], proc_img))
                if len(pairs) >= num_samples:
                    break
        
        if not pairs:
            display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan pasangan gambar yang cocok untuk komparasi"))
            return
        
        # Tampilkan deskripsi
        display(create_info_alert(
            f"Komparasi {len(pairs)} sampel dataset: mentah vs preprocessed",
            "info"
        ))
        
        # Visualisasi komparasi
        fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 4*len(pairs)))
        if len(pairs) == 1:
            axes = axes.reshape(1, 2)
            
        for i, (raw_path, proc_path) in enumerate(pairs):
            # Load gambar raw
            try:
                raw_img = cv2.imread(str(raw_path))
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                # Load gambar preprocessed
                if proc_path.suffix == '.npy':
                    proc_img = np.load(str(proc_path))
                    # Denormalisasi jika perlu
                    if proc_img.dtype == np.float32 and proc_img.max() <= 1.0:
                        proc_img = (proc_img * 255).astype(np.uint8)
                else:
                    proc_img = cv2.imread(str(proc_path))
                    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
                
                # Tampilkan gambar
                axes[i, 0].imshow(raw_img)
                axes[i, 0].set_title(f"Mentah: {raw_path.name}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(proc_img)
                axes[i, 1].set_title(f"Preprocessed: {proc_path.name}")
                axes[i, 1].axis('off')
                
            except Exception as e:
                axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Informasi detail untuk setiap pasangan
        for raw_path, proc_path in pairs:
            try:
                raw_img = cv2.imread(str(raw_path))
                raw_h, raw_w = raw_img.shape[:2]
                
                if proc_path.suffix == '.npy':
                    proc_img = np.load(str(proc_path))
                    proc_h, proc_w = proc_img.shape[:2]
                else:
                    proc_img = cv2.imread(str(proc_path))
                    proc_h, proc_w = proc_img.shape[:2]
                
                display(HTML(f"""
                <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                    <p><strong>{raw_path.stem}</strong></p>
                    <p>Mentah: {raw_w}x{raw_h} piksel | Preprocessed: {proc_w}x{proc_h} piksel</p>
                    <p>Rasio kompresi: {(proc_img.nbytes / raw_img.nbytes):.2f}x</p>
                </div>
                """))
            except Exception:
                pass

def get_preprocessing_stats(ui_components: Dict[str, Any], preprocessed_dir: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        
    Returns:
        Dictionary statistik preprocessing
    """
    stats = {
        'splits': {},
        'total': {
            'images': 0,
            'labels': 0
        }
    }
    
    # Cek setiap split
    for split in ['train', 'valid', 'test']:
        split_dir = Path(preprocessed_dir) / split
        if not split_dir.exists():
            stats['splits'][split] = {'exists': False, 'images': 0, 'labels': 0}
            continue
            
        # Hitung gambar dan label
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.npy'))) if images_dir.exists() else 0
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # Update statistik
        stats['splits'][split] = {
            'exists': True,
            'images': num_images,
            'labels': num_labels,
            'complete': num_images > 0 and num_labels > 0 and num_images == num_labels
        }
        
        # Update total
        stats['total']['images'] += num_images
        stats['total']['labels'] += num_labels
    
    # Dataset dianggap valid jika minimal ada 1 split dengan data lengkap
    stats['valid'] = any(split_info.get('complete', False) for split_info in stats['splits'].values())
    
    return stats