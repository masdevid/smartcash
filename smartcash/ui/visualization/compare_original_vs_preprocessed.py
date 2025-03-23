"""
File: smartcash/ui/visualization/compare_original_vs_preprocessed.py
Deskripsi: Utilitas untuk menampilkan komparasi gambar original dengan gambar preprocessed
"""
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_original_vs_preprocessed(ui_components: Dict[str, Any], raw_dir: str, preprocessed_dir: str, num_samples: int = 3):
    """
    Komparasi sampel dataset original dengan yang telah dipreprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        raw_dir: Direktori dataset original
        preprocessed_dir: Direktori dataset preprocessed
        num_samples: Jumlah sampel yang akan divisualisasikan
    """
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
                
                # Tampilkan gambar dengan nama file pendek
                raw_name = raw_path.name
                if len(raw_name) > 10:
                    raw_name = f"...{raw_name[-10:]}"
                    
                proc_name = proc_path.name
                if len(proc_name) > 10:
                    proc_name = f"...{proc_name[-10:]}"
                
                axes[i, 0].imshow(raw_img)
                axes[i, 0].set_title(f"Mentah: {raw_name}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(proc_img)
                axes[i, 1].set_title(f"Preprocessed: {proc_name}")
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
                
                # Tampilkan nama file yang pendek
                raw_name = raw_path.stem
                if len(raw_name) > 10:
                    raw_name = f"...{raw_name[-10:]}"
                
                display(HTML(f"""
                <div style="margin:10px 0; padding:5px; border-left:3px solid {COLORS['primary']}; background-color:{COLORS['light']}">
                    <p style="color:{COLORS['dark']};"><strong>{raw_name}</strong></p>
                    <p style="color:{COLORS['dark']};">Mentah: {raw_w}x{raw_h} piksel | Preprocessed: {proc_w}x{proc_h} piksel</p>
                    <p style="color:{COLORS['dark']};">Rasio kompresi: {(proc_img.nbytes / raw_img.nbytes):.2f}x</p>
                </div>
                """))
            except Exception:
                pass