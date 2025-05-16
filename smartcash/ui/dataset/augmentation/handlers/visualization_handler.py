"""
File: smartcash/ui/dataset/augmentation/handlers/visualization_handler.py
Deskripsi: Handler untuk visualisasi hasil augmentasi dataset
"""

from typing import Dict, Any
import os
import glob
import random
import traceback
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_text

def visualize_augmented_images(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Visualisasikan hasil augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict dengan status dan pesan hasil visualisasi
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan direktori augmentasi
        data_dir = ui_components.get('data_dir', 'data')
        aug_dir = os.path.join(data_dir, 'augmented')
        
        # Periksa apakah direktori ada
        if not os.path.exists(aug_dir):
            return {
                'status': 'error',
                'message': f'Direktori augmentasi tidak ditemukan: {aug_dir}',
                'error': 'DirectoryNotFound'
            }
        
        # Dapatkan daftar gambar
        image_files = glob.glob(os.path.join(aug_dir, 'images', '*.jpg')) + \
                     glob.glob(os.path.join(aug_dir, 'images', '*.png')) + \
                     glob.glob(os.path.join(aug_dir, 'images', '*.jpeg'))
        
        if not image_files:
            return {
                'status': 'error',
                'message': 'Tidak ada gambar hasil augmentasi yang ditemukan',
                'error': 'NoImagesFound'
            }
        
        # Pilih beberapa gambar secara acak untuk ditampilkan
        num_samples = min(10, len(image_files))
        sample_images = random.sample(image_files, num_samples)
        
        # Tampilkan gambar dalam grid
        with ui_components.get('visualization_container', widgets.Output()):
            clear_output(wait=True)
            
            # Buat grid untuk menampilkan gambar
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for i, img_path in enumerate(sample_images):
                if i < len(axes):
                    try:
                        from PIL import Image
                        import numpy as np
                        
                        # Baca gambar
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        # Tampilkan gambar
                        axes[i].imshow(img_array)
                        axes[i].set_title(os.path.basename(img_path))
                        axes[i].axis('off')
                    except Exception as e:
                        logger.error(f"{ICONS['error']} Gagal memuat gambar {img_path}: {str(e)}")
                        axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                                    horizontalalignment='center',
                                    verticalalignment='center')
                        axes[i].axis('off')
            
            # Sembunyikan axes yang tidak digunakan
            for i in range(len(sample_images), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Tampilkan tombol visualisasi jika ada
        if 'visualization_buttons' in ui_components:
            ui_components['visualization_buttons'].layout.display = 'flex'
        
        return {
            'status': 'success',
            'message': f'Berhasil memvisualisasikan {num_samples} gambar hasil augmentasi',
            'num_images': len(image_files),
            'num_samples': num_samples
        }
    except Exception as e:
        logger.error(f"{ICONS['error']} Gagal memvisualisasikan hasil augmentasi: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return {
            'status': 'error',
            'message': f'Gagal memvisualisasikan hasil augmentasi: {str(e)}',
            'error': str(e)
        }
