"""
File: smartcash/examples/visualization_augmentation_example.py
Deskripsi: Contoh penggunaan visualisasi augmentasi untuk dataset SmartCash
"""

import os
import sys
from pathlib import Path

# Tambahkan path root ke sys.path
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from smartcash.ui.dataset.augmentation.visualization.visualization_initializer import initialize_augmentation_visualization

def main():
    """
    Contoh penggunaan visualisasi augmentasi
    """
    # Konfigurasi path data
    data_dir = "data/dataset"  # Sesuaikan dengan path dataset Anda
    preprocessed_dir = "data/preprocessed"  # Sesuaikan dengan path data preprocessed Anda
    
    # Buat konfigurasi kustom
    custom_config = {
        'data_dir': data_dir,
        'preprocessed_dir': preprocessed_dir,
        'visualization': {
            'sample_count': 3,
            'show_bboxes': True,
            'show_original': True,
            'save_visualizations': False,
            'vis_dir': 'visualizations/augmentation'
        }
    }
    
    # Inisialisasi UI visualisasi augmentasi
    initialize_augmentation_visualization(config=custom_config)
    
    print("✅ UI visualisasi augmentasi telah diinisialisasi.")
    print("📝 Petunjuk penggunaan:")
    print("1. Gunakan tab 'Sampel Augmentasi' untuk melihat contoh hasil augmentasi pada beberapa gambar.")
    print("2. Gunakan tab 'Variasi Augmentasi' untuk melihat berbagai variasi augmentasi pada satu gambar.")
    print("3. Gunakan tab 'Perbandingan Preprocess vs Augmentasi' untuk membandingkan gambar asli, preprocessed, dan hasil augmentasi.")
    print("4. Gunakan tab 'Dampak Augmentasi' untuk melihat dampak berbagai jenis augmentasi pada satu gambar.")

if __name__ == "__main__":
    main()
