"""
File: smartcash/examples/visualisasi_augmentasi_fixed.py
Deskripsi: Contoh penggunaan visualisasi augmentasi untuk dataset SmartCash (versi Python script)
"""

import os
import sys
from pathlib import Path

# Tambahkan path root ke sys.path
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

def main():
    """
    Contoh penggunaan visualisasi augmentasi
    """
    print("# Visualisasi Augmentasi Dataset SmartCash")
    print("Script ini mendemonstrasikan penggunaan visualisasi augmentasi untuk dataset SmartCash dalam deteksi mata uang.")
    
    # Instalasi dependensi jika belum ada
    print("\n## 1. Instalasi Dependensi")
    # Gunakan %pip install di notebook, tapi di script kita gunakan subprocess
    print("# Menginstal dependensi yang diperlukan...")
    
    # Persiapan dataset
    print("\n## 2. Persiapan Dataset")
    # Contoh: Unduh dataset contoh jika diperlukan
    # Untuk tujuan demonstrasi, kita akan menggunakan path dataset yang sudah ada
    data_dir = "data/dataset"  # Sesuaikan dengan path dataset Anda
    preprocessed_dir = "data/preprocessed"  # Sesuaikan dengan path data preprocessed Anda
    
    # Buat direktori jika belum ada
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    print(f"Dataset path: {data_dir}")
    print(f"Preprocessed path: {preprocessed_dir}")
    
    # Inisialisasi UI visualisasi augmentasi
    print("\n## 3. Inisialisasi UI Visualisasi Augmentasi")
    
    from smartcash.ui.dataset.augmentation.visualization.visualization_initializer import initialize_augmentation_visualization
    
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
    print("Menginisialisasi UI visualisasi augmentasi...")
    initialize_augmentation_visualization(config=custom_config)
    
    # Petunjuk penggunaan
    print("\n## 4. Petunjuk Penggunaan")
    print("UI visualisasi augmentasi menyediakan beberapa tab untuk memvisualisasikan augmentasi pada dataset:")
    print("1. **Sampel Augmentasi**: Melihat contoh hasil augmentasi pada beberapa gambar.")
    print("2. **Variasi Augmentasi**: Melihat berbagai variasi augmentasi pada satu gambar.")
    print("3. **Perbandingan Preprocess vs Augmentasi**: Membandingkan gambar asli, preprocessed, dan hasil augmentasi.")
    print("4. **Dampak Augmentasi**: Melihat dampak berbagai jenis augmentasi pada satu gambar.")
    
    # Contoh penggunaan langsung API visualisasi
    print("\n## 5. Contoh Penggunaan Langsung API Visualisasi")
    
    from smartcash.ui.dataset.augmentation.visualization.visualization_manager import AugmentationVisualizationManager
    
    # Dapatkan instance manager
    manager = AugmentationVisualizationManager.get_instance()
    
    # Contoh: Visualisasi sampel augmentasi secara langsung
    print("Memvisualisasikan sampel augmentasi...")
    sample_handler = manager.sample_handler
    result = sample_handler.visualize_augmentation_samples(
        data_dir=data_dir,
        aug_types=['combined'],  # Jenis augmentasi: 'combined', 'position', 'lighting'
        split='train',
        num_samples=2
    )
    
    # Tampilkan hasil
    if result['status'] == 'success':
        print(f"Berhasil memvisualisasikan {len(result['figures'])} sampel augmentasi")
    else:
        print(f"Gagal memvisualisasikan sampel augmentasi: {result.get('message', 'Unknown error')}")
    
    # Visualisasi perbandingan preprocess vs augmentasi
    print("\n## 6. Visualisasi Perbandingan Preprocess vs Augmentasi")
    print("Memvisualisasikan perbandingan preprocess vs augmentasi...")
    
    compare_handler = manager.compare_handler
    result = compare_handler.visualize_preprocess_vs_augmentation(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        aug_type='combined',
        split='train',
        num_samples=2
    )
    
    # Tampilkan hasil
    if result['status'] == 'success':
        print("Berhasil memvisualisasikan perbandingan preprocess vs augmentasi")
    else:
        print(f"Gagal memvisualisasikan perbandingan: {result.get('message', 'Unknown error')}")
    
    # Visualisasi dampak berbagai jenis augmentasi
    print("\n## 7. Visualisasi Dampak Berbagai Jenis Augmentasi")
    print("Memvisualisasikan dampak berbagai jenis augmentasi...")
    
    result = compare_handler.visualize_augmentation_impact(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        aug_types=['combined', 'position', 'lighting'],
        split='train'
    )
    
    # Tampilkan hasil
    if result['status'] == 'success':
        print("Berhasil memvisualisasikan dampak berbagai jenis augmentasi")
    else:
        print(f"Gagal memvisualisasikan dampak augmentasi: {result.get('message', 'Unknown error')}")
    
    # Kesimpulan
    print("\n## 8. Kesimpulan")
    print("Visualisasi augmentasi membantu memahami bagaimana augmentasi mempengaruhi gambar dan label dalam dataset.")
    print("Dengan memahami dampak augmentasi, kita dapat memilih jenis augmentasi yang tepat untuk meningkatkan performa model deteksi mata uang.")
    
    print("\nBeberapa manfaat visualisasi augmentasi:")
    print("- Memahami bagaimana augmentasi mempengaruhi gambar dan label")
    print("- Memilih jenis augmentasi yang tepat untuk dataset")
    print("- Mendeteksi masalah dalam pipeline augmentasi")
    print("- Meningkatkan interpretabilitas model")

if __name__ == "__main__":
    main()
