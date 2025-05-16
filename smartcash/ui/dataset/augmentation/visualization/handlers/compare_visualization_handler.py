"""
File: smartcash/ui/dataset/augmentation/visualization/handlers/compare_visualization_handler.py
Deskripsi: Handler untuk visualisasi perbandingan preprocess vs augmentasi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.bbox_utils import draw_bboxes
from smartcash.dataset.utils.image_utils import load_image, resize_image
from smartcash.ui.dataset.augmentation.visualization.visualization_base import AugmentationVisualizationBase


class CompareVisualizationHandler(AugmentationVisualizationBase):
    """Handler untuk visualisasi perbandingan preprocess vs augmentasi"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi handler visualisasi perbandingan.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
        """
        super().__init__(config, logger)
        self.logger = logger or get_logger("compare_visualization")
        
    def visualize_preprocess_vs_augmentation(self, original_dir: str, preprocessed_dir: str, 
                                        aug_type: str = 'combined', split: str = 'train', 
                                        n_samples: int = None) -> Dict:
        """
        Visualisasikan perbandingan antara gambar original, preprocessed, dan hasil augmentasi.
        
        Args:
            original_dir: Direktori dataset original
            preprocessed_dir: Direktori dataset preprocessed
            aug_type: Jenis augmentasi
            split: Split dataset (train, valid, test)
            n_samples: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Dictionary dengan hasil visualisasi
        """
        self.logger.info(f"🎨 Memvisualisasikan perbandingan preprocess vs augmentasi '{aug_type}'")
        
        if n_samples is None:
            n_samples = self.sample_count
        
        # Load sample data dari direktori original
        original_samples = self.load_sample_data(original_dir, split, n_samples)
        
        if not original_samples:
            self.logger.warning(f"⚠️ Tidak ada sampel original yang ditemukan di {original_dir}/{split}")
            return {
                'status': 'error',
                'message': f"Tidak ada sampel yang ditemukan di {original_dir}/{split}"
            }
        
        # Load sample data dari direktori preprocessed    
        preprocessed_samples = self.load_sample_data(preprocessed_dir, split, n_samples)
        
        # Jika tidak ada sampel preprocessed, kembalikan error
        if not preprocessed_samples:
            self.logger.warning(f"⚠️ Tidak ada sampel preprocessed yang ditemukan di {preprocessed_dir}/{split}")
            return {"status": "error", "message": f"Tidak ada sampel preprocessed yang ditemukan di {preprocessed_dir}/{split}"}
            
        # Visualisasikan perbandingan
        return self._visualize_comparison(original_samples, preprocessed_samples, aug_type, f"Perbandingan Original vs Preprocessed vs Augmented ({aug_type})")
        
    def _create_visualization_figure(self, n_samples: int, n_cols: int = 3, title: str = None) -> Tuple[plt.Figure, np.ndarray]:
        """Buat figure dan axes untuk visualisasi
        
        Args:
            n_samples: Jumlah sampel yang akan divisualisasikan
            n_cols: Jumlah kolom dalam figure
            title: Judul figure (opsional)
            
        Returns:
            Tuple berisi figure dan axes
        """
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Pastikan axes dalam bentuk 2D array
        if n_samples == 1:
            axes = np.array([axes])
            
        return fig, axes
    
    def _visualize_image(self, ax: plt.Axes, image: np.ndarray, title: str, labels: List = None) -> None:
        """Visualisasikan gambar pada axes
        
        Args:
            ax: Axes untuk menampilkan gambar
            image: Gambar yang akan ditampilkan
            title: Judul untuk gambar
            labels: Label bounding box (opsional)
        """
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
        
        # Gambar bbox jika ada dan show_bboxes aktif
        if labels is not None and self.show_bboxes:
            draw_bboxes(ax, image, labels)
    
    def _visualize_comparison(self, original_samples: List[Dict], preprocessed_samples: List[Dict], 
                              aug_type: str = 'combined', title: str = None) -> Dict:
        """
        Visualisasikan perbandingan antara gambar original, preprocessed, dan hasil augmentasi.
        
        Args:
            original_samples: List data sampel original
            preprocessed_samples: List data sampel preprocessed
            aug_type: Jenis augmentasi
            title: Judul visualisasi
            
        Returns:
            Dictionary dengan hasil visualisasi
        """
        # Pastikan jumlah sampel sama
        n_samples = min(len(original_samples), len(preprocessed_samples))
        
        # Buat figure
        fig, axes = self._create_visualization_figure(n_samples, 3, title)
            
        # Visualisasikan setiap sampel
        for i in range(n_samples):
            # Dapatkan data sampel original
            original_sample = original_samples[i]
            original_image = original_sample['image']
            original_labels = original_sample['labels']
            filename = original_sample['filename']
            
            # Dapatkan data sampel preprocessed
            preprocessed_sample = preprocessed_samples[i]
            preprocessed_image = preprocessed_sample['image']
            preprocessed_labels = preprocessed_sample['labels']
            
            # Terapkan augmentasi pada gambar preprocessed
            aug_image, aug_labels = self.apply_augmentation(preprocessed_image, preprocessed_labels, aug_type)
            
            # Tampilkan gambar original
            self._visualize_image(axes[i, 0], original_image, f"Original: {filename}", original_labels)
                
            # Tampilkan gambar preprocessed
            self._visualize_image(axes[i, 1], preprocessed_image, f"Preprocessed: {filename}", preprocessed_labels)
                
            # Tampilkan gambar hasil augmentasi
            self._visualize_image(axes[i, 2], aug_image, f"Augmented ({aug_type}): {filename}", aug_labels)
        
        # Sesuaikan layout
        plt.tight_layout()
        
        # Simpan figure
        if self.save_visualizations:
            output_filename = f"comparison_{aug_type}.png"
            self.save_figure(fig, output_filename)
            
        return {
            "aug_type": aug_type,
            "figure": fig,
            "n_samples": n_samples
        }
        
    def visualize_augmentation_impact(self, preprocessed_dir: str, aug_types: List[str] = None, split: str = 'train') -> Dict:
        """
        Visualisasikan dampak berbagai jenis augmentasi pada satu gambar.
        
        Args:
            preprocessed_dir: Direktori dataset preprocessed
            aug_types: Jenis augmentasi yang akan divisualisasikan
            split: Split dataset (train, valid, test)
            
        Returns:
            Dictionary dengan hasil visualisasi
        """
        self.logger.info(f"🎨 Memvisualisasikan dampak berbagai jenis augmentasi")
        
        # Default aug_types jika tidak disediakan
        if not aug_types:
            aug_types = ['combined', 'position', 'lighting']
            
        # Muat satu sampel data dari dataset preprocessed
        preprocessed_samples = self.load_sample_data(preprocessed_dir, split, 1)
        
        if not preprocessed_samples:
            self.logger.warning(f"⚠️ Tidak ada sampel yang ditemukan di {preprocessed_dir}/{split}")
            return {"status": "error", "message": f"Tidak ada sampel yang ditemukan di {preprocessed_dir}/{split}"}
            
        # Dapatkan data sampel
        sample = preprocessed_samples[0]
        image = sample['image']
        labels = sample['labels']
        filename = sample['filename']
        
        # Buat figure
        n_aug_types = len(aug_types)
        fig, axes = plt.subplots(1, n_aug_types + 1, figsize=(5 * (n_aug_types + 1), 5))
        fig.suptitle(f"Dampak Berbagai Jenis Augmentasi pada Gambar: {filename}", fontsize=16)
        
        # Tampilkan gambar preprocessed
        self._visualize_image(axes[0], image, "Preprocessed", labels)
            
        # Visualisasikan setiap jenis augmentasi
        for i, aug_type in enumerate(aug_types):
            # Terapkan augmentasi
            aug_image, aug_labels = self.apply_augmentation(image, labels, aug_type)
            
            # Tampilkan gambar hasil augmentasi
            self._visualize_image(axes[i + 1], aug_image, f"Augmented ({aug_type})", aug_labels)
        
        # Sesuaikan layout
        plt.tight_layout()
        
        # Simpan figure
        if self.save_visualizations:
            output_filename = f"augmentation_impact_{filename}.png"
            self.save_figure(fig, output_filename)
            
        return {
            "status": "success",
            "aug_types": aug_types,
            "figure": fig,
            "filename": filename
        }
