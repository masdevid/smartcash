"""
File: smartcash/ui/dataset/augmentation/visualization/handlers/sample_visualization_handler.py
Deskripsi: Handler untuk visualisasi sampel hasil augmentasi
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
from smartcash.ui.dataset.augmentation.visualization.visualization_base import AugmentationVisualizationBase


class SampleVisualizationHandler(AugmentationVisualizationBase):
    """Handler untuk visualisasi sampel hasil augmentasi"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi handler visualisasi sampel.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
        """
        super().__init__(config, logger)
        self.logger = logger or get_logger("sample_visualization")
        
    def visualize_augmentation_samples(self, data_dir: str, aug_types: List[str] = None, 
                                      split: str = 'train', num_samples: int = None) -> Dict:
        """
        Visualisasikan sampel hasil augmentasi.
        
        Args:
            data_dir: Direktori dataset
            aug_types: Jenis augmentasi yang akan divisualisasikan
            split: Split dataset (train, valid, test)
            num_samples: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Dictionary dengan hasil visualisasi
        """
        # Default aug_types jika tidak disediakan
        if not aug_types:
            aug_types = ['combined', 'position', 'lighting']
            
        # Gunakan sample_count dari konfigurasi jika num_samples tidak disediakan
        if num_samples is None:
            num_samples = self.sample_count
            
        # Muat sampel data
        self.logger.info(f"🔍 Memuat {num_samples} sampel dari split {split}")
        samples = self.load_sample_data(data_dir, split, num_samples)
        
        if not samples:
            self.logger.warning(f"⚠️ Tidak ada sampel yang ditemukan di {data_dir}/{split}")
            return {"status": "error", "message": f"Tidak ada sampel yang ditemukan di {data_dir}/{split}"}
            
        # Visualisasikan sampel untuk setiap jenis augmentasi
        results = []
        for aug_type in aug_types:
            result = self._visualize_single_augmentation_type(samples, aug_type)
            results.append(result)
            
        return {
            "status": "success", 
            "message": f"Berhasil memvisualisasikan {len(samples)} sampel untuk {len(aug_types)} jenis augmentasi",
            "results": results
        }
        
    def _create_visualization_figure(self, n_samples: int) -> Tuple[plt.Figure, np.ndarray]:
        """Buat figure dan axes untuk visualisasi
        
        Args:
            n_samples: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Tuple berisi figure dan axes
        """
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 5 * n_samples))
        
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
    
    def _visualize_single_augmentation_type(self, samples: List[Dict], aug_type: str) -> Dict:
        """Visualisasikan sampel dengan satu jenis augmentasi
        
        Args:
            samples: List sampel yang akan divisualisasikan
            aug_type: Jenis augmentasi yang akan diterapkan
            
        Returns:
            Dict berisi informasi visualisasi
        """
        n_samples = len(samples)
        
        # Buat figure dan axes
        fig, axes = self._create_visualization_figure(n_samples)
        
        # Visualisasikan setiap sampel
        for i, sample in enumerate(samples):
            image = sample['image']
            labels = sample['labels']
            filename = sample['filename']
            
            # Terapkan augmentasi
            aug_image, aug_labels = self.apply_augmentation(image, labels, aug_type)
            
            # Tampilkan gambar asli
            self._visualize_image(axes[i, 0], image, f"Original: {filename}", labels)
                
            # Tampilkan gambar hasil augmentasi
            self._visualize_image(axes[i, 1], aug_image, f"Augmented ({aug_type}): {filename}", aug_labels)
        
        # Atur layout
        plt.tight_layout()
        
        # Simpan figure jika diperlukan
        if self.save_visualizations:
            self.save_figure(fig, f"augmentation_samples_{aug_type}.png")
        
        return {
            'aug_type': aug_type,
            'n_samples': n_samples,
            'figure': fig
        }
        
    def visualize_augmentation_variations(self, data_dir: str, aug_type: str, split: str = 'train', n_variations: int = 5) -> Dict:
        """Visualisasikan variasi augmentasi untuk satu sampel
        
        Args:
            data_dir: Direktori dataset
            aug_type: Jenis augmentasi yang akan diterapkan
            split: Split dataset (train, val, test)
            n_variations: Jumlah variasi augmentasi yang akan ditampilkan
            
        Returns:
            Dict berisi informasi visualisasi
        """
        # Muat satu sampel
        samples = self.load_sample_data(data_dir, split, 1)
        
        if not samples:
            self.logger.warning(f"⚠️ Tidak ada sampel yang ditemukan di {data_dir}/{split}")
            return {'status': 'error'}
        
        sample = samples[0]
        image = sample['image']
        labels = sample['labels']
        filename = sample['filename']
        
        # Buat figure dengan 2 baris, 3 kolom
        # Baris 1: Original, Variation 1, Variation 2
        # Baris 2: Variation 3, Variation 4, Variation 5
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Tampilkan gambar asli
        self._visualize_image(axes[0, 0], image, f"Original: {filename}", labels)
        
        # Tampilkan variasi augmentasi
        variation_idx = 0
        for i in range(2):
            for j in range(3):
                # Lewati gambar asli (0,0)
                if i == 0 and j == 0:
                    continue
                    
                # Hentikan jika sudah mencapai jumlah variasi yang diminta
                if variation_idx >= n_variations:
                    break
                    
                # Terapkan augmentasi
                aug_image, aug_labels = self.apply_augmentation(image, labels, aug_type)
                
                # Tampilkan gambar hasil augmentasi
                self._visualize_image(axes[i, j], aug_image, f"Variation {variation_idx+1}", aug_labels)
                    
                variation_idx += 1
        
        # Atur layout
        plt.tight_layout()
        
        # Simpan figure jika diperlukan
        if self.save_visualizations:
            self.save_figure(fig, f"augmentation_variations_{aug_type}.png")
        
        return {
            'status': 'success',
            'aug_type': aug_type,
            'figure': fig,
            'filename': filename
        }
