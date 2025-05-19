"""
File: smartcash/ui/dataset/augmentation/visualization/visualization_base.py
Deskripsi: Kelas dasar untuk visualisasi augmentasi dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.utils.image_utils import load_image, resize_image
from smartcash.dataset.utils.bbox_utils import load_yolo_labels, draw_bboxes


class AugmentationVisualizationBase:
    """Kelas dasar untuk visualisasi augmentasi dataset"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi kelas dasar visualisasi augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("augmentation_visualization")
        
        # Inisialisasi komponen-komponen utama
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        
        # Dapatkan konfigurasi visualisasi
        self.vis_config = self.config.get('visualization', {})
        self.sample_count = self.vis_config.get('sample_count', 5)
        self.show_bboxes = self.vis_config.get('show_bboxes', True)
        self.show_original = self.vis_config.get('show_original', True)
        self.save_visualizations = self.vis_config.get('save_visualizations', True)
        self.vis_dir = self.vis_config.get('vis_dir', 'visualizations/augmentation')
        
        # Pastikan direktori visualisasi ada
        if self.save_visualizations:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        self.logger.info(f"🎨 Visualisasi augmentasi siap digunakan")
    
    def apply_augmentation(self, image: np.ndarray, labels: List = None, aug_type: str = 'combined') -> Tuple[np.ndarray, List]:
        """
        Terapkan augmentasi pada gambar dan label.
        
        Args:
            image: Gambar yang akan diaugmentasi
            labels: Label YOLO (opsional)
            aug_type: Jenis augmentasi
            
        Returns:
            Tuple gambar dan label yang sudah diaugmentasi
        """
        # Buat pipeline augmentasi
        pipeline = self.pipeline_factory.create_pipeline(
            augmentation_types=[aug_type],
            img_size=(image.shape[1], image.shape[0]),
            include_normalize=False
        )
        
        # Persiapkan data untuk augmentasi
        if labels:
            # Format data untuk albumentations dengan bbox
            bboxes = []
            class_labels = []
            
            for label in labels:
                class_id, x_center, y_center, width, height = label
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
                
            # Terapkan augmentasi dengan bbox
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # Kembalikan hasil augmentasi
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']
            
            # Rekonstruksi label
            aug_labels = []
            for i, bbox in enumerate(aug_bboxes):
                x_center, y_center, width, height = bbox
                aug_labels.append([aug_class_labels[i], x_center, y_center, width, height])
                
            return aug_image, aug_labels
        else:
            # Terapkan augmentasi tanpa bbox
            augmented = pipeline(image=image)
            return augmented['image'], None
    
    def load_sample_data(self, data_dir: str, split: str = 'train', num_samples: int = 5) -> List[Dict]:
        """
        Muat data sampel untuk visualisasi.
        
        Args:
            data_dir: Direktori dataset
            split: Split dataset (train, valid, test)
            num_samples: Jumlah sampel yang akan dimuat
            
        Returns:
            List data sampel
        """
        # Setup path
        images_dir = os.path.join(data_dir, split, 'images')
        labels_dir = os.path.join(data_dir, split, 'labels')
        
        # Cek apakah direktori ada
        if not os.path.exists(images_dir):
            self.logger.warning(f"⚠️ Direktori gambar tidak ditemukan: {images_dir}")
            return []
            
        # Dapatkan daftar file gambar
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Pilih sampel secara acak
        if len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        # Muat data sampel
        samples = []
        for img_file in image_files:
            # Path file
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            # Muat gambar
            image = load_image(img_path)
            if image is None:
                continue
                
            # Muat label jika ada
            labels = None
            if os.path.exists(label_path) and self.show_bboxes:
                labels = load_yolo_labels(label_path)
                
            # Tambahkan ke sampel
            samples.append({
                'image': image,
                'labels': labels,
                'img_path': img_path,
                'label_path': label_path,
                'filename': img_file
            })
            
        return samples
    
    def create_figure(self, title: str = None, figsize: Tuple[int, int] = (15, 10)) -> Tuple:
        """
        Buat figure matplotlib untuk visualisasi.
        
        Args:
            title: Judul figure
            figsize: Ukuran figure
            
        Returns:
            Tuple (fig, ax)
        """
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=16)
        return fig, ax
    
    def save_figure(self, fig, filename: str) -> None:
        """
        Simpan figure ke file.
        
        Args:
            fig: Figure matplotlib
            filename: Nama file
        """
        if self.save_visualizations:
            save_path = os.path.join(self.vis_dir, filename)
            fig.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"✅ Figure disimpan ke {save_path}")
