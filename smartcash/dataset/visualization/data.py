# File: smartcash/utils/visualization/data.py
# Deskripsi: Utilitas untuk visualisasi data dan dataset

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import Counter

from smartcash.common.logger import SmartCashLogger, get_logger
import albumentations as A


class DataVisualizationHelper:
    """Helper untuk visualisasi dataset dan data-related components."""
    
    def __init__(self, output_dir: str, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi helper visualisasi data.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil visualisasi
            logger: Logger untuk output informasi
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("DataVisualization")
        
        # Buat direktori output jika belum ada
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"📊 Visualisasi akan disimpan di: {self.output_dir}")
    
    def _get_save_path(self, save_path: Optional[str], default_name: str) -> str:
        """Mendapatkan path untuk menyimpan visualisasi."""
        if save_path:
            # Gunakan path yang disediakan
            full_path = Path(save_path)
            # Pastikan direktori ada
            os.makedirs(full_path.parent, exist_ok=True)
        else:
            # Gunakan default path
            full_path = self.output_dir / default_name
        
        return str(full_path)
    
    def plot_class_distribution(self, class_stats: Dict[str, int], 
                               title: str = "Distribusi Kelas", 
                               save_path: Optional[str] = None,
                               top_n: int = 10,
                               figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Visualisasikan distribusi kelas dalam dataset.
        
        Args:
            class_stats: Dictionary dengan class_name: count
            title: Judul plot
            save_path: Path untuk menyimpan plot (opsional)
            top_n: Jumlah kelas teratas untuk ditampilkan
            figsize: Ukuran figure (width, height)
            
        Returns:
            Path lengkap ke file visualisasi yang disimpan
        """
        plt.figure(figsize=figsize)
        
        # Sorting dan ambil top N
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        if top_n and len(sorted_classes) > top_n:
            top_classes = sorted_classes[:top_n]
            other_sum = sum(count for _, count in sorted_classes[top_n:])
            # Tambahkan kategori 'Other' jika ada kelas yang tidak ditampilkan
            if other_sum > 0:
                top_classes.append(('Lainnya', other_sum))
        else:
            top_classes = sorted_classes
        
        # Plot distribusi
        labels, values = zip(*top_classes)
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        
        plt.bar(labels, values, color=colors)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Kelas', fontsize=12)
        plt.ylabel('Jumlah Sampel', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tambahkan nilai di atas setiap bar
        for i, v in enumerate(values):
            plt.text(i, v + max(values) * 0.01, str(v), 
                     ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Simpan plot
        save_path = self._get_save_path(save_path, "class_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Plot distribusi kelas disimpan ke: {save_path}")
        return save_path
    
    def plot_layer_distribution(self, layer_stats: Dict[str, Dict[str, int]],
                              title: str = "Distribusi Layer",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> str:
        """
        Visualisasikan distribusi layer dan kelas per layer.
        
        Args:
            layer_stats: Dictionary dengan layer: {class_name: count}
            title: Judul plot
            save_path: Path untuk menyimpan plot (opsional)
            figsize: Ukuran figure (width, height)
            
        Returns:
            Path lengkap ke file visualisasi yang disimpan
        """
        plt.figure(figsize=figsize)
        
        # Hitung total per layer
        layer_totals = {layer: sum(counts.values()) for layer, counts in layer_stats.items()}
        sorted_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Plot distribusi
        labels, values = zip(*sorted_layers)
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        
        plt.bar(labels, values, color=colors)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Jumlah Deteksi', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tambahkan nilai di atas setiap bar
        for i, v in enumerate(values):
            plt.text(i, v + max(values) * 0.01, str(v), 
                     ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Simpan plot
        save_path = self._get_save_path(save_path, "layer_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Plot distribusi layer disimpan ke: {save_path}")
        return save_path
    
    def plot_sample_images(self, data_dir: str, num_samples: int = 9,
                         classes: Optional[List[str]] = None,
                         random_seed: int = 42, 
                         figsize: Tuple[int, int] = (15, 15),
                         save_path: Optional[str] = None) -> str:
        """
        Visualisasikan sampel gambar dengan bounding box.
        
        Args:
            data_dir: Direktori dataset
            num_samples: Jumlah sampel yang akan ditampilkan
            classes: List kelas untuk difilter (opsional)
            random_seed: Seed untuk random sampling
            figsize: Ukuran figure (width, height)
            save_path: Path untuk menyimpan plot (opsional)
            
        Returns:
            Path lengkap ke file visualisasi yang disimpan
        """
        import cv2
        import glob
        import os
        
        random.seed(random_seed)
        
        # Cari semua file gambar
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(data_dir, 'images', ext)))
        
        if len(image_files) == 0:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {data_dir}/images")
            return ""
        
        # Acak dan batasi jumlah sampel
        random.shuffle(image_files)
        samples = image_files[:min(num_samples, len(image_files))]
        
        # Set up plot grid
        rows = cols = int(np.ceil(np.sqrt(len(samples))))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        for i, img_path in enumerate(samples):
            if i >= len(axes):
                break
                
            # Baca gambar
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Baca label
            label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            
            # Tampilkan gambar
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(os.path.basename(img_path), fontsize=8)
            
            # Parse label file jika ada
            h, w = img.shape[:2]
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # format: class_id center_x center_y width height
                            cls_id = int(parts[0])
                            x_center, y_center = float(parts[1]) * w, float(parts[2]) * h
                            width, height = float(parts[3]) * w, float(parts[4]) * h
                            
                            # Konversi ke koordinat x, y (top-left)
                            x = x_center - (width / 2)
                            y = y_center - (height / 2)
                            
                            # Tambahkan bounding box
                            rect = patches.Rectangle(
                                (x, y), width, height, 
                                linewidth=2, 
                                edgecolor='r', 
                                facecolor='none'
                            )
                            ax.add_patch(rect)
            
            ax.axis('off')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        # Simpan plot
        save_path = self._get_save_path(save_path, "sample_images.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"🖼️ Plot sampel gambar disimpan ke: {save_path}")
        return save_path
    
    def plot_augmentation_comparison(self, image_path: str,
                               augmentation_types: List[str] = ['original', 'lighting', 'geometric', 'combined'],
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Visualisasikan perbandingan berbagai jenis augmentasi pada gambar.
        
        Args:
            image_path: Path ke gambar yang akan diaugmentasi
            augmentation_types: List jenis augmentasi
            save_path: Path untuk menyimpan plot (opsional)
            figsize: Ukuran figure (width, height)
            
        Returns:
            Path lengkap ke file visualisasi yang disimpan
        """
        import cv2
        
        # Definisi augmentasi
        augmentations = {
            'original': None,
            'lighting': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ]),
            'geometric': A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1),
                A.Affine(shear=10, p=1),
            ]),
            'combined': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=1),
            ])
        }
        
        # Baca gambar
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Set up plot
        num_types = len(augmentation_types)
        rows = int(np.ceil(num_types / 2))
        cols = 2 if num_types > 1 else 1
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Buat augmentasi dan tampilkan
        for i, aug_type in enumerate(augmentation_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if aug_type == 'original' or aug_type not in augmentations:
                # Tampilkan gambar asli
                augmented = img
                title = "Gambar Asli" if aug_type == 'original' else aug_type
            else:
                # Terapkan augmentasi
                augmented = augmentations[aug_type](image=img)['image']
                title = f"Augmentasi: {aug_type}"
            
            ax.imshow(augmented)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        # Simpan plot
        save_path = self._get_save_path(save_path, "augmentation_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"🔄 Plot perbandingan augmentasi disimpan ke: {save_path}")
        return save_path