# File: smartcash/handlers/dataset/visualizations/sample/sample_grid_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk sampel gambar dataset dalam bentuk grid

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase

class SampleGridVisualizer(VisualizationBase):
    """
    Visualizer untuk menampilkan sampel-sampel gambar dari dataset dalam bentuk grid.
    Mendukung filter berdasarkan kelas dan layer, serta visualisasi anotasi bounding box.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi SampleGridVisualizer.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(data_dir, output_dir, logger)
        
        self.logger.info(f"🖼️ SampleGridVisualizer diinisialisasi")
    
    def visualize_samples(
        self,
        split: str = 'train',
        num_samples: int = 9,
        classes: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        random_seed: int = 42,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None,
        show_annotations: bool = True,
        num_workers: int = 4
    ) -> str:
        """
        Visualisasikan sampel gambar dari dataset dengan bounding box dalam grid.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            num_samples: Jumlah sampel yang akan ditampilkan
            classes: Filter kelas tertentu (opsional)
            layers: Filter layer tertentu (opsional)
            random_seed: Seed untuk random state
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_annotations: Tampilkan anotasi bounding box
            num_workers: Jumlah worker untuk proses paralel
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        random.seed(random_seed)
        
        self.logger.info(f"🖼️ Memvisualisasikan {num_samples} sampel untuk split: {split}")
        
        # Set default layers jika tidak ditentukan
        if layers is None:
            layers = self.layer_config.get_layer_names()
        
        # Cari gambar dan label
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            raise ValueError(f"Direktori {split}/images atau {split}/labels tidak ditemukan")
        
        # Cari semua file gambar
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(list(images_dir.glob(f"*.{ext}")))
        
        if not image_files:
            raise ValueError(f"Tidak ada file gambar di {images_dir}")
        
        # Filter sampel berdasarkan kelas jika ditentukan
        filtered_samples = []
        
        if classes:
            # Konversi nama kelas ke ID dengan layer
            class_ids = []
            for cls_name in classes:
                for layer in layers:
                    layer_config = self.layer_config.get_layer_config(layer)
                    if cls_name in layer_config['classes']:
                        idx = layer_config['classes'].index(cls_name)
                        if idx < len(layer_config['class_ids']):
                            class_ids.append(layer_config['class_ids'][idx])
                
            if not class_ids:
                self.logger.warning(f"⚠️ Tidak ada kelas yang cocok dengan filter: {classes}")
                # Gunakan semua sampel jika tidak ada kelas yang cocok
                filtered_samples = [(img, labels_dir / f"{img.stem}.txt") 
                                    for img in image_files 
                                    if (labels_dir / f"{img.stem}.txt").exists()]
            else:
                # Filter sampel berdasarkan kelas
                for img_path in image_files:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        # Cek apakah label mengandung kelas yang dicari
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and len(parts) >= 5:
                                    try:
                                        cls_id = int(float(parts[0]))
                                        if cls_id in class_ids:
                                            filtered_samples.append((img_path, label_path))
                                            break
                                    except (ValueError, IndexError):
                                        pass
        else:
            # Gunakan semua sampel dengan label yang valid
            filtered_samples = [(img, labels_dir / f"{img.stem}.txt") 
                                for img in image_files 
                                if (labels_dir / f"{img.stem}.txt").exists()]
        
        if not filtered_samples:
            raise ValueError(f"Tidak ada sampel yang memenuhi kriteria filter")
        
        # Pilih sampel secara acak
        if len(filtered_samples) > num_samples:
            samples = random.sample(filtered_samples, num_samples)
        else:
            samples = filtered_samples
            self.logger.info(f"⚠️ Hanya tersedia {len(samples)} sampel, lebih sedikit dari yang diminta ({num_samples})")
        
        # Tentukan grid layout
        rows = int(np.ceil(np.sqrt(len(samples))))
        cols = int(np.ceil(len(samples) / rows))
        
        # Buat figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes jika multi-dimensi
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Proses gambar secara paralel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Buat future untuk setiap gambar
            futures = []
            for i, (img_path, label_path) in enumerate(samples):
                if i < len(axes):
                    future = executor.submit(
                        self._process_sample_image,
                        img_path,
                        label_path,
                        layers,
                        show_annotations
                    )
                    futures.append((i, future))
            
            # Proses hasil saat selesai
            for i, future in futures:
                try:
                    img, boxes_info = future.result()
                    
                    # Tampilkan gambar
                    axes[i].imshow(img)
                    
                    # Tambahkan bounding box jika diminta
                    if show_annotations:
                        for box_info in boxes_info:
                            x, y, w, h = box_info['box']
                            layer = box_info['layer']
                            cls_name = box_info['class_name']
                            
                            # Pilih warna berdasarkan layer
                            color = self.layer_colors.get(layer, self.layer_colors['default'])
                            
                            # Buat rectangle
                            rect = patches.Rectangle(
                                (x, y),
                                w,
                                h,
                                linewidth=2,
                                edgecolor=color,
                                facecolor='none'
                            )
                            axes[i].add_patch(rect)
                            
                            # Tambahkan label
                            # Buat background untuk teks
                            text_color = 'white'
                            text_bg_color = to_rgba(color, alpha=0.8)
                            
                            # Tentukan posisi label (di atas box jika ada ruang, di dalam jika tidak)
                            text_y = y - 5 if y > 15 else y + 15
                            
                            # Tambahkan label kelas
                            axes[i].text(
                                x, text_y, cls_name,
                                color=text_color,
                                fontsize=10,
                                fontweight='bold',
                                backgroundcolor=text_bg_color,
                                bbox=dict(facecolor=text_bg_color, edgecolor='none', pad=2)
                            )
                    
                    # Tambahkan judul dengan nama file
                    axes[i].set_title(img_path.name, fontsize=10)
                    
                    # Matikan axis
                    axes[i].axis('off')
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memproses sampel {i}: {str(e)}")
                    axes[i].text(0.5, 0.5, "Error", ha='center', va='center')
                    axes[i].axis('off')
        
        # Sembunyikan axes yang tidak terpakai
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        # Tambahkan judul keseluruhan
        filters = []
        if classes:
            filters.append(f"Kelas: {', '.join(classes)}")
        if layers and len(layers) < len(self.layer_config.get_layer_names()):
            filters.append(f"Layer: {', '.join(layers)}")
            
        filter_str = f" ({' - '.join(filters)})" if filters else ""
        plt.suptitle(f"Sampel Dataset - {split.capitalize()}{filter_str}", fontsize=16)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Beri ruang untuk judul
        
        # Simpan gambar
        if save_path is None:
            class_str = f"_cls{'_'.join(classes)}" if classes else ""
            layer_str = f"_layer{'_'.join(layers)}" if layers and len(layers) < len(self.layer_config.get_layer_names()) else ""
            timestamp = self._get_timestamp()
            
            filename = f"samples_{split}{class_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        return self.save_plot(fig, save_path)
    
    def _process_sample_image(
        self,
        img_path: Path,
        label_path: Path,
        layers: List[str],
        show_annotations: bool
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Proses satu sampel gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            label_path: Path ke file label
            layers: List layer yang akan divisualisasikan
            show_annotations: Apakah menampilkan anotasi
            
        Returns:
            Tuple (array gambar, list info bounding box)
        """
        # Baca gambar
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Gagal membaca gambar: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inisialisasi list untuk menyimpan informasi box
        boxes_info = []
        
        # Hanya proses label jika show_annotations aktif
        if show_annotations and label_path.exists():
            # Cek ukuran gambar
            img_h, img_w = img.shape[:2]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            
                            # Filter berdasarkan layer
                            layer = self._get_layer_for_class(cls_id)
                            if layer not in layers:
                                continue
                                
                            # Format YOLO: class_id, x_center, y_center, width, height
                            x_center = float(parts[1]) * img_w
                            y_center = float(parts[2]) * img_h
                            width = float(parts[3]) * img_w
                            height = float(parts[4]) * img_h
                            
                            # Konversi ke format (x, y, width, height) untuk rectangle
                            x = x_center - width / 2
                            y = y_center - height / 2
                            
                            # Ambil nama kelas
                            cls_name = self._get_class_name(cls_id)
                            
                            boxes_info.append({
                                'box': (x, y, width, height),
                                'class_id': cls_id,
                                'class_name': cls_name,
                                'layer': layer,
                            })
                        except (ValueError, IndexError):
                            # Skip label yang tidak valid
                            continue
        
        return img, boxes_info
    
    def visualize_class_samples(
        self,
        class_name: str,
        split: str = 'train',
        num_samples: int = 9,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None,
        show_annotations: bool = True,
        crop_to_bbox: bool = False,
        zoom_factor: float = 1.2
    ) -> str:
        """
        Visualisasikan sampel gambar untuk satu kelas spesifik.
        
        Args:
            class_name: Nama kelas yang akan divisualisasikan
            split: Split dataset ('train', 'val', 'test')
            num_samples: Jumlah sampel yang akan ditampilkan
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_annotations: Tampilkan anotasi bounding box
            crop_to_bbox: Jika True, crop gambar ke sekitar bbox (dengan zoom factor)
            zoom_factor: Faktor zoom untuk crop (1.0 = crop tepat di bbox)
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        # Cari class_id dari class_name
        class_id = None
        class_layer = None
        
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if class_name in layer_config['classes']:
                idx = layer_config['classes'].index(class_name)
                if idx < len(layer_config['class_ids']):
                    class_id = layer_config['class_ids'][idx]
                    class_layer = layer_name
                    break
        
        if class_id is None:
            raise ValueError(f"Kelas '{class_name}' tidak ditemukan dalam konfigurasi layer")
        
        self.logger.info(f"🖼️ Memvisualisasikan {num_samples} sampel untuk kelas: {class_name} (ID: {class_id}, Layer: {class_layer})")
        
        # Cari gambar dan label
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            raise ValueError(f"Direktori {split}/images atau {split}/labels tidak ditemukan")
        
        # Cari semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Filter sampel berdasarkan kelas
        class_samples = []
        
        for label_path in label_files:
            img_stem = label_path.stem
            
            # Cari file gambar yang sesuai
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                img_candidate = images_dir / f"{img_stem}{ext}"
                if img_candidate.exists():
                    img_path = img_candidate
                    break
            
            if img_path is None:
                continue
            
            # Baca label untuk mencari class_id
            with open(label_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            line_cls_id = int(float(parts[0]))
                            if line_cls_id == class_id:
                                # Temukan bbox untuk kelas ini
                                bbox = [
                                    float(parts[1]),  # x_center
                                    float(parts[2]),  # y_center
                                    float(parts[3]),  # width
                                    float(parts[4])   # height
                                ]
                                class_samples.append((img_path, label_path, line_idx, bbox))
                                break
                        except (ValueError, IndexError):
                            continue
        
        if not class_samples:
            raise ValueError(f"Tidak ada sampel untuk kelas '{class_name}' di split '{split}'")
            
        self.logger.info(f"🔍 Ditemukan {len(class_samples)} sampel untuk kelas '{class_name}'")
        
        # Pilih sampel secara acak
        if len(class_samples) > num_samples:
            samples = random.sample(class_samples, num_samples)
        else:
            samples = class_samples
            if len(samples) < num_samples:
                self.logger.info(f"⚠️ Hanya tersedia {len(samples)} sampel, lebih sedikit dari yang diminta ({num_samples})")
        
        # Tentukan grid layout
        rows = int(np.ceil(np.sqrt(len(samples))))
        cols = int(np.ceil(len(samples) / rows))
        
        # Buat figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes jika multi-dimensi
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Proses setiap sampel
        for i, (img_path, label_path, line_idx, bbox) in enumerate(samples):
            if i < len(axes):
                try:
                    # Baca gambar
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError(f"Gagal membaca gambar: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Ukuran gambar
                    img_h, img_w = img.shape[:2]
                    
                    # Konversi koordinat YOLO ke piksel
                    x_center, y_center, width, height = bbox
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h
                    
                    # Koordinat bbox (x, y) adalah pojok kiri atas
                    x = x_center - width / 2
                    y = y_center - height / 2
                    
                    # Crop gambar jika diminta
                    if crop_to_bbox:
                        # Hitung margin dengan zoom factor
                        margin_w = (zoom_factor - 1.0) * width / 2
                        margin_h = (zoom_factor - 1.0) * height / 2
                        
                        # Koordinat crop (pastikan dalam batas gambar)
                        crop_x1 = max(0, int(x - margin_w))
                        crop_y1 = max(0, int(y - margin_h))
                        crop_x2 = min(img_w, int(x + width + margin_w))
                        crop_y2 = min(img_h, int(y + height + margin_h))
                        
                        # Crop gambar
                        img_cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # Sesuaikan koordinat untuk gambar yang telah di-crop
                        x -= crop_x1
                        y -= crop_y1
                        
                        # Gunakan gambar yang telah di-crop
                        img = img_cropped
                    
                    # Tampilkan gambar di subplot
                    axes[i].imshow(img)
                    
                    # Tambahkan bounding box jika diminta
                    if show_annotations:
                        # Buat rectangle
                        rect = patches.Rectangle(
                            (x, y),
                            width,
                            height,
                            linewidth=2,
                            edgecolor=self.layer_colors.get(class_layer, self.layer_colors['default']),
                            facecolor='none'
                        )
                        axes[i].add_patch(rect)
                    
                    # Tambahkan judul dengan nama file
                    axes[i].set_title(f"{img_path.name}", fontsize=10)
                    
                    # Matikan axis
                    axes[i].axis('off')
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memproses sampel {i}: {str(e)}")
                    axes[i].text(0.5, 0.5, "Error", ha='center', va='center')
                    axes[i].axis('off')
        
        # Sembunyikan axes yang tidak terpakai
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        # Tambahkan judul keseluruhan
        crop_str = " (Cropped)" if crop_to_bbox else ""
        plt.suptitle(f"Sampel Kelas: {class_name} - {split.capitalize()}{crop_str}", fontsize=16)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Beri ruang untuk judul
        
        # Simpan gambar
        if save_path is None:
            crop_str = "_cropped" if crop_to_bbox else ""
            timestamp = self._get_timestamp()
            
            # Bersihkan nama kelas untuk nama file
            safe_class_name = class_name.replace('/', '_').replace(' ', '_')
            filename = f"samples_class_{safe_class_name}_{split}{crop_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        return self.save_plot(fig, save_path)