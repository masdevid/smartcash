# File: smartcash/handlers/dataset/visualizations/sample_image_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualisasi sampel gambar dari dataset dengan anotasi

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase


class SampleImageVisualizer(VisualizationBase):
    """
    Visualisasi sampel gambar dari dataset dengan anotasi bounding box.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi visualizer sampel gambar.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset
            output_dir: Direktori output untuk visualisasi
            logger: Logger kustom (opsional)
        """
        super().__init__(
            config=config,
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger or get_logger("sample_image_visualizer")
        )
        
        # Siapkan palet warna untuk layer (konsisten)
        self.layer_colors = {
            'banknote': '#FF5555',  # Merah muda
            'nominal': '#5555FF',   # Biru
            'security': '#55AA55',  # Hijau
            'watermark': '#AA55AA', # Ungu
            'default': '#AAAAAA'    # Abu-abu untuk layer yang tidak dikenal
        }
        
        # Layer class mapping untuk pencarian cepat
        self.class_to_layer = {}
        self.class_to_name = {}
        
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            for i, cls_id in enumerate(layer_config['class_ids']):
                self.class_to_layer[cls_id] = layer_name
                if i < len(layer_config['classes']):
                    self.class_to_name[cls_id] = layer_config['classes'][i]
        
        self.logger.info(f"🖼️ SampleImageVisualizer diinisialisasi: {self.data_dir}")
    
    def visualize_samples(
        self,
        split: str,
        num_samples: int = 9,
        classes: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        random_seed: int = 42,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None,
        num_workers: int = 4
    ) -> str:
        """
        Visualisasikan sampel gambar dari dataset dengan bounding box.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            num_samples: Jumlah sampel yang akan ditampilkan
            classes: Filter kelas tertentu (opsional)
            layers: Filter layer tertentu (opsional)
            random_seed: Seed untuk random state
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            num_workers: Jumlah worker untuk proses paralel
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        random.seed(random_seed)
        
        self.logger.info(f"🖼️ Memvisualisasikan {num_samples} sampel untuk split: {split}")
        
        try:
            # Set default layers jika tidak ditentukan
            if layers is None:
                layers = self.config.get('layers', ['banknote'])
            
            # Cari gambar dan label
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
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
                # Konversi nama kelas ke ID
                class_ids = []
                for cls_name in classes:
                    for layer in layers:
                        layer_config = self.layer_config.get_layer_config(layer)
                        if cls_name in layer_config['classes']:
                            idx = layer_config['classes'].index(cls_name)
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
                            layers
                        )
                        futures.append((i, future))
                
                # Proses hasil saat selesai
                for i, future in futures:
                    try:
                        img, boxes_info = future.result()
                        
                        # Tampilkan gambar
                        axes[i].imshow(img)
                        
                        # Tambahkan bounding box
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
            plt.suptitle(f"Sampel Dataset - {split.capitalize()}", fontsize=16)
            
            # Tight layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Beri ruang untuk judul
            
            # Simpan gambar
            if save_path is None:
                save_path = str(self.output_dir / f"{split}_samples.png")
                
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Visualisasi sampel tersimpan di: {save_path}")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi sampel: {str(e)}")
            # Return path dummy jika gagal
            return ""
    
    def _process_sample_image(
        self,
        img_path: Path,
        label_path: Path,
        layers: List[str]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Proses satu sampel gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            label_path: Path ke file label
            layers: List layer yang akan divisualisasikan
            
        Returns:
            Tuple (array gambar, list info bounding box)
        """
        # Baca gambar
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cek ukuran gambar
        img_h, img_w = img.shape[:2]
        
        # Baca label dan filter berdasarkan layer
        boxes_info = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            
                            # Filter berdasarkan layer
                            if cls_id in self.class_to_layer:
                                layer = self.class_to_layer[cls_id]
                                if layer in layers:
                                    # Format YOLO: class_id, x_center, y_center, width, height
                                    x_center = float(parts[1]) * img_w
                                    y_center = float(parts[2]) * img_h
                                    width = float(parts[3]) * img_w
                                    height = float(parts[4]) * img_h
                                    
                                    # Konversi ke format (x, y, width, height) untuk rectangle
                                    x = x_center - width / 2
                                    y = y_center - height / 2
                                    
                                    # Ambil nama kelas jika tersedia
                                    cls_name = self.class_to_name.get(cls_id, f"Class-{cls_id}")
                                    
                                    boxes_info.append({
                                        'box': (x, y, width, height),
                                        'class_id': cls_id,
                                        'class_name': cls_name,
                                        'layer': layer,
                                    })
                        except (ValueError, IndexError):
                            # Skip label yang tidak valid
                            pass
        
        return img, boxes_info