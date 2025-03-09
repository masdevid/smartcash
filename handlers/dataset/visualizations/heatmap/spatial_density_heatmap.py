# File: smartcash/handlers/dataset/visualizations/heatmap/spatial_density_heatmap.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk heatmap kepadatan spasial objek dalam dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase

class SpatialDensityHeatmap(VisualizationBase):
    """
    Visualizer untuk membuat heatmap distribusi spasial objek dalam dataset.
    
    Memungkinkan melihat di mana posisi objek cenderung muncul dalam gambar,
    berguna untuk memahami bias spasial dalam dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None,
        resolution: Tuple[int, int] = (32, 32)  # Resolusi default heatmap
    ):
        """
        Inisialisasi SpatialDensityHeatmap.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
            resolution: Resolusi heatmap (jumlah sel sumbu X dan Y)
        """
        super().__init__(data_dir, output_dir, logger)
        self.resolution = resolution
        
        self.logger.info(f"🔥 SpatialDensityHeatmap diinisialisasi dengan resolusi {resolution}")
    
    def generate_heatmap(
        self,
        split: str = 'train',
        class_filter: Optional[List[int]] = None,
        layer_filter: Optional[List[str]] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        show_plot: bool = False,
        alpha: float = 0.7,
        colormap: str = 'hot',
        gaussian_sigma: Optional[float] = 1.5
    ) -> str:
        """
        Membuat visualisasi heatmap spasial dari dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            class_filter: Filter berdasarkan ID kelas tertentu (opsional)
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            normalize: Normalisasi intensitas heatmap
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_plot: Tampilkan plot secara langsung
            alpha: Nilai transparansi untuk heatmap
            colormap: Colormap untuk heatmap
            gaussian_sigma: Gaussian blur untuk smoothing heatmap (None = no smoothing)
            
        Returns:
            Path file gambar yang disimpan
        """
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Inisialisasi array 2D untuk heatmap
        heatmap = np.zeros(self.resolution)
        
        # Log informasi filter
        filter_info = []
        if class_filter:
            class_names = [self._get_class_name(idx) for idx in class_filter]
            filter_info.append(f"Kelas: {', '.join(class_names[:5])}")
            if len(class_names) > 5:
                filter_info.append(f"dan {len(class_names) - 5} lainnya")
        
        if layer_filter:
            filter_info.append(f"Layer: {', '.join(layer_filter)}")
            
        filter_str = " (" + "; ".join(filter_info) + ")" if filter_info else ""
        self.logger.info(f"🔍 Membuat heatmap spasial untuk split '{split}'{filter_str}")
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        total_objects = 0
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Memproses {split}"):
            if not label_path.exists():
                continue
                
            # Baca label
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Format: class_id, x_center, y_center, width, height
                            cls_id = int(float(parts[0]))
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            
                            # Terapkan filter kelas jika ada
                            if class_filter and cls_id not in class_filter:
                                continue
                                
                            # Terapkan filter layer jika ada
                            if layer_filter:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_filter:
                                    continue
                            
                            # Konversi koordinat dinormalisasi ([0-1]) ke indeks grid heatmap
                            grid_x = int(x_center * (self.resolution[0] - 1))
                            grid_y = int(y_center * (self.resolution[1] - 1))
                            
                            # Tambahkan ke heatmap
                            heatmap[grid_y, grid_x] += 1
                            total_objects += 1
                            
                        except (ValueError, IndexError):
                            # Skip entri yang tidak valid
                            continue
        
        # Terapkan gaussian blur jika diperlukan
        if gaussian_sigma is not None and gaussian_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
            except ImportError:
                self.logger.warning("⚠️ scipy.ndimage tidak ditemukan, gaussian blur tidak diterapkan")
        
        # Normalisasi heatmap
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Buat visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap menggunakan imshow
        im = ax.imshow(
            heatmap, 
            cmap=colormap, 
            interpolation='nearest',
            alpha=alpha,
            extent=[0, 1, 1, 0]  # Sesuai dengan koordinat YOLO
        )
        
        # Tambahkan colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Kepadatan Normalisasi' if normalize else 'Jumlah Objek')
        
        # Grid dan labels
        ax.grid(False)
        ax.set_xlabel('Posisi X')
        ax.set_ylabel('Posisi Y')
        
        # Judul dan informasi
        title_parts = [f"Distribusi Spasial Objek - {split.capitalize()}"]
        if class_filter:
            title_parts.append(f"Kelas: {len(class_filter)} dipilih")
        if layer_filter:
            title_parts.append(f"Layer: {', '.join(layer_filter)}")
            
        ax.set_title(" | ".join(title_parts))
        
        # Tambahkan informasi total objek
        ax.text(
            0.02, 0.02, 
            f"Total objek: {total_objects}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            class_str = f"_cls{len(class_filter) if class_filter else 0}"
            layer_str = f"_layer{'_'.join(layer_filter)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"heatmap_spatial_{split}{class_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout()
        
        result_path = self.save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return result_path
    
    def generate_multi_class_heatmap(
        self,
        split: str = 'train',
        top_n_classes: int = 4,
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
        gaussian_sigma: float = 1.0
    ) -> str:
        """
        Membuat visualisasi heatmap spasial untuk beberapa kelas teratas sekaligus.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            top_n_classes: Jumlah kelas teratas yang akan divisualisasikan
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            gaussian_sigma: Gaussian blur untuk smoothing heatmap
            
        Returns:
            Path file gambar yang disimpan
        """
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Hitung frekuensi kelas
        class_counts = {}
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar untuk menghitung frekuensi kelas
        for label_path in tqdm(label_files, desc=f"Menghitung frekuensi kelas di {split}"):
            if not label_path.exists():
                continue
                
            # Baca label
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            
                            # Terapkan filter layer jika ada
                            if layer_filter:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_filter:
                                    continue
                            
                            # Tambahkan ke penghitung
                            if cls_id not in class_counts:
                                class_counts[cls_id] = 0
                            class_counts[cls_id] += 1
                            
                        except (ValueError, IndexError):
                            continue
        
        # Temukan kelas teratas
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_classes]
        
        if not top_classes:
            self.logger.warning(f"⚠️ Tidak ada kelas yang ditemukan di split '{split}' dengan filter yang diberikan")
            return ""
            
        # Log kelas teratas
        self.logger.info(f"📊 Memvisualisasikan {len(top_classes)} kelas teratas:")
        for cls_id, count in top_classes:
            class_name = self._get_class_name(cls_id)
            self.logger.info(f"   • {class_name}: {count} objek")
        
        # Buat subplot grid
        n_cols = min(2, top_n_classes)
        n_rows = (top_n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Jika hanya satu subplot, pastikan axes adalah array
        if top_n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Iterasi untuk setiap kelas teratas
        for i, (cls_id, count) in enumerate(top_classes):
            class_name = self._get_class_name(cls_id)
            
            # Inisialisasi heatmap untuk kelas ini
            heatmap = np.zeros(self.resolution)
            
            # Progress bar untuk mengumpulkan data spasial
            for label_path in tqdm(label_files, desc=f"Memproses {class_name}"):
                if not label_path.exists():
                    continue
                    
                # Baca label
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Format: class_id, x_center, y_center, width, height
                                file_cls_id = int(float(parts[0]))
                                
                                # Hanya proses jika kelas cocok
                                if file_cls_id != cls_id:
                                    continue
                                    
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                
                                # Konversi koordinat dinormalisasi ([0-1]) ke indeks grid heatmap
                                grid_x = int(x_center * (self.resolution[0] - 1))
                                grid_y = int(y_center * (self.resolution[1] - 1))
                                
                                # Tambahkan ke heatmap
                                heatmap[grid_y, grid_x] += 1
                                
                            except (ValueError, IndexError):
                                continue
            
            # Terapkan gaussian blur
            if gaussian_sigma > 0:
                try:
                    from scipy.ndimage import gaussian_filter
                    heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
                except ImportError:
                    pass
            
            # Normalisasi heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
                
            # Plot pada subplot yang sesuai
            ax = axes[i]
            im = ax.imshow(
                heatmap, 
                cmap='hot', 
                interpolation='nearest',
                alpha=0.7,
                extent=[0, 1, 1, 0]  # Sesuai dengan koordinat YOLO
            )
            
            # Tambahkan colorbar
            fig.colorbar(im, ax=ax, label='Kepadatan')
            
            # Judul dan label
            ax.set_title(f"{class_name} ({count} objek)")
            ax.set_xlabel('Posisi X')
            ax.set_ylabel('Posisi Y')
            ax.grid(False)
        
        # Nonaktifkan subplot kosong
        for i in range(len(top_classes), len(axes)):
            axes[i].set_visible(False)
            
        # Tambahkan judul utama
        layer_str = f" (Layer: {', '.join(layer_filter)})" if layer_filter else ""
        plt.suptitle(f"Distribusi Spasial per Kelas - {split.capitalize()}{layer_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = f"_layer{'_'.join(layer_filter)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"heatmap_spatial_multiclass_{split}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Buat ruang untuk judul utama
        
        return self.save_plot(fig, save_path)