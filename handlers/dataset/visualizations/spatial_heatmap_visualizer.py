# File: handlers/dataset/visualizations/spatial_heatmap_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk heatmap spasial distribusi objek dalam dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase

class SpatialHeatmapVisualizer(VisualizationBase):
    """
    Visualizer untuk membuat heatmap distribusi spasial objek dalam dataset.
    
    Memungkinkan untuk melihat di mana posisi objek cenderung muncul dalam gambar,
    berguna untuk memahami bias spasial dalam dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        resolution: Tuple[int, int] = (32, 32)  # Resolusi default heatmap
    ):
        """
        Inisialisasi SpatialHeatmapVisualizer.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
            resolution: Resolusi heatmap (jumlah sel sumbu X dan Y)
        """
        super().__init__(data_dir, output_dir, logger)
        self.resolution = resolution
        self.layer_config = get_layer_config()
        
        self.logger.info(f"🔥 SpatialHeatmapVisualizer diinisialisasi dengan resolusi {resolution}")
    
    def generate_heatmap(
        self,
        split: str = 'train',
        class_filter: Optional[List[str]] = None,
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
            class_filter: Filter berdasarkan kelas tertentu (opsional)
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
        
        # Siapkan filter kelas dan layer
        class_indices = self._prepare_class_filter(class_filter)
        layer_names = self._prepare_layer_filter(layer_filter) 
        
        # Log informasi filter
        filter_info = []
        if class_indices:
            class_names = [self._get_class_name(idx) for idx in class_indices]
            filter_info.append(f"Kelas: {', '.join(class_names[:5])}")
            if len(class_names) > 5:
                filter_info.append(f"dan {len(class_names) - 5} lainnya")
        
        if layer_names:
            filter_info.append(f"Layer: {', '.join(layer_names)}")
            
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
                            if class_indices and cls_id not in class_indices:
                                continue
                                
                            # Terapkan filter layer jika ada
                            if layer_names:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_names:
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
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
        
        # Normalisasi heatmap
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Buat visualisasi
        plt.figure(figsize=figsize)
        
        # Plot heatmap menggunakan imshow
        im = plt.imshow(
            heatmap, 
            cmap=colormap, 
            interpolation='nearest',
            alpha=alpha,
            extent=[0, 1, 1, 0]  # Sesuai dengan koordinat YOLO
        )
        
        # Tambahkan colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Kepadatan Normalisasi' if normalize else 'Jumlah Objek')
        
        # Grid dan labels
        plt.grid(False)
        plt.xlabel('Posisi X')
        plt.ylabel('Posisi Y')
        
        # Judul dan informasi
        title_parts = [f"Distribusi Spasial Objek - {split.capitalize()}"]
        if class_filter:
            title_parts.append(f"Kelas: {len(class_indices)} dipilih")
        if layer_filter:
            title_parts.append(f"Layer: {', '.join(layer_names)}")
            
        plt.title(" | ".join(title_parts))
        
        # Tambahkan informasi total objek
        plt.text(
            0.02, 0.02, 
            f"Total objek: {total_objects}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            class_str = f"_cls{len(class_indices)}" if class_indices else ""
            layer_str = f"_layer{'_'.join(layer_names)}" if layer_names else ""
            timestamp = self._get_timestamp()
            
            filename = f"heatmap_spatial_{split}{class_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        self.logger.success(f"✅ Heatmap spasial berhasil dibuat: {save_path}")
        
        return save_path
    
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
        
        # Siapkan filter layer
        layer_names = self._prepare_layer_filter(layer_filter)
        
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
                            if layer_names:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_names:
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
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
            
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
        layer_str = f" (Layer: {', '.join(layer_names)})" if layer_names else ""
        plt.suptitle(f"Distribusi Spasial per Kelas - {split.capitalize()}{layer_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = f"_layer{'_'.join(layer_names)}" if layer_names else ""
            timestamp = self._get_timestamp()
            
            filename = f"heatmap_spatial_multiclass_{split}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Buat ruang untuk judul utama
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
            
        self.logger.success(f"✅ Heatmap spasial multi-kelas berhasil dibuat: {save_path}")
        
        return save_path
    
    def generate_size_heatmap(
        self,
        split: str = 'train',
        class_filter: Optional[List[str]] = None,
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> str:
        """
        Membuat visualisasi heatmap distribusi ukuran objek.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            class_filter: Filter berdasarkan kelas tertentu (opsional)
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            normalize: Normalisasi intensitas heatmap
            
        Returns:
            Path file gambar yang disimpan
        """
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Siapkan filter kelas dan layer
        class_indices = self._prepare_class_filter(class_filter)
        layer_names = self._prepare_layer_filter(layer_filter)
        
        # Siapkan array untuk ukuran objek
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Memproses ukuran objek di {split}"):
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
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Terapkan filter kelas jika ada
                            if class_indices and cls_id not in class_indices:
                                continue
                                
                            # Terapkan filter layer jika ada
                            if layer_names:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_names:
                                    continue
                            
                            # Catat dimensi objek
                            widths.append(width)
                            heights.append(height)
                            areas.append(width * height)
                            aspect_ratios.append(width / height if height > 0 else 0)
                            
                        except (ValueError, IndexError, ZeroDivisionError):
                            continue
        
        if not widths:
            self.logger.warning(f"⚠️ Tidak ada objek yang memenuhi filter di split '{split}'")
            return ""
        
        # Log statistik
        self.logger.info(
            f"📊 Statistik ukuran objek (N={len(widths)}):\n"
            f"   • Width: min={min(widths):.3f}, max={max(widths):.3f}, avg={np.mean(widths):.3f}\n"
            f"   • Height: min={min(heights):.3f}, max={max(heights):.3f}, avg={np.mean(heights):.3f}\n"
            f"   • Area: min={min(areas):.5f}, max={max(areas):.5f}, avg={np.mean(areas):.5f}\n"
            f"   • Aspect Ratio: min={min(aspect_ratios):.3f}, max={max(aspect_ratios):.3f}, avg={np.mean(aspect_ratios):.3f}"
        )
        
        # Buat subplot untuk tiga heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. 2D Heatmap untuk width vs height
        h = ax1.hist2d(
            widths, 
            heights, 
            bins=20, 
            cmap='viridis',
            cmin=1
        )
        ax1.set_title('Distribusi Dimensi Objek')
        ax1.set_xlabel('Lebar (dinormalisasi)')
        ax1.set_ylabel('Tinggi (dinormalisasi)')
        cbar1 = plt.colorbar(h[3], ax=ax1)
        cbar1.set_label('Jumlah Objek')
        
        # 2. Scatter plot untuk area vs aspect ratio
        scatter = ax2.scatter(
            areas, 
            aspect_ratios, 
            c=areas, 
            s=20, 
            cmap='plasma',
            alpha=0.6
        )
        ax2.set_title('Area vs Rasio Aspek')
        ax2.set_xlabel('Area (lebar × tinggi)')
        ax2.set_ylabel('Rasio Aspek (lebar / tinggi)')
        cbar2 = plt.colorbar(scatter, ax=ax2)
        cbar2.set_label('Area')
        
        # Tambahkan judul utama
        filter_parts = []
        if class_indices:
            class_names = [self._get_class_name(idx) for idx in class_indices]
            if len(class_names) <= 3:
                filter_parts.append(f"Kelas: {', '.join(class_names)}")
            else:
                filter_parts.append(f"Kelas: {len(class_names)} dipilih")
                
        if layer_names:
            filter_parts.append(f"Layer: {', '.join(layer_names)}")
            
        filter_str = f" ({', '.join(filter_parts)})" if filter_parts else ""
        
        plt.suptitle(f"Analisis Ukuran Objek - {split.capitalize()}{filter_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            class_str = f"_cls{len(class_indices)}" if class_indices else ""
            layer_str = f"_layer{'_'.join(layer_names)}" if layer_names else ""
            timestamp = self._get_timestamp()
            
            filename = f"size_analysis_{split}{class_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Buat ruang untuk judul utama
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
            
        self.logger.success(f"✅ Analisis ukuran objek berhasil dibuat: {save_path}")
        
        return save_path
    
    def _prepare_class_filter(self, class_filter: Optional[List[str]]) -> List[int]:
        """
        Persiapkan filter kelas dengan mengkonversi nama kelas ke ID.
        
        Args:
            class_filter: List nama kelas untuk filter
            
        Returns:
            List ID kelas
        """
        if not class_filter:
            return []
            
        class_indices = []
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            classes = layer_config['classes']
            class_ids = layer_config['class_ids']
            
            for i, class_name in enumerate(classes):
                if class_name in class_filter and i < len(class_ids):
                    class_indices.append(class_ids[i])
        
        return class_indices
    
    def _prepare_layer_filter(self, layer_filter: Optional[List[str]]) -> List[str]:
        """
        Persiapkan filter layer.
        
        Args:
            layer_filter: List nama layer untuk filter
            
        Returns:
            List nama layer yang valid
        """
        if not layer_filter:
            return []
            
        valid_layers = self.layer_config.get_layer_names()
        return [layer for layer in layer_filter if layer in valid_layers]
    
    def _get_class_name(self, cls_id: int) -> str:
        """
        Dapatkan nama kelas dari ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama kelas
        """
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            class_ids = layer_config['class_ids']
            classes = layer_config['classes']
            
            if cls_id in class_ids:
                idx = class_ids.index(cls_id)
                if idx < len(classes):
                    return classes[idx]
                    
        return f"Kelas-{cls_id}"
    
    def _get_layer_for_class(self, cls_id: int) -> Optional[str]:
        """
        Dapatkan nama layer untuk ID kelas tertentu.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama layer atau None jika tidak ditemukan
        """
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                return layer_name
                
        return None