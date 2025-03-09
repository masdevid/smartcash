# File: smartcash/handlers/dataset/visualizations/heatmap/size_distribution_heatmap.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk heatmap distribusi ukuran objek dalam dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase

class SizeDistributionHeatmap(VisualizationBase):
    """
    Visualizer untuk membuat heatmap distribusi ukuran objek dalam dataset.
    
    Memungkinkan melihat distribusi ukuran bounding box, rasio aspek,
    dan hubungan antara ukuran dan kelas objek.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi SizeDistributionHeatmap.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(data_dir, output_dir, logger)
        
        self.logger.info(f"📏 SizeDistributionHeatmap diinisialisasi")
    
    def generate_size_heatmap(
        self,
        split: str = 'train',
        class_filter: Optional[List[int]] = None,
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Membuat visualisasi heatmap distribusi ukuran objek.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            class_filter: Filter berdasarkan ID kelas tertentu (opsional)
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_plot: Tampilkan plot secara langsung
            
        Returns:
            Path file gambar yang disimpan
        """
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
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
        self.logger.info(f"📊 Membuat distribusi ukuran objek untuk split '{split}'{filter_str}")
        
        # Siapkan array untuk ukuran objek
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        class_sizes = {}  # Dict untuk ukuran per kelas
        
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
                            if class_filter and cls_id not in class_filter:
                                continue
                                
                            # Terapkan filter layer jika ada
                            if layer_filter:
                                class_layer = self._get_layer_for_class(cls_id)
                                if class_layer not in layer_filter:
                                    continue
                            
                            # Catat dimensi objek
                            widths.append(width)
                            heights.append(height)
                            area = width * height
                            areas.append(area)
                            aspect_ratio = width / height if height > 0 else 0
                            aspect_ratios.append(aspect_ratio)
                            
                            # Catat untuk analisis per kelas
                            class_name = self._get_class_name(cls_id)
                            if class_name not in class_sizes:
                                class_sizes[class_name] = {'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': []}
                            
                            class_sizes[class_name]['widths'].append(width)
                            class_sizes[class_name]['heights'].append(height)
                            class_sizes[class_name]['areas'].append(area)
                            class_sizes[class_name]['aspect_ratios'].append(aspect_ratio)
                            
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
        if class_filter:
            class_names = [self._get_class_name(idx) for idx in class_filter]
            if len(class_names) <= 3:
                filter_parts.append(f"Kelas: {', '.join(class_names)}")
            else:
                filter_parts.append(f"Kelas: {len(class_names)} dipilih")
                
        if layer_filter:
            filter_parts.append(f"Layer: {', '.join(layer_filter)}")
            
        filter_str = f" ({', '.join(filter_parts)})" if filter_parts else ""
        
        plt.suptitle(f"Analisis Ukuran Objek - {split.capitalize()}{filter_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            class_str = f"_cls{len(class_filter)}" if class_filter else ""
            layer_str = f"_layer{'_'.join(layer_filter)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"size_analysis_{split}{class_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Buat ruang untuk judul utama
        
        result_path = self.save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return result_path
    
    def generate_class_size_comparison(
        self,
        split: str = 'train',
        top_n_classes: int = 10,
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
        metric: str = 'area'  # 'area', 'width', 'height', atau 'aspect_ratio'
    ) -> str:
        """
        Membuat visualisasi perbandingan ukuran objek antar kelas.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            top_n_classes: Jumlah kelas teratas untuk ditampilkan
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            metric: Metrik ukuran yang dianalisis ('area', 'width', 'height', 'aspect_ratio')
            
        Returns:
            Path file gambar yang disimpan
        """
        # Validasi metrik
        valid_metrics = ['area', 'width', 'height', 'aspect_ratio']
        if metric not in valid_metrics:
            raise ValueError(f"Metrik tidak valid: {metric}. Gunakan salah satu dari {valid_metrics}")
        
        metric_labels = {
            'area': 'Area (lebar × tinggi)',
            'width': 'Lebar',
            'height': 'Tinggi',
            'aspect_ratio': 'Rasio Aspek (lebar / tinggi)'
        }
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        self.logger.info(f"📊 Membandingkan {metric_labels[metric]} objek antar kelas di split '{split}'")
        
        # Hitung frekuensi kelas dan ukuran
        class_metrics = {}  # {class_id: {count: N, metrics: [values]}}
        
        # Terapkan filter layer jika ada
        valid_layers = layer_filter or self.layer_config.get_layer_names()
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Menganalisis ukuran objek di {split}"):
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
                            
                            # Terapkan filter layer jika ada
                            class_layer = self._get_layer_for_class(cls_id)
                            if class_layer not in valid_layers:
                                continue
                            
                            # Hitung metrik yang diminta
                            if metric == 'area':
                                metric_value = width * height
                            elif metric == 'width':
                                metric_value = width
                            elif metric == 'height':
                                metric_value = height
                            elif metric == 'aspect_ratio':
                                metric_value = width / height if height > 0 else 0
                            
                            # Catat untuk analisis per kelas
                            if cls_id not in class_metrics:
                                class_metrics[cls_id] = {'count': 0, 'metrics': []}
                            
                            class_metrics[cls_id]['count'] += 1
                            class_metrics[cls_id]['metrics'].append(metric_value)
                            
                        except (ValueError, IndexError, ZeroDivisionError):
                            continue
        
        if not class_metrics:
            self.logger.warning(f"⚠️ Tidak ada objek yang memenuhi filter di split '{split}'")
            return ""
        
        # Hitung statistik untuk setiap kelas
        stats_by_class = {}
        for cls_id, data in class_metrics.items():
            if data['count'] > 0:
                metrics = np.array(data['metrics'])
                
                stats_by_class[cls_id] = {
                    'count': data['count'],
                    'mean': np.mean(metrics),
                    'median': np.median(metrics),
                    'std': np.std(metrics),
                    'min': np.min(metrics),
                    'max': np.max(metrics),
                    'name': self._get_class_name(cls_id)
                }
        
        # Pilih kelas terbanyak untuk visualisasi
        top_classes = sorted(stats_by_class.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n_classes]
        
        # Siapkan data untuk box plot
        boxplot_data = []
        class_labels = []
        
        for cls_id, stats in top_classes:
            boxplot_data.append(class_metrics[cls_id]['metrics'])
            class_labels.append(f"{stats['name']} (n={stats['count']})")
        
        # Buat visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot
        bp = ax.boxplot(
            boxplot_data,
            vert=True,
            patch_artist=True,
            labels=class_labels,
            showfliers=True
        )
        
        # Warnai box plots
        for i, box in enumerate(bp['boxes']):
            color_idx = i % len(self.color_palette)
            box.set(facecolor=self.color_palette[color_idx])
        
        # Tambahkan judul dan label
        ax.set_title(f"Distribusi {metric_labels[metric]} per Kelas - {split.capitalize()}")
        ax.set_ylabel(metric_labels[metric])
        
        # Rotasi label jika terlalu banyak
        if len(class_labels) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Tambahkan informasi filter layer
        if layer_filter:
            plt.figtext(
                0.98, 0.02, 
                f"Filter layer: {', '.join(valid_layers)}",
                fontsize=10,
                ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Tambahkan grid
        ax.yaxis.grid(True)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = f"_layer{'_'.join(valid_layers)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"size_comparison_{metric}_{split}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout()
        
        return self.save_plot(fig, save_path)
    
    def generate_size_distribution_matrix(
        self,
        split: str = 'train',
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None,
        top_n_classes: int = 6
    ) -> str:
        """
        Membuat visualisasi matriks distribusi ukuran untuk beberapa kelas sekaligus.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            layer_filter: Filter berdasarkan layer tertentu (opsional)
            figsize: Ukuran gambar output
            save_path: Path untuk menyimpan visualisasi (opsional)
            top_n_classes: Jumlah kelas teratas untuk ditampilkan
            
        Returns:
            Path file gambar yang disimpan
        """
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        self.logger.info(f"📊 Membuat matriks distribusi ukuran per kelas di split '{split}'")
        
        # Terapkan filter layer jika ada
        valid_layers = layer_filter or self.layer_config.get_layer_names()
        
        # Kumpulkan ukuran dan frekuensi kelas
        class_sizes = {}  # {class_id: {count: N, widths: [], heights: []}}
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Menganalisis ukuran objek di {split}"):
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
                            
                            # Terapkan filter layer jika ada
                            class_layer = self._get_layer_for_class(cls_id)
                            if class_layer not in valid_layers:
                                continue
                            
                            # Catat untuk analisis per kelas
                            if cls_id not in class_sizes:
                                class_sizes[cls_id] = {'count': 0, 'widths': [], 'heights': []}
                            
                            class_sizes[cls_id]['count'] += 1
                            class_sizes[cls_id]['widths'].append(width)
                            class_sizes[cls_id]['heights'].append(height)
                            
                        except (ValueError, IndexError, ZeroDivisionError):
                            continue
        
        if not class_sizes:
            self.logger.warning(f"⚠️ Tidak ada objek yang memenuhi filter di split '{split}'")
            return ""
        
        # Pilih kelas teratas berdasarkan jumlah
        top_classes = sorted(
            [(cls_id, data['count'], self._get_class_name(cls_id)) 
             for cls_id, data in class_sizes.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n_classes]
        
        # Siapkan jumlah baris dan kolom untuk subplots
        n_rows = (len(top_classes) + 1) // 2
        n_cols = min(2, len(top_classes))
        
        # Buat grid subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes jika perlu
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot untuk setiap kelas teratas
        for i, (cls_id, count, class_name) in enumerate(top_classes):
            if i < len(axes):
                ax = axes[i]
                widths = class_sizes[cls_id]['widths']
                heights = class_sizes[cls_id]['heights']
                
                # Heatmap 2D untuk width vs height
                h = ax.hist2d(
                    widths, 
                    heights, 
                    bins=15, 
                    cmap='viridis',
                    cmin=1
                )
                ax.set_title(f"{class_name} (n={count})")
                ax.set_xlabel('Lebar')
                ax.set_ylabel('Tinggi')
                fig.colorbar(h[3], ax=ax)
        
        # Hide unused subplots
        for i in range(len(top_classes), len(axes)):
            axes[i].set_visible(False)
        
        # Tambahkan judul utama
        layer_str = f" (Layer: {', '.join(valid_layers)})" if layer_filter else ""
        plt.suptitle(f"Distribusi Ukuran Objek per Kelas - {split.capitalize()}{layer_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = f"_layer{'_'.join(valid_layers)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"size_matrix_{split}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
            
        # Simpan plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Buat ruang untuk judul utama
        
        return self.save_plot(fig, save_path)