# File: smartcash/handlers/dataset/visualizations/heatmap/class_density_heatmap.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk heatmap kepadatan kelas dalam dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase

class ClassDensityHeatmap(VisualizationBase):
    """
    Visualizer untuk membuat heatmap kepadatan kelas dalam dataset.
    
    Memungkinkan melihat distribusi kelas di berbagai bagian dataset,
    seperti distribusi lintas split atau kepadatan kelas untuk setiap layer.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi ClassDensityHeatmap.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(data_dir, output_dir, logger)
        
        self.logger.info(f"🔥 ClassDensityHeatmap diinisialisasi")
    
    def generate_class_layer_heatmap(
        self,
        split: str = 'train',
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        show_plot: bool = False,
        cmap: str = "YlGnBu"
    ) -> str:
        """
        Visualisasikan heatmap korelasi antara kelas dan layer.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            normalize: Normalisasi nilai heatmap (persentase vs. jumlah absolut)
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_plot: Tampilkan plot secara langsung
            cmap: Colormap untuk heatmap
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"🔥 Membuat heatmap kelas-layer untuk split: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Dapatkan semua layer dan kelas
        layers = self.layer_config.get_layer_names()
        classes_by_layer = {}
        class_names = {}
        
        for layer in layers:
            layer_config = self.layer_config.get_layer_config(layer)
            classes_by_layer[layer] = layer_config['class_ids']
            
            for i, cls_id in enumerate(layer_config['class_ids']):
                if i < len(layer_config['classes']):
                    class_names[cls_id] = layer_config['classes'][i]
        
        # Inisialisasi matriks kelas-layer
        matrix = {}
        for layer in layers:
            matrix[layer] = {}
            for cls_id, name in class_names.items():
                matrix[layer][name] = 0
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        total_objects = 0
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Menganalisis label {split}"):
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
                            
                            # Cari layer untuk kelas ini
                            for layer, class_ids in classes_by_layer.items():
                                if cls_id in class_ids:
                                    # Tambahkan ke matriks
                                    class_name = class_names.get(cls_id, f"Class-{cls_id}")
                                    matrix[layer][class_name] += 1
                                    total_objects += 1
                                    break
                                    
                        except (ValueError, IndexError):
                            # Skip entri yang tidak valid
                            continue
        
        # Konversi matriks ke DataFrame untuk visualisasi
        data = []
        for layer, classes in matrix.items():
            for class_name, count in classes.items():
                if count > 0:  # Hanya sertakan kelas dengan objek
                    data.append({
                        'Layer': layer,
                        'Kelas': class_name,
                        'Jumlah': count
                    })
        
        df = pd.DataFrame(data)
        
        # Pivot table untuk heatmap
        pivot_df = df.pivot_table(
            values='Jumlah',
            index='Layer',
            columns='Kelas',
            fill_value=0
        )
        
        # Normalisasi jika diminta
        if normalize and total_objects > 0:
            pivot_df = (pivot_df / total_objects) * 100
        
        # Buat visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        
        # Buat heatmap
        heatmap = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f" if normalize else "d",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={'label': 'Persentase (%)' if normalize else 'Jumlah Objek'}
        )
        
        # Tambahkan detail plot
        plt.title(f'Distribusi Kelas-Layer - {split.capitalize()}')
        plt.ylabel('Layer')
        plt.xlabel('Kelas')
        
        # Rotate x labels if too many
        plt.xticks(rotation=45, ha='right')
        
        # Tambahkan informasi total objek
        plt.figtext(
            0.02, 0.02, 
            f"Total objek: {total_objects}",
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            norm_str = "_norm" if normalize else ""
            timestamp = self._get_timestamp()
            
            filename = f"heatmap_class_layer_{split}{norm_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan plot
        plt.tight_layout()
        
        result_path = self.save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return result_path
    
    def generate_class_distribution_comparison(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        top_n: int = 15,
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show_percentages: bool = True,
        sort_by: str = 'total'  # 'total' atau 'name'
    ) -> str:
        """
        Visualisasikan dan bandingkan distribusi kelas di berbagai split dataset.
        
        Args:
            splits: List split yang akan dibandingkan
            top_n: Jumlah kelas teratas yang akan ditampilkan
            layer_filter: Filter berdasarkan layer tertentu
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_percentages: Tampilkan persentase pada bar
            sort_by: Urutkan berdasarkan ('total' atau 'name')
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Membandingkan distribusi kelas pada split: {', '.join(splits)}")
        
        # Inisialisasi penghitung kelas per split
        class_counts = {split: Counter() for split in splits}
        total_counts = {split: 0 for split in splits}
        class_names_map = {}  # Map dari class_id ke nama kelas
        
        # Kumpulkan layer yang difilter
        if layer_filter:
            valid_layers = [layer for layer in layer_filter if layer in self.layer_config.get_layer_names()]
            if not valid_layers:
                self.logger.warning(f"⚠️ Tidak ada layer valid dalam filter: {layer_filter}")
                valid_layers = self.layer_config.get_layer_names()
        else:
            valid_layers = self.layer_config.get_layer_names()
        
        # Proses setiap split
        for split in splits:
            split_dir = self._get_split_path(split)
            labels_dir = split_dir / 'labels'
            
            if not labels_dir.exists():
                self.logger.warning(f"⚠️ Direktori label tidak ditemukan untuk split {split}: {labels_dir}")
                continue
            
            # Baca semua file label
            label_files = list(labels_dir.glob('*.txt'))
            
            # Progress bar
            for label_path in tqdm(label_files, desc=f"Menganalisis split {split}"):
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
                                
                                # Terapkan filter layer jika ada
                                layer = self._get_layer_for_class(cls_id)
                                if layer not in valid_layers:
                                    continue
                                
                                # Tambahkan ke penghitung
                                class_counts[split][cls_id] += 1
                                total_counts[split] += 1
                                
                                # Catat nama kelas jika belum ada
                                if cls_id not in class_names_map:
                                    class_names_map[cls_id] = self._get_class_name(cls_id)
                                    
                            except (ValueError, IndexError):
                                continue
        
        # Gabungkan semua kelas yang ditemukan
        all_classes = set()
        for counter in class_counts.values():
            all_classes.update(counter.keys())
        
        # Hitung total per kelas (seluruh split)
        class_totals = Counter()
        for counter in class_counts.values():
            class_totals.update(counter)
        
        # Pilih kelas teratas berdasarkan kriteria
        if sort_by == 'total':
            # Urutkan berdasarkan jumlah total
            top_classes = [cls_id for cls_id, _ in class_totals.most_common(top_n)]
        else:
            # Urutkan berdasarkan nama kelas
            sorted_classes = sorted(all_classes, key=lambda x: class_names_map.get(x, f"Class-{x}"))
            top_classes = sorted_classes[:top_n]
        
        # Siapkan data untuk visualisasi
        data = []
        for cls_id in top_classes:
            class_name = class_names_map.get(cls_id, f"Class-{cls_id}")
            
            for split in splits:
                count = class_counts[split][cls_id]
                percentage = (count / max(1, total_counts[split])) * 100
                
                data.append({
                    'Kelas': class_name,
                    'Split': split,
                    'Jumlah': count,
                    'Persentase': percentage
                })
        
        df = pd.DataFrame(data)
        
        # Buat visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot diagram batang
        if show_percentages:
            barplot = sns.barplot(
                x='Kelas',
                y='Persentase',
                hue='Split',
                data=df,
                palette=self.color_palette[:len(splits)],
                ax=ax
            )
            ax.set_ylabel('Persentase (%)')
        else:
            barplot = sns.barplot(
                x='Kelas',
                y='Jumlah',
                hue='Split',
                data=df,
                palette=self.color_palette[:len(splits)],
                ax=ax
            )
            ax.set_ylabel('Jumlah Objek')
        
        # Rotasi label
        plt.xticks(rotation=45, ha='right')
        
        # Tambahkan judul dan label
        ax.set_title(f'Perbandingan Distribusi Kelas')
        ax.set_xlabel('Kelas')
        
        # Tambahkan informasi jumlah objek
        info_text = "Total objek:"
        for split in splits:
            info_text += f" {split}: {total_counts[split]},"
        info_text = info_text.rstrip(',')
        
        plt.figtext(
            0.02, 0.02, 
            info_text,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Tambahkan informasi filter layer
        if layer_filter:
            plt.figtext(
                0.98, 0.02, 
                f"Filter layer: {', '.join(valid_layers)}",
                fontsize=10,
                ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            pct_str = "_pct" if show_percentages else ""
            splits_str = "_".join(splits)
            layer_str = f"_layer{'_'.join(valid_layers)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"distribution_class_comparison{pct_str}_{splits_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan plot
        plt.tight_layout()
        
        return self.save_plot(fig, save_path)
    
    def generate_class_count_distribution(
        self,
        split: str = 'train',
        layer_filter: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None,
        top_n: int = 20,
        horizontal: bool = True,
        show_percentages: bool = True
    ) -> str:
        """
        Visualisasikan distribusi jumlah objek per kelas untuk satu split dataset.
        
        Args:
            split: Split dataset yang akan divisualisasikan
            layer_filter: Filter berdasarkan layer tertentu
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            top_n: Jumlah kelas teratas yang akan ditampilkan
            horizontal: Jika True, gunakan barchart horizontal
            show_percentages: Tampilkan persentase pada bar
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        self.logger.info(f"📊 Memvisualisasikan distribusi kelas untuk split: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        labels_dir = split_dir / 'labels'
        
        if not labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Kumpulkan layer yang difilter
        if layer_filter:
            valid_layers = [layer for layer in layer_filter if layer in self.layer_config.get_layer_names()]
            if not valid_layers:
                self.logger.warning(f"⚠️ Tidak ada layer valid dalam filter: {layer_filter}")
                valid_layers = self.layer_config.get_layer_names()
        else:
            valid_layers = self.layer_config.get_layer_names()
        
        # Inisialisasi penghitung kelas
        class_counts = Counter()
        total_objects = 0
        class_names = {}  # Map dari class_id ke nama kelas
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        for label_path in tqdm(label_files, desc=f"Menganalisis split {split}"):
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
                            
                            # Terapkan filter layer jika ada
                            layer = self._get_layer_for_class(cls_id)
                            if layer not in valid_layers:
                                continue
                            
                            # Tambahkan ke penghitung
                            class_counts[cls_id] += 1
                            total_objects += 1
                            
                            # Catat nama kelas
                            if cls_id not in class_names:
                                class_names[cls_id] = self._get_class_name(cls_id)
                                
                        except (ValueError, IndexError):
                            continue
        
        # Pilih kelas teratas
        top_classes = class_counts.most_common(top_n)
        
        # Siapkan data untuk visualisasi
        class_names_list = [class_names.get(cls_id, f"Class-{cls_id}") for cls_id, _ in top_classes]
        counts = [count for _, count in top_classes]
        
        # Hitung persentase
        percentages = [(count / total_objects) * 100 for count in counts]
        
        # Buat DataFrame
        df = pd.DataFrame({
            'Kelas': class_names_list,
            'Jumlah': counts,
            'Persentase': percentages
        })
        
        # Buat visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        
        # Pilih orientasi batang
        if horizontal:
            if show_percentages:
                barplot = sns.barplot(
                    y='Kelas',
                    x='Persentase',
                    data=df,
                    palette=self.color_palette[:min(len(df), 15)],
                    ax=ax
                )
                
                # Tambahkan label persentase pada batang
                for i, p in enumerate(barplot.patches):
                    width = p.get_width()
                    ax.text(
                        width + 0.5,
                        p.get_y() + p.get_height()/2,
                        f"{width:.1f}% ({counts[i]})",
                        ha='left',
                        va='center'
                    )
                
                ax.set_xlabel('Persentase (%)')
            else:
                barplot = sns.barplot(
                    y='Kelas',
                    x='Jumlah',
                    data=df,
                    palette=self.color_palette[:min(len(df), 15)],
                    ax=ax
                )
                
                ax.set_xlabel('Jumlah Objek')
        else:
            # Orientasi vertikal
            if show_percentages:
                barplot = sns.barplot(
                    x='Kelas',
                    y='Persentase',
                    data=df,
                    palette=self.color_palette[:min(len(df), 15)],
                    ax=ax
                )
                
                # Tambahkan label persentase pada batang
                for i, p in enumerate(barplot.patches):
                    height = p.get_height()
                    ax.text(
                        p.get_x() + p.get_width()/2,
                        height + 0.5,
                        f"{height:.1f}%\n({counts[i]})",
                        ha='center',
                        va='bottom'
                    )
                
                ax.set_ylabel('Persentase (%)')
                plt.xticks(rotation=45, ha='right')
            else:
                barplot = sns.barplot(
                    x='Kelas',
                    y='Jumlah',
                    data=df,
                    palette=self.color_palette[:min(len(df), 15)],
                    ax=ax
                )
                
                ax.set_ylabel('Jumlah Objek')
                plt.xticks(rotation=45, ha='right')
        
        # Tambahkan judul
        layer_str = f" (Layer: {', '.join(valid_layers)})" if layer_filter else ""
        ax.set_title(f'Distribusi Kelas - {split.capitalize()}{layer_str}')
        
        # Tambahkan informasi total objek
        ax.text(
            0.02, 0.02,
            f"Total objek: {total_objects}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            pct_str = "_pct" if show_percentages else ""
            layer_str = f"_layer{'_'.join(valid_layers)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            
            filename = f"distribution_class_{split}{pct_str}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan plot
        plt.tight_layout()
        
        return self.save_plot(fig, save_path)