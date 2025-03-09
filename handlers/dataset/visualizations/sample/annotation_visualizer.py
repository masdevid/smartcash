# File: smartcash/handlers/dataset/visualizations/sample/annotation_visualizer.py
# Author: Alfrida Sabar
# Deskripsi: Visualizer untuk anotasi gambar dari dataset dengan berbagai opsi visualisasi

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import random

from smartcash.utils.logger import get_logger
from smartcash.handlers.dataset.visualizations.visualization_base import VisualizationBase


class AnnotationVisualizer(VisualizationBase):
    """
    Visualizer khusus untuk anotasi gambar dari dataset.
    
    Mendukung berbagai mode visualisasi termasuk:
    - Visualisasi layer terpisah
    - Visualisasi hanya kelas spesifik
    - Visualisasi dengan warna berbeda per kelas/layer
    - Indikator kualitas anotasi
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi AnnotationVisualizer.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(data_dir, output_dir, logger)
        
        # Setup warna untuk berbagai kelas
        self.class_colors = {}  # Akan diinisialisasi dinamis saat pertama digunakan
        
        self.logger.info(f"🎨 AnnotationVisualizer diinisialisasi")
    
    def visualize_single_image_annotations(
        self,
        image_path: Union[str, Path],
        label_path: Optional[Union[str, Path]] = None,
        show_color_per_class: bool = True,
        layer_filter: Optional[List[str]] = None,
        class_filter: Optional[List[int]] = None,
        show_labels: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        dpi: int = 150
    ) -> str:
        """
        Visualisasikan anotasi untuk satu gambar tunggal.
        
        Args:
            image_path: Path ke gambar
            label_path: Path ke file label (jika None, dicari dari nama gambar)
            show_color_per_class: Gunakan warna berbeda per kelas (jika False, per layer)
            layer_filter: Filter berdasarkan layer tertentu
            class_filter: Filter berdasarkan ID kelas tertentu
            show_labels: Tampilkan label teks
            figsize: Ukuran gambar
            save_path: Path untuk menyimpan visualisasi (opsional)
            title: Judul custom (opsional)
            dpi: Resolusi output
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        # Konversi ke Path
        image_path = Path(image_path)
        
        # Tentukan label path jika tidak disediakan
        if label_path is None:
            label_path = Path(os.path.splitext(str(image_path))[0] + '.txt')
        else:
            label_path = Path(label_path)
        
        # Pastikan gambar dan label ada
        if not image_path.exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")
        
        if not label_path.exists():
            self.logger.warning(f"⚠️ File label tidak ditemukan: {label_path}. Visualisasi tanpa anotasi.")
        
        self.logger.info(f"🎨 Memvisualisasikan anotasi untuk: {image_path.name}")
        
        # Baca gambar
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Gagal membaca gambar: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error saat membaca gambar: {str(e)}")
        
        # Siapkan visualisasi
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        
        # Set judul
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Anotasi: {image_path.name}")
        
        # Proses label
        if label_path.exists():
            # Ukuran gambar untuk normalisasi
            img_h, img_w = img.shape[:2]
            
            # Siapkan set kelas yang ditemukan untuk legenda
            found_classes = set()
            found_layers = set()
            
            # Baca label
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            # Format YOLO: class_id, x_center, y_center, width, height
                            cls_id = int(float(parts[0]))
                            
                            # Filter berdasarkan kelas jika diperlukan
                            if class_filter and cls_id not in class_filter:
                                continue
                            
                            # Dapatkan layer untuk kelas ini
                            layer = self._get_layer_for_class(cls_id)
                            
                            # Filter berdasarkan layer jika diperlukan
                            if layer_filter and layer not in layer_filter:
                                continue
                            
                            # Dapatkan nama kelas
                            class_name = self._get_class_name(cls_id)
                            
                            # Koordinat dan dimensi bbox
                            x_center = float(parts[1]) * img_w
                            y_center = float(parts[2]) * img_h
                            width = float(parts[3]) * img_w
                            height = float(parts[4]) * img_h
                            
                            # Konversi ke format (x, y, width, height) untuk rectangle
                            x = x_center - width / 2
                            y = y_center - height / 2
                            
                            # Pilih warna berdasarkan mode
                            if show_color_per_class:
                                # Warna per kelas: gunakan konfigurasi atau generate dinamis
                                if cls_id not in self.class_colors:
                                    # Generate random color
                                    r = random.random()
                                    g = random.random()
                                    b = random.random()
                                    self.class_colors[cls_id] = (r, g, b)
                                
                                color = self.class_colors[cls_id]
                            else:
                                # Warna per layer
                                color = self.layer_colors.get(layer, self.layer_colors['default'])
                            
                            # Buat dan tambahkan rectangle
                            rect = patches.Rectangle(
                                (x, y),
                                width,
                                height,
                                linewidth=2,
                                edgecolor=color,
                                facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            # Tambahkan label teks jika diminta
                            if show_labels:
                                # Tentukan posisi label (di atas box)
                                text_y = y - 5 if y > 15 else y + 15
                                
                                # Create background for text
                                text_bg_color = to_rgba(color, alpha=0.8)
                                
                                # Tentukan label teks
                                if show_color_per_class:
                                    label_text = class_name
                                else:
                                    label_text = f"{class_name} ({layer})"
                                
                                # Tambahkan teks
                                ax.text(
                                    x, text_y,
                                    label_text,
                                    fontsize=10,
                                    color='white',
                                    fontweight='bold',
                                    bbox=dict(
                                        facecolor=text_bg_color,
                                        edgecolor='none',
                                        boxstyle='round,pad=0.3'
                                    )
                                )
                            
                            # Catat kelas dan layer untuk legenda
                            found_classes.add((cls_id, class_name))
                            found_layers.add(layer)
                                
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"⚠️ Error membaca label: {str(e)}")
                            continue
            
            # Tambahkan legenda
            if found_classes and show_color_per_class:
                legend_elements = []
                for cls_id, class_name in sorted(found_classes, key=lambda x: x[0]):
                    color = self.class_colors[cls_id]
                    legend_elements.append(
                        patches.Patch(facecolor=to_rgba(color, alpha=0.5), edgecolor=color, label=f"{class_name}")
                    )
                
                if legend_elements:
                    ax.legend(
                        handles=legend_elements,
                        loc='upper right',
                        title="Kelas"
                    )
            elif found_layers and not show_color_per_class:
                legend_elements = []
                for layer in sorted(found_layers):
                    color = self.layer_colors.get(layer, self.layer_colors['default'])
                    legend_elements.append(
                        patches.Patch(facecolor=to_rgba(color, alpha=0.5), edgecolor=color, label=layer)
                    )
                
                if legend_elements:
                    ax.legend(
                        handles=legend_elements,
                        loc='upper right',
                        title="Layer"
                    )
        
        # Nonaktifkan tick label
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tambahkan informasi filter
        filters = []
        if layer_filter:
            filters.append(f"Layer: {', '.join(layer_filter)}")
        if class_filter:
            class_names = [self._get_class_name(cls_id) for cls_id in class_filter]
            if len(class_names) <= 5:
                filters.append(f"Kelas: {', '.join(class_names)}")
            else:
                filters.append(f"Kelas: {len(class_names)} dipilih")
                
        if filters:
            filter_text = " | ".join(filters)
            plt.figtext(
                0.5, 0.01,
                filter_text,
                ha='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            mode = "class" if show_color_per_class else "layer"
            timestamp = self._get_timestamp()
            filename = f"annotation_{image_path.stem}_{mode}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan dan return path
        return self.save_plot(fig, save_path, dpi=dpi)
    
    def visualize_layer_comparison(
        self,
        image_path: Union[str, Path],
        label_path: Optional[Union[str, Path]] = None,
        layers: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
        show_labels: bool = True
    ) -> str:
        """
        Visualisasikan anotasi untuk beberapa layer secara side-by-side.
        
        Args:
            image_path: Path ke gambar
            label_path: Path ke file label (jika None, dicari dari nama gambar)
            layers: List layer yang akan divisualisasikan (jika None, semua layer)
            figsize: Ukuran gambar
            save_path: Path untuk menyimpan visualisasi (opsional)
            show_labels: Tampilkan label teks
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        # Konversi ke Path
        image_path = Path(image_path)
        
        # Tentukan label path jika tidak disediakan
        if label_path is None:
            label_path = Path(os.path.splitext(str(image_path))[0] + '.txt')
        else:
            label_path = Path(label_path)
        
        # Pastikan gambar dan label ada
        if not image_path.exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")
        
        if not label_path.exists():
            self.logger.warning(f"⚠️ File label tidak ditemukan: {label_path}. Visualisasi tanpa anotasi.")
        
        # Tentukan layer yang akan divisualisasikan
        if not layers:
            layers = self.layer_config.get_layer_names()
        
        self.logger.info(f"🎨 Memvisualisasikan perbandingan layer untuk: {image_path.name}")
        
        # Baca gambar
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Gagal membaca gambar: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error saat membaca gambar: {str(e)}")
        
        # Buat subplot untuk setiap layer + satu untuk semua layer
        n_cols = min(3, len(layers) + 1)
        n_rows = (len(layers) + 2) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot semua layer terlebih dahulu
        ax_all = axes[0]
        ax_all.imshow(img)
        ax_all.set_title("Semua Layer")
        
        # Proses label untuk semua layer
        if label_path.exists():
            # Ukuran gambar untuk normalisasi
            img_h, img_w = img.shape[:2]
            
            # Siapkan data anotasi per layer
            annotations_by_layer = {layer: [] for layer in layers}
            
            # Baca label
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            # Format YOLO: class_id, x_center, y_center, width, height
                            cls_id = int(float(parts[0]))
                            
                            # Dapatkan layer untuk kelas ini
                            layer = self._get_layer_for_class(cls_id)
                            
                            # Skip jika layer tidak dalam daftar
                            if layer not in layers:
                                continue
                            
                            # Dapatkan nama kelas
                            class_name = self._get_class_name(cls_id)
                            
                            # Koordinat dan dimensi bbox
                            x_center = float(parts[1]) * img_w
                            y_center = float(parts[2]) * img_h
                            width = float(parts[3]) * img_w
                            height = float(parts[4]) * img_h
                            
                            # Konversi ke format (x, y, width, height) untuk rectangle
                            x = x_center - width / 2
                            y = y_center - height / 2
                            
                            # Simpan data anotasi
                            annotations_by_layer[layer].append({
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'class_id': cls_id,
                                'class_name': class_name,
                                'layer': layer
                            })
                            
                            # Tambahkan ke plot semua layer
                            color = self.layer_colors.get(layer, self.layer_colors['default'])
                            
                            # Buat dan tambahkan rectangle
                            rect = patches.Rectangle(
                                (x, y),
                                width,
                                height,
                                linewidth=2,
                                edgecolor=color,
                                facecolor='none'
                            )
                            ax_all.add_patch(rect)
                            
                            # Tambahkan label teks jika diminta
                            if show_labels:
                                # Tentukan posisi label (di atas box)
                                text_y = y - 5 if y > 15 else y + 15
                                
                                # Create background for text
                                text_bg_color = to_rgba(color, alpha=0.8)
                                
                                # Tambahkan teks
                                ax_all.text(
                                    x, text_y,
                                    class_name,
                                    fontsize=9,
                                    color='white',
                                    fontweight='bold',
                                    bbox=dict(
                                        facecolor=text_bg_color,
                                        edgecolor='none',
                                        boxstyle='round,pad=0.2'
                                    )
                                )
                                
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"⚠️ Error membaca label: {str(e)}")
                            continue
            
            # Tambahkan legenda untuk semua layer
            legend_elements = []
            for layer in layers:
                color = self.layer_colors.get(layer, self.layer_colors['default'])
                legend_elements.append(
                    patches.Patch(facecolor=to_rgba(color, alpha=0.5), edgecolor=color, label=layer)
                )
            
            if legend_elements:
                ax_all.legend(
                    handles=legend_elements,
                    loc='upper right',
                    title="Layer"
                )
            
            # Plot setiap layer secara terpisah
            for i, layer in enumerate(layers):
                if i + 1 < len(axes):
                    ax = axes[i + 1]
                    ax.imshow(img)
                    ax.set_title(f"Layer: {layer}")
                    
                    color = self.layer_colors.get(layer, self.layer_colors['default'])
                    
                    # Tambahkan anotasi untuk layer ini
                    for anno in annotations_by_layer[layer]:
                        # Buat dan tambahkan rectangle
                        rect = patches.Rectangle(
                            (anno['x'], anno['y']),
                            anno['width'],
                            anno['height'],
                            linewidth=2,
                            edgecolor=color,
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Tambahkan label teks jika diminta
                        if show_labels:
                            # Tentukan posisi label (di atas box)
                            text_y = anno['y'] - 5 if anno['y'] > 15 else anno['y'] + 15
                            
                            # Create background for text
                            text_bg_color = to_rgba(color, alpha=0.8)
                            
                            # Tambahkan teks
                            ax.text(
                                anno['x'], text_y,
                                anno['class_name'],
                                fontsize=9,
                                color='white',
                                fontweight='bold',
                                bbox=dict(
                                    facecolor=text_bg_color,
                                    edgecolor='none',
                                    boxstyle='round,pad=0.2'
                                )
                            )
                    
                    # Tambahkan informasi jumlah objek
                    num_objects = len(annotations_by_layer[layer])
                    ax.text(
                        0.05, 0.95,
                        f"Objek: {num_objects}",
                        transform=ax.transAxes,
                        fontsize=10,
                        va='top',
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
        
        # Nonaktifkan axis untuk semua subplot
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Sembunyikan axis yang tidak terpakai
        for i in range(len(layers) + 1, len(axes)):
            axes[i].axis('off')
        
        # Tambahkan judul keseluruhan
        plt.suptitle(f"Perbandingan Layer Anotasi: {image_path.name}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = '_'.join([l[:3] for l in layers])  # Singkatan nama layer
            timestamp = self._get_timestamp()
            filename = f"layer_comparison_{image_path.stem}_{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan dan return path
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Beri ruang untuk judul
        return self.save_plot(fig, save_path)
    
    def visualize_annotation_density(
        self,
        image_path: Union[str, Path],
        label_path: Optional[Union[str, Path]] = None,
        include_original: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        layer_filter: Optional[List[str]] = None,
        alpha: float = 0.7
    ) -> str:
        """
        Visualisasikan heatmap kepadatan anotasi untuk satu gambar.
        
        Args:
            image_path: Path ke gambar
            label_path: Path ke file label (jika None, dicari dari nama gambar)
            include_original: Tampilkan overlay pada gambar asli
            figsize: Ukuran gambar
            save_path: Path untuk menyimpan visualisasi (opsional)
            layer_filter: Filter berdasarkan layer tertentu
            alpha: Transparansi heatmap
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        # Konversi ke Path
        image_path = Path(image_path)
        
        # Tentukan label path jika tidak disediakan
        if label_path is None:
            label_path = Path(os.path.splitext(str(image_path))[0] + '.txt')
        else:
            label_path = Path(label_path)
        
        # Pastikan gambar dan label ada
        if not image_path.exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")
        
        if not label_path.exists():
            self.logger.warning(f"⚠️ File label tidak ditemukan: {label_path}. Visualisasi tanpa anotasi.")
            return ""
        
        self.logger.info(f"🔥 Memvisualisasikan kepadatan anotasi untuk: {image_path.name}")
        
        # Baca gambar
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Gagal membaca gambar: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error saat membaca gambar: {str(e)}")
        
        # Ukuran gambar untuk normalisasi
        img_h, img_w = img.shape[:2]
        
        # Buat heatmap kosong
        heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Proses label
        if label_path.exists():
            # Siapkan set layer yang valid
            valid_layers = layer_filter or self.layer_config.get_layer_names()
            
            # Baca label
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            # Format YOLO: class_id, x_center, y_center, width, height
                            cls_id = int(float(parts[0]))
                            
                            # Dapatkan layer untuk kelas ini
                            layer = self._get_layer_for_class(cls_id)
                            
                            # Filter berdasarkan layer jika diperlukan
                            if layer not in valid_layers:
                                continue
                            
                            # Koordinat dan dimensi bbox
                            x_center = float(parts[1]) * img_w
                            y_center = float(parts[2]) * img_h
                            width = float(parts[3]) * img_w
                            height = float(parts[4]) * img_h
                            
                            # Konversi ke format (x, y, width, height) untuk rectangle
                            x1 = int(max(0, x_center - width / 2))
                            y1 = int(max(0, y_center - height / 2))
                            x2 = int(min(img_w - 1, x_center + width / 2))
                            y2 = int(min(img_h - 1, y_center + height / 2))
                            
                            # Tambahkan ke heatmap
                            heatmap[y1:y2, x1:x2] += 1
                                
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"⚠️ Error membaca label: {str(e)}")
                            continue
        
        # Normalisasi heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Setup plot
        if include_original:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Gambar asli
            ax1.imshow(img)
            ax1.set_title("Gambar Asli")
            
            # Overlay heatmap
            ax2.imshow(img)
            im = ax2.imshow(heatmap, cmap='hot', alpha=alpha)
            ax2.set_title("Kepadatan Anotasi")
            
            # Tambahkan colorbar
            cbar = fig.colorbar(im, ax=ax2)
            cbar.set_label('Kepadatan Relatif')
            
            # Nonaktifkan axis
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Overlay heatmap pada gambar asli
            ax.imshow(img)
            im = ax.imshow(heatmap, cmap='hot', alpha=alpha)
            
            # Tambahkan colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Kepadatan Relatif')
            
            # Nonaktifkan axis
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Tambahkan judul
        layer_str = f" (Layer: {', '.join(valid_layers)})" if layer_filter else ""
        plt.suptitle(f"Kepadatan Anotasi: {image_path.name}{layer_str}", fontsize=16)
        
        # Tentukan path simpan jika tidak diberikan
        if save_path is None:
            layer_str = f"_layer{'_'.join(valid_layers)}" if layer_filter else ""
            timestamp = self._get_timestamp()
            filename = f"density_{image_path.stem}{layer_str}_{timestamp}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        # Simpan dan return path
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Beri ruang untuk judul
        return self.save_plot(fig, save_path)