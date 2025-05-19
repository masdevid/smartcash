"""
File: smartcash/ui/dataset/visualization/handlers/visualization_tab_handler.py
Deskripsi: Handler untuk tab visualisasi dataset
"""

import os
from typing import Dict, Any, List, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import threading

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.dataset.visualization.dashboard.class_visualizer import ClassVisualizer
from smartcash.dataset.visualization.dashboard.layer_visualizer import LayerVisualizer
from smartcash.dataset.visualization.dashboard.bbox_visualizer import BBoxVisualizer
from smartcash.dataset.visualization.dashboard.split_visualizer import SplitVisualizer

logger = get_logger(__name__)


def on_distribution_click(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol distribusi kelas.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        output = ui_components['visualization_components']['distribution_tab']['output']
        
        with output:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi kelas..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Hitung distribusi kelas
            labels_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(labels_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori labels tidak ditemukan"))
                return
            
            # Hitung jumlah file per kelas
            class_counts = {}
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(labels_path, split)
                if os.path.exists(split_path):
                    for label_file in os.listdir(split_path):
                        if label_file.endswith('.txt'):
                            with open(os.path.join(split_path, label_file), 'r') as f:
                                for line in f:
                                    class_id = int(line.strip().split()[0])
                                    class_name = f"Kelas {class_id}"
                                    if class_name not in class_counts:
                                        class_counts[class_name] = 0
                                    class_counts[class_name] += 1
            
            # Tampilkan distribusi kelas
            clear_output(wait=True)
            
            if not class_counts:
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Tidak ada data kelas ditemukan"))
                return
            
            # Gunakan ClassVisualizer untuk plot distribusi kelas
            class_visualizer = ClassVisualizer()
            
            # Plot distribusi kelas
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # Buat data untuk visualizer
            class_data = {name: count for name, count in class_counts.items()}
            class_visualizer.plot_class_distribution(ax, class_data, title="Distribusi Kelas")
            
            plt.tight_layout()
            display(plt.gcf())
            
            # Tampilkan tabel distribusi kelas
            df = pd.DataFrame(list(class_counts.items()), columns=['Kelas', 'Jumlah Objek'])
            df = df.sort_values('Jumlah Objek', ascending=False)
            
            # Tambahkan kolom persentase
            total = df['Jumlah Objek'].sum()
            df['Persentase'] = (df['Jumlah Objek'] / total * 100).round(2).astype(str) + '%'
            
            display(df)
            
    except Exception as e:
        with ui_components['visualization_components']['distribution_tab']['output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi kelas: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))


def on_split_distribution_click(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol distribusi split.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        output = ui_components['visualization_components']['split_tab']['output']
        
        with output:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi split..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Path untuk images dan labels
            images_path = os.path.join(dataset_path, 'images')
            labels_path = os.path.join(dataset_path, 'labels')
            
            # Cek apakah direktori ada
            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori images atau labels tidak ditemukan"))
                return
            
            # Hitung jumlah gambar dan label per split
            split_stats = {
                'train': {'images': 0, 'labels': 0, 'objects': 0},
                'val': {'images': 0, 'labels': 0, 'objects': 0},
                'test': {'images': 0, 'labels': 0, 'objects': 0}
            }
            
            for split in ['train', 'val', 'test']:
                # Hitung jumlah gambar
                split_images_path = os.path.join(images_path, split)
                if os.path.exists(split_images_path):
                    image_files = [f for f in os.listdir(split_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    split_stats[split]['images'] = len(image_files)
                
                # Hitung jumlah label dan objek
                split_labels_path = os.path.join(labels_path, split)
                if os.path.exists(split_labels_path):
                    label_files = [f for f in os.listdir(split_labels_path) if f.endswith('.txt')]
                    split_stats[split]['labels'] = len(label_files)
                    
                    # Hitung jumlah objek
                    for label_file in label_files:
                        with open(os.path.join(split_labels_path, label_file), 'r') as f:
                            split_stats[split]['objects'] += len(f.readlines())
            
            # Tampilkan distribusi split
            clear_output(wait=True)
            
            # Gunakan SplitVisualizer untuk plot distribusi split
            split_visualizer = SplitVisualizer()
            
            # Plot distribusi split
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            split_visualizer.plot_split_distribution(ax, split_stats)
            
            plt.tight_layout()
            display(plt.gcf())
            
            # Buat tabel statistik
            data = []
            for split, stats in split_stats.items():
                data.append([
                    split.capitalize(),
                    stats['images'],
                    stats['labels'],
                    stats['objects'],
                    round(stats['objects'] / stats['images'], 2) if stats['images'] > 0 else 0
                ])
            
            # Tambahkan total
            total_images = sum(split_stats[split]['images'] for split in split_stats)
            total_labels = sum(split_stats[split]['labels'] for split in split_stats)
            total_objects = sum(split_stats[split]['objects'] for split in split_stats)
            data.append([
                'Total',
                total_images,
                total_labels,
                total_objects,
                round(total_objects / total_images, 2) if total_images > 0 else 0
            ])
            
            df = pd.DataFrame(data, columns=['Split', 'Jumlah Gambar', 'Jumlah Label', 'Jumlah Objek', 'Objek/Gambar'])
            display(df)
            
    except Exception as e:
        with ui_components['visualization_components']['split_tab']['output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi split: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))


def on_layer_distribution_click(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol distribusi layer deteksi.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        output = ui_components['visualization_components']['layer_tab']['output']
        
        with output:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat distribusi layer..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Hitung distribusi layer
            labels_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(labels_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori labels tidak ditemukan"))
                return
            
            # Gunakan LayerVisualizer untuk analisis layer
            layer_visualizer = LayerVisualizer()
            
            # Hitung distribusi layer per split
            layer_stats = {'small': 0, 'medium': 0, 'large': 0}
            layer_stats_per_split = {
                'train': {'small': 0, 'medium': 0, 'large': 0},
                'val': {'small': 0, 'medium': 0, 'large': 0},
                'test': {'small': 0, 'medium': 0, 'large': 0}
            }
            
            # Fungsi untuk menentukan ukuran layer
            def get_layer_size(width, height):
                area = width * height
                if area < 0.02:  # 2% dari area gambar
                    return 'small'
                elif area < 0.1:  # 10% dari area gambar
                    return 'medium'
                else:
                    return 'large'
            
            # Hitung distribusi layer
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(labels_path, split)
                if os.path.exists(split_path):
                    for label_file in os.listdir(split_path):
                        if label_file.endswith('.txt'):
                            with open(os.path.join(split_path, label_file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        # Format YOLOv5: class x_center y_center width height
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        layer_size = get_layer_size(width, height)
                                        
                                        # Update statistik
                                        layer_stats[layer_size] += 1
                                        layer_stats_per_split[split][layer_size] += 1
            
            # Tampilkan distribusi layer
            clear_output(wait=True)
            
            # Plot distribusi layer keseluruhan
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(1, 2, 1)
            
            # Plot distribusi layer
            sizes = list(layer_stats.keys())
            counts = list(layer_stats.values())
            
            ax1.bar(sizes, counts, color=sns.color_palette("viridis", 3))
            ax1.set_title('Distribusi Ukuran Objek', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Ukuran Objek')
            ax1.set_ylabel('Jumlah Objek')
            
            # Tambahkan nilai di atas bar
            for i, count in enumerate(counts):
                ax1.text(i, count + max(counts) * 0.02, str(count), ha='center')
            
            # Plot distribusi layer per split
            ax2 = plt.subplot(1, 2, 2)
            
            # Prepare data
            splits = list(layer_stats_per_split.keys())
            small_counts = [stats['small'] for stats in layer_stats_per_split.values()]
            medium_counts = [stats['medium'] for stats in layer_stats_per_split.values()]
            large_counts = [stats['large'] for stats in layer_stats_per_split.values()]
            
            # Plot stacked bar
            width = 0.6
            x = np.arange(len(splits))
            
            ax2.bar(x, small_counts, width, label='Small', color=sns.color_palette("viridis", 3)[0])
            ax2.bar(x, medium_counts, width, bottom=small_counts, label='Medium', color=sns.color_palette("viridis", 3)[1])
            ax2.bar(x, large_counts, width, bottom=[s+m for s, m in zip(small_counts, medium_counts)], label='Large', color=sns.color_palette("viridis", 3)[2])
            
            ax2.set_title('Distribusi Ukuran Objek per Split', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Split')
            ax2.set_ylabel('Jumlah Objek')
            ax2.set_xticks(x)
            ax2.set_xticklabels(splits)
            ax2.legend()
            
            plt.tight_layout()
            display(plt.gcf())
            
            # Buat tabel statistik
            data = []
            for split, stats in layer_stats_per_split.items():
                data.append([
                    split.capitalize(),
                    stats['small'],
                    stats['medium'],
                    stats['large'],
                    sum(stats.values())
                ])
            
            # Tambahkan total
            data.append([
                'Total',
                layer_stats['small'],
                layer_stats['medium'],
                layer_stats['large'],
                sum(layer_stats.values())
            ])
            
            df = pd.DataFrame(data, columns=['Split', 'Small', 'Medium', 'Large', 'Total'])
            display(df)
            
    except Exception as e:
        with ui_components['visualization_components']['layer_tab']['output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi layer: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))


def on_heatmap_click(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol heatmap deteksi.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan output widget
        output = ui_components['visualization_components']['heatmap_tab']['output']
        
        with output:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat heatmap deteksi..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Hitung heatmap deteksi
            labels_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(labels_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori labels tidak ditemukan"))
                return
            
            # Gunakan BBoxVisualizer untuk membuat heatmap
            bbox_visualizer = BBoxVisualizer()
            
            # Kumpulkan semua koordinat bounding box
            all_boxes = []
            
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(labels_path, split)
                if os.path.exists(split_path):
                    for label_file in os.listdir(split_path):
                        if label_file.endswith('.txt'):
                            with open(os.path.join(split_path, label_file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        # Format YOLOv5: class x_center y_center width height
                                        x_center = float(parts[1])
                                        y_center = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        
                                        # Tambahkan ke list
                                        all_boxes.append({
                                            'x_center': x_center,
                                            'y_center': y_center,
                                            'width': width,
                                            'height': height
                                        })
            
            # Tampilkan heatmap
            clear_output(wait=True)
            
            if not all_boxes:
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Tidak ada data bounding box ditemukan"))
                return
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            
            # Plot heatmap pusat objek
            plt.subplot(2, 2, 1)
            x_centers = [box['x_center'] for box in all_boxes]
            y_centers = [box['y_center'] for box in all_boxes]
            
            plt.hist2d(x_centers, y_centers, bins=50, cmap='viridis')
            plt.colorbar(label='Jumlah Objek')
            plt.title('Heatmap Pusat Objek', fontsize=12, fontweight='bold')
            plt.xlabel('X Center')
            plt.ylabel('Y Center')
            plt.gca().invert_yaxis()  # Invert y-axis untuk visualisasi yang lebih intuitif
            
            # Plot distribusi ukuran objek
            plt.subplot(2, 2, 2)
            widths = [box['width'] for box in all_boxes]
            heights = [box['height'] for box in all_boxes]
            
            plt.scatter(widths, heights, alpha=0.5, s=5)
            plt.title('Distribusi Ukuran Objek', fontsize=12, fontweight='bold')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot distribusi x_center
            plt.subplot(2, 2, 3)
            plt.hist(x_centers, bins=50, color='skyblue', edgecolor='black')
            plt.title('Distribusi X Center', fontsize=12, fontweight='bold')
            plt.xlabel('X Center')
            plt.ylabel('Frekuensi')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot distribusi y_center
            plt.subplot(2, 2, 4)
            plt.hist(y_centers, bins=50, color='salmon', edgecolor='black')
            plt.title('Distribusi Y Center', fontsize=12, fontweight='bold')
            plt.xlabel('Y Center')
            plt.ylabel('Frekuensi')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            display(plt.gcf())
            
            # Tampilkan statistik bounding box
            stats = {
                'x_center_mean': np.mean(x_centers),
                'y_center_mean': np.mean(y_centers),
                'width_mean': np.mean(widths),
                'height_mean': np.mean(heights),
                'width_height_ratio_mean': np.mean([w/h if h > 0 else 0 for w, h in zip(widths, heights)]),
                'area_mean': np.mean([w*h for w, h in zip(widths, heights)]),
                'count': len(all_boxes)
            }
            
            # Tampilkan statistik dalam bentuk tabel
            stats_df = pd.DataFrame([
                ['Jumlah Objek', stats['count']],
                ['Rata-rata X Center', round(stats['x_center_mean'], 4)],
                ['Rata-rata Y Center', round(stats['y_center_mean'], 4)],
                ['Rata-rata Width', round(stats['width_mean'], 4)],
                ['Rata-rata Height', round(stats['height_mean'], 4)],
                ['Rata-rata Rasio Width/Height', round(stats['width_height_ratio_mean'], 4)],
                ['Rata-rata Area', round(stats['area_mean'], 4)]
            ], columns=['Metrik', 'Nilai'])
            
            display(stats_df)
            
    except Exception as e:
        with ui_components['visualization_components']['heatmap_tab']['output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan heatmap deteksi: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))


def setup_visualization_tab_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk tab visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol distribusi kelas
    ui_components['visualization_components']['distribution_tab']['button'].on_click(
        lambda b: on_distribution_click(b, ui_components)
    )
    
    # Setup handler untuk tombol distribusi split
    ui_components['visualization_components']['split_tab']['button'].on_click(
        lambda b: on_split_distribution_click(b, ui_components)
    )
    
    # Setup handler untuk tombol distribusi layer
    ui_components['visualization_components']['layer_tab']['button'].on_click(
        lambda b: on_layer_distribution_click(b, ui_components)
    )
    
    # Setup handler untuk tombol heatmap
    ui_components['visualization_components']['heatmap_tab']['button'].on_click(
        lambda b: on_heatmap_click(b, ui_components)
    )
    
    return ui_components
