"""
File: smartcash/ui/dataset/visualization/handlers/visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset dengan pendekatan SRP
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import pandas as pd
import seaborn as sns

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.common.config.manager import get_config_manager

logger = get_logger(__name__)

def setup_visualization_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Setup handler untuk tombol distribusi kelas
    ui_components['class_distribution_button'].on_click(
        lambda b: on_class_distribution_click(b, ui_components)
    )
    
    # Setup handler untuk tombol sampel gambar
    ui_components['sample_images_button'].on_click(
        lambda b: on_sample_images_click(b, ui_components)
    )
    
    # Setup handler untuk tombol statistik dataset
    ui_components['stats_button'].on_click(
        lambda b: on_stats_click(b, ui_components)
    )
    
    return ui_components

def on_class_distribution_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol distribusi kelas.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        with ui_components['class_distribution_output']:
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
            
            # Plot distribusi kelas
            plt.figure(figsize=(10, 6))
            plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
            plt.title('Distribusi Kelas Dataset')
            plt.xlabel('Kelas')
            plt.ylabel('Jumlah Objek')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Tampilkan plot
            display(plt.gcf())
            
            # Tampilkan tabel distribusi
            df = pd.DataFrame(list(class_counts.items()), columns=['Kelas', 'Jumlah Objek'])
            df['Persentase'] = df['Jumlah Objek'] / df['Jumlah Objek'].sum() * 100
            df['Persentase'] = df['Persentase'].round(2).astype(str) + '%'
            display(df)
            
    except Exception as e:
        with ui_components['class_distribution_output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan distribusi kelas: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_sample_images_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol sampel gambar.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        with ui_components['sample_images_output']:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat sampel gambar..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Cari gambar dari direktori images
            images_path = os.path.join(dataset_path, 'images')
            if not os.path.exists(images_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori images tidak ditemukan"))
                return
            
            # Ambil sampel gambar dari setiap split
            sample_images = []
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(images_path, split)
                if os.path.exists(split_path):
                    image_files = [f for f in os.listdir(split_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        # Ambil maksimal 3 sampel dari setiap split
                        samples = random.sample(image_files, min(3, len(image_files)))
                        for sample in samples:
                            sample_images.append((os.path.join(split_path, sample), f"{split}/{sample}"))
            
            # Tampilkan sampel gambar
            clear_output(wait=True)
            
            if not sample_images:
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Tidak ada gambar ditemukan"))
                return
            
            # Plot sampel gambar
            n_samples = len(sample_images)
            n_cols = min(3, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, (img_path, img_title) in enumerate(sample_images):
                plt.subplot(n_rows, n_cols, i + 1)
                img = Image.open(img_path)
                plt.imshow(np.array(img))
                plt.title(img_title)
                plt.axis('off')
            
            plt.tight_layout()
            display(plt.gcf())
            
    except Exception as e:
        with ui_components['sample_images_output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan sampel gambar: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))

def on_stats_click(b, ui_components: Dict[str, Any]):
    """
    Handler untuk tombol statistik dataset.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    try:
        with ui_components['stats_output']:
            clear_output(wait=True)
            
            # Tampilkan status loading
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memuat statistik dataset..."))
            
            # Dapatkan path dataset
            config_manager = get_config_manager()
            dataset_path = config_manager.get('dataset_path', None)
            
            if not dataset_path or not os.path.exists(dataset_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Path dataset tidak valid"))
                return
            
            # Hitung statistik dataset
            images_path = os.path.join(dataset_path, 'images')
            labels_path = os.path.join(dataset_path, 'labels')
            
            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Direktori images atau labels tidak ditemukan"))
                return
            
            # Hitung jumlah gambar dan label per split
            stats = {
                'train': {'images': 0, 'labels': 0, 'objects': 0},
                'val': {'images': 0, 'labels': 0, 'objects': 0},
                'test': {'images': 0, 'labels': 0, 'objects': 0}
            }
            
            for split in ['train', 'val', 'test']:
                # Hitung jumlah gambar
                split_images_path = os.path.join(images_path, split)
                if os.path.exists(split_images_path):
                    image_files = [f for f in os.listdir(split_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    stats[split]['images'] = len(image_files)
                
                # Hitung jumlah label dan objek
                split_labels_path = os.path.join(labels_path, split)
                if os.path.exists(split_labels_path):
                    label_files = [f for f in os.listdir(split_labels_path) if f.endswith('.txt')]
                    stats[split]['labels'] = len(label_files)
                    
                    # Hitung jumlah objek
                    for label_file in label_files:
                        with open(os.path.join(split_labels_path, label_file), 'r') as f:
                            stats[split]['objects'] += len(f.readlines())
            
            # Tampilkan statistik dataset
            clear_output(wait=True)
            
            # Buat tabel statistik
            data = []
            for split, split_stats in stats.items():
                data.append([
                    split.capitalize(),
                    split_stats['images'],
                    split_stats['labels'],
                    split_stats['objects'],
                    round(split_stats['objects'] / split_stats['images'], 2) if split_stats['images'] > 0 else 0
                ])
            
            # Tambahkan total
            total_images = sum(stats[split]['images'] for split in stats)
            total_labels = sum(stats[split]['labels'] for split in stats)
            total_objects = sum(stats[split]['objects'] for split in stats)
            data.append([
                'Total',
                total_images,
                total_labels,
                total_objects,
                round(total_objects / total_images, 2) if total_images > 0 else 0
            ])
            
            df = pd.DataFrame(data, columns=['Split', 'Jumlah Gambar', 'Jumlah Label', 'Jumlah Objek', 'Objek/Gambar'])
            display(df)
            
            # Plot distribusi split
            plt.figure(figsize=(10, 6))
            sns.barplot(x=['Train', 'Validation', 'Test'], 
                        y=[stats['train']['images'], stats['val']['images'], stats['test']['images']])
            plt.title('Distribusi Split Dataset')
            plt.xlabel('Split')
            plt.ylabel('Jumlah Gambar')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            display(plt.gcf())
            
            # Plot distribusi objek per split
            plt.figure(figsize=(10, 6))
            sns.barplot(x=['Train', 'Validation', 'Test'], 
                        y=[stats['train']['objects'], stats['val']['objects'], stats['test']['objects']])
            plt.title('Distribusi Objek per Split')
            plt.xlabel('Split')
            plt.ylabel('Jumlah Objek')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            display(plt.gcf())
            
    except Exception as e:
        with ui_components['stats_output']:
            clear_output(wait=True)
            logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan statistik dataset: {str(e)}")
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
