"""
File: smartcash/ui/handlers/visualization_handler.py
Deskripsi: Handler visualisasi yang dapat digunakan bersama untuk berbagai modul dataset
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output
from pathlib import Path
import ipywidgets as widgets
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_visualization_handlers(ui_components: Dict[str, Any], module_name: str, env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset yang dapat digunakan bersama.
    
    Args:
        ui_components: Dictionary komponen UI
        module_name: Nama modul (augmentation, preprocessing, dll)
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk visualisasi sampel dataset
    def on_visualize_click(b):
        """Handler untuk visualisasi sampel dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi sampel {module_name}..."))
        
        try:
            # Dapatkan parameter untuk visualisasi
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
            
            # Cek prefix untuk augmentasi jika ada
            aug_prefix = None
            if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 2:
                aug_prefix = ui_components['aug_options'].children[2].value
            else:
                aug_prefix = 'aug'
            
            # Cek lokasi sampel (prioritas ke preprocessed)
            train_images_dir = Path(preprocessed_dir) / 'train' / 'images'
            augmented_images_dir = Path(augmented_dir) / 'images'
            
            # Pilih lokasi yang memiliki sampel
            target_dir = None
            if train_images_dir.exists() and list(train_images_dir.glob(f"{aug_prefix}_*.jpg")):
                target_dir = train_images_dir
            elif augmented_images_dir.exists() and list(augmented_images_dir.glob(f"{aug_prefix}_*.jpg")):
                target_dir = augmented_images_dir
            else:
                # Coba cari di data_dir jika modul bukan augmentasi
                if module_name != 'augmentation':
                    for root, _, files in os.walk(data_dir):
                        if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                            target_dir = Path(root)
                            break
            
            if target_dir is None:
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file gambar yang sesuai"))
                return
            
            # Import dan visualisasi
            try:
                from smartcash.ui.charts.visualize_augmented_samples import visualize_augmented_samples
                visualize_augmented_samples(target_dir, output_widget, ui_components, 5)
            except ImportError:
                # Fallback: Visualisasi standar
                visualize_samples(target_dir, output_widget, 5, aug_prefix)
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat visualisasi sampel: {str(e)}")
    
    # Handler untuk komparasi dataset
    def on_compare_click(b):
        """Handler untuk komparasi dataset original vs processed."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan komparasi original vs processed..."))
        
        try:
            # Dapatkan parameter
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek direktori
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Import helper untuk komparasi
            try:
                from smartcash.ui.charts.comparison_visualizer import compare_original_vs_processed
                
                # Buat wrapper untuk ui_components
                vis_ui_components = {
                    "visualization_container": output_widget,
                    "logger": logger, 
                    "status": output_widget,
                    "data_dir": data_dir,
                    "preprocessed_dir": preprocessed_dir
                }
                
                # Visualisasi komparasi
                compare_original_vs_processed(vis_ui_components)
            except ImportError:
                # Fallback: Visualisasi standar
                compare_original_vs_processed_fallback(data_dir, preprocessed_dir, output_widget)
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Handler untuk distribusi kelas
    def on_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan visualisasi distribusi kelas..."))
        
        try:
            # Dapatkan parameter
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek prefix untuk augmentasi jika ada
            aug_prefix = None
            if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 2:
                aug_prefix = ui_components['aug_options'].children[2].value
            else:
                aug_prefix = 'aug'
            
            # Cek direktori
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display(create_status_indicator('warning', f"{ICONS['warning']} Direktori tidak ditemukan: {preprocessed_dir}"))
                return
            
            # Import helper untuk distribusi kelas
            try:
                from smartcash.ui.charts.visualization_integrator import create_distribution_visualizations
                
                # Buat wrapper untuk ui_components
                vis_ui_components = {
                    "visualization_container": output_widget,
                    "logger": logger, 
                    "status": output_widget,
                    "data_dir": data_dir,
                    "preprocessed_dir": preprocessed_dir,
                    "aug_options": ui_components.get('aug_options')
                }
                
                # Visualisasi distribusi kelas
                create_distribution_visualizations(
                    ui_components=vis_ui_components,
                    dataset_dir=preprocessed_dir,
                    split_name='train',
                    aug_prefix=aug_prefix,
                    orig_prefix='rp',
                    target_count=1000
                )
            except ImportError:
                # Fallback: Visualisasi standar
                visualize_class_distribution_fallback(preprocessed_dir, output_widget)
            
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
                
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error visualisasi distribusi kelas: {str(e)}")
    
    # Register handlers untuk tombol visualisasi
    visualization_handlers = {
        'visualize_button': on_visualize_click,
        'compare_button': on_compare_click,
        'distribution_button': on_distribution_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in visualization_handlers.items() if button in ui_components]
    
    # Tambahkan handlers ke UI components
    ui_components.update({
        'on_visualize_click': on_visualize_click,
        'on_compare_click': on_compare_click,
        'on_distribution_click': on_distribution_click
    })
    
    return ui_components

# Fungsi helper untuk visualisasi sampel
def visualize_samples(image_dir: Path, output_widget: widgets.Output, num_samples: int = 5, prefix: str = None):
    """
    Visualisasi sampel gambar dari direktori.
    
    Args:
        image_dir: Direktori gambar
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang ditampilkan
        prefix: Prefix nama file (opsional)
    """
    try:
        # Cari file gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            if prefix:
                image_files.extend(list(image_dir.glob(f"{prefix}*{ext}")))
            else:
                image_files.extend(list(image_dir.glob(f"*{ext}")))
        
        if not image_files:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada gambar ditemukan di {image_dir}"))
            return
        
        # Ambil sampel acak
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Tampilkan sampel
        with output_widget:
            clear_output(wait=True)
            
            # Gunakan ThreadPoolExecutor untuk loading gambar secara paralel
            fig, axes = plt.subplots(1, len(samples), figsize=(15, 4))
            if len(samples) == 1:
                axes = [axes]
            
            def load_and_display(idx, img_path):
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    axes[idx].imshow(np.array(img))
                    axes[idx].set_title(f"Sample {idx+1}")
                    axes[idx].axis('off')
                except Exception as e:
                    axes[idx].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                    axes[idx].axis('off')
            
            with ThreadPoolExecutor(max_workers=min(len(samples), 4)) as executor:
                futures = [executor.submit(load_and_display, i, img_path) for i, img_path in enumerate(samples)]
                for future in futures:
                    future.result()
            
            plt.tight_layout()
            plt.show()
            
            display(widgets.HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                border-left:4px solid {COLORS['alert_success_text']};">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan {len(samples)} sampel dari {len(image_files)} gambar.</p>
                </div>
            """))
    except Exception as e:
        with output_widget:
            display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi sampel: {str(e)}"))

# Fungsi helper untuk komparasi
def compare_original_vs_processed_fallback(data_dir: str, processed_dir: str, output_widget: widgets.Output):
    """
    Komparasi gambar original vs processed.
    
    Args:
        data_dir: Direktori data original
        processed_dir: Direktori data processed
        output_widget: Widget output untuk menampilkan visualisasi
    """
    try:
        # Cari direktori gambar
        original_dir = None
        processed_train_dir = None
        
        # Cek struktur direktori untuk original
        for root, dirs, files in os.walk(data_dir):
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                original_dir = Path(root)
                break
        
        # Cek struktur direktori untuk processed
        processed_train_dir = Path(processed_dir) / 'train' / 'images'
        if not processed_train_dir.exists():
            for root, dirs, files in os.walk(processed_dir):
                if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    processed_train_dir = Path(root)
                    break
        
        if not original_dir or not processed_train_dir:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak dapat menemukan direktori gambar yang sesuai"))
            return
        
        # Cari file gambar
        original_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            original_files.extend(list(original_dir.glob(f"*{ext}")))
        
        processed_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            processed_files.extend(list(processed_train_dir.glob(f"*{ext}")))
        
        if not original_files or not processed_files:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak cukup gambar untuk komparasi"))
            return
        
        # Ambil sampel acak
        num_samples = 3
        original_samples = random.sample(original_files, min(num_samples, len(original_files)))
        processed_samples = random.sample(processed_files, min(num_samples, len(processed_files)))
        
        # Tampilkan komparasi
        with output_widget:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
            
            def load_and_display_row(row_idx, img_paths):
                for col_idx, img_path in enumerate(img_paths[:num_samples]):
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        axes[row_idx, col_idx].imshow(np.array(img))
                        axes[row_idx, col_idx].set_title(f"{'Original' if row_idx == 0 else 'Processed'} {col_idx+1}")
                        axes[row_idx, col_idx].axis('off')
                    except Exception as e:
                        axes[row_idx, col_idx].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        axes[row_idx, col_idx].axis('off')
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(load_and_display_row, 0, original_samples),
                    executor.submit(load_and_display_row, 1, processed_samples)
                ]
                for future in futures:
                    future.result()
            
            plt.tight_layout()
            plt.show()
            
            display(widgets.HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                border-left:4px solid {COLORS['alert_success_text']};">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan komparasi {num_samples} sampel gambar.</p>
                </div>
            """))
    except Exception as e:
        with output_widget:
            display(create_status_indicator('error', f"{ICONS['error']} Error saat komparasi: {str(e)}"))

# Fungsi helper untuk distribusi kelas
def visualize_class_distribution_fallback(dataset_dir: str, output_widget: widgets.Output):
    """
    Visualisasi distribusi kelas dataset.
    
    Args:
        dataset_dir: Direktori dataset
        output_widget: Widget output untuk menampilkan visualisasi
    """
    try:
        # Cari direktori label
        labels_dir = Path(dataset_dir) / 'train' / 'labels'
        if not labels_dir.exists():
            for root, dirs, files in os.walk(dataset_dir):
                if any(f.endswith('.txt') for f in files):
                    labels_dir = Path(root)
                    break
        
        if not labels_dir.exists():
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak dapat menemukan direktori label"))
            return
        
        # Hitung distribusi kelas
        class_counts = {}
        label_files = list(labels_dir.glob('*.txt'))
        
        def process_label_file(file_path):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            return class_id
            except Exception:
                return None
        
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            class_ids = list(executor.map(process_label_file, label_files))
        
        for class_id in class_ids:
            if class_id is not None:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        if not class_counts:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ada data kelas ditemukan"))
            return
        
        # Tampilkan distribusi
        with output_widget:
            clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            bars = ax.bar(classes, counts, color='skyblue')
            
            # Tambahkan label di atas bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            ax.set_xlabel('Class ID')
            ax.set_ylabel('Count')
            ax.set_title('Distribusi Kelas Dataset')
            ax.set_xticks(classes)
            
            plt.tight_layout()
            plt.show()
            
            total_samples = sum(counts)
            display(widgets.HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                border-left:4px solid {COLORS['alert_success_text']};">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan distribusi {len(classes)} kelas dari {total_samples} sampel.</p>
                </div>
            """))
    except Exception as e:
        with output_widget:
            display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
