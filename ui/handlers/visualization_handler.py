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
            
            # Jika modul adalah preprocessing, prioritaskan mencari gambar di preprocessed_dir
            if module_name == 'preprocessing':
                # Cari gambar di preprocessed_dir dengan berbagai ekstensi
                has_images_in_preprocessed = False
                if train_images_dir.exists():
                    for ext in ['.jpg', '.jpeg', '.png']:
                        if list(train_images_dir.glob(f'*{ext}')):
                            has_images_in_preprocessed = True
                            target_dir = train_images_dir
                            if logger: logger.info(f"✅ Menggunakan gambar dari {train_images_dir}")
                            break
                
                # Jika tidak ada di train/images, coba cari di preprocessed_dir langsung
                if not has_images_in_preprocessed:
                    preprocessed_path = Path(preprocessed_dir)
                    if preprocessed_path.exists():
                        # Cari di semua subdirektori preprocessed
                        for root, _, files in os.walk(preprocessed_path):
                            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                                target_dir = Path(root)
                                if logger: logger.info(f"✅ Menggunakan gambar dari {root}")
                                break
                
                # Jika masih tidak ada, cari di data_dir
                if not target_dir:
                    for root, _, files in os.walk(data_dir):
                        if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                            target_dir = Path(root)
                            if logger: logger.info(f"✅ Menggunakan gambar dari {root}")
                            break
            # Untuk modul augmentasi, cari gambar dengan prefix augmentasi
            elif module_name == 'augmentation':
                # Cari gambar dengan prefix augmentasi di berbagai lokasi
                aug_files_train = []
                aug_files_augmented = []
                
                # Cek di direktori train/images
                if train_images_dir.exists():
                    aug_files_train = list(train_images_dir.glob(f"{aug_prefix}_*.jpg")) + \
                                    list(train_images_dir.glob(f"{aug_prefix}_*.jpeg")) + \
                                    list(train_images_dir.glob(f"{aug_prefix}_*.png"))
                
                # Cek di direktori augmented/images
                if augmented_images_dir.exists():
                    aug_files_augmented = list(augmented_images_dir.glob(f"{aug_prefix}_*.jpg")) + \
                                        list(augmented_images_dir.glob(f"{aug_prefix}_*.jpeg")) + \
                                        list(augmented_images_dir.glob(f"{aug_prefix}_*.png"))
                
                # Pilih lokasi yang memiliki sampel augmentasi
                if aug_files_train:
                    target_dir = train_images_dir
                    if logger: logger.info(f"✅ Menggunakan {len(aug_files_train)} gambar augmentasi dari {train_images_dir}")
                elif aug_files_augmented:
                    target_dir = augmented_images_dir
                    if logger: logger.info(f"✅ Menggunakan {len(aug_files_augmented)} gambar augmentasi dari {augmented_images_dir}")
                else:
                    # Jika tidak ada gambar augmentasi, gunakan gambar biasa
                    if train_images_dir.exists() and any(train_images_dir.glob('*.jpg')):
                        target_dir = train_images_dir
                        if logger: logger.info(f"⚠️ Tidak ada gambar augmentasi, menggunakan gambar biasa dari {train_images_dir}")
                    elif augmented_images_dir.exists() and any(augmented_images_dir.glob('*.jpg')):
                        target_dir = augmented_images_dir
                        if logger: logger.info(f"⚠️ Tidak ada gambar augmentasi, menggunakan gambar biasa dari {augmented_images_dir}")
            # Untuk modul lain, cari di data_dir
            else:
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
                if module_name == 'augmentation':
                    # Untuk modul augmentasi, teruskan prefix augmentasi
                    visualize_samples(target_dir, output_widget, 5, aug_prefix)
                    if logger: logger.info(f"🔍 Menampilkan sampel dengan prefix augmentasi: {aug_prefix}")
                else:
                    # Untuk modul lain, tidak perlu prefix
                    visualize_samples(target_dir, output_widget, 5)
            
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
def visualize_samples(target_dir, output_widget: widgets.Output, num_samples: int = 4, aug_prefix: str = None):
    """
    Visualisasi sampel gambar dari direktori.
    
    Args:
        target_dir: Direktori target (Path atau str)
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang akan ditampilkan
        aug_prefix: Prefix untuk gambar augmentasi (jika ada)
    """
    try:
        # Cari file gambar
        image_files = []
        augmented_files = []
        original_files = []
        
        # Cari semua file gambar
        for ext in ['.jpg', '.jpeg', '.png']:
            all_files = list(Path(target_dir).glob(f"*{ext}"))
            image_files.extend(all_files)
            
            # Pisahkan file augmentasi dan original jika ada aug_prefix
            if aug_prefix:
                for f in all_files:
                    if f.name.startswith(f"{aug_prefix}_"):
                        augmented_files.append(f)
                    else:
                        original_files.append(f)
        
        if not image_files:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak ditemukan file gambar di {target_dir}"))
            return
        
        # Prioritaskan file augmentasi jika tersedia dan diminta
        samples_to_use = []
        if aug_prefix and augmented_files:
            # Gunakan file augmentasi jika tersedia
            samples_to_use = augmented_files
            sample_type = "augmentasi"
        else:
            # Gunakan semua file jika tidak ada augmentasi atau tidak diminta
            samples_to_use = image_files
            sample_type = "original"
        
        # Ambil sampel acak
        samples = random.sample(samples_to_use, min(num_samples, len(samples_to_use)))
        
        # Tampilkan sampel
        with output_widget:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(1, len(samples), figsize=(15, 4))
            if len(samples) == 1:
                axes = [axes]
            
            def load_and_display(idx, img_path):
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    axes[idx].imshow(np.array(img))
                    
                    # Ekstrak nama file yang lebih pendek dan informatif
                    filename = Path(img_path).name
                    # Tentukan tipe gambar (augmentasi atau original)
                    img_type = "Aug" if aug_prefix and filename.startswith(f"{aug_prefix}_") else "Orig"
                    
                    # Truncate nama file jika terlalu panjang (max 15 karakter)
                    if len(filename) > 15:
                        filename = filename[:12] + '...'
                    
                    # Tampilkan informasi yang lebih jelas
                    axes[idx].set_title(f"{img_type}: {filename}", fontsize=9)
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
            
            # Tampilkan informasi yang lebih jelas tentang jenis gambar
            sample_type_info = "augmentasi" if aug_prefix and any(s.name.startswith(f"{aug_prefix}_") for s in samples) else "original"
            
            display(widgets.HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                border-left:4px solid {COLORS['alert_success_text']}">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan {len(samples)} sampel gambar {sample_type_info} dari total {len(image_files)} gambar.</p>
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
        
        # Mencoba mencari pasangan gambar yang sesuai berdasarkan nama file
        matched_pairs = []
        num_samples = 3
        
        # Ekstrak nama file tanpa ekstensi dan path
        original_names = {f.stem.split('_')[0]: f for f in original_files}
        processed_names = {f.stem.split('_')[0]: f for f in processed_files}
        
        # Cari pasangan yang cocok
        common_names = set(original_names.keys()) & set(processed_names.keys())
        
        # Jika ada pasangan yang cocok, gunakan itu
        if common_names:
            # Ambil sampel dari pasangan yang cocok
            sample_names = random.sample(list(common_names), min(num_samples, len(common_names)))
            matched_pairs = [(original_names[name], processed_names[name]) for name in sample_names]
        
        # Jika tidak ada pasangan yang cocok atau tidak cukup, gunakan sampel acak
        if len(matched_pairs) < num_samples:
            # Ambil sampel acak untuk sisa yang dibutuhkan
            remaining = num_samples - len(matched_pairs)
            original_samples = random.sample(original_files, min(remaining, len(original_files)))
            processed_samples = random.sample(processed_files, min(remaining, len(processed_files)))
            
            # Tambahkan ke pasangan yang sudah ada
            for i in range(min(len(original_samples), len(processed_samples))):
                matched_pairs.append((original_samples[i], processed_samples[i]))
        
        # Pisahkan pasangan menjadi dua list
        original_samples = [pair[0] for pair in matched_pairs]
        processed_samples = [pair[1] for pair in matched_pairs]
        
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
                        
                        # Ekstrak nama file yang lebih pendek dan informatif
                        filename = Path(img_path).name
                        # Truncate nama file jika terlalu panjang (max 15 karakter)
                        if len(filename) > 15:
                            filename = filename[:12] + '...'
                            
                        # Tampilkan tipe dan nama file
                        axes[row_idx, col_idx].set_title(
                            f"{'Original' if row_idx == 0 else 'Processed'} {col_idx+1}\n{filename}", 
                            fontsize=8
                        )
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
                border-left:4px solid {COLORS['alert_success_text']}">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan komparasi {len(original_samples)} pasang gambar (original vs processed).</p>
                <p style="margin:5px 0; font-size:0.9em;">Direktori original: {original_dir}</p>
                <p style="margin:5px 0; font-size:0.9em;">Direktori processed: {processed_train_dir}</p>
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
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mencari direktori label..."))
        
        # Cari direktori label
        labels_dir = None
        dataset_path = Path(dataset_dir)
        
        # Cek lokasi potensial untuk label secara langsung tanpa rekursi
        potential_label_dirs = [
            dataset_path / 'train' / 'labels',
            dataset_path / 'labels',
            dataset_path / 'val' / 'labels',
            dataset_path / 'test' / 'labels'
        ]
        
        # Cek setiap direktori potensial
        for dir_path in potential_label_dirs:
            if dir_path.exists() and any(dir_path.glob('*.txt')):
                labels_dir = dir_path
                break
        
        if not labels_dir:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak dapat menemukan direktori label di {dataset_dir}"))
            return
        
        # Hitung distribusi kelas dengan cara yang lebih efisien dan aman
        class_counts = {}
        label_files = list(labels_dir.glob('*.txt'))
        
        # Batasi jumlah file yang diproses jika terlalu banyak
        max_files_to_process = 1000
        if len(label_files) > max_files_to_process:
            if logger: logger.info(f"⚠️ Membatasi pemrosesan ke {max_files_to_process} dari {len(label_files)} file label")
            label_files = random.sample(label_files, max_files_to_process)
        
        # Update status
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Menghitung distribusi kelas dari {len(label_files)} file..."))
        
        # Proses file label secara langsung dan sederhana (tanpa rekursi atau threading)
        # Batasi jumlah file yang diproses untuk menghindari masalah performa
        processed_count = 0
        for file_path in label_files:
            try:
                # Baca file secara efisien
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts and len(parts) > 0:
                                try:
                                    class_id = int(parts[0])
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                except (ValueError, IndexError):
                                    continue  # Abaikan baris yang tidak valid
                processed_count += 1
                # Update progress setiap 100 file
                if processed_count % 100 == 0:
                    with output_widget:
                        clear_output(wait=True)
                        display(create_status_indicator('info', f"{ICONS['processing']} Menghitung distribusi kelas... ({processed_count}/{len(label_files)} file)"))
            except Exception as e:
                # Abaikan file yang bermasalah
                continue
        
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
            
            # Tambahkan label di atas bar dan warna berbeda untuk setiap bar
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(classes)))
            for i, (bar, count) in enumerate(zip(bars, counts)):
                bar.set_color(colors[i])
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Kelas', fontweight='bold')
            ax.set_ylabel('Jumlah', fontweight='bold')
            ax.set_title('Distribusi Kelas Dataset', fontsize=14, fontweight='bold')
            
            # Tambahkan grid untuk memudahkan pembacaan
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(classes)
            
            plt.tight_layout()
            plt.show()
            
            total_samples = sum(counts)
            display(widgets.HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                border-left:4px solid {COLORS['alert_success_text']};">
                <p style="margin:5px 0">{ICONS['success']} Berhasil menampilkan distribusi {len(classes)} kelas dari {total_samples} sampel.</p>
                <p style="margin:5px 0; font-size:0.9em;">Direktori label: {labels_dir}</p>
                <p style="margin:5px 0; font-size:0.9em;">Total file label: {len(label_files)}</p>
                </div>
            """))
    except Exception as e:
        with output_widget:
            display(create_status_indicator('error', f"{ICONS['error']} Error saat visualisasi distribusi: {str(e)}"))
