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
            
            # Cek lokasi sampel dengan pendekatan yang lebih efisien
            train_images_dir = Path(preprocessed_dir) / 'train' / 'images'
            augmented_images_dir = Path(augmented_dir) / 'images'
            
            # Pilih lokasi yang memiliki sampel
            target_dir = None
            
            # Jika modul adalah preprocessing, prioritaskan mencari gambar di preprocessed_dir
            if module_name == 'preprocessing':
                # Cari gambar di lokasi yang paling mungkin terlebih dahulu
                potential_dirs = [
                    train_images_dir,  # Prioritas tertinggi
                    Path(preprocessed_dir) / 'val' / 'images',
                    Path(preprocessed_dir) / 'test' / 'images',
                    Path(preprocessed_dir) / 'images',
                    Path(data_dir) / 'images',
                    Path(data_dir)  # Fallback
                ]
                
                # Cek setiap direktori potensial secara efisien
                for dir_path in potential_dirs:
                    if dir_path.exists():
                        # Cek apakah ada file gambar (hanya cek beberapa file pertama)
                        try:
                            # Gunakan os.listdir yang lebih efisien daripada Path.glob
                            import os
                            files = os.listdir(str(dir_path))[:20]  # Batasi pencarian
                            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                                target_dir = dir_path
                                if logger: logger.info(f"✅ Menggunakan gambar dari {dir_path}")
                                break
                        except Exception as e:
                            if logger: logger.debug(f"⚠️ Error saat memeriksa {dir_path}: {str(e)}")
                            continue
            # Untuk modul augmentasi, cari gambar dengan prefix augmentasi dengan pendekatan yang lebih efisien
            elif module_name == 'augmentation':
                # Cari gambar dengan prefix augmentasi di berbagai lokasi
                potential_dirs = [
                    train_images_dir,  # Prioritas tertinggi
                    augmented_images_dir,
                    Path(preprocessed_dir) / 'images',
                    Path(augmented_dir)
                ]
                
                # Cek setiap direktori potensial untuk file augmentasi
                for dir_path in potential_dirs:
                    if not dir_path.exists():
                        continue
                        
                    try:
                        # Gunakan os.listdir yang lebih efisien
                        import os
                        files = os.listdir(str(dir_path))
                        
                        # Cek apakah ada file augmentasi
                        aug_files = [f for f in files if f.startswith(f"{aug_prefix}_") and 
                                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        if aug_files:
                            target_dir = dir_path
                            if logger: logger.info(f"✅ Menggunakan {len(aug_files)} gambar augmentasi dari {dir_path}")
                            break
                            
                        # Jika tidak ada file augmentasi tetapi ada file gambar biasa
                        normal_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if normal_files and not target_dir:  # Simpan sebagai fallback
                            target_dir = dir_path
                            # Jangan log dulu, kita masih mencari file augmentasi di direktori lain
                    except Exception as e:
                        if logger: logger.debug(f"⚠️ Error saat memeriksa {dir_path}: {str(e)}")
                        continue
                
                # Jika tidak menemukan file augmentasi tetapi menemukan file gambar biasa
                if target_dir and not any(f.startswith(f"{aug_prefix}_") for f in os.listdir(str(target_dir)) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))):
                    # Hanya tampilkan pesan ini jika kita menggunakan gambar biasa
                    if logger: logger.info(f"⚠️ Tidak ada gambar augmentasi dengan prefix '{aug_prefix}', menggunakan gambar biasa dari {target_dir}")
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
    Visualisasi sampel gambar dari direktori dengan pendekatan yang lebih efisien.
    
    Args:
        target_dir: Direktori target (Path atau str)
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang akan ditampilkan
        aug_prefix: Prefix untuk gambar augmentasi (jika ada)
    """
    try:
        # Gunakan pendekatan yang lebih efisien untuk mencari file gambar
        target_dir_path = Path(target_dir)
        if not target_dir_path.exists():
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Direktori {target_dir} tidak ditemukan"))
            return
        
        # Gunakan os.listdir yang lebih efisien daripada Path.glob
        import os
        try:
            all_files = os.listdir(str(target_dir_path))
        except Exception as e:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Error saat membaca direktori {target_dir}: {str(e)}"))
            return
        
        # Filter hanya file gambar
        image_files = []
        augmented_files = []
        original_files = []
        
        for filename in all_files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = target_dir_path / filename
                image_files.append(full_path)
                
                # Pisahkan file augmentasi dan original jika ada aug_prefix
                if aug_prefix and filename.startswith(f"{aug_prefix}_"):
                    augmented_files.append(full_path)
                else:
                    original_files.append(full_path)
        
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
    Visualisasi distribusi kelas dataset dengan pendekatan yang aman tanpa rekursi.
    
    Args:
        dataset_dir: Direktori dataset
        output_widget: Widget output untuk menampilkan visualisasi
    """
    try:
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mencari direktori label..."))
        
        # Cari direktori label dengan pendekatan yang aman
        labels_dir = None
        dataset_path = Path(dataset_dir)
        
        # Daftar lokasi potensial untuk label tanpa rekursi
        potential_label_dirs = [
            dataset_path / 'train' / 'labels',
            dataset_path / 'labels',
            dataset_path / 'val' / 'labels',
            dataset_path / 'test' / 'labels'
        ]
        
        # Cek setiap direktori potensial dengan pendekatan yang aman
        for dir_path in potential_label_dirs:
            try:
                if dir_path.exists():
                    # Cek apakah ada file .txt dengan cara yang aman
                    txt_files = list(dir_path.glob('*.txt'))
                    if txt_files and len(txt_files) > 0:
                        labels_dir = dir_path
                        break
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat memeriksa direktori {dir_path}: {str(e)}")
                continue
        
        if not labels_dir:
            with output_widget:
                display(create_status_indicator('warning', f"{ICONS['warning']} Tidak dapat menemukan direktori label di {dataset_dir}"))
            return
        
        # Hitung distribusi kelas dengan pendekatan yang aman
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mengumpulkan file label..."))
        
        # Gunakan pendekatan yang aman untuk mendapatkan daftar file
        try:
            label_files = []
            # Gunakan os.listdir yang lebih aman daripada Path.glob
            import os
            for file_name in os.listdir(str(labels_dir)):
                if file_name.endswith('.txt'):
                    label_files.append(os.path.join(str(labels_dir), file_name))
        except Exception as e:
            with output_widget:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat mengumpulkan file label: {str(e)}"))
            return
        
        # Batasi jumlah file yang diproses jika terlalu banyak
        max_files_to_process = 1000
        if len(label_files) > max_files_to_process:
            if logger: logger.info(f"{ICONS['warning']} Membatasi pemrosesan ke {max_files_to_process} dari {len(label_files)} file label")
            import random
            random.seed(42)  # Untuk hasil yang konsisten
            label_files = random.sample(label_files, max_files_to_process)
        
        # Update status
        with output_widget:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Menghitung distribusi kelas dari {len(label_files)} file..."))
        
        # Proses file label dengan pendekatan yang aman
        class_counts = {}
        processed_count = 0
        
        for file_path in label_files:
            try:
                # Baca file dengan pendekatan yang aman
                with open(file_path, 'r') as f:
                    # Batasi jumlah baris yang dibaca per file
                    for i, line in enumerate(f):
                        if i >= 1000:  # Batasi maksimal 1000 baris per file
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if not parts or len(parts) == 0:
                            continue
                            
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
                # Log error dan lanjutkan
                if logger: logger.debug(f"{ICONS['warning']} Error saat memproses file {file_path}: {str(e)}")
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
