"""
File: smartcash/ui/handlers/visualization_handler.py
Deskripsi: Handler visualisasi yang dapat digunakan bersama untuk berbagai modul dataset
"""

from typing import Dict, Any, Optional, Callable, List, Union
from IPython.display import display, clear_output
from pathlib import Path
import ipywidgets as widgets
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS, COLORS, ALERT_STYLES
from smartcash.ui.utils.alert_utils import (
    create_status_indicator, create_info_alert, 
    create_info_box, create_info_log, create_alert_html
)
from smartcash.ui.handlers.status_handler import create_status_panel, update_status_panel

# Fungsi helper untuk visualisasi
def display_no_data_message(output_widget, title="Tidak Ada Data", message="", detail_message=""):
    """
    Menampilkan pesan informasi ketika tidak ada data yang tersedia dengan format yang konsisten.
    Menggunakan create_info_alert dari alert_utils untuk konsistensi UI.
    
    Args:
        output_widget: Widget output untuk menampilkan pesan
        title: Judul pesan
        message: Pesan utama
        detail_message: Pesan detail tambahan (opsional)
    """
    with output_widget:
        clear_output(wait=True)
        full_message = f"<strong>{title}</strong>"
        if message:
            full_message += f"<br>{message}"
        if detail_message:
            full_message += f"<br><small>{detail_message}</small>"
        
        # Menggunakan create_info_alert dari alert_utils
        alert = create_info_alert(full_message, alert_type="warning")
        display(alert)

def find_valid_image_directory(base_dirs, extensions=None):
    """
    Mencari direktori gambar yang valid dari daftar direktori potensial.
    Menggunakan one-liner style dan utilitas dari dataset_constants.
    
    Args:
        base_dirs: List direktori yang akan dicari
        extensions: List ekstensi file yang dicari (default: ['.jpg', '.jpeg', '.png'])
        
    Returns:
        Path ke direktori yang valid atau None jika tidak ditemukan
    """
    from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
    extensions = extensions or [ext[1:] for ext in IMG_EXTENSIONS]  # Hapus * dari '*.jpg'
    
    # Gunakan list comprehension dan any untuk memeriksa direktori
    valid_dirs = [Path(d) if not isinstance(d, Path) else d for d in base_dirs if isinstance(d, (str, Path))]
    valid_dirs = [d for d in valid_dirs if d.exists() and any(any(d.glob(f"*{ext}")) for ext in extensions)]
    
    # Return direktori pertama yang valid atau None jika tidak ada
    return valid_dirs[0] if valid_dirs else None

def find_valid_label_directory(base_dirs):
    """
    Mencari direktori label yang valid dari daftar direktori potensial.
    Menggunakan one-liner style untuk kode yang lebih ringkas.
    
    Args:
        base_dirs: List direktori yang akan dicari
        
    Returns:
        Path ke direktori yang valid atau None jika tidak ditemukan
    """
    # Konversi ke Path dan filter direktori yang valid dengan file .txt
    valid_dirs = [Path(d) if not isinstance(d, Path) else d for d in base_dirs if isinstance(d, (str, Path))]
    valid_dirs = [d for d in valid_dirs if d.exists() and any(d.glob("*.txt"))]
    
    # Return direktori pertama yang valid atau None jika tidak ada
    return valid_dirs[0] if valid_dirs else None

def display_visualization_status(output_widget, status_type="info", title="", messages: Optional[List[str]]=None):
    """
    Menampilkan status visualisasi dengan format yang konsisten.
    Menggunakan create_info_box dari alert_utils untuk tampilan yang lebih informatif.
    
    Args:
        output_widget: Widget output untuk menampilkan status
        status_type: Tipe status ("info", "success", "warning", "error")
        title: Judul status
        messages: List pesan yang akan ditampilkan
    """
    if messages is None:
        messages = []
    
    # Buat konten HTML untuk pesan
    content = ""
    if messages:
        content = "<ul style='margin-top: 5px; margin-bottom: 0;'>"
        for msg in messages:
            content += f"<li>{msg}</li>"
        content += "</ul>"
    
    # Gunakan create_info_box dari alert_utils untuk tampilan yang lebih baik
    with output_widget:
        clear_output(wait=True)
        # Jika tidak ada title tapi ada messages, gunakan message pertama sebagai title
        if not title and messages:
            title = messages[0]
            content = "" if len(messages) == 1 else "<ul style='margin-top: 5px; margin-bottom: 0;'>"
            for i, msg in enumerate(messages[1:], 1):
                content += f"<li>{msg}</li>"
            content += "</ul>" if len(messages) > 1 else ""
        
        # Jika konten kosong, gunakan create_info_alert yang lebih sederhana
        if not content:
            alert = create_info_alert(title, alert_type=status_type)
            display(alert)
        else:
            # Gunakan create_info_box untuk tampilan yang lebih informatif
            info_box = create_info_box(title, content, style=status_type)
            display(info_box)

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
            display_visualization_status(
                output_widget,
                status_type="info",
                title="Mencari sampel untuk visualisasi..."
            )
        
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
                
                # Gunakan fungsi helper untuk mencari direktori gambar yang valid
                target_dir = find_valid_image_directory(potential_dirs)
                if target_dir and logger: 
                    logger.info(f"✅ Menggunakan gambar dari {target_dir}")
                        
                # Jika tidak ditemukan, tampilkan pesan informatif
                if target_dir is None:
                    if logger: logger.warning(f"⚠️ Tidak ada gambar ditemukan untuk visualisasi")
                    display_no_data_message(
                        output_widget,
                        message="Tidak ada gambar yang ditemukan untuk divisualisasikan.",
                        detail_message="Silakan lakukan preprocessing dataset terlebih dahulu."
                    )
                    return
            # Jika modul adalah augmentation, prioritaskan mencari gambar di augmented_dir
            elif module_name == 'augmentation':
                # Jika label_dir tidak ada, coba cari di lokasi standar
                if show_labels and not label_dir:
                    potential_label_dirs = [
                        image_dir.parent / 'labels',
                        image_dir.parent.parent / 'labels',
                        image_dir.parent / 'train' / 'labels',
                        image_dir.parent / 'val' / 'labels',
                        image_dir.parent / 'test' / 'labels'
                    ]
                    
                    # Gunakan fungsi helper untuk mencari direktori label yang valid
                    label_dir = find_valid_label_directory(potential_label_dirs)
                    if label_dir and logger:
                        logger.info(f"✅ Menggunakan label dari {label_dir}")
                        # Jika tidak ditemukan, tampilkan pesan informatif
                        if target_dir is None:
                            if logger: logger.warning(f"⚠️ Tidak ada gambar ditemukan untuk visualisasi")
                            display_no_data_message(
                                output_widget,
                                message="Tidak ada gambar yang ditemukan untuk divisualisasikan.",
                                detail_message="Silakan lakukan augmentasi dataset terlebih dahulu."
                            )
                            return
                # Cari gambar di lokasi yang paling mungkin terlebih dahulu
                potential_dirs = [
                    augmented_images_dir,  # Prioritas tertinggi
                    Path(augmented_dir) / 'train' / 'images',
                    Path(preprocessed_dir) / 'train' / 'images',
                    Path(data_dir) / 'images',
                    Path(data_dir)  # Fallback
                ]
                
                # Gunakan fungsi helper untuk mencari direktori gambar yang valid
                target_dir = find_valid_image_directory(potential_dirs)
                if target_dir and logger: 
                    logger.info(f"✅ Menggunakan gambar dari {target_dir}")
                    
                # Jika tidak ditemukan, tampilkan pesan informatif
                if target_dir is None:
                    if logger: logger.warning(f"⚠️ Tidak ada gambar ditemukan untuk visualisasi")
                    display_no_data_message(
                        output_widget,
                        message="Tidak ada gambar yang ditemukan untuk divisualisasikan.",
                        detail_message="Silakan lakukan augmentasi dataset terlebih dahulu."
                    )
                    return
            # Jika modul adalah split, prioritaskan mencari gambar di data_dir
            else:  # split atau default
                # Cari gambar di lokasi yang paling mungkin terlebih dahulu
                potential_dirs = [
                    Path(data_dir) / 'images',  # Prioritas tertinggi
                    Path(data_dir),
                    train_images_dir,  # Fallback ke preprocessed jika perlu
                    Path(preprocessed_dir) / 'images'
                ]
                
                # Gunakan fungsi helper untuk mencari direktori gambar yang valid
                target_dir = find_valid_image_directory(potential_dirs)
                if target_dir and logger: 
                    logger.info(f"✅ Menggunakan gambar dari {target_dir}")
                
                # Jika tidak ditemukan, tampilkan pesan informatif
                if target_dir is None:
                    if logger: logger.warning(f"⚠️ Tidak ada gambar ditemukan untuk visualisasi")
                    display_no_data_message(
                        output_widget,
                        message="Tidak ada gambar yang ditemukan untuk divisualisasikan.",
                        detail_message="Silakan pastikan dataset telah diupload dengan benar."
                    )
                    return
            # Periksa apakah ada file augmentasi di direktori target jika modul adalah augmentation
            if module_name == 'augmentation' and target_dir:
                try:
                    import os
                    has_aug_files = any(f.startswith(f"{aug_prefix}_") for f in os.listdir(str(target_dir)) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                    if not has_aug_files and logger:
                        logger.info(f"ℹ️ Tidak menemukan gambar augmentasi, menggunakan gambar normal dari {target_dir}")
                except Exception as e:
                    if logger: logger.debug(f"⚠️ Error saat memeriksa file augmentasi: {str(e)}")
            # Jika tidak ada direktori target yang valid, coba cari di data_dir dengan metode alternatif
            if target_dir is None:
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
            display_visualization_status(
                output_widget,
                status_type="info",
                title="Mempersiapkan komparasi original vs processed..."
            )
        
        try:
            # Dapatkan parameter
            data_dir = ui_components.get('data_dir', 'data')
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Cek direktori
            if not Path(preprocessed_dir).exists():
                with output_widget:
                    display_visualization_status(
                output_widget,
                status_type="warning",
                title="Direktori tidak ditemukan",
                messages=[str(preprocessed_dir)]
            )
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
                display_visualization_status(
                    output_widget,
                    status_type="error",
                    title="Error saat komparasi",
                    messages=[str(e)]
                )
            if logger: logger.error(f"{ICONS['error']} Error saat komparasi dataset: {str(e)}")
    
    # Handler untuk distribusi kelas
    def on_distribution_click(b):
        """Handler untuk visualisasi distribusi kelas dataset."""
        output_widget = ui_components.get('visualization_container', ui_components.get('status'))
        
        with output_widget:
            clear_output(wait=True)
            display_visualization_status(
                output_widget,
                status_type="info",
                title="Mempersiapkan visualisasi distribusi kelas..."
            )
        
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
                    display_visualization_status(
                output_widget,
                status_type="warning",
                title="Direktori tidak ditemukan",
                messages=[str(preprocessed_dir)]
            )
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
                display_visualization_status(
                    output_widget,
                    status_type="error",
                    title="Error saat visualisasi distribusi",
                    messages=[str(e)]
                )
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

# Fungsi helper untuk visualisasi sampel dengan anotasi
def visualize_images_with_annotations(output_widget, image_dir, label_dir=None, num_samples=5, show_labels=True, fig_size=(15, 10), random_seed=None, class_names=None, logger=None, aug_prefix=None):
    """
    Menampilkan gambar dengan anotasi bounding box.
    
    Args:
        output_widget: Widget output untuk menampilkan gambar
        image_dir: Direktori gambar
        label_dir: Direktori label (opsional)
        num_samples: Jumlah sampel yang akan ditampilkan
        show_labels: Apakah menampilkan label atau tidak
        fig_size: Ukuran gambar
        random_seed: Seed untuk random sampling
        class_names: Daftar nama kelas
        logger: Logger untuk mencatat aktivitas
        aug_prefix: Prefix untuk file augmentasi (opsional)
    """
    try:
        # Validasi parameter
        if not image_dir:
            if logger: logger.warning(f"⚠️ Direktori gambar tidak valid: {image_dir}")
            display_visualization_status(
                output_widget, 
                status_type="warning", 
                title="Direktori gambar tidak valid", 
                messages=[]
            )
            return
            
        # Konversi ke Path
        image_dir = Path(image_dir)
        if label_dir:
            label_dir = Path(label_dir)
        
        # Cek apakah direktori ada
        if not image_dir.exists():
            if logger: logger.warning(f"⚠️ Direktori gambar tidak ditemukan: {image_dir}")
            display_visualization_status(
                output_widget, 
                status_type="warning", 
                title="Direktori gambar tidak ditemukan", 
                messages=[str(image_dir)]
            )
            return
        
        # Filter hanya file gambar
        image_files = []
        augmented_files = []
        original_files = []
        
        for filename in image_dir.iterdir():
            if filename.is_file() and filename.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_files.append(filename)
                
                # Pisahkan file augmentasi dan original jika ada aug_prefix
                if aug_prefix and filename.name.startswith(f"{aug_prefix}_"):
                    augmented_files.append(filename)
                else:
                    original_files.append(filename)
        
        if not image_files:
            if logger: logger.warning(f"⚠️ Tidak ada gambar ditemukan di {image_dir}")
            display_visualization_status(
                output_widget, 
                status_type="warning", 
                title="Tidak ada gambar ditemukan", 
                messages=[str(image_dir)]
            )
            return
            
        # Jika label_dir tidak ada, coba cari di lokasi standar
        if show_labels and not label_dir:
            potential_label_dirs = [
                image_dir.parent / 'labels',
                image_dir.parent.parent / 'labels',
                image_dir.parent / 'train' / 'labels',
                image_dir.parent / 'val' / 'labels',
                image_dir.parent / 'test' / 'labels'
            ]
            
            # Gunakan fungsi helper untuk mencari direktori label yang valid
            label_dir = find_valid_label_directory(potential_label_dirs)
            if label_dir and logger:
                logger.info(f"✅ Menggunakan label dari {label_dir}")
                
        # Implementasi visualisasi gambar dengan anotasi di sini
        # ...
        
        # Contoh implementasi sederhana
        with output_widget:
            clear_output(wait=True)
            display_visualization_status(
                output_widget, 
                status_type="info", 
                title="Visualisasi Sampel", 
                messages=[f"Menampilkan {num_samples} sampel dari {len(image_files)} gambar di {image_dir}"]
            )
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat visualisasi gambar: {str(e)}")
        with output_widget:
            clear_output(wait=True)
            display_visualization_status(
                output_widget, 
                status_type="error", 
                title="Error saat visualisasi gambar", 
                messages=[str(e)]
            )
def visualize_samples(target_dir, output_widget: widgets.Output, num_samples: int = 4, aug_prefix: str = None):
    """
    Visualisasi sampel gambar dari direktori dengan pendekatan yang lebih efisien.
    Menggunakan one-liner style dan utilitas dari dataset_utils.
    
    Args:
        target_dir: Direktori target (Path atau str)
        output_widget: Widget output untuk menampilkan visualisasi
        num_samples: Jumlah sampel yang akan ditampilkan
        aug_prefix: Prefix untuk gambar augmentasi (jika ada)
    """
    try:
        # Import utilitas yang diperlukan
        from smartcash.dataset.utils.data_utils import load_image, find_image_files
        from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
        
        # Validasi parameter dengan one-liner
        if not target_dir: 
            with output_widget: clear_output(wait=True); display(create_info_alert("Direktori target tidak valid", alert_type="warning")); return
            
        # Konversi ke Path dan validasi direktori
        target_dir = Path(target_dir) if not isinstance(target_dir, Path) else target_dir
        if not target_dir.exists():
            with output_widget: clear_output(wait=True); display(create_info_alert(f"Direktori tidak ditemukan: {target_dir}", alert_type="warning")); return
            
        # Tampilkan status pencarian sampel
        with output_widget: clear_output(wait=True); display(create_info_alert(f"Mencari sampel untuk visualisasi...", alert_type="info"))
            
        # Cari file gambar dengan list comprehension
        extensions = [ext[1:] for ext in IMG_EXTENSIONS]  # Hapus * dari '*.jpg'
        image_files = [f for ext in extensions for f in target_dir.glob(f"*{ext}")]
            
        # Validasi hasil pencarian
        if not image_files:
            with output_widget: clear_output(wait=True); display(create_info_alert(f"Tidak ada gambar ditemukan di {target_dir}", alert_type="warning")); return
            
        # Filter file berdasarkan prefix jika ada dengan list comprehension
        if aug_prefix and any(f.name.startswith(f"{aug_prefix}_") for f in image_files):
            image_files = [f for f in image_files if f.name.startswith(f"{aug_prefix}_")]
                
        # Batasi jumlah sampel dengan one-liner
        import random; random.seed(42)  # Untuk hasil yang konsisten
        image_files = random.sample(image_files, min(len(image_files), num_samples))
        
        # Tampilkan sampel dengan matplotlib
        with output_widget:
            clear_output(wait=True)
            
            # Buat grid untuk menampilkan gambar dengan one-liner
            cols = min(2, num_samples)
            rows = (num_samples + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            axes = np.array([axes]) if rows == 1 and cols == 1 else (axes.flatten() if rows == 1 or cols == 1 else axes)
                
            # Definisikan fungsi helper untuk load dan display gambar
            def load_and_display(idx, img_path):
                try:
                    # Baca gambar dengan utilitas dari dataset_utils
                    img = load_image(str(img_path), convert_rgb=True) if 'load_image' in locals() else plt.imread(img_path)
                    
                    # Tentukan posisi subplot dengan one-liner
                    ax = axes[idx] if axes.ndim == 1 else axes[idx // cols, idx % cols]
                    
                    # Tampilkan gambar dan set properti dengan one-liner
                    ax.imshow(img); ax.set_title(img_path.name, fontsize=10); ax.axis('off')
                except Exception as e:
                    # Tampilkan error placeholder dengan one-liner
                    ax = axes[idx] if axes.ndim == 1 else axes[idx // cols, idx % cols]
                    ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10); ax.axis('off')
            
            # Load dan tampilkan gambar dengan ThreadPoolExecutor untuk paralelisme
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
                list(executor.map(lambda x: load_and_display(*x), enumerate(image_files)))
            
            # Sembunyikan axis yang tidak digunakan dengan list comprehension
            [axes[i].axis('off') if axes.ndim == 1 else axes[i // cols, i % cols].axis('off') for i in range(len(image_files), rows * cols)]
                
            # Tampilkan plot dan informasi
            plt.tight_layout(); plt.show()
            display(create_info_alert(f"Menampilkan {len(image_files)} sampel dari {target_dir}", alert_type="success"))
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
            display(create_info_alert(f"Error saat visualisasi sampel: {str(e)}", alert_type="error"))

# Fungsi helper untuk komparasi
def compare_original_vs_processed_fallback(data_dir: str, processed_dir: str, output_widget: widgets.Output):
    """
    Komparasi gambar original vs processed dengan pendekatan yang lebih efisien.
    Menggunakan one-liner style dan utilitas dari dataset_utils.
    
    Args:
        data_dir: Direktori data original
        processed_dir: Direktori data processed
        output_widget: Widget output untuk menampilkan visualisasi
    """
    try:
        # Import utilitas yang diperlukan
        from smartcash.dataset.utils.data_utils import load_image, find_image_files
        from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
        
        # Validasi direktori dengan one-liner
        if not os.path.exists(data_dir):
            with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="warning", title="Direktori original tidak ditemukan", messages=[str(data_dir)]); return
            
        if not os.path.exists(processed_dir):
            with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="warning", title="Direktori processed tidak ditemukan", messages=[str(processed_dir)]); return
            
        # Tampilkan status pencarian file
        with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="info", title="Mencari file gambar untuk komparasi...")
            
        # Cari direktori gambar dengan fungsi helper
        original_dir = Path(data_dir) / 'images' if (Path(data_dir) / 'images').exists() else Path(data_dir)
        processed_dir = next((d for d in [Path(processed_dir) / 'images', Path(processed_dir) / 'train' / 'images'] if d.exists()), Path(processed_dir))
            
        # Cari file gambar dengan list comprehension dan fungsi helper
        extensions = [ext[1:] for ext in IMG_EXTENSIONS]  # Hapus * dari '*.jpg'
        original_files = [f for ext in extensions for f in original_dir.glob(f"*.{ext}")]
        processed_files = [f for ext in extensions for f in processed_dir.glob(f"*.{ext}")]
            
        # Validasi hasil pencarian dengan one-liner
        if not original_files:
            with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="warning", title="Tidak ada file gambar di direktori original", messages=[str(original_dir)]); return
            
        if not processed_files:
            with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="warning", title="Tidak ada file gambar di direktori processed", messages=[str(processed_dir)]); return
            
        # Cari file yang ada di kedua direktori dengan set operations
        common_names = {f.stem for f in original_files} & {f.stem for f in processed_files}
        
        if not common_names:
            with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="warning", title="Tidak ada file gambar yang sama di kedua direktori"); return
            
        # Ambil sampel untuk komparasi dengan one-liner
        import random; random.seed(42)  # Untuk hasil yang konsisten
        sample_names = random.sample(list(common_names), min(5, len(common_names)))
        
        # Buat pasangan file dengan list comprehension dan dictionary comprehension
        original_dict = {f.stem: f for f in original_files}
        processed_dict = {f.stem: f for f in processed_files}
        comparison_pairs = [(original_dict[name], processed_dict[name]) for name in sample_names if name in original_dict and name in processed_dict]
                
        # Tampilkan komparasi dengan matplotlib
        with output_widget:
            clear_output(wait=True)
            
            # Buat grid untuk menampilkan gambar dengan one-liner
            fig, axes = plt.subplots(len(comparison_pairs), 2, figsize=(12, 4 * len(comparison_pairs)))
            axes = np.array([axes]) if len(comparison_pairs) == 1 else axes
                
            # Definisikan fungsi helper untuk load dan display gambar
            def load_and_display_row(row_idx, img_paths):
                original_path, processed_path = img_paths
                
                # Load dan tampilkan original dengan one-liner
                try:
                    img_original = load_image(str(original_path), convert_rgb=True) if 'load_image' in locals() else plt.imread(original_path)
                    axes[row_idx, 0].imshow(img_original); axes[row_idx, 0].set_title(f"Original: {Path(original_path).name}", fontsize=10); axes[row_idx, 0].axis('off')
                except Exception as e:
                    axes[row_idx, 0].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[row_idx, 0].transAxes, color='red'); axes[row_idx, 0].axis('off')
                
                # Load dan tampilkan processed dengan one-liner
                try:
                    img_processed = load_image(str(processed_path), convert_rgb=True) if 'load_image' in locals() else plt.imread(processed_path)
                    axes[row_idx, 1].imshow(img_processed); axes[row_idx, 1].set_title(f"Processed: {Path(processed_path).name}", fontsize=10); axes[row_idx, 1].axis('off')
                except Exception as e:
                    axes[row_idx, 1].text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=axes[row_idx, 1].transAxes, color='red'); axes[row_idx, 1].axis('off')
            
            # Load dan tampilkan gambar dengan ThreadPoolExecutor untuk paralelisme
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
                list(executor.map(lambda x: load_and_display_row(*x), enumerate(comparison_pairs)))
            
            # Tampilkan plot dan informasi
            plt.tight_layout(); plt.show()
            display_visualization_status(output_widget, status_type="success", title=f"Berhasil menampilkan {len(comparison_pairs)} pasang gambar untuk komparasi")
    except Exception as e:
        with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="error", title="Error saat komparasi", messages=[str(e)])
            
def visualize_class_distribution_fallback(dataset_dir: str, output_widget: widgets.Output):
    """
    Visualisasi distribusi kelas dari dataset dengan pendekatan yang lebih efisien.
    Menggunakan utilitas dari dataset_utils dan one-liner style.
    
    Args:
        dataset_dir: Direktori dataset
        output_widget: Widget output untuk menampilkan hasil
    """
    try:
        # Import utilitas yang diperlukan
        from smartcash.dataset.utils.statistics.class_stats import ClassStatistics
        from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
        import logging
        logger = logging.getLogger('visualization')
        
        # Tampilkan status pencarian direktori label dengan one-liner
        with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="info", title="Mencari direktori label...")
        
        # Cari direktori label dengan pendekatan yang efisien menggunakan find_valid_label_directory
        dataset_path = Path(dataset_dir)
        potential_label_dirs = [dataset_path / 'labels', dataset_path / 'train' / 'labels', dataset_path / 'val' / 'labels', dataset_path / 'test' / 'labels']
        labels_dir = find_valid_label_directory(potential_label_dirs)
        
        # Validasi hasil pencarian dengan one-liner
        if labels_dir is None:
            display_no_data_message(output_widget, message="Tidak dapat menemukan direktori label.", detail_message="Pastikan dataset memiliki direktori 'labels' dengan file .txt di dalamnya."); return
        
        # Tampilkan status pengumpulan file label dengan one-liner
        with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="info", title="Mengumpulkan file label...")
        
        # Kumpulkan file label dengan list comprehension dan batasi jumlah file jika terlalu banyak
        try:
            label_files = [os.path.join(str(labels_dir), f) for f in os.listdir(str(labels_dir)) if f.endswith('.txt')]
            if len(label_files) > 5000: random.seed(42); label_files = random.sample(label_files, 5000)
        except Exception as e:
            with output_widget: display_visualization_status(output_widget, status_type="error", title="Error saat mengumpulkan file label", messages=[str(e)]); return
        
        # Tampilkan status perhitungan distribusi kelas dengan one-liner
        with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="info", title="Menghitung distribusi kelas", messages=[f"Memproses {len(label_files)} file label..."])
        
        # Gunakan ThreadPoolExecutor untuk memproses file secara paralel dengan Counter untuk thread-safety
        from collections import Counter
        class_counts = Counter()
        processed_count = 0
        
        # Definisikan fungsi untuk memproses file label dengan one-liner
        def process_label_file(file_path):
            nonlocal processed_count
            try:
                # Baca file dan ekstrak class_id dengan list comprehension dan one-liner
                with open(file_path, 'r') as f: class_ids = [int(line.split()[0]) for line in f.read().strip().split('\n') if line.strip() and len(line.split()) >= 5 and line.split()[0].isdigit()][:1000]
                
                # Update class_counts secara thread-safe dengan Counter.update
                class_counts.update(class_ids)
                
                # Update progress dengan one-liner
                processed_count += 1
                if processed_count % 100 == 0: 
                    with output_widget: clear_output(wait=True); display_visualization_status(output_widget, status_type="info", title="Menghitung distribusi kelas...", messages=[f"Memproses {processed_count} dari {len(label_files)} file"])
            except Exception as e:
                if logger: logger.debug(f"{ICONS['warning']} Error saat memproses file {file_path}: {str(e)}")
        
        # Proses file secara paralel dengan ThreadPoolExecutor dan one-liner
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor: list(executor.map(process_label_file, label_files))
        
        # Validasi hasil dengan one-liner
        if not class_counts:
            display_no_data_message(output_widget, message="Tidak ada data kelas ditemukan dalam file label.", detail_message="Silakan pastikan dataset telah diproses dengan benar dan memiliki anotasi kelas."); return
        
        # Tampilkan distribusi kelas dengan matplotlib dan one-liner
        with output_widget:
            clear_output(wait=True)
            
            # Buat visualisasi dengan matplotlib dan one-liner
            fig, ax = plt.subplots(figsize=(10, 6))
            classes, counts = list(class_counts.keys()), list(class_counts.values())
            bars = ax.bar(classes, counts, color=plt.cm.viridis(np.linspace(0, 0.8, len(classes))))
            
            # Tambahkan styling pada bar chart dengan list comprehension
            [ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{count}', ha='center', va='bottom', fontweight='bold') for bar, count in zip(bars, counts)]
            
            # Tambahkan label dan judul dengan one-liner
            ax.set_xlabel('Kelas', fontweight='bold'); ax.set_ylabel('Jumlah', fontweight='bold')
            ax.set_title('Distribusi Kelas Dataset', fontsize=14, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7); ax.set_xticks(classes)
            
            # Tampilkan plot dan statistik dengan one-liner
            plt.tight_layout(); plt.show()
            display_visualization_status(output_widget, status_type="info", title="Status Visualisasi: Visualisasi menggunakan data aktual dari dataset", 
                                      messages=[f"Berhasil menampilkan distribusi {len(classes)} kelas dari {sum(counts)} sampel.", f"Direktori label: {labels_dir}", f"Total file label: {len(label_files)}"])
    except Exception as e:
        with output_widget: display_visualization_status(output_widget, status_type="error", title="Error saat visualisasi distribusi", messages=[str(e)])
