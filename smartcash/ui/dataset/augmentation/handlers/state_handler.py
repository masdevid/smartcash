"""
File: smartcash/ui/dataset/augmentation/handlers/state_handler.py
Deskripsi: Handler state untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def detect_augmentation_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deteksi status data augmentasi dan update tampilan UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Dapatkan paths dan prefix dari ui_components
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        aug_prefix = 'aug'
        
        # Ambil prefix dari UI jika tersedia
        if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 2:
            aug_prefix = ui_components['aug_options'].children[2].value
        
        # Gunakan Google Drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_path = str(env_manager.drive_path)
                preprocessed_dir = os.path.join(drive_path, 'SmartCash', preprocessed_dir)
                augmented_dir = os.path.join(drive_path, 'SmartCash', augmented_dir)
                if logger: logger.debug(f"🔍 Mencari data augmentasi di drive: {preprocessed_dir}, {augmented_dir}")
        except (ImportError, AttributeError):
            pass
        
        # Konversi ke path absolut
        abs_preprocessed_dir = os.path.abspath(preprocessed_dir)
        abs_augmented_dir = os.path.abspath(augmented_dir)
        
        # Flag untuk status
        is_augmented = False
        
        # Pattern file yang dicari
        pattern = f"{aug_prefix}_*.jpg"
        
        # Cek di kedua lokasi - preprocessed dir
        preprocessed_path = Path(preprocessed_dir)
        for split in DEFAULT_SPLITS:
            split_images_dir = preprocessed_path / split / 'images'
            if split_images_dir.exists():
                aug_files = list(split_images_dir.glob(pattern))
                if aug_files:
                    is_augmented = True
                    if logger: logger.info(f"✅ Data augmentasi ditemukan di {split}: {len(aug_files)} file")
                    break
        
        # Jika tidak ditemukan di preprocessed, cek di augmented dir
        if not is_augmented:
            augmented_path = Path(augmented_dir)
            augmented_images_dir = augmented_path / 'images'
            if augmented_images_dir.exists():
                aug_files = list(augmented_images_dir.glob(pattern))
                if aug_files:
                    is_augmented = True
                    if logger: logger.info(f"✅ Data augmentasi ditemukan di {augmented_dir}: {len(aug_files)} file")
        
        # Update UI berdasarkan hasil deteksi
        ui_components['is_augmented'] = is_augmented
        
        if is_augmented:
            # Update status panel
            message = f"Dataset augmentasi tersedia dengan prefix: {aug_prefix}"
            update_status_panel(ui_components, "success", f"{ICONS['success']} {message}")
            
            # Tampilkan tombol yang relevan
            ui_components['cleanup_button'].layout.display = 'block'
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            # Tampilkan tombol visualisasi individual
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].disabled = False
            
            # Persiapkan container visualisasi dan summary
            for container in ['visualization_container', 'summary_container']:
                if container in ui_components:
                    ui_components[container].layout.display = 'block'
            
            # Tampilkan summary
            generate_augmentation_summary(ui_components, preprocessed_dir, augmented_dir)
        else:
            # Update status panel
            message = "Belum ada data augmentasi. Silakan jalankan augmentasi."
            update_status_panel(ui_components, "info", f"{ICONS['info']} {message}")
            
            # Sembunyikan tombol visualisasi dan cleanup
            ui_components['cleanup_button'].layout.display = 'none'
            ui_components['visualization_buttons'].layout.display = 'none'
            
            # Sembunyikan container
            for container in ['visualization_container', 'summary_container']:
                if container in ui_components:
                    ui_components[container].layout.display = 'none'
                    
            if logger: logger.info(f"ℹ️ Tidak ditemukan data augmentasi dengan prefix {aug_prefix}")
    
    except Exception as e:
        # Log error
        if logger: logger.warning(f"⚠️ Error saat mendeteksi status augmentasi: {str(e)}")
        
        # Tampilkan pesan error di status panel
        try:
            update_status_panel(ui_components, "warning", f"{ICONS['warning']} Error saat mendeteksi status augmentasi")
        except Exception:
            pass
        
        # Pastikan UI tetap berfungsi meskipun ada error
        ui_components['is_augmented'] = False
        
        # Sembunyikan tombol yang tidak perlu
        for btn_name in ['cleanup_button', 'visualization_buttons']:
            if btn_name in ui_components:
                ui_components[btn_name].layout.display = 'none'
        
        # Sembunyikan container
        for container in ['visualization_container', 'summary_container']:
            if container in ui_components:
                ui_components[container].layout.display = 'none'
    
    return ui_components

def generate_augmentation_summary(ui_components: Dict[str, Any], preprocessed_dir: Optional[str] = None, augmented_dir: Optional[str] = None) -> None:
    """
    Generate dan tampilkan ringkasan dataset augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        augmented_dir: Direktori dataset augmented
    """
    # Validasi parameter untuk mencegah error
    if not ui_components:
        return
    
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    
    try:
        # Gunakan preprocessed_dir dan augmented_dir dari parameter atau ui_components
        preprocessed_dir = preprocessed_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = augmented_dir or ui_components.get('augmented_dir', 'data/augmented')
        
        # Dapatkan prefix augmentasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        orig_prefix = 'rp'  # Default preprocessed prefix
        
        # Hitung jumlah file
        preprocessed_path = Path(preprocessed_dir)
        augmented_path = Path(augmented_dir)
        
        # Inisialisasi counters
        orig_files = 0
        aug_files = 0
        
        # Cek di preprocessed dir (untuk file original dan augmented)
        for split in DEFAULT_SPLITS:
            split_images_dir = preprocessed_path / split / 'images'
            if split_images_dir.exists():
                # Count original files
                orig_files += len(list(split_images_dir.glob(f"{orig_prefix}_*.jpg")))
                
                # Count augmented files
                aug_files += len(list(split_images_dir.glob(f"{aug_prefix}_*.jpg")))
        
        # Cek juga di augmented dir
        augmented_images_dir = augmented_path / 'images'
        if augmented_images_dir.exists():
            aug_files += len(list(augmented_images_dir.glob(f"{aug_prefix}_*.jpg")))
        
        # Hitung durasi (tidak diketahui dari loaded files)
        duration = 0
        
        # Dapatkan jenis augmentasi dengan validasi yang lebih kuat
        aug_types = []
        if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children'):
            try:
                aug_value = ui_components['aug_options'].children[0].value
                if aug_value is not None:
                    if isinstance(aug_value, (list, tuple)):
                        aug_types = list(aug_value)
                    elif isinstance(aug_value, str):
                        aug_types = [aug_value]
                    else:
                        # Jika tipe tidak dikenali, gunakan default
                        aug_types = ['Combined (Recommended)']
                else:
                    # Jika nilai None, gunakan default
                    aug_types = ['Combined (Recommended)']
            except Exception as e:
                # Jika terjadi error, gunakan default
                aug_types = ['Combined (Recommended)']
                if logger: logger.warning(f"{ICONS['warning']} Error saat mendapatkan aug_types: {str(e)}, menggunakan default")
        
        # Buat summary
        summary = {
            'original': orig_files,
            'generated': aug_files,
            'total_files': orig_files + aug_files,
            'duration': duration,
            'augmentation_types': aug_types,
            'output_dir': preprocessed_dir
        }
        
        # Tampilkan summary jika ditemukan data
        if aug_files > 0 and 'summary_container' in ui_components:
            with ui_components['summary_container']:
                clear_output(wait=True)
                
                # Header
                display(widgets.HTML(f"<h3 style='color:{COLORS['dark']}'>{ICONS['stats']} Ringkasan Augmentasi</h3>"))
                
                # Metrics grid
                metrics_container = widgets.HBox(layout=widgets.Layout(
                    display='flex', flex_flow='row wrap', align_items='flex-start',
                    justify_content='space-around', width='100%', margin='10px 0'
                ))
                
                # Create metric displays
                from smartcash.ui.utils.metric_utils import create_metric_display
                metrics_container.children = [
                    create_metric_display("File Original", summary['original']),
                    create_metric_display("File Augmentasi", summary['generated']),
                    create_metric_display("Total File", summary['total_files'])
                ]
                
                display(metrics_container)
                
                # Jenis augmentasi - pastikan aug_types selalu valid
                if not aug_types:
                    aug_types = ['Combined (Recommended)']
                
                display(widgets.HTML(f"""
                <div style="margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 5px;">
                    <p style="margin: 0;"><strong>Jenis augmentasi:</strong> {', '.join(aug_types)}</p>
                </div>
                """))
                
                # Path output
                display(widgets.HTML(f"""
                <div style="margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 5px;">
                    <p style="margin: 0;"><strong>Path Output:</strong> {preprocessed_dir}</p>
                </div>
                """))
                
            # Aktifkan tombol visualisasi
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].disabled = False
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat membuat ringkasan augmentasi: {str(e)}")