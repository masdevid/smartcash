"""
File: smartcash/ui/dataset/augmentation_initialization.py
Deskripsi: Inisialisasi komponen untuk augmentasi dataset dengan fokus pada data preprocessed
"""

from typing import Dict, Any
from pathlib import Path
import os
from IPython.display import display, HTML
import ipywidgets as widgets


def detect_augmentation_state(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen augmentasi dataset dengan deteksi multi-lokasi."""
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    logger = ui_components.get('logger')
    
    try:
        # Gunakan drive_utils standar untuk deteksi Google Drive
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        # Default lokasi preprocessed dari config
        preprocessed_dir = config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        file_prefix = config.get('preprocessing', {}).get('file_prefix', 'rp')
        
        # Default lokasi temp augmentasi dari config
        augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
        
        # Gunakan Google Drive jika tersedia
        drive_mounted, drive_path = detect_drive_mount()
        if drive_mounted and drive_path:
            smartcash_dir = f"{drive_path}/SmartCash"
            preprocessed_dir = f"{smartcash_dir}/{preprocessed_dir}"
            augmented_dir = f"{smartcash_dir}/{augmented_dir}"
        
        # Konversi ke path absolut untuk ditampilkan
        abs_preprocessed_dir = os.path.abspath(preprocessed_dir)
        abs_augmented_dir = os.path.abspath(augmented_dir)
        
        # Update status panel dengan utils standar
        update_status_panel(
            ui_components, 
            "info", 
            f"Augmentasi akan dilakukan dari sumber: <strong>{abs_preprocessed_dir}</strong> dan disimpan sementara di <strong>{abs_augmented_dir}</strong>"
        )
        
        # Cek status augmented data yang sudah ada (di kedua lokasi)
        preprocessed_path = Path(preprocessed_dir)
        augmented_path = Path(augmented_dir)
        is_augmented = False
        
        # Periksa apakah ada gambar augmentasi di direktori preprocessed
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Cek lokasi 1: folder preprocessed (prioritas)
        if preprocessed_path.exists():
            for split in ['train', 'valid', 'test']:
                images_dir = preprocessed_path / split / 'images'
                if images_dir.exists():
                    augmented_files = list(images_dir.glob(f"{aug_prefix}_*.*"))
                    if augmented_files:
                        is_augmented = True
                        if logger: logger.info(f"{ICONS['folder']} Ditemukan {len(augmented_files)} file augmentasi di {images_dir}")
                        break
        
        # Cek lokasi 2: folder temp augmentasi (backup)
        if not is_augmented and augmented_path.exists():
            images_dir = augmented_path / 'images'
            if images_dir.exists():
                augmented_files = list(images_dir.glob(f"{aug_prefix}_*.*"))
                if augmented_files:
                    is_augmented = True
                    if logger: logger.info(f"{ICONS['folder']} Ditemukan {len(augmented_files)} file augmentasi di {images_dir}")
        
        if is_augmented:
            # Tampilkan informasi dengan utilitas standar
            update_status_panel(
                ui_components,
                "success",
                f"Data augmentasi sudah tersedia dengan prefix: {aug_prefix}"
            )
            
            # Tampilkan tombol cleanup dan visualisasi
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
        
        # Cari file preprocessed dengan prefix rp (fokus hanya di train split)
        preprocessed_found = False
        rp_files_count = 0
        train_images_dir = preprocessed_path / 'train' / 'images'
        if train_images_dir.exists():
            rp_files = list(train_images_dir.glob(f"{file_prefix}_*.*"))
            if rp_files:
                preprocessed_found = True
                rp_files_count = len(rp_files)
                update_status_panel(
                    ui_components,
                    "info",
                    f"Ditemukan {rp_files_count} file sumber dengan prefix {file_prefix} di split train"
                )
                if logger: logger.info(f"{ICONS['file']} Ditemukan {rp_files_count} file sumber di {train_images_dir}")
                        
        if not preprocessed_found:
            update_status_panel(
                ui_components,
                "warning",
                f"Tidak ditemukan file sumber dengan prefix {file_prefix} di split train. Jalankan preprocessing terlebih dahulu."
            )
            if logger: logger.warning(f"{ICONS['warning']} Tidak ditemukan file sumber dengan prefix {file_prefix} di split train")
        
        # Store info di ui_components
        ui_components.update({
            'preprocessed_dir': preprocessed_dir,
            'augmented_dir': augmented_dir,
            'file_prefix': file_prefix,
            'is_augmented': is_augmented,
            'preprocessed_found': preprocessed_found,
            'rp_files_count': rp_files_count
        })
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Inisialisasi augmentasi: {str(e)}")
    
    return ui_components

def update_status_panel(ui_components, status_type, message):
    """Update status panel dengan pesan dan jenis status, menggunakan alert_utils standar."""
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika alert_utils tidak tersedia
        from smartcash.ui.utils.constants import ALERT_STYLES, ICONS
        
        if 'status_panel' in ui_components:
            style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
            bg_color = style.get('bg_color', '#d1ecf1')
            text_color = style.get('text_color', '#0c5460')
            border_color = style.get('border_color', '#0c5460') 
            icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
            
            ui_components['status_panel'].value = f"""
            <div style="padding: 10px; background-color: {bg_color}; 
                        color: {text_color}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {border_color};">
                <p style="margin:5px 0">{icon} {message}</p>
            </div>
            """