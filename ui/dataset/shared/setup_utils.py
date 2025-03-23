"""
File: smartcash/ui/dataset/shared/setup_utils.py
Deskripsi: Utilitas bersama untuk konfigurasi dan setup awal modul preprocessing/augmentasi
"""

from typing import Dict, Any, Tuple
import os
from pathlib import Path
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.shared.status_panel import update_status_panel

def detect_module_state(ui_components: Dict[str, Any], module_type: str = 'preprocessing', 
                       file_pattern: str = None) -> Dict[str, Any]:
    """
    Deteksi status data yang telah diproses untuk berbagai modul.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        file_pattern: Pola file untuk dicari (opsional)
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Gunakan drive_utils standar untuk deteksi Google Drive
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        # Default lokasi dan pattern berdasarkan module_type
        if module_type == 'preprocessing':
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            # Cek semua format file yang didukung
            file_pattern = file_pattern or ['*.jpg', '*.png', '*.npy']
        else:  # augmentation
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')  # Utamakan preprocessed untuk augmentasi
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            file_pattern = file_pattern or [f"{aug_prefix}_*.jpg", f"{aug_prefix}_*.png"]
            
        # Gunakan Google Drive jika tersedia
        drive_mounted, drive_path = detect_drive_mount()
        if drive_mounted and drive_path:
            smartcash_dir = f"{drive_path}/SmartCash"
            target_dir = f"{smartcash_dir}/{target_dir}"
        
        # Konversi ke path absolut
        abs_target_dir = os.path.abspath(target_dir)
        
        # PERBAIKAN: Deteksi lebih komprehensif untuk berbagai split dan format file
        is_processed = False
        
        # Pastikan file_pattern adalah list
        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]
            
        for split in ['train', 'valid', 'test']:
            images_path = Path(target_dir) / split / 'images'
            if not images_path.exists():
                continue
                
            # Cek semua format file yang didukung
            for pattern in file_pattern:
                files = list(images_path.glob(pattern))
                if files:
                    is_processed = True
                    if logger: logger.info(f"{ICONS['folder']} Ditemukan {len(files)} file {pattern} di {images_path}")
                    break
            
            if is_processed:
                break
        
        # Tambahan untuk augmentation: cek juga di direktori temp
        if module_type == 'augmentation' and not is_processed:
            temp_dir = ui_components.get('augmented_dir', 'data/augmented')
            
            # Gunakan Google Drive jika tersedia
            if drive_mounted and drive_path:
                temp_dir = f"{drive_path}/SmartCash/{temp_dir}"
                
            images_path = Path(temp_dir) / 'images'
            if images_path.exists():
                for pattern in file_pattern:
                    files = list(images_path.glob(pattern))
                    if files:
                        is_processed = True
                        if logger: logger.info(f"{ICONS['folder']} Ditemukan {len(files)} file {pattern} di {images_path}")
                        break
        
        # Update status panel berdasarkan hasil deteksi
        if is_processed:
            message = (
                f"Dataset preprocessed sudah tersedia di: {abs_target_dir}" if module_type == 'preprocessing'
                else f"Data augmentasi sudah tersedia dengan prefix: {aug_prefix}"
            )
            
            update_status_panel(ui_components, "success", f"{ICONS['success']} {message}")
            
            # Tampilkan tombol yang relevan
            ui_components['cleanup_button'].layout.display = 'block'
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            # Tampilkan tombol visualisasi individual
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].layout.display = 'inline-flex'
                    
            # Tampilkan container visualisasi
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
            
            # Update summary jika terdeteksi data
            try:
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    # Generate summary berdasarkan module_type
                    if module_type == 'preprocessing':
                        from smartcash.ui.dataset.shared.summary_handler import generate_preprocessing_summary
                        summary = generate_preprocessing_summary(target_dir)
                    else:
                        from smartcash.ui.dataset.shared.summary_handler import generate_augmentation_summary
                        summary = generate_augmentation_summary(ui_components.get('augmented_dir', 'data/augmented'))
                    
                    ui_components['update_summary'](summary)
                    
                    # Tampilkan summary container
                    if 'summary_container' in ui_components:
                        ui_components['summary_container'].layout.display = 'block'
            except Exception as e:
                if logger: logger.debug(f"{ICONS['info']} {str(e)}")
        
        # Simpan flags di ui_components
        if module_type == 'preprocessing':
            ui_components['is_preprocessed'] = is_processed
        else:
            ui_components['is_augmented'] = is_processed
    
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Inisialisasi {module_type}: {str(e)}")
    
    return ui_components

def setup_manager(ui_components: Dict[str, Any], config: Dict[str, Any], module_type: str = 'preprocessing') -> Any:
    """
    Setup manager untuk modul tertentu dengan fallback.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Manager instance atau None
    """
    logger = ui_components.get('logger')
    
    try:
        # Coba setup manager berdasarkan module_type
        if module_type == 'preprocessing':
            # Import dataset manager dengan fallback
            from smartcash.ui.utils.fallback_utils import get_dataset_manager
            manager = get_dataset_manager(config, logger)
            manager_key = 'dataset_manager'
        else:
            # Import augmentation service
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Dapatkan paths dan config yang diperlukan
            data_dir = ui_components.get('data_dir', 'data')
            
            # Ambil num_workers dari UI jika tersedia
            num_workers = 4  # Default value
            if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 5:
                num_workers = ui_components['aug_options'].children[5].value
            
            # Buat instance AugmentationService dengan explicit parameter
            manager = AugmentationService(config, data_dir, logger, num_workers)
            manager_key = 'augmentation_manager'
            
        # Simpan manager di ui_components
        ui_components[manager_key] = manager
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](manager)
            
        if logger: logger.info(f"{ICONS['success']} {manager_key} berhasil diinisialisasi")
        
        return manager
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error inisialisasi {module_type} manager: {str(e)}")
        return None