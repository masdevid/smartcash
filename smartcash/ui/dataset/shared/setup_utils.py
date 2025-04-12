"""
File: smartcash/ui/dataset/shared/setup_utils.py
Deskripsi: Utilitas bersama untuk konfigurasi dan setup awal modul preprocessing/augmentasi
dengan deteksi otomatis status data
"""

from typing import Dict, Any, Optional, List, Union
import os
from pathlib import Path
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.shared.status_panel import update_status_panel
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def detect_module_state(ui_components: Dict[str, Any], module_type: str = 'preprocessing',
                      file_pattern: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Deteksi status dataset yang telah diproses dengan auto-detect pattern.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        file_pattern: Pattern file yang dicari (string atau list)
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Deteksi Google Drive dengan utilitas standar
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        # Tentukan lokasi dan pattern berdasarkan module_type
        if module_type == 'preprocessing':
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            # Pattern untuk files preprocessing (multi-format support)
            file_pattern = file_pattern or ['*.jpg', '*.png', '*.npy']
        else:  # augmentation
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')  # Utamakan preprocessed
            # Dapatkan prefix augmentasi
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            file_pattern = file_pattern or [f"{aug_prefix}_*.jpg", f"{aug_prefix}_*.png", f"{aug_prefix}_*.npy"]
        
        # Cek Google Drive jika tersedia
        drive_mounted, drive_path = detect_drive_mount()
        if drive_mounted and drive_path:
            smartcash_dir = os.path.join(drive_path, 'SmartCash')
            drive_target_dir = os.path.join(smartcash_dir, target_dir)
            
            # Jika jalur sama (symlink), gunakan lokal saja
            if os.path.realpath(target_dir) == os.path.realpath(drive_target_dir):
                if logger: logger.debug(f"🔄 Path lokal dan drive identik: {target_dir}, menggunakan lokal")
            else:
                # Gunakan jalur drive jika tersedia file di sana
                target_dir = drive_target_dir
        
        # Pastikan file_pattern adalah list
        if isinstance(file_pattern, str):
            file_pattern = [file_pattern]
        
        # Cek keberadaan file terolah dengan deteksi multi-split dan multi-format
        is_processed = False
        found_files_count = 0
        
        # Cek di setiap split dengan multi-pattern
        for split in DEFAULT_SPLITS:
            images_path = Path(target_dir) / split / 'images'
            if not images_path.exists():
                continue
            
            # Cek setiap pattern file yang didukung
            for pattern in file_pattern:
                files = list(images_path.glob(pattern))
                found_files_count += len(files)
                if files:
                    is_processed = True
                    if logger: logger.debug(f"{ICONS['folder']} Ditemukan {len(files)} file {pattern} di {images_path}")
            
            # Optimization: Break early jika sudah ditemukan
            if is_processed and found_files_count >= 5:  # Cukup untuk visualisasi
                break
        
        # Khusus augmentation: cek juga di direktori temp
        if module_type == 'augmentation' and not is_processed:
            temp_dir = ui_components.get('augmented_dir', 'data/augmented')
            
            # Cek di Google Drive jika tersedia
            if drive_mounted and drive_path:
                drive_temp_dir = os.path.join(drive_path, 'SmartCash', temp_dir)
                if os.path.realpath(temp_dir) != os.path.realpath(drive_temp_dir):
                    temp_dir = drive_temp_dir
            
            # Cek files di folder temp
            images_path = Path(temp_dir) / 'images'
            if images_path.exists():
                for pattern in file_pattern:
                    files = list(images_path.glob(pattern))
                    found_files_count += len(files)
                    if files:
                        is_processed = True
                        if logger: logger.debug(f"{ICONS['folder']} Ditemukan {len(files)} file {pattern} di {images_path}")
                        break
        
        # Update UI berdasarkan hasil deteksi
        if is_processed:
            # Dapatkan absolute path untuk pesan yang jelas
            abs_target_dir = os.path.abspath(target_dir)
            
            # Tampilkan pesan berdasarkan module_type
            message = (
                f"Dataset preprocessed sudah tersedia di: {abs_target_dir}" if module_type == 'preprocessing'
                else f"Data augmentasi tersedia dengan prefix: {aug_prefix}"
            )
            
            # Update status panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} {message}")
            
            # Tampilkan tombol-tombol yang relevan
            ui_components['cleanup_button'].layout.display = 'block'
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            # Tampilkan visualization container
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'block'
            
            # Update summary jika terdeteksi data
            try:
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    # Generate summary berdasarkan tipe modul
                    if module_type == 'preprocessing':
                        from smartcash.ui.dataset.shared.summary_handler import generate_preprocessing_summary
                        summary = generate_preprocessing_summary(target_dir)
                    else:
                        from smartcash.ui.dataset.shared.summary_handler import generate_augmentation_summary
                        summary = generate_augmentation_summary(target_dir, ui_components.get('augmented_dir', 'data/augmented'))
                    
                    # Update UI dengan summary
                    ui_components['update_summary'](summary)
                    
                    # Tampilkan summary container
                    if 'summary_container' in ui_components:
                        ui_components['summary_container'].layout.display = 'block'
            except Exception as e:
                if logger: logger.debug(f"{ICONS['info']} Tidak bisa mengupdate summary: {str(e)}")
        else:
            # Dataset belum diproses, sembunyikan tombol-tombol yang tidak relevan
            for component in ['cleanup_button', 'visualization_buttons']:
                if component in ui_components:
                    ui_components[component].layout.display = 'none'
            
            # Sembunyikan container-container yang tidak relevan
            for container in ['visualization_container', 'summary_container']:
                if container in ui_components:
                    ui_components[container].layout.display = 'none'
        
        # Simpan flags ke ui_components
        is_flag_key = 'is_preprocessed' if module_type == 'preprocessing' else 'is_augmented'
        ui_components[is_flag_key] = is_processed
    
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Deteksi status {module_type}: {str(e)}")
    
    return ui_components

def setup_manager(ui_components: Dict[str, Any], config: Dict[str, Any] = None, 
                module_type: str = 'preprocessing') -> Any:
    """
    Setup manager untuk modul (DatasetManager/AugmentationService) dengan fallback.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Manager instance (DatasetManager atau AugmentationService) atau None jika gagal
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    try:
        # Tentukan tipe manager berdasarkan module_type
        if module_type == 'preprocessing':
            # Import dan initialize DatasetManager
            from smartcash.ui.utils.fallback_utils import get_dataset_manager
            manager = get_dataset_manager(config, logger)
            manager_key = 'dataset_manager'
        else:
            # Import dan initialize AugmentationService
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Dapatkan parameter yang diperlukan
            data_dir = ui_components.get('data_dir', 'data')
            
            # Dapatkan num_workers dari UI jika tersedia
            num_workers = 4  # Default
            if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 5:
                num_workers = ui_components['aug_options'].children[5].value
            
            # Buat instance
            manager = AugmentationService(config, data_dir, logger, num_workers)
            manager_key = 'augmentation_manager'
        
        # Tambahkan manager ke ui_components
        ui_components[manager_key] = manager
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](manager)
            
        if logger: logger.info(f"{ICONS['success']} {manager_key.replace('_', ' ').title()} berhasil diinisialisasi")
        
        return manager
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error inisialisasi {module_type} manager: {str(e)}")
        return None

def register_resource(ui_components: Dict[str, Any], resource: Any, 
                      cleanup_func: Optional[callable] = None) -> None:
    """
    Register resource untuk auto-cleanup.
    
    Args:
        ui_components: Dictionary komponen UI
        resource: Resource yang perlu di-cleanup
        cleanup_func: Fungsi cleanup (opsional, default: resource.close())
    """
    # Buat list resources jika belum ada
    if 'resources' not in ui_components:
        ui_components['resources'] = []
    
    # Tambahkan resource dan cleanup function
    ui_components['resources'].append((resource, cleanup_func))