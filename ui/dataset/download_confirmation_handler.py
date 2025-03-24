"""
File: smartcash/ui/dataset/download_confirmation_handler.py
Deskripsi: Handler untuk menampilkan dialog konfirmasi download dengan integrasi utils standar
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def check_existing_dataset(data_dir: str = "data") -> bool:
    """
    Cek apakah dataset sudah ada di direktori data dengan validasi lebih baik.
    
    Args:
        data_dir: Path direktori data
        
    Returns:
        bool: True jika dataset terdeteksi
    """
    # Gunakan utils standar untuk path handling
    from smartcash.ui.utils.file_utils import list_files
    
    required_dirs = [os.path.join(data_dir, split, folder) for split in DEFAULT_SPLITS for folder in ['images', 'labels']]
    existing_dirs = sum(1 for dir_path in required_dirs if os.path.isdir(dir_path))
    
    # Cek juga apakah ada file gambar dengan utils standar
    has_images = False
    for split in DEFAULT_SPLITS:
        img_dir = os.path.join(data_dir, split, 'images')
        if os.path.isdir(img_dir):
            image_files = list_files(img_dir, pattern="*.jpg", recursive=False)
            if not image_files:
                image_files = list_files(img_dir, pattern="*.png", recursive=False)
            if image_files: has_images = True; break
    
    # Dataset dianggap valid jika minimal 4 folder ada dan minimal ada 1 file gambar
    return existing_dirs >= 4 and has_images

def get_dataset_stats(data_dir: str = "data") -> Dict[str, Any]:
    """Dapatkan statistik dataset yang ada menggunakan utils standar."""
    # Gunakan utils standar untuk analisis file
    from smartcash.ui.utils.file_utils import list_files
    
    stats = {'total_images': 0, 'train': 0, 'valid': 0, 'test': 0}
    
    for split in DEFAULT_SPLITS:
        img_dir = os.path.join(data_dir, split, 'images')
        if os.path.isdir(img_dir):
            image_files = list_files(img_dir, pattern="*.jpg", recursive=False)
            if not image_files:
                image_files = list_files(img_dir, pattern="*.png", recursive=False)
            image_count = len(image_files)
            stats[split] = image_count
            stats['total_images'] += image_count
    
    return stats

def setup_confirmation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk konfirmasi download dengan utils standar."""
    # Gunakan komponen standar dari utils
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.helpers.ui_helpers import create_confirmation_dialog
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    # Dapatkan logger dari UI components
    logger = ui_components.get('logger')
    
    # Buat container konfirmasi jika tidak ada
    if 'confirmation_container' not in ui_components:
        import ipywidgets as widgets
        ui_components['confirmation_container'] = widgets.Box(
            layout=widgets.Layout(
                display='none',
                flex_flow='column',
                border='1px solid #ddd',
                margin='10px 0',
                padding='10px',
                border_radius='5px',
                background_color='#f8f9fa'
            )
        )
        
        # Tambahkan ke UI utama di atas output box
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            output_index = -1
            for i, child in enumerate(ui_components['ui'].children):
                if child is ui_components.get('status'):
                    output_index = i
                    break
            
            if output_index > 0:
                children_list = list(ui_components['ui'].children)
                children_list.insert(output_index, ui_components['confirmation_container'])
                ui_components['ui'].children = tuple(children_list)
    
    # Handler asli untuk tombol download
    original_click_handler = ui_components.get('on_download_click')
    ui_components['original_download_handler'] = original_click_handler
    
    # Handler untuk download button dengan konfirmasi
    def on_download_click_with_confirmation(b):
        # Dapatkan lokasi data menggunakan drive utils standar
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
        drive_mounted, drive_path = detect_drive_mount()
        if drive_mounted and drive_path:
            data_dir = f"{drive_path}/data"
        
        # Cek dataset existing
        existing_dataset = check_existing_dataset(data_dir)
        
        if existing_dataset:
            # Dataset sudah ada, tampilkan konfirmasi dengan komponen standar
            stats = get_dataset_stats(data_dir)
            
            # Gunakan fungsi helper untuk membuat dialog konfirmasi
            def on_confirm():
                # Sembunyikan dialog
                ui_components['confirmation_container'].layout.display = 'none'
                
                # Progress bar sebaiknya reset dulu
                if 'progress_bar' in ui_components: ui_components['progress_bar'].value = 0
                
                # Tentukan option yang dipilih dan jalankan handler yang sesuai
                download_option = ui_components.get('download_options').value
                
                try:
                    if download_option == 'Roboflow (Online)':
                        from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
                        download_from_roboflow(ui_components, env, config)
                    elif download_option == 'Local Data (Upload)':
                        from smartcash.ui.dataset.local_upload_handler import process_local_upload
                        process_local_upload(ui_components, env, config)
                except Exception as e:
                    # Gunakan error handler standar
                    from smartcash.ui.handlers.error_handler import handle_ui_error
                    handle_ui_error(e, ui_components['status'], True, f"{ICONS['error']} Error saat download dataset")
            
            def on_cancel():
                # Sembunyikan dialog
                ui_components['confirmation_container'].layout.display = 'none'
                # Tampilkan pesan dibatalkan dengan utils standar
                with ui_components['status']:
                    display(create_status_indicator("info", f"{ICONS['info']} Download dibatalkan"))
            
            # Buat konten konfirmasi dengan helper
            confirmation_dialog = create_confirmation_dialog(
                "Konfirmasi Download Ulang",
                f"Dataset sudah tersedia dengan {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']}). Apakah Anda yakin ingin mendownload ulang?",
                on_confirm, on_cancel, "Ya, Download Ulang", "Batal"
            )
            
            # Tampilkan dialog
            ui_components['confirmation_container'].children = [confirmation_dialog]
            ui_components['confirmation_container'].layout.display = 'flex'
            
        else:
            # Dataset belum ada, langsung download
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['download']} Dataset belum ada, memulai download..."))
            
            # Jalankan download original
            if original_click_handler:
                original_click_handler(b)
    
    # Update handler untuk tombol download
    if 'download_button' in ui_components:
        # Hapus handler yang ada dengan aman
        if hasattr(ui_components['download_button'], '_click_handlers'):
            try:
                ui_components['download_button']._click_handlers.callbacks.clear()
            except (AttributeError, IndexError):
                pass
            
        # Registrasi handler baru
        ui_components['download_button'].on_click(on_download_click_with_confirmation)
    
    # Tambahkan fungsi utilitas ke ui_components
    ui_components.update({
        'check_existing_dataset': check_existing_dataset,
        'get_dataset_stats': get_dataset_stats,
        'on_download_click_with_confirmation': on_download_click_with_confirmation
    })
    
    return ui_components