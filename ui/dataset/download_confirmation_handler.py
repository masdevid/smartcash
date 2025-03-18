"""
File: smartcash/ui/dataset/download_confirmation_handler.py
Deskripsi: Handler untuk menampilkan dialog konfirmasi sebelum download dataset dan pengecekan dataset yang sudah ada
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML

def check_existing_dataset(data_dir: str = "data") -> bool:
    """
    Cek apakah dataset sudah ada di direktori data
    
    Args:
        data_dir: Direktori data
        
    Returns:
        Boolean menunjukkan keberadaan dataset
    """
    required_dirs = [
        os.path.join(data_dir, split, folder)
        for split in ['train', 'valid', 'test']
        for folder in ['images', 'labels']
    ]
    
    # Dataset dianggap ada jika minimal 4 dari 6 folder yang dibutuhkan ada
    existing_dirs = sum(1 for dir_path in required_dirs if os.path.isdir(dir_path))
    return existing_dirs >= 4

def get_dataset_stats(data_dir: str = "data") -> Dict[str, Any]:
    """
    Dapatkan statistik dataset yang ada
    
    Args:
        data_dir: Direktori data
        
    Returns:
        Dictionary berisi statistik dataset
    """
    stats = {'total_images': 0, 'train': 0, 'valid': 0, 'test': 0}
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(data_dir, split, 'images')
        if os.path.isdir(img_dir):
            image_count = len([f for f in os.listdir(img_dir) 
                             if os.path.isfile(os.path.join(img_dir, f)) and 
                             f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats[split] = image_count
            stats['total_images'] += image_count
    
    return stats

def setup_confirmation_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk konfirmasi download dan pengecekan dataset
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS, COLORS
        from smartcash.ui.components.helpers import create_confirmation_dialog
    except ImportError:
        # Fallback tanpa komponen UI
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        ICONS = {
            'warning': '⚠️',
            'success': '✅',
            'info': 'ℹ️',
            'folder': '📁',
            'download': '📥'
        }
        
        COLORS = {
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
        
        def create_confirmation_dialog(title, message, on_confirm, on_cancel=None, 
                                     confirm_label="Konfirmasi", cancel_label="Batal"):
            # Fallback tanpa dialog konfirmasi
            on_confirm()
            return None
    
    # Cek apakah kita memiliki fungsi asli download dari roboflow handler
    original_roboflow_handler = None
    if 'download_button' in ui_components and hasattr(ui_components['download_button'], '_click_handlers'):
        original_roboflow_handler = ui_components['download_button']._click_handlers.callbacks[0]
    
    # Handler untuk download button dengan konfirmasi
    def on_download_click_with_confirmation(b):
        # Dapatkan lokasi data
        data_dir = config.get('data', {}).get('dir', 'data')
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
            data_dir = str(env.drive_path / 'data')
        
        # Cek dataset yang sudah ada
        existing_dataset = check_existing_dataset(data_dir)
        
        if existing_dataset:
            # Dataset sudah ada, tampilkan konfirmasi
            stats = get_dataset_stats(data_dir)
            with ui_components['status']:
                display(create_status_indicator("info", 
                    f"{ICONS['folder']} Dataset sudah ada dengan {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})"))
            
            def on_confirm_redownload():
                # Jalankan download original
                if original_roboflow_handler:
                    original_roboflow_handler(b)
                else:
                    # Fallback jika tidak menemukan handler asli
                    from smartcash.ui.dataset.download_click_handler import on_download_click
                    on_download_click(b)
            
            def on_cancel_download():
                with ui_components['status']:
                    display(create_status_indicator("info", f"{ICONS['info']} Download dibatalkan"))
            
            # Tampilkan dialog konfirmasi
            confirmation_dialog = create_confirmation_dialog(
                "Konfirmasi Download Ulang",
                f"Dataset sudah tersedia dengan {stats['total_images']} gambar. Apakah Anda yakin ingin mendownload ulang?",
                on_confirm_redownload,
                on_cancel_download,
                "Ya, Download Ulang",
                "Batal"
            )
            
            with ui_components['status']:
                display(confirmation_dialog)
        else:
            # Dataset belum ada, langsung download
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['download']} Dataset belum ada, memulai download..."))
            
            # Jalankan download original
            if original_roboflow_handler:
                original_roboflow_handler(b)
            else:
                # Fallback jika tidak menemukan handler asli
                from smartcash.ui.dataset.download_click_handler import on_download_click
                on_download_click(b)
    
    # Update handler untuk tombol download
    if 'download_button' in ui_components:
        # Hapus semua handler yang ada
        if hasattr(ui_components['download_button'], '_click_handlers'):
            ui_components['download_button']._click_handlers.callbacks.clear()
            
        # Registrasi handler baru dengan konfirmasi
        ui_components['download_button'].on_click(on_download_click_with_confirmation)
        
        # Tambahkan referensi handler asli untuk digunakan nanti
        ui_components['original_download_handler'] = original_roboflow_handler
    
    # Tambahkan fungsi utilitas ke ui_components
    ui_components['check_existing_dataset'] = check_existing_dataset
    ui_components['get_dataset_stats'] = get_dataset_stats
    
    return ui_components