"""
File: smartcash/ui/dataset/download_confirmation_handler.py
Deskripsi: Handler untuk menampilkan dialog konfirmasi di atas output box
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML
def check_existing_dataset(data_dir: str = "data") -> bool:
    """
    Cek apakah dataset sudah ada di direktori data dengan validasi lebih baik.
    
    Args:
        data_dir: Path direktori data
        
    Returns:
        bool: True jika dataset terdeteksi
    """
    required_dirs = [os.path.join(data_dir, split, folder) for split in ['train', 'valid', 'test'] for folder in ['images', 'labels']]
    
    # Tentukan minimal 4 folder yang harus ada untuk mendeteksi dataset
    existing_dirs = sum(1 for dir_path in required_dirs if os.path.isdir(dir_path))
    
    # Cek juga apakah ada file gambar di salah satu folder images
    has_images = False
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(data_dir, split, 'images')
        if os.path.isdir(img_dir):
            files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if files: has_images = True; break
    
    # Dataset dianggap valid jika minimal 4 folder ada dan minimal ada 1 file gambar
    return existing_dirs >= 4 and has_images

def get_dataset_stats(data_dir: str = "data") -> Dict[str, Any]:
    """Dapatkan statistik dataset yang ada"""
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
    """Setup handler untuk konfirmasi download"""
    try:
        import ipywidgets as widgets
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS, COLORS
        
        # Buat container untuk dialog konfirmasi di atas output
        if 'confirmation_container' not in ui_components:
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
            
            # Tambahkan ke UI utama, pastikan ada di atas output box
            if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
                # Cari posisi output box
                output_index = -1
                for i, child in enumerate(ui_components['ui'].children):
                    if child is ui_components.get('status'):
                        output_index = i
                        break
                
                # Tambahkan dialog container sebelum output
                if output_index > 0:
                    children_list = list(ui_components['ui'].children)
                    children_list.insert(output_index, ui_components['confirmation_container'])
                    ui_components['ui'].children = tuple(children_list)
    except ImportError:
        # Fallback jika ipywidgets tidak tersedia
        pass
    
    # Cek apakah kita memiliki fungsi asli download dari click handler
    original_click_handler = None
    if 'on_download_click' in ui_components and callable(ui_components['on_download_click']):
        original_click_handler = ui_components['on_download_click']
    
    # Simpan handler asli ke UI components agar dapat diakses oleh handler lain
    ui_components['original_download_handler'] = original_click_handler
    
    # Handler untuk download button dengan konfirmasi
    def on_download_click_with_confirmation(b):
        # Dapatkan lokasi data
        data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
            data_dir = str(env.drive_path / 'data')
        
        # Cek dataset yang sudah ada
        existing_dataset = check_existing_dataset(data_dir)
        
        if existing_dataset:
            # Dataset sudah ada, tampilkan konfirmasi
            stats = get_dataset_stats(data_dir)
            
            try:
                import ipywidgets as widgets
                # Buat dialog konfirmasi
                confirmation_title = widgets.HTML(f"<h3 style='margin-top:0'>{ICONS['warning']} Konfirmasi Download Ulang</h3>")
                confirmation_msg = widgets.HTML(f"<p>Dataset sudah tersedia dengan {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']}). Apakah Anda yakin ingin mendownload ulang?</p>")
                
                # Tombol aksi
                btn_cancel = widgets.Button(description='Batal', button_style='warning', layout=widgets.Layout(margin='5px'))
                btn_confirm = widgets.Button(description='Ya, Download Ulang', button_style='danger', layout=widgets.Layout(margin='5px'))
                btn_container = widgets.HBox([btn_cancel, btn_confirm])
                
                # Fungsi aksi
                def on_confirm(b):
                    # Sembunyikan dialog
                    ui_components['confirmation_container'].layout.display = 'none'
                    
                    # Progress bar sebaiknya reset dulu
                    if 'progress_bar' in ui_components: ui_components['progress_bar'].value = 0
                    
                    # Tentukan option yang dipilih
                    download_option = ui_components.get('download_options').value
                    
                    # Jalankan handler yang sesuai berdasarkan option yang dipilih
                    try:
                        if download_option == 'Roboflow (Online)':
                            from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
                            download_from_roboflow(ui_components, env, config)
                        elif download_option == 'Local Data (Upload)':
                            from smartcash.ui.dataset.local_upload_handler import process_local_upload
                            process_local_upload(ui_components, env, config)
                    except Exception as e:
                        with ui_components['status']:
                            from smartcash.ui.components.alerts import create_status_indicator
                            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
                            
                        # Log error
                        if 'logger' in ui_components:
                            ui_components['logger'].error(f"{ICONS['error']} Error saat download dataset: {str(e)}")
                
                def on_cancel(b):
                    # Sembunyikan dialog
                    ui_components['confirmation_container'].layout.display = 'none'
                    # Tampilkan pesan dibatalkan
                    with ui_components['status']:
                        display(create_status_indicator("info", f"{ICONS['info']} Download dibatalkan"))
                
                # Register handler
                btn_confirm.on_click(on_confirm)
                btn_cancel.on_click(on_cancel)
                
                # Tampilkan dialog di container
                ui_components['confirmation_container'].children = [confirmation_title, confirmation_msg, btn_container]
                ui_components['confirmation_container'].layout.display = 'flex'
                
            except (ImportError, AttributeError):
                # Fallback ke tampilan di output
                if original_click_handler:
                    original_click_handler(b)
        else:
            # Dataset belum ada, langsung download
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['download']} Dataset belum ada, memulai download..."))
            
            # Jalankan download original
            if original_click_handler:
                original_click_handler(b)
            elif 'on_download_click' in ui_components and callable(ui_components['on_download_click']):
                ui_components['on_download_click'](b)
    
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
    ui_components['check_existing_dataset'] = check_existing_dataset
    ui_components['get_dataset_stats'] = get_dataset_stats
    
    return ui_components
