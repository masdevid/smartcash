"""
File: smartcash/ui/dataset/local_upload_handler.py
Deskripsi: Handler untuk upload dataset dari file lokal
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def process_local_upload(ui_components: Dict[str, Any], env=None, config=None):
    """
    Proses upload dataset dari file lokal - fungsi koordinator utama
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.download_initialization import update_status_panel
    except ImportError:
        # Fallback
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def update_status_panel(ui_components, status_type, message):
            pass
            
        ICONS = {
            'upload': '📤',
            'folder': '📂',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }
    
    if 'dataset_manager' not in ui_components:
        with ui_components['status']:
            display(create_status_indicator("error", f"{ICONS['error']} DatasetManager tidak tersedia"))
        return
    
    # Dapatkan info file
    file_data = get_upload_file_data(ui_components)
    if not file_data:
        return
    
    # Process upload
    with ui_components['status']:
        display(create_status_indicator("info", 
            f"{ICONS['upload']} Memproses file upload: {file_data['name']} ({file_data['size']/1024:.1f} KB)"))
    
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 25
    
    try:
        # Simpan file dan ekstrak
        target_dir = file_data['target_dir']
        zip_path = save_uploaded_file(ui_components, file_data)
        extracted_dir = extract_dataset(ui_components, zip_path, target_dir)
        handle_successful_upload(ui_components, file_data, target_dir)
    except Exception as e:
        handle_upload_error(ui_components, e)

def get_upload_file_data(ui_components):
    """
    Dapatkan data file yang diupload
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi data file atau None jika gagal
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        ICONS = {'warning': '⚠️'}
    
    upload_widget = ui_components['local_upload'].children[0]
    target_dir = ui_components['local_upload'].children[1].value
    
    if not upload_widget.value:
        with ui_components['status']:
            display(create_status_indicator("warning", f"{ICONS['warning']} Silahkan pilih file ZIP untuk diupload"))
        return None
    
    # Extract file info
    file_info = next(iter(upload_widget.value.values()))
    file_name = file_info.get('metadata', {}).get('name', 'unknown.zip')
    file_size = file_info.get('metadata', {}).get('size', 0)
    file_content = file_info.get('content', b'')
    
    return {
        'name': file_name,
        'size': file_size,
        'content': file_content,
        'target_dir': target_dir
    }

def save_uploaded_file(ui_components, file_data):
    """
    Simpan file yang diupload ke disk
    
    Args:
        ui_components: Dictionary komponen UI
        file_data: Dictionary berisi data file
        
    Returns:
        Path file yang disimpan
    """
    target_dir = file_data['target_dir']
    file_name = file_data['name']
    file_content = file_data['content']
    
    # Pastikan direktori target ada
    os.makedirs(target_dir, exist_ok=True)
    temp_zip_path = os.path.join(target_dir, file_name)
    
    # Simpan file terlebih dahulu
    with open(temp_zip_path, 'wb') as f:
        f.write(file_content)
    
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 50
        
    return temp_zip_path

def extract_dataset(ui_components, zip_path, target_dir):
    """
    Ekstrak dataset dari file ZIP
    
    Args:
        ui_components: Dictionary komponen UI
        zip_path: Path file ZIP
        target_dir: Direktori tujuan
        
    Returns:
        Path direktori hasil ekstraksi
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        ICONS = {'folder': '📂'}
    
    with ui_components['status']:
        display(create_status_indicator("info", f"{ICONS['folder']} Ekstraksi dan validasi file..."))
    
    dataset_manager = ui_components['dataset_manager']
    return dataset_manager.import_dataset_from_zip(
        zip_path=zip_path,
        target_dir=target_dir,
        format="yolov5pytorch"
    )

def handle_successful_upload(ui_components, file_data, target_dir):
    """
    Tangani upload yang berhasil
    
    Args:
        ui_components: Dictionary komponen UI
        file_data: Dictionary berisi data file
        target_dir: Direktori tujuan
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.download_initialization import update_status_panel
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def update_status_panel(ui_comp, status_type, message):
            pass
            
        ICONS = {'success': '✅'}
    
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 100
    
    with ui_components['status']:
        display(create_status_indicator("success", f"{ICONS['success']} File berhasil diproses: {file_data['name']}"))
        
        # Validasi struktur dataset
        if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
            ui_components['validate_dataset_structure'](target_dir)
    
    # Update status panel
    update_status_panel(ui_components, "success", f"{ICONS['success']} Dataset lokal siap digunakan")
    
    # Notify event if observer available
    if 'observer_manager' in ui_components:
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="DOWNLOAD_END",
                sender="local_upload_handler",
                message="Dataset berhasil diekstrak",
                dataset_path=target_dir
            )
        except ImportError:
            pass

def handle_upload_error(ui_components, error):
    """
    Tangani error saat upload
    
    Args:
        ui_components: Dictionary komponen UI
        error: Exception yang terjadi
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.download_initialization import update_status_panel
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def update_status_panel(ui_comp, status_type, message):
            pass
            
        ICONS = {'error': '❌'}
    
    with ui_components['status']:
        display(create_status_indicator("error", f"{ICONS['error']} Error: {str(error)}"))
    
    # Update status panel
    update_status_panel(ui_components, "error", f"{ICONS['error']} Proses file gagal")
    
    # Notify error event if observer available
    if 'observer_manager' in ui_components:
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="DOWNLOAD_ERROR",
                sender="local_upload_handler",
                message=f"Error: {str(error)}"
            )
        except ImportError:
            pass