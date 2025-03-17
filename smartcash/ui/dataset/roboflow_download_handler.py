"""
File: smartcash/ui/dataset/roboflow_download_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow
"""

from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML

def download_from_roboflow(ui_components: Dict[str, Any], env=None, config=None):
    """
    Download dataset dari Roboflow - fungsi koordinator utama
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.download_initialization import update_status_panel, get_api_key_from_secret
    except ImportError:
        # Fallback
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def update_status_panel(ui_components, status_type, message):
            pass
        
        def get_api_key_from_secret():
            return None
            
        ICONS = {
            'key': '🔑',
            'folder': '📁',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
    
    if 'dataset_manager' not in ui_components:
        with ui_components['status']:
            display(create_status_indicator("error", f"{ICONS['error']} DatasetManager tidak tersedia"))
        return
    
    # Get download settings
    settings = get_roboflow_settings(ui_components)
    if not settings:
        return
        
    # Update config
    data_dir = setup_data_directory(ui_components, env, config)
    if not data_dir:
        return
    
    # Download dataset
    with ui_components['status']:
        display(create_status_indicator("info", 
            f"{ICONS['key']} Mengunduh dataset dari Roboflow ({settings['workspace']}/{settings['project']} v{settings['version']})..."))
    
    try:
        dataset_paths = execute_download(ui_components, settings, data_dir)
        handle_successful_download(ui_components, dataset_paths, data_dir)
    except Exception as e:
        handle_download_error(ui_components, e)

def get_roboflow_settings(ui_components):
    """
    Ambil pengaturan download Roboflow
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi pengaturan Roboflow atau None jika gagal
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        ICONS = {'error': '❌'}
    
    # Get settings
    api_settings = ui_components['roboflow_settings'].children
    api_key = api_settings[0].value
    workspace = api_settings[1].value
    project = api_settings[2].value
    version = api_settings[3].value
    
    # Try to get API key from Google Secret if not provided
    if not api_key:
        from smartcash.ui.dataset.download_initialization import get_api_key_from_secret
        api_key = get_api_key_from_secret()
        if api_key:
            api_settings[0].value = api_key
        else:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} API Key Roboflow tidak tersedia"))
            return None
    
    # Update config dengan pengaturan baru
    if config and 'data' in config and 'roboflow' in config['data']:
        config['data']['roboflow'].update({
            'api_key': api_key,
            'workspace': workspace,
            'project': project,
            'version': version
        })
        
        if 'dataset_manager' in ui_components:
            ui_components['dataset_manager'].config = config
    
    return {
        'api_key': api_key,
        'workspace': workspace,
        'project': project,
        'version': version
    }

def setup_data_directory(ui_components, env, config):
    """
    Setup direktori data, dengan prioritas Google Drive jika tersedia
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Path direktori data atau None jika gagal
    """
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        ICONS = {
            'folder': '📁',
            'success': '✅',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
    
    data_dir = config.get('data', {}).get('dir', 'data')
    
    # Cek apakah kita bisa menggunakan Google Drive
    if env and hasattr(env, 'is_colab') and env.is_colab:
        # Gunakan direktori Drive yang sudah terhubung
        if hasattr(env, 'drive_path') and env.is_drive_mounted:
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset"))
            data_dir = str(env.drive_path / 'data')
        # Coba hubungkan ke Drive
        elif hasattr(env, 'mount_drive'):
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['folder']} Mencoba menghubungkan ke Google Drive..."))
            try:
                env.mount_drive()
                if env.is_drive_mounted:
                    data_dir = str(env.drive_path / 'data')
                    display(create_status_indicator("success", f"{ICONS['success']} Google Drive berhasil terhubung"))
            except Exception as e:
                with ui_components['status']:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Tidak dapat menghubungkan ke Google Drive: {str(e)}"))
                display(create_status_indicator("info", f"{ICONS['folder']} Menggunakan penyimpanan lokal"))
    
    # Pastikan direktori ada
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    return data_dir

def execute_download(ui_components, settings, data_dir):
    """
    Eksekusi download dari Roboflow
    
    Args:
        ui_components: Dictionary komponen UI
        settings: Pengaturan Roboflow
        data_dir: Direktori data
        
    Returns:
        List path dataset atau None jika gagal
    """
    dataset_manager = ui_components['dataset_manager']
    
    # Download dataset
    return dataset_manager.download_from_roboflow(
        api_key=settings['api_key'],
        workspace=settings['workspace'],
        project=settings['project'],
        version=settings['version'],
        output_format="yolov5pytorch",
        output_dir=data_dir
    )

def handle_successful_download(ui_components, dataset_paths, data_dir):
    """
    Tangani download yang berhasil
    
    Args:
        ui_components: Dictionary komponen UI
        dataset_paths: Path dataset yang di-download
        data_dir: Direktori data
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
    
    with ui_components['status']:
        display(create_status_indicator("success", f"{ICONS['success']} Dataset berhasil diunduh ke {data_dir}"))
        
        # Validasi struktur dataset
        if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
            ui_components['validate_dataset_structure'](data_dir)
    
    # Update status panel
    update_status_panel(ui_components, "success", f"{ICONS['success']} Dataset siap digunakan: {len(dataset_paths)} splits")
    
    # Notify event if observer available
    if 'observer_manager' in ui_components:
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="DOWNLOAD_END",
                sender="download_handler",
                message="Download dataset berhasil",
                dataset_path=data_dir
            )
        except ImportError:
            pass

def handle_download_error(ui_components, error):
    """
    Tangani error saat download
    
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
    update_status_panel(ui_components, "error", f"{ICONS['error']} Download dataset gagal")
    
    # Notify error event if observer available
    if 'observer_manager' in ui_components:
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="DOWNLOAD_ERROR",
                sender="download_handler",
                message=f"Error: {str(error)}"
            )
        except ImportError:
            pass