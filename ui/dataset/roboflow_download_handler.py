"""
File: smartcash/ui/dataset/roboflow_download_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow yang diperbaiki untuk mencegah index out of range
"""

from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.alerts import create_status_indicator

def download_from_roboflow(ui_components: Dict[str, Any], env=None, config=None):
    """
    Download dataset dari Roboflow - fungsi koordinator utama
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
    """
    logger = ui_components.get('logger')
    
    # Cek ketersediaan dataset manager
    if 'dataset_manager' not in ui_components:
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} DatasetManager tidak tersedia"))
        return
    
    # Get Roboflow settings
    settings = get_roboflow_settings(ui_components)
    if not settings: return
    
    # Setup data directory
    data_dir = setup_data_directory(ui_components, env, config)
    if not data_dir: return
    
    # Log download
    if logger: logger.info(f"🔑 Mengunduh dataset dari Roboflow ({settings['workspace']}/{settings['project']} v{settings['version']})")
    else:
        with ui_components['status']: display(create_status_indicator("info", f"{ICONS['key']} Mengunduh dataset dari Roboflow ({settings['workspace']}/{settings['project']} v{settings['version']})..."))
    
    try:
        # Eksekusi download
        dataset_paths = ui_components['dataset_manager'].download_from_roboflow(
            api_key=settings['api_key'],
            workspace=settings['workspace'],
            project=settings['project'],
            version=settings['version'],
            output_format="yolov5pytorch",
            output_dir=data_dir
        )
        
        # Handle sukses
        with ui_components['status']: display(create_status_indicator("success", f"{ICONS['success']} Dataset berhasil diunduh ke {data_dir}"))
        
        # Validasi struktur
        if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
            ui_components['validate_dataset_structure'](data_dir)
        
        # Update status panel
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_status_indicator(
                "success", f"{ICONS['success']} Dataset siap digunakan: {len(dataset_paths) if dataset_paths else 0} splits"
            ).value
            
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
            except ImportError: pass
    except Exception as e:
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        if logger: logger.error(f"{ICONS['error']} Error saat download: {str(e)}")
        if 'status_panel' in ui_components: ui_components['status_panel'].value = create_status_indicator("error", f"{ICONS['error']} Download dataset gagal").value

def get_roboflow_settings(ui_components):
    """
    Ambil pengaturan download Roboflow dengan verifikasi ketersediaan komponen
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi pengaturan Roboflow atau None jika gagal
    """
    # Periksa keberadaan komponen terlebih dahulu
    if 'roboflow_settings' not in ui_components:
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Konfigurasi Roboflow tidak ditemukan"))
        return None
    
    roboflow_settings = ui_components['roboflow_settings']
    if not hasattr(roboflow_settings, 'children') or len(roboflow_settings.children) < 4:
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Konfigurasi Roboflow tidak lengkap"))
        return None
    
    # Get settings dengan aman
    api_settings = roboflow_settings.children
    api_key = api_settings[0].value if len(api_settings) > 0 else ""
    workspace = api_settings[1].value if len(api_settings) > 1 else "smartcash-wo2us"
    project = api_settings[2].value if len(api_settings) > 2 else "rupiah-emisi-2022"
    version = api_settings[3].value if len(api_settings) > 3 else "3"
    
    # Try to get API key from Google Secret if not provided
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get('ROBOFLOW_API_KEY')
            if api_key:
                if len(api_settings) > 0:
                    api_settings[0].value = api_key
            else:
                with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} API Key Roboflow tidak tersedia"))
                return None
        except Exception:
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} API Key Roboflow tidak tersedia"))
            return None
    
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
    logger = ui_components.get('logger')
    # Default data directory jika config tidak tersedia
    data_dir = "data" 
    
    # Coba ambil dari config jika tersedia
    if config and isinstance(config, dict) and 'data' in config:
        data_dir = config.get('data', {}).get('dir', 'data')
    
    # Gunakan Google Drive jika tersedia
    if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
        data_dir = str(env.drive_path / 'data')
        if logger: logger.info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset")
    
    # Pastikan direktori ada
    try: Path(data_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e: 
        with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Error membuat direktori: {str(e)}"))
        return None
    return data_dir