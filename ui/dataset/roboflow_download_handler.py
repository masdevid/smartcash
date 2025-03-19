"""
File: smartcash/ui/dataset/roboflow_download_handler.py (perbaikan Google Secret API key)
Deskripsi: Mengembalikan fitur cek API key dari Google Secret dan sembunyikan field API jika tersedia
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.exceptions import DatasetError

def download_from_roboflow(
    ui_components: Dict[str, Any],
    env=None,
    config=None
) -> Dict[str, Any]:
    """
    Download dataset dari Roboflow menggunakan DatasetManager.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi informasi hasil download
        
    Raises:
        DatasetError: Jika terjadi error saat download
    """
    status_widget = ui_components.get('status')
    logger = ui_components.get('logger')
    
    try:
        # Ekstrak parameter dari UI components
        if 'roboflow_settings' not in ui_components or not hasattr(ui_components['roboflow_settings'], 'children'):
            raise DatasetError("Komponen roboflow_settings tidak ditemukan")
            
        settings = ui_components['roboflow_settings'].children
        
        # Memastikan indeks valid
        if len(settings) < 4:
            raise DatasetError("Komponen roboflow_settings tidak lengkap")
            
        # Ekstrak nilai dari komponen UI
        api_key = settings[0].value
        workspace = settings[1].value
        project = settings[2].value
        version = settings[3].value
        format = "yolov5pytorch"  # Default format
        
        # Cek API key dari Google Secret jika tidak diisi
        secret_api_key = None
        if not api_key:
            try:
                from google.colab import userdata
                secret_api_key = userdata.get('ROBOFLOW_API_KEY')
                if secret_api_key:
                    if logger:
                        logger.info(f"🔑 Menggunakan API key dari Google Secret")
                    
                    # Sembunyikan field API key dan tampilkan pesan bahwa sudah menggunakan secret
                    settings[0].layout.display = 'none'
                    
                    # Tambahkan pesan informasi tentang penggunaan secret
                    if 'status_panel' in ui_components:
                        ui_components['status_panel'].value = f"""
                        <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                                  color:{COLORS['alert_info_text']}; margin:10px 0; border-radius:4px; 
                                  border-left:4px solid {COLORS['alert_info_text']};">
                            <p style="margin:5px 0">{ICONS['info']} Menggunakan API key dari Google Secret. Workspace: {workspace}, Project: {project}</p>
                        </div>
                        """
                    
                    api_key = secret_api_key
                else:
                    raise DatasetError("API Key Roboflow diperlukan. Isi field API Key atau tambahkan sebagai Google Secret.")
            except ImportError:
                # Tidak berjalan di Google Colab
                raise DatasetError("API Key Roboflow diperlukan")
        
        # Tampilkan status loading
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                              color:{COLORS['alert_info_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['download']} Memulai download dataset dari Roboflow...</p>
                        <p style="margin:5px 0">Workspace: {workspace}, Project: {project}, Version: {version}</p>
                    </div>
                """))
        
        # Update progress bar jika ada
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 10
        
        # Dapatkan dataset_manager, prioritaskan yang sudah ada
        dataset_manager = ui_components.get('dataset_manager')
        if not dataset_manager:
            from smartcash.dataset.manager import DatasetManager
            # Gunakan config dari parameter jika ada
            dataset_manager = DatasetManager(config=config)
            ui_components['dataset_manager'] = dataset_manager
            
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 30
        
        # Tentukan output_dir
        output_dir = ui_components.get('data_dir', 'data/')
        
        # Dapatkan downloader service
        downloader = dataset_manager.get_service('downloader')
        
        # Download dataset
        result = downloader.download_dataset(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            format=format,
            output_dir=output_dir
        )
        
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 90
        
        # Tampilkan hasil sukses
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                              color:{COLORS['alert_success_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['success']} Dataset berhasil didownload!</p>
                        <p>Project: {project} (v{version}) dari workspace {workspace}</p>
                        <p>Format: {format}</p>
                        <p>Output: {output_dir or 'data/'}</p>
                    </div>
                """))
        
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
            
        # Notifikasi selesai jika ada observer
        if 'observer_manager' in ui_components:
            try:
                from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
                EventDispatcher.notify(
                    event_type="DOWNLOAD_COMPLETE",
                    sender="download_handler",
                    message=f"Download dataset dari Roboflow selesai: {project} (v{version})"
                )
            except ImportError:
                pass
        
        return {"status": "success", "result": result}
        
    except DatasetError as e:
        # Dataset manager sudah menangani banyak exceptions dengan DatasetError
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['error']} {str(e)}</p>
                    </div>
                """))
        raise
        
    except Exception as e:
        # Tangani exception lain
        error_message = f"Error saat download dataset: {str(e)}"
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['error']} {error_message}</p>
                    </div>
                """))
        raise DatasetError(error_message)