"""
File: smartcash/ui_handlers/dataset_download.py
Author: Refactored
Deskripsi: Handler untuk UI download dataset SmartCash dengan implementasi ObserverManager.
"""

import threading
from pathlib import Path
from IPython.display import display, clear_output

from smartcash.utils.ui_utils import create_status_indicator, update_output_area

def setup_dataset_download_handlers(ui_components, config=None):
    """Setup handlers untuk UI download dataset."""
    if config is None:
        config = {
            'data': {
                'source': 'roboflow',
                'roboflow': {'api_key': '', 'workspace': 'smartcash-wo2us', 
                            'project': 'rupiah-emisi-2022', 'version': '3'},
            },
            'data_dir': 'data'
        }
    
    # Coba import dependencies
    dataset_manager = None
    logger = None
    observer_manager = None
    env_manager = None
    
    try:
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.logger import get_logger
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        from smartcash.utils.environment_manager import EnvironmentManager
        
        logger = get_logger("dataset_download")
        observer_manager = ObserverManager(auto_register=True)
        env_manager = EnvironmentManager(logger=logger)
        dataset_manager = DatasetManager(config, logger=logger)
        
        # Simpan ke ui_components
        ui_components['dataset_manager'] = dataset_manager
        ui_components['observer_manager'] = observer_manager
        
    except ImportError as e:
        if 'env' in ui_components and 'logger' in ui_components['env']:
            ui_components['env']['logger'].warning(f"⚠️ Beberapa dependencies tidak tersedia: {str(e)}")
    
    # Setup observer untuk download progress
    download_observers_group = "download_observers"
    
    if observer_manager:
        # Callback untuk update progress
        def update_progress_callback(event_type, sender, progress=0, total=100, message=None, **kwargs):
            ui_components['download_progress'].value = int(progress * 100 / total) if total > 0 else 0
            ui_components['download_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
            if message:
                update_output_area(ui_components['download_status'], message, status="info")
        
        # Register observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.DOWNLOAD_PROGRESS,
            callback=update_progress_callback,
            name="DownloadProgressObserver",
            group=download_observers_group
        )
        
        # Observer untuk notifikasi download start/end
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.DOWNLOAD_START,
                EventTopics.DOWNLOAD_END,
                EventTopics.DOWNLOAD_ERROR,
                EventTopics.DOWNLOAD_COMPLETE
            ],
            log_level="info",
            name="DownloadLoggerObserver",
            group=download_observers_group
        )
    
    # Cek dan tambahkan API key dari Google Colab Secret
    try:
        from google.colab import userdata
        roboflow_api_key = userdata.get('ROBOFLOW_API_KEY')
        if roboflow_api_key:
            ui_components['roboflow_settings'].children[0].value = roboflow_api_key
    except:
        pass
    
    # Handler untuk source selection
    def update_download_options(change):
        if change['new'] == 'Roboflow (Online)':
            # Create API key info
            api_key_info = create_status_indicator("info", "ℹ️ API Key diperlukan untuk download dari Roboflow")
            ui_components['download_settings_container'].children = [ui_components['roboflow_settings'], api_key_info]
        else:  # Local Data (Upload)
            ui_components['download_settings_container'].children = [ui_components['local_upload']]
    
    ui_components['download_options'].observe(update_download_options, names='value')
    
    # Trigger initial update untuk tampilan awal
    update_download_options({'new': ui_components['download_options'].value})
    
    # Handler untuk download dataset
    def on_download_click(b):
        with ui_components['download_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai download dataset..."))
            ui_components['download_progress'].value = 0
            
            try:
                # Ambil opsi download yang dipilih
                download_option = ui_components['download_options'].value
                
                # Notifikasi awal download
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.DOWNLOAD_START,
                        sender="download_handler",
                        message=f"Memulai download dataset dari {download_option}"
                    )
                
                if download_option == 'Roboflow (Online)':
                    # Ambil settings
                    api_settings = ui_components['roboflow_settings'].children
                    api_key = api_settings[0].value
                    workspace = api_settings[1].value
                    project = api_settings[2].value
                    version = api_settings[3].value
                    
                    # Coba ambil API key dari secrets jika tidak ada
                    if not api_key:
                        try:
                            from google.colab import userdata
                            api_key = userdata.get('ROBOFLOW_API_KEY')
                        except:
                            pass
                            
                        if not api_key:
                            display(create_status_indicator("error", 
                                "❌ API Key Roboflow tidak tersedia"))
                            return
                    
                    # Update config
                    config['data']['roboflow']['api_key'] = api_key
                    config['data']['roboflow']['workspace'] = workspace
                    config['data']['roboflow']['project'] = project
                    config['data']['roboflow']['version'] = version
                    
                    # Download dengan DatasetManager
                    if dataset_manager:
                        display(create_status_indicator("info", "🔑 Mengunduh dataset dari Roboflow..."))
                        
                        # Gunakan env_manager untuk path
                        data_dir = 'data'
                        if env_manager:
                            if env_manager.is_colab and not env_manager.is_drive_mounted:
                                display(create_status_indicator("info", "📁 Menghubungkan ke Google Drive..."))
                                env_manager.mount_drive()
                            
                            if env_manager.is_drive_mounted:
                                data_dir = str(env_manager.drive_path / 'data')
                                display(create_status_indicator("info", "📁 Menggunakan Google Drive untuk penyimpanan"))
                        
                        # Pastikan direktori ada
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                        config['data_dir'] = data_dir
                        
                        # Download dataset
                        try:
                            # Update dengan config baru
                            dataset_manager.config = config
                            dataset_paths = dataset_manager.pull_dataset(
                                format="yolov5pytorch",
                                api_key=api_key,
                                workspace=workspace,
                                project=project,
                                version=version,
                                show_progress=True
                            )
                            
                            display(create_status_indicator("success", 
                                f"✅ Dataset berhasil diunduh ke {data_dir}"))
                            
                            # Notify success
                            if observer_manager:
                                EventDispatcher.notify(
                                    event_type=EventTopics.DOWNLOAD_END,
                                    sender="download_handler",
                                    message="Download dataset berhasil",
                                    dataset_path=data_dir
                                )
                        except Exception as e:
                            display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                            if observer_manager:
                                EventDispatcher.notify(
                                    event_type=EventTopics.DOWNLOAD_ERROR,
                                    sender="download_handler",
                                    message=f"Error: {str(e)}"
                                )
                    else:
                        display(create_status_indicator("error", "❌ DatasetManager tidak tersedia"))
                
                elif download_option == 'Local Data (Upload)':
                    # Ambil file yang diupload
                    upload_widget = ui_components['local_upload'].children[0]
                    target_dir = ui_components['local_upload'].children[1].value
                    
                    if not upload_widget.value:
                        display(create_status_indicator("warning", "⚠️ Silahkan pilih file ZIP untuk diupload"))
                        return
                    
                    # Process upload
                    if dataset_manager:
                        display(create_status_indicator("info", "📤 Memproses file upload..."))
                        ui_components['download_progress'].value = 50
                        
                        try:
                            # Info file
                            file_info = next(iter(upload_widget.value.values()))
                            file_name = file_info.get('metadata', {}).get('name', 'unknown.zip')
                            file_size = file_info.get('metadata', {}).get('size', 0)
                            file_content = file_info.get('content', b'')
                            
                            # Save dan proses file
                            import os
                            os.makedirs(os.path.dirname(os.path.join(target_dir, file_name)), exist_ok=True)
                            temp_zip_path = os.path.join(target_dir, file_name)
                            
                            with open(temp_zip_path, 'wb') as f:
                                f.write(file_content)
                            
                            # Import dataset from zip
                            display(create_status_indicator("info", "📂 Ekstraksi file..."))
                            ui_components['download_progress'].value = 75
                            
                            imported_dir = dataset_manager.import_from_zip(
                                zip_path=temp_zip_path,
                                target_dir=target_dir,
                                format="yolov5pytorch"
                            )
                            
                            ui_components['download_progress'].value = 100
                            display(create_status_indicator("success", 
                                f"✅ File berhasil diproses: {file_name} ({file_size/1024:.1f} KB)"))
                            
                        except Exception as e:
                            display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                    else:
                        display(create_status_indicator("error", "❌ DatasetManager tidak tersedia"))
            
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Register handlers
    ui_components['download_button'].on_click(on_download_click)
    
    # Cleanup function
    def cleanup():
        if observer_manager:
            try:
                observer_manager.unregister_group(download_observers_group)
                
                if dataset_manager and hasattr(dataset_manager, 'unregister_observers'):
                    dataset_manager.unregister_observers("dataset_download")
                    
                    if hasattr(dataset_manager, 'loading_facade') and hasattr(dataset_manager.loading_facade, 'downloader'):
                        downloader = dataset_manager.loading_facade.downloader
                        if hasattr(downloader, 'unregister_observers'):
                            downloader.unregister_observers()
                
                if logger:
                    logger.info("✅ Observer untuk dataset download telah dibersihkan")
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat membersihkan observer: {str(e)}")
    
    ui_components['cleanup'] = cleanup
    
    # Update UI dari config
    if config and 'data' in config and 'roboflow' in config['data']:
        ui_components['roboflow_settings'].children[1].value = config['data']['roboflow'].get('workspace', 'smartcash-wo2us')
        ui_components['roboflow_settings'].children[2].value = config['data']['roboflow'].get('project', 'rupiah-emisi-2022')
        ui_components['roboflow_settings'].children[3].value = str(config['data']['roboflow'].get('version', '3'))
    
    return ui_components