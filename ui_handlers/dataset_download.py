"""
File: smartcash/ui_handlers/dataset_download.py
Author: Refactored for optimization
Deskripsi: Handler untuk UI download dataset SmartCash dengan implementasi optimized.
"""

from IPython.display import display, clear_output, HTML
from pathlib import Path
import os
from ipywidgets import widgets
def setup_dataset_download_handlers(ui_components, config):
    """Setup handlers untuk UI download dataset."""
    if not config or not isinstance(config, dict) or 'data' not in config:
        raise ValueError("Config tidak valid! Pastikan config berisi konfigurasi data.")
    
    # Import dependencies
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        from smartcash.utils.ui_utils import create_status_indicator
        from smartcash.utils.environment_manager import EnvironmentManager
        
        logger = get_logger("dataset_download")
        dataset_manager = DatasetManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        env_manager = EnvironmentManager(logger=logger)
        
        # Store for cleanup
        ui_components['dataset_manager'] = dataset_manager
        ui_components['observer_manager'] = observer_manager
        
    except ImportError as e:
        # Fallback for missing dependencies
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
            
        logger = None
        dataset_manager = None
        observer_manager = None
        env_manager = None
        
        with ui_components['download_status']:
            display(create_status_indicator("warning", f"⚠️ Limited functionality available: {str(e)}"))
    
    # Check for API key from Google Colab Secret
    api_key_info = widgets.HTML(
        """<div style="padding: 10px; border-left: 4px solid #856404; 
                     color: #856404; margin: 5px 0; border-radius: 4px; background-color: #fff3cd">
                <p><i>⚠️ API Key diperlukan untuk download dari Roboflow</i></p>
            </div>"""
    )
    
    # Cek dan tambahkan API key dari Google Colab Secret
    has_api_key = False
    try:
        from google.colab import userdata
        roboflow_api_key = userdata.get('ROBOFLOW_API_KEY')
        if roboflow_api_key:
            ui_components['roboflow_settings'].children[0].value = roboflow_api_key
            has_api_key = True
            api_key_info = widgets.HTML(
                """<div style="padding: 10px; border-left: 4px solid #0c5460; 
                     color: #0c5460; margin: 5px 0; border-radius: 4px; background-color: #d1ecf1">
                    <p><i>ℹ️ API Key Roboflow tersedia dari Google Secret.</i></p>
                </div>"""
            )
    except:
        pass
    
    # Setup observer for progress updates
    observer_group = "download_observers"
    if observer_manager:
        observer_manager.unregister_group(observer_group)
        
        # Progress callback
        observer_manager.create_simple_observer(
            event_type=EventTopics.DOWNLOAD_PROGRESS,
            callback=lambda _, __, progress=0, total=100, message=None, **kwargs: 
                update_progress(ui_components, progress, total, message),
            name="DownloadProgressObserver",
            group=observer_group
        )
        
        # Event observer
        observer_manager.create_logging_observer(
            event_types=[EventTopics.DOWNLOAD_START, EventTopics.DOWNLOAD_END, 
                        EventTopics.DOWNLOAD_ERROR],
            log_level="info",
            name="DownloadLoggerObserver",
            group=observer_group
        )
    
    # Progress update helper
    def update_progress(ui, progress, total, message=None):
        if 'download_progress' in ui:
            ui['download_progress'].value = int(progress * 100 / total) if total > 0 else 0
            ui['download_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
            
        if message and 'download_status' in ui:
            with ui['download_status']:
                display(create_status_indicator("info", message))
    
    # Handle download button click
    def on_download_click(b):
        with ui_components['download_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai download dataset..."))
            ui_components['download_progress'].value = 0
            
            try:
                download_option = ui_components['download_options'].value
                
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.DOWNLOAD_START,
                        sender="download_handler",
                        message=f"Memulai download dataset dari {download_option}"
                    )
                
                if download_option == 'Roboflow (Online)':
                    download_from_roboflow()
                elif download_option == 'Local Data (Upload)':
                    process_local_upload()
                    
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Download from Roboflow
    def download_from_roboflow():
        if not dataset_manager:
            display(create_status_indicator("error", "❌ DatasetManager tidak tersedia"))
            return
            
        # Get settings
        api_settings = ui_components['roboflow_settings'].children
        api_key = api_settings[0].value
        workspace = api_settings[1].value
        project = api_settings[2].value
        version = api_settings[3].value
        
        # Try to get API key from secrets if not provided
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get('ROBOFLOW_API_KEY')
                if api_key:
                    api_settings[0].value = api_key
            except:
                pass
                
            if not api_key:
                display(create_status_indicator("error", "❌ API Key Roboflow tidak tersedia"))
                return
        
        # Update config
        config['data']['roboflow'].update({
            'api_key': api_key,
            'workspace': workspace,
            'project': project,
            'version': version
        })
        
        # Setup data directory
        data_dir = config.get('data_dir', 'data')
        if env_manager:
            if env_manager.is_colab and not env_manager.is_drive_mounted:
                display(create_status_indicator("info", "📁 Menghubungkan ke Google Drive..."))
                env_manager.mount_drive()
                
            if env_manager.is_drive_mounted:
                data_dir = str(env_manager.drive_path / 'data')
                display(create_status_indicator("info", "📁 Menggunakan Google Drive untuk penyimpanan"))
        
        # Ensure directory exists
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        config['data_dir'] = data_dir
        
        # Download dataset
        display(create_status_indicator("info", "🔑 Mengunduh dataset dari Roboflow..."))
        try:
            dataset_manager.config = config
            dataset_paths = dataset_manager.pull_dataset(
                format="yolov5",
                api_key=api_key,
                workspace=workspace,
                project=project,
                version=version,
                show_progress=True
            )
            
            display(create_status_indicator("success", f"✅ Dataset berhasil diunduh ke {data_dir}"))
            validate_dataset_structure(data_dir)
            
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
    
    # Process local upload
    def process_local_upload():
        if not dataset_manager:
            display(create_status_indicator("error", "❌ DatasetManager tidak tersedia"))
            return
            
        upload_widget = ui_components['local_upload'].children[0]
        target_dir = ui_components['local_upload'].children[1].value
        
        if not upload_widget.value:
            display(create_status_indicator("warning", "⚠️ Silahkan pilih file ZIP untuk diupload"))
            return
        
        # Extract file info
        file_info = next(iter(upload_widget.value.values()))
        file_name = file_info.get('metadata', {}).get('name', 'unknown.zip')
        file_size = file_info.get('metadata', {}).get('size', 0)
        file_content = file_info.get('content', b'')
        
        # Process upload
        display(create_status_indicator("info", "📤 Memproses file upload..."))
        ui_components['download_progress'].value = 50
        
        try:
            # Save uploaded file
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
                format="yolov5"
            )
            
            ui_components['download_progress'].value = 100
            display(create_status_indicator("success", 
                f"✅ File berhasil diproses: {file_name} ({file_size/1024:.1f} KB)"))
            validate_dataset_structure(target_dir)
            
        except Exception as e:
            display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Verify dataset structure
    def validate_dataset_structure(data_dir):
        splits = ['train', 'valid', 'test']
        valid_structure = True
        
        for split in splits:
            split_dir = Path(data_dir) / split
            if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                valid_structure = False
                break
        
        if valid_structure:
            display(create_status_indicator("success", "✅ Struktur dataset valid dan siap digunakan"))
        else:
            display(create_status_indicator("warning", 
                "⚠️ Struktur dataset belum lengkap, mungkin perlu validasi"))
    
    # Update UI from config
    if 'data' in config and 'roboflow' in config['data']:
        roboflow_config = config['data']['roboflow']
        if 'roboflow_settings' in ui_components:
            api_settings = ui_components['roboflow_settings'].children
            api_settings[1].value = roboflow_config.get('workspace', 'smartcash-wo2us')
            api_settings[2].value = roboflow_config.get('project', 'rupiah-emisi-2022')
            api_settings[3].value = str(roboflow_config.get('version', '3'))
    
    # Handle download option change
    def on_download_option_change(change):
        if change['new'] == 'Roboflow (Online)':
            ui_components['download_settings_container'].children = [ui_components['roboflow_settings'], api_key_info]
        elif change['new'] == 'Local Data (Upload)':
            ui_components['download_settings_container'].children = [ui_components['local_upload']]
    
    # Register event handlers
    ui_components['download_button'].on_click(on_download_click)
    ui_components['download_options'].observe(on_download_option_change, names='value')
    
    # Initial UI setup
    if 'download_settings_container' in ui_components:
        ui_components['download_settings_container'].children = [ui_components['roboflow_settings'], api_key_info]
    
    # Cleanup function
    def cleanup():
        if observer_manager:
            observer_manager.unregister_group(observer_group)
            
            # Clean up other resources
            if dataset_manager and hasattr(dataset_manager, 'unregister_observers'):
                dataset_manager.unregister_observers("dataset_download")
            
            if logger:
                logger.info("✅ Observer untuk dataset download telah dibersihkan")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components