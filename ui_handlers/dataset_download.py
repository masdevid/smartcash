"""
File: smartcash/ui_handlers/dataset_download.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI download dataset SmartCash dengan implementasi ObserverManager
           dan perbaikan untuk upload file lokal.
"""

import os
import time
import tempfile
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

def create_status_indicator(status, message):
    """Buat indikator status dengan styling konsisten."""
    status_styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = status_styles.get(status, status_styles['info'])
    
    return HTML(f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """)

def setup_download_handlers(ui_components, config=None):
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
    
    # Coba import DatasetManager dan utilitas
    dataset_manager = None
    logger = None
    observer_manager = None
    try:
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.logger import get_logger
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        
        logger = get_logger("dataset_download")
        dataset_manager = DatasetManager(config, logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        
    except ImportError as e:
        print(f"Info: {str(e)}")
    
    # Tambahkan API key info widget
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
    
    # Setup observer untuk download progress
    download_observers_group = "download_observers"
    if observer_manager:
        # Callback untuk update progress
        def update_progress_callback(event_type, sender, progress=0, total=100, message=None, **kwargs):
            ui_components['download_progress'].value = int(progress * 100 / total) if total > 0 else 0
            ui_components['download_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
            if message:
                with ui_components['download_status']:
                    display(create_status_indicator("info", message))
        
        # Register observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.DOWNLOAD_PROGRESS,
            callback=update_progress_callback,
            name="DownloadProgressObserver",
            group=download_observers_group
        )
    
    # Handler untuk download dataset
    def on_download_click(b):
        # Disable tombol download saat sedang berjalan
        ui_components['download_button'].disabled = True
        ui_components['download_progress'].layout.visibility = 'visible'
        
        try:
            with ui_components['download_status']:
                clear_output()
                display(create_status_indicator("info", "🔄 Memulai download dataset..."))
                ui_components['download_progress'].value = 0
                
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
                            if api_key:
                                api_settings[0].value = api_key
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
                        
                        # Use EnvironmentManager to detect Colab and handle Drive
                        try:
                            from smartcash.utils.environment_manager import EnvironmentManager
                            env_manager = EnvironmentManager()
                            
                            # Check if in Colab and mount Drive if needed
                            if env_manager.is_colab and not env_manager.is_drive_mounted:
                                display(create_status_indicator("info", "📁 Menghubungkan ke Google Drive..."))
                                env_manager.mount_drive()
                            
                            # Use environment-appropriate path
                            if env_manager.is_drive_mounted:
                                data_dir = str(env_manager.drive_path / 'data')
                                display(create_status_indicator("info", "📁 Menggunakan Google Drive untuk penyimpanan"))
                            else:
                                data_dir = config.get('data_dir', 'data')
                        except Exception as e:
                            # Fallback to config data_dir if EnvironmentManager fails
                            data_dir = config.get('data_dir', 'data')
                            if logger:
                                logger.warning(f"⚠️ Gagal menggunakan EnvironmentManager: {str(e)}")
                        
                        # Ensure data directory exists
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                        config['data_dir'] = data_dir
                        
                        # Download dataset
                        try:
                            # Update with new config
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
                            
                            # Verify dataset structure
                            valid_structure = True
                            for split in ['train', 'valid', 'test']:
                                split_dir = Path(data_dir) / split
                                if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                                    valid_structure = False
                                    break
                            
                            # Verifikasi jumlah file setiap split
                            file_stats = {}
                            for split in ['train', 'valid', 'test']:
                                split_dir = Path(data_dir) / split
                                img_count = sum(1 for _ in (split_dir / 'images').glob('*.*')) if (split_dir / 'images').exists() else 0
                                label_count = sum(1 for _ in (split_dir / 'labels').glob('*.txt')) if (split_dir / 'labels').exists() else 0
                                file_stats[split] = (img_count, label_count)
                                
                            # Log hasil verifikasi
                            display(create_status_indicator("info", 
                                f"📊 Statistik dataset: Train {file_stats['train'][0]} gambar, "
                                f"Valid {file_stats['valid'][0]} gambar, Test {file_stats['test'][0]} gambar"))
                            
                            if valid_structure and all(stats[0] > 0 and stats[1] > 0 for stats in file_stats.values()):
                                display(create_status_indicator("success", 
                                    "✅ Struktur dataset valid dan siap digunakan"))
                            else:
                                display(create_status_indicator("warning", 
                                    "⚠️ Struktur dataset belum lengkap, mungkin perlu validasi"))
                                
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
                    
                    # Buat target directory jika belum ada
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                    
                    # Process upload
                    if dataset_manager:
                        display(create_status_indicator("info", "📤 Memproses file upload..."))
                        ui_components['download_progress'].value = 20
                        
                        try:
                            # Info file pertama yang diupload
                            file_key = next(iter(upload_widget.value))
                            file_info = upload_widget.value[file_key]
                            file_name = file_info.get('metadata', {}).get('name', 'unknown.zip')
                            file_size = file_info.get('metadata', {}).get('size', 0)
                            file_content = file_info.get('content', b'')
                            
                            # Pastikan file adalah ZIP
                            if not file_name.lower().endswith('.zip'):
                                display(create_status_indicator("error", 
                                    "❌ File harus berformat ZIP (.zip)"))
                                return
                            
                            # Buat temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                                temp_path = temp_file.name
                                temp_file.write(file_content)
                                
                            ui_components['download_progress'].value = 50
                            display(create_status_indicator("info", 
                                f"📦 File {file_name} ({file_size/1024/1024:.2f} MB) berhasil diunggah"))
                            
                            # Import dataset from zip
                            display(create_status_indicator("info", "📂 Ekstraksi dan proses file..."))
                            ui_components['download_progress'].value = 75
                            
                            from smartcash.handlers.dataset.core.dataset_downloader import DatasetDownloader
                            downloader = DatasetDownloader(config, logger=logger)
                            
                            # Import dataset
                            imported_dir = downloader.import_from_zip(
                                zip_path=temp_path,
                                target_dir=target_dir
                            )
                            
                            # Update config data_dir
                            config['data_dir'] = target_dir
                            if dataset_manager:
                                dataset_manager.config = config
                            
                            # Verifikasi hasil import
                            valid_structure = True
                            file_stats = {}
                            
                            for split in ['train', 'valid', 'test']:
                                split_dir = Path(target_dir) / split
                                if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                                    valid_structure = False
                                img_count = sum(1 for _ in (split_dir / 'images').glob('*.*')) if (split_dir / 'images').exists() else 0
                                label_count = sum(1 for _ in (split_dir / 'labels').glob('*.txt')) if (split_dir / 'labels').exists() else 0
                                file_stats[split] = (img_count, label_count)
                                
                            # Hapus temporary file
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                                
                            ui_components['download_progress'].value = 100
                            display(create_status_indicator("success", 
                                f"✅ Dataset berhasil diimport ke {target_dir}"))
                                
                            # Log hasil verifikasi
                            display(create_status_indicator("info", 
                                f"📊 Statistik dataset: Train {file_stats['train'][0]} gambar, "
                                f"Valid {file_stats['valid'][0]} gambar, Test {file_stats['test'][0]} gambar"))
                            
                            if valid_structure and all(stats[0] > 0 for split, stats in file_stats.items() if split == 'train'):
                                display(create_status_indicator("success", 
                                    "✅ Dataset valid dan siap digunakan"))
                            else:
                                display(create_status_indicator("warning", 
                                    "⚠️ Struktur dataset belum lengkap, mungkin perlu validasi"))
                            
                        except Exception as e:
                            display(create_status_indicator("error", f"❌ Error: {str(e)}"))
                    else:
                        display(create_status_indicator("error", "❌ DatasetManager tidak tersedia"))
        
        except Exception as e:
            with ui_components['download_status']:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
        
        finally:
            # Enable kembali tombol download
            ui_components['download_button'].disabled = False
    
    # Register handlers
    ui_components['download_button'].on_click(on_download_click)
    
    # Update UI dengan nilai config
    if config and not has_api_key:
        api_settings = ui_components['roboflow_settings'].children
        api_settings[1].value = config['data']['roboflow'].get('workspace', 'smartcash-wo2us')
        api_settings[2].value = config['data']['roboflow'].get('project', 'rupiah-emisi-2022')
        api_settings[3].value = str(config['data']['roboflow'].get('version', '3'))
    
    # Cleanup function
    def cleanup():
        if observer_manager:
            observer_manager.unregister_group(download_observers_group)
    
    ui_components['cleanup'] = cleanup
    
    # Add API key info to Roboflow settings container
    ui_components['download_settings_container'] = ui_components.get('download_settings_container')
    if ui_components.get('download_settings_container') is None:
        from ipywidgets import VBox
        ui_components['download_settings_container'] = VBox([ui_components['roboflow_settings'], api_key_info])
        
        # Update downloads option handler
        def on_download_option_change(change):
            if change['new'] == 'Roboflow (Online)':
                ui_components['download_settings_container'].children = [ui_components['roboflow_settings'], api_key_info]
            elif change['new'] == 'Local Data (Upload)':
                ui_components['download_settings_container'].children = [ui_components['local_upload']]
            else:
                ui_components['download_settings_container'].children = [ui_components['sample_data']]
        
        ui_components['download_options'].observe(on_download_option_change, names='value')
    
    return ui_components