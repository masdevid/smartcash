"""
File: smartcash/ui/dataset/roboflow_download_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow dengan integrasi utils standar dan dukungan backup opsional
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output

from smartcash.common.exceptions import DatasetError

def download_from_roboflow(
    ui_components: Dict[str, Any],
    env=None,
    config=None
) -> Dict[str, Any]:
    """
    Download dataset dari Roboflow menggunakan DatasetManager dan utils standar.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi informasi hasil download
        
    Raises:
        DatasetError: Jika terjadi error saat download
    """
    # Import komponen UI standar
    from smartcash.ui.utils.constants import ICONS
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.utils.fallback_utils import update_status_panel
    from smartcash.ui.handlers.error_handler import handle_ui_error
    
    status_widget = ui_components.get('status')
    logger = ui_components.get('logger')
    
    try:
        # Ekstrak parameter dari UI components dengan validasi standar
        if 'roboflow_settings' not in ui_components or not hasattr(ui_components['roboflow_settings'], 'children'):
            raise DatasetError("Komponen roboflow_settings tidak ditemukan")
            
        settings = ui_components['roboflow_settings'].children
        
        # Memastikan indeks valid dengan utils standar
        if len(settings) < 4:
            raise DatasetError("Komponen roboflow_settings tidak lengkap")
            
        # Ekstrak nilai dari komponen UI dengan validasi standar
        api_key = settings[0].value if len(settings) > 0 else ""
        workspace = settings[1].value if len(settings) > 1 else "smartcash-wo2us"
        project = settings[2].value if len(settings) > 2 else "rupiah-emisi-2022"
        version = settings[3].value if len(settings) > 3 else "3"
        format = "yolov5pytorch"  # Default format
        
        # Cek apakah backup diaktifkan dengan standar
        backup_enabled = False
        if len(settings) > 4 and hasattr(settings[4], 'value'):
            backup_enabled = settings[4].value
        
        # Cek API key dari Google Secret dengan utils standar
        if not api_key:
            from smartcash.ui.utils.fallback_utils import import_with_fallback
            userdata_module = import_with_fallback('google.colab.userdata')
            
            if userdata_module:
                api_key = userdata_module.get('ROBOFLOW_API_KEY')
                
            if not api_key:
                raise DatasetError("API Key Roboflow diperlukan. Isi field API Key atau tambahkan sebagai Google Secret.")
        
        # Tampilkan status loading dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS['download']} Memulai download dataset dari Roboflow...\nWorkspace: {workspace}, Project: {project}, Version: {version}\nBackup: {'Aktif' if backup_enabled else 'Tidak aktif'}",
                    "info"
                ))
        
        # Update status panel dengan utils standar 
        update_status_panel(ui_components, "info", f"{ICONS['download']} Memulai download dari Roboflow ({project} v{version})")
        
        # Update progress dengan progress handler atau langsung
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(10, 100, "Inisialisasi download...")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 10
                ui_components['progress_bar'].description = 'Download: 10%'
        
        # Dapatkan dataset_manager dengan utils standar
        from smartcash.ui.utils.fallback_utils import get_dataset_manager
        
        dataset_manager = ui_components.get('dataset_manager') or get_dataset_manager(config, logger)
        
        if not dataset_manager:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config=config)
            ui_components['dataset_manager'] = dataset_manager
        
        # Update progress dengan progress handler atau langsung
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(30, 100, "Mempersiapkan download...")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 30
                ui_components['progress_bar'].description = 'Download: 30%'
        
        # Mendapatkan download_service dari dataset_manager
        try:
            download_service = dataset_manager.get_service('downloader')
        except Exception as e:
            # Fallback dengan utils standar jika gagal mendapatkan service
            from smartcash.dataset.services.downloader.download_service import DownloadService
            
            # Updated code
            output_dir = ui_components.get('data_dir', 'data/')
            if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
                output_dir = str(env.drive_path / 'data')
            
            download_service = DownloadService(
                output_dir=output_dir,
                config=config,
                logger=logger
            )
        
        # Update progress dengan progress handler atau langsung
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(40, 100, "Mendownload dataset...")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 40
                ui_components['progress_bar'].description = 'Download: 40%'
            
        # Notifikasi observer dengan observer standar
        try:
            from smartcash.components.observer import notify
            notify(
                event_type="DOWNLOAD_PROGRESS",
                sender="download_handler",
                message=f"Mendownload dataset {project} (v{version})",
                progress=40,
                total=100
            )
        except ImportError:
            pass
            
        # Download menggunakan download_service dengan parameter backup_existing
        result = download_service.download_from_roboflow(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            format=format,
            backup_existing=backup_enabled
        )
        
        # Export ke folder data standar
        output_dir = ui_components.get('dir', 'data/')
        
        # Update progress dengan utils standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(80, 100, "Mengekspor dataset...")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 80
                ui_components['progress_bar'].description = 'Download: 80%'
            
        # Notifikasi observer dengan observer standar
        try:
            from smartcash.components.observer import notify
            notify(
                event_type="EXPORT_START",
                sender="download_handler",
                message=f"Mengekspor dataset {project} (v{version}) ke struktur lokal",
                progress=80,
                total=100
            )
        except ImportError:
            pass
            
        # Export ke struktur lokal standar (dengan backup optional)
        export_result = download_service.export_to_local(
            source_dir=result['output_dir'],
            output_dir=output_dir,
            backup_existing=backup_enabled
        )
        
        # Update progress dengan utils standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(100, 100, "Download selesai!")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 100
                ui_components['progress_bar'].description = 'Download: 100%'
            
        # Tampilkan hasil sukses dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS['success']} Dataset berhasil didownload!\nProject: {project} (v{version}) dari workspace {workspace}\nFormat: {format}\nOutput: {output_dir}\nFiles: {export_result.get('copied', 0)} file disalin",
                    "success"
                ))
        
        # Update status panel dengan utils standar
        update_status_panel(ui_components, "success", f"{ICONS['success']} Dataset berhasil didownload ke {output_dir}")
            
        # Notifikasi selesai dengan observer standar
        try:
            from smartcash.components.observer import notify
            notify(
                event_type="DOWNLOAD_COMPLETE",
                sender="download_handler",
                message=f"Download dataset dari Roboflow selesai: {project} (v{version})"
            )
        except ImportError:
            pass
        
        return {
            'status': 'success',
            'download_result': result,
            'export_result': export_result,
            'output_dir': output_dir
        }
        
    except DatasetError as e:
        # Dataset manager error dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(f"{ICONS['error']} {str(e)}", "error"))
        
        # Update status panel dengan utils standar
        update_status_panel(ui_components, "error", f"{ICONS['error']} Error: {str(e)}")
        
        # Reset progress dengan utils standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].reset_progress_bar()
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 0
                
        # Notifikasi error dengan observer standar
        try:
            from smartcash.components.observer import notify
            notify(
                event_type="DOWNLOAD_ERROR",
                sender="download_handler",
                message=f"Error download dataset: {str(e)}"
            )
        except ImportError:
            pass
            
        raise
        
    except Exception as e:
        # Tangani exception lain dengan utils standar
        error_message = f"Error saat download dataset: {str(e)}"
        
        # Handle error dengan utils standar
        handle_ui_error(e, status_widget, True, error_message)
        
        # Update status panel dengan utils standar
        update_status_panel(ui_components, "error", f"{ICONS['error']} Error: {error_message}")
        
        # Notifikasi error dengan observer standar
        try:
            from smartcash.components.observer import notify
            notify(
                event_type="DOWNLOAD_ERROR",
                sender="download_handler",
                message=error_message
            )
        except ImportError:
            pass
            
        raise DatasetError(error_message)