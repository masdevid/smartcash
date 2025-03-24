"""
File: smartcash/ui/dataset/local_upload_handler.py
Deskripsi: Handler untuk upload dataset lokal dengan integrasi utils standar dan dukungan backup opsional
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import tempfile

from smartcash.common.exceptions import DatasetError

def process_local_upload(
    ui_components: Dict[str, Any],
    env=None,
    config=None
) -> Dict[str, Any]:
    """
    Proses upload dataset lokal menggunakan DatasetManager dan DownloadService.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi informasi hasil proses upload
        
    Raises:
        DatasetError: Jika terjadi error saat proses file
    """
    # Gunakan utils standar untuk UI components
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Gunakan utils standar untuk status dan error handling
    from smartcash.ui.handlers.error_handler import handle_ui_error
    from smartcash.ui.dataset.download_initialization import update_status_panel
    
    status_widget = ui_components.get('status')
    logger = ui_components.get('logger')
    
    # Validasi: mendapatkan file upload widget dari local_upload
    local_upload = ui_components.get('local_upload')
    if not local_upload or not hasattr(local_upload, 'children') or len(local_upload.children) < 1:
        raise DatasetError("Komponen local_upload tidak ditemukan atau tidak valid")
        
    file_upload = local_upload.children[0]  # FileUpload widget seharusnya di indeks 0
    
    if not file_upload or not hasattr(file_upload, 'value') or not file_upload.value:
        raise DatasetError("Tidak ada file yang dipilih")
        
    try:
        # Cek apakah backup diaktifkan dengan utils standar
        backup_enabled = False
        if len(local_upload.children) > 2 and hasattr(local_upload.children[2], 'value'):
            backup_enabled = local_upload.children[2].value
        
        # Tampilkan status loading dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS['upload']} Memproses file dataset...\nBackup: {'Aktif' if backup_enabled else 'Tidak aktif'}",
                    "info"
                ))
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['upload']} Memproses file dataset...")
        
        # Dapatkan dataset_manager menggunakan utils standar
        from smartcash.ui.utils.fallback_utils import get_dataset_manager
        
        # Gunakan dataset_manager yang sudah ada atau buat baru
        dataset_manager = ui_components.get('dataset_manager') or get_dataset_manager(config, logger)
        if not dataset_manager:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config=config)
            ui_components['dataset_manager'] = dataset_manager
        
        # Notify progress dengan utils standar
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="UPLOAD_START",
                sender="upload_handler",
                message=f"Memulai proses upload dataset lokal"
            )
        except ImportError:
            pass
        
        # Dapatkan download_service langsung dari manager
        download_service = dataset_manager.get_service('downloader')
        
        # Simpan file sementara dan proses menggunakan utils standar
        from smartcash.ui.utils.file_utils import save_uploaded_file
        
        temp_dir = tempfile.mkdtemp()
        
        # Dapatkan file pertama dari file_upload
        file_key = list(file_upload.value.keys())[0]
        file_content = file_upload.value[file_key]['content']
        file_name = file_upload.value[file_key]['metadata']['name']
        
        # Simpan file dengan utils standar
        file_saved, file_path = save_uploaded_file(file_content, file_name, temp_dir, True)
        if not file_saved:
            raise DatasetError(f"Gagal menyimpan file upload: {file_path}")
        
        # Dapatkan output_dir dengan utils standar
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        output_dir = ui_components.get('data_dir', 'data/')
        
        # Jika ada di local_upload child ke-1 (target_dir), gunakan itu
        if len(local_upload.children) > 1 and hasattr(local_upload.children[1], 'value'):
            target_dir_input = local_upload.children[1]
            if target_dir_input.value:
                output_dir = target_dir_input.value
        
        # Update progress dengan utils standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(50, 100)
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 50
            
            if 'progress_label' in ui_components:
                ui_components['progress_label'].value = "Memproses upload dataset..."
            
        # Proses file ZIP atau folder dataset
        if file_name.lower().endswith('.zip'):
            # Gunakan import_from_zip dari download_service dengan opsi backup
            result = download_service.import_from_zip(
                zip_file=file_path,
                target_dir=output_dir,
                remove_zip=True,
                show_progress=True,
                backup_existing=backup_enabled
            )
            
            # Update progress dengan utils standar
            if 'progress_handler' in ui_components:
                ui_components['progress_handler'].update_progress_bar(90, 100, 
                                                                   "Finalisasi dataset...")
            else:
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = 90
                    
                if 'progress_label' in ui_components:
                    ui_components['progress_label'].value = "Finalisasi dataset..."
        else:
            # Jika bukan ZIP, coba gunakan fungsi lain dengan utils standar
            from smartcash.ui.utils.file_utils import list_files
            os.makedirs(output_dir, exist_ok=True)
            
            # Simpan file langsung
            from shutil import copy2
            copy2(file_path, os.path.join(output_dir, file_name))
            result = {"success": True, "message": f"File {file_name} berhasil disalin ke {output_dir}"}
        
        # Tampilkan hasil sukses dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(
                    f"{ICONS['success']} Dataset berhasil diproses!\nFile: {file_name}\nOutput: {output_dir}\nBackup: {'Ya' if backup_enabled else 'Tidak'}",
                    "success"
                ))
        
        # Update status panel
        update_status_panel(ui_components, "success", f"{ICONS['success']} File dataset berhasil diproses")
        
        # Bersihkan file sementara dengan utils standar
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        # Update progress dengan utils standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].update_progress_bar(100, 100, 
                                                               "Dataset siap digunakan")
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 100
                
            if 'progress_label' in ui_components:
                ui_components['progress_label'].value = "Dataset siap digunakan"
            
        # Notifikasi selesai dengan komponen standar
        try:
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(
                event_type="UPLOAD_COMPLETE",
                sender="upload_handler",
                message=f"File dataset {file_name} berhasil diproses ke {output_dir}"
            )
        except ImportError:
            pass
        
        return result
        
    except DatasetError as e:
        # Dataset manager error dengan utils standar
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(create_info_alert(f"{ICONS['error']} {str(e)}", "error"))
        
        # Reset progress dengan komponen standar
        if 'progress_handler' in ui_components:
            ui_components['progress_handler'].reset_progress_bar()
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 0
                
            if 'progress_label' in ui_components:
                ui_components['progress_label'].value = "Error: proses dibatalkan"
                
        # Update status panel
        update_status_panel(ui_components, "error", f"{ICONS['error']} Error: {str(e)}")
            
        raise
        
    except Exception as e:
        # Tangani exception lain dengan utils standar
        error_message = f"Error saat memproses file dataset: {str(e)}"
        
        # Handle with utils standar
        handle_ui_error(e, status_widget, True, error_message)
            
        # Update status panel
        update_status_panel(ui_components, "error", f"{ICONS['error']} Error: {error_message}")
            
        raise DatasetError(error_message)