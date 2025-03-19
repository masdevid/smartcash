"""
File: smartcash/ui/dataset/local_upload_handler.py
Deskripsi: Handler untuk upload dataset lokal (ZIP/folder) memanfaatkan DownloadService dengan opsi backup opsional
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import tempfile

from smartcash.ui.utils.constants import COLORS, ICONS
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
    status_widget = ui_components.get('status')
    
    # Validasi: mendapatkan file upload widget dari local_upload
    local_upload = ui_components.get('local_upload')
    if not local_upload or not hasattr(local_upload, 'children') or len(local_upload.children) < 1:
        raise DatasetError("Komponen local_upload tidak ditemukan atau tidak valid")
        
    file_upload = local_upload.children[0]  # FileUpload widget seharusnya di indeks 0
    
    if not file_upload or not hasattr(file_upload, 'value') or not file_upload.value:
        raise DatasetError("Tidak ada file yang dipilih")
        
    try:
        # Cek apakah backup diaktifkan (defaultnya False)
        backup_enabled = False
        if len(local_upload.children) > 2 and hasattr(local_upload.children[2], 'value'):
            backup_enabled = local_upload.children[2].value
        
        # Tampilkan status loading
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                              color:{COLORS['alert_info_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['upload']} Memproses file dataset...</p>
                        <p style="margin:5px 0">Backup: {'Aktif' if backup_enabled else 'Tidak aktif'}</p>
                    </div>
                """))
        
        # Dapatkan dataset_manager dan download_service
        from smartcash.dataset.manager import DatasetManager
        
        # Gunakan dataset_manager yang sudah ada atau buat baru
        dataset_manager = ui_components.get('dataset_manager')
        if not dataset_manager:
            # Buat instance DatasetManager
            dataset_manager = DatasetManager(config=config)
            # Tambahkan ke ui_components untuk penggunaan berikutnya
            ui_components['dataset_manager'] = dataset_manager
        
        # Dapatkan download_service langsung dari manager
        download_service = dataset_manager.get_service('downloader')
        
        # Simpan file sementara dan proses
        temp_dir = tempfile.mkdtemp()
        
        # Dapatkan file pertama dari file_upload
        file_key = list(file_upload.value.keys())[0]
        file_content = file_upload.value[file_key]['content']
        file_name = file_upload.value[file_key]['metadata']['name']
        file_path = os.path.join(temp_dir, file_name)
        
        # Simpan file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Dapatkan output_dir dari UI components, config, atau dari local_upload
        output_dir = ui_components.get('data_dir', 'data/')
        
        # Jika ada di local_upload child ke-1 (target_dir), gunakan itu
        if len(local_upload.children) > 1 and hasattr(local_upload.children[1], 'value'):
            target_dir_input = local_upload.children[1]
            if target_dir_input.value:
                output_dir = target_dir_input.value
        
        # Update progress indicator
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
            
            # Update progress indicator
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 90
                
            if 'progress_label' in ui_components:
                ui_components['progress_label'].value = "Finalisasi dataset..."
        else:
            # Jika bukan ZIP, coba gunakan fungsi lain yang tersedia
            # atau langsung copy file ke direktori output
            from shutil import copy2
            os.makedirs(output_dir, exist_ok=True)
            copy2(file_path, os.path.join(output_dir, file_name))
            result = {"success": True, "message": f"File {file_name} berhasil disalin ke {output_dir}"}
        
        # Tampilkan hasil sukses
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                              color:{COLORS['alert_success_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['success']} Dataset berhasil diproses!</p>
                        <p>File: {file_name}</p>
                        <p>Output: {output_dir}</p>
                        <p>Backup: {'Ya' if backup_enabled else 'Tidak'}</p>
                    </div>
                """))
        
        # Bersihkan file sementara
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        # Update progress indicator
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
            
        if 'progress_label' in ui_components:
            ui_components['progress_label'].value = "Dataset siap digunakan"
            
        # Notifikasi selesai jika ada observer
        if 'observer_manager' in ui_components:
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
        
        # Reset progress indicator
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            
        if 'progress_label' in ui_components:
            ui_components['progress_label'].value = "Error: proses dibatalkan"
                
        raise
        
    except Exception as e:
        # Tangani exception lain
        error_message = f"Error saat memproses file dataset: {str(e)}"
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
                
        # Reset progress indicator
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            
        if 'progress_label' in ui_components:
            ui_components['progress_label'].value = "Error: proses dibatalkan"
            
        # Update status panel jika tersedia
        try:
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(ui_components, "error", f"❌ Error: {error_message}")
        except ImportError:
            pass
            
        raise DatasetError(error_message)