"""
File: smartcash/ui/dataset/local_upload_handler.py (perbaikan)
Deskripsi: Perbaikan handler upload dataset lokal untuk memanfaatkan layanan dari DatasetManager
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import tempfile
from pathlib import Path

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
        # Tampilkan status loading
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                              color:{COLORS['alert_info_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['upload']} Memproses file dataset...</p>
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
        downloader = dataset_manager.get_service('downloader')
        
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
            ui_components['progress_bar'].value = 30
            
        # Proses file ZIP atau folder dataset
        if file_name.lower().endswith('.zip'):
            # Gunakan import_from_zip dari download_service yang sudah dikonfigurasi sebelumnya
            result = downloader.import_from_zip(
                zip_path=file_path,
                target_dir=output_dir,
                remove_zip=True,
                show_progress=True
            )
            
            # Update progress indicator
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 90
        else:
            # Jika bukan ZIP, gunakan utilitas yang ada dari FileProcessor
            file_processor = downloader.processor
            file_processor.clean_existing_download(output_dir)
            
            # Copy file
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            shutil.copy2(file_path, os.path.join(output_dir, file_name))
            
            # Fix struktur jika memungkinkan
            fix_result = file_processor.fix_dataset_structure(output_dir)
            
            result = {
                "status": "success" if fix_result else "warning",
                "message": f"File {file_name} berhasil disalin ke {output_dir}" + 
                           (". Struktur dataset diperbaiki." if fix_result else "")
            }
        
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
                        <p>Status: {result.get('status', 'success')}</p>
                    </div>
                """))
        
        # Bersihkan file sementara
        try:
            os.remove(file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].warning(f"⚠️ Gagal membersihkan file sementara: {str(e)}")
        
        # Update progress indicator
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
            
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
        raise DatasetError(error_message)