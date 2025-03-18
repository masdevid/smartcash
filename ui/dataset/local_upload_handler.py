"""
File: smartcash/ui/dataset/local_upload_handler.py
Deskripsi: Handler untuk upload dataset lokal (ZIP/folder) memanfaatkan DownloadService
"""

import os
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output
import tempfile

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.exceptions import DatasetError

def process_local_upload(
    ui_components: Dict[str, Any],
    env=None
) -> Dict[str, Any]:
    """
    Proses upload dataset lokal menggunakan DatasetManager dan DownloadService.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        
    Returns:
        Dictionary berisi informasi hasil proses upload
        
    Raises:
        DatasetError: Jika terjadi error saat proses file
    """
    status_widget = ui_components.get('status')
    file_upload = ui_components.get('file_upload')
    
    if not file_upload or not file_upload.value:
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
        
        # Coba dapatkan config dari UI components atau gunakan default
        config = ui_components.get('config', {})
        
        # Buat instance DatasetManager
        dataset_manager = DatasetManager(config=config)
        
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
        
        # Dapatkan output_dir dari config atau UI components
        output_dir = config.get('dataset_dir', 'data/')
        if 'output_dir_input' in ui_components and hasattr(ui_components['output_dir_input'], 'value'):
            output_dir = ui_components['output_dir_input'].value or output_dir
        
        # Proses file ZIP atau folder dataset
        if file_name.lower().endswith('.zip'):
            # Gunakan process_zip_file dari download_service
            result = download_service.process_zip_file(
                zip_path=file_path,
                output_dir=output_dir,
                extract_only=False,
                validate_after=True
            )
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
                    </div>
                """))
        
        # Bersihkan file sementara
        os.remove(file_path)
        os.rmdir(temp_dir)
        
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