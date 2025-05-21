"""
File: smartcash/ui/dataset/download/handlers/download_executor.py
Deskripsi: Handler untuk eksekusi download dataset dari berbagai sumber
"""

import os
from typing import Dict, Any, Optional
from smartcash.dataset.manager import DatasetManager
from smartcash.ui.dataset.download.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.download.utils.logger_helper import log_message
from smartcash.ui.dataset.download.utils.progress_manager import update_progress
from smartcash.ui.dataset.download.utils.ui_state_manager import reset_ui_after_download

def download_from_roboflow(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download dataset dari Roboflow menggunakan dataset manager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Hasil download
    """
    # Tampilkan progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
    
    # Kirim notifikasi progress dimulai dan log message
    update_progress(ui_components, 0, "Mempersiapkan download dataset...")
    log_message(ui_components, "Mempersiapkan download dataset...", "info", "🔄")
    
    # Ambil parameter dari UI
    params = {
        'workspace': ui_components['workspace'].value,
        'project': ui_components['project'].value,
        'version': ui_components['version'].value,
        'api_key': ui_components['api_key'].value,
        'output_dir': ui_components['output_dir'].value
    }
    
    # Parameter tambahan jika tersedia
    if 'backup_checkbox' in ui_components and hasattr(ui_components['backup_checkbox'], 'value'):
        backup_existing = ui_components['backup_checkbox'].value
    else:
        backup_existing = False
    
    if 'backup_dir' in ui_components and hasattr(ui_components['backup_dir'], 'value'):
        backup_dir = ui_components['backup_dir'].value
    else:
        backup_dir = None
    
    # Validasi parameter
    if not params['api_key']:
        log_message(ui_components, "API Key tidak ditemukan. Mohon masukkan API Key Roboflow.", "error", "❌")
        reset_ui_after_download(ui_components)
        return {"status": "error", "message": "API Key tidak ditemukan"}
    
    # Pastikan output_dir valid dan ada
    output_dir = params['output_dir']
    if not _validate_output_directory(ui_components, output_dir):
        reset_ui_after_download(ui_components)
        return {"status": "error", "message": "Gagal membuat/mengakses direktori output"}
    
    # Jalankan download
    try:
        # Update progress
        update_progress(ui_components, 10, "Memulai download dari Roboflow...")
        
        dataset_manager = DatasetManager()
        
        # Register observer jika ada
        observer_manager = register_ui_observers(ui_components)
        
        # Coba dapatkan service downloader dan set observer
        try:
            downloader_service = dataset_manager.get_service('downloader')
            downloader_service.set_observer_manager(observer_manager)
        except Exception as e:
            log_message(ui_components, f"Gagal mengatur observer untuk downloader: {str(e)}", "warning", "⚠️")
        
        # Update progress lagi
        update_progress(ui_components, 20, "Mendownload dataset dari Roboflow...")
        
        # Jalankan download dengan parameter yang sesuai
        return _execute_download_with_parameters(ui_components, dataset_manager, params, backup_existing, backup_dir)
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat proses download dataset: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        reset_ui_after_download(ui_components)
        return {"status": "error", "message": error_msg}

def _validate_output_directory(ui_components: Dict[str, Any], output_dir: str) -> bool:
    """
    Validasi direktori output dan buat jika belum ada.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Path direktori output
        
    Returns:
        bool: True jika direktori valid dan siap digunakan, False jika tidak
    """
    try:
        # Buat direktori jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        
        # Periksa apakah direktori dapat ditulis
        test_file = os.path.join(output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        log_message(ui_components, f"Direktori output dibuat/ditemukan: {output_dir}", "info", "📁")
        return True
    except Exception as e:
        error_msg = f"Gagal membuat direktori output: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Coba gunakan direktori alternatif
        alt_output_dir = os.path.join(os.path.expanduser('~'), 'smartcash_downloads')
        try:
            os.makedirs(alt_output_dir, exist_ok=True)
            ui_components['output_dir'].value = alt_output_dir
            log_message(ui_components, f"Menggunakan direktori output alternatif: {alt_output_dir}", "warning", "⚠️")
            return True
        except Exception as e2:
            log_message(ui_components, f"Gagal membuat direktori alternatif: {str(e2)}", "error", "❌")
            return False

def _execute_download_with_parameters(
    ui_components: Dict[str, Any], 
    dataset_manager: Any, 
    params: Dict[str, str],
    backup_existing: bool,
    backup_dir: Optional[str]
) -> Dict[str, Any]:
    """
    Eksekusi download dengan parameter yang tepat, memeriksa signature method.
    
    Args:
        ui_components: Dictionary komponen UI
        dataset_manager: Instance DatasetManager
        params: Parameter download dasar
        backup_existing: Flag backup data yang sudah ada
        backup_dir: Direktori backup opsional
        
    Returns:
        Dict[str, Any]: Hasil download
    """
    # Periksa signature method
    import inspect
    try:
        signature = inspect.signature(dataset_manager.download_from_roboflow)
        valid_params = {}
        
        # Tambahkan parameter yang valid
        for param_name, param in signature.parameters.items():
            if param_name in params:
                valid_params[param_name] = params[param_name]
        
        # Tambahkan parameter opsional jika didukung
        if 'backup_existing' in signature.parameters:
            valid_params['backup_existing'] = backup_existing
        
        if 'backup_dir' in signature.parameters and backup_dir:
            valid_params['backup_dir'] = backup_dir
            
        # Tambahkan parameter show_progress jika didukung
        if 'show_progress' in signature.parameters:
            valid_params['show_progress'] = True
            
        # Tambahkan parameter verify_integrity jika didukung
        if 'verify_integrity' in signature.parameters:
            valid_params['verify_integrity'] = True
            
        log_message(ui_components, f"Mendownload dataset dengan parameter: {', '.join(valid_params.keys())}", "info", "📥")
        
        # Jalankan download
        result = dataset_manager.download_from_roboflow(**valid_params)
        
        # Update progress setelah download selesai
        update_progress(ui_components, 90, "Download selesai, memproses hasil...")
        
        return result
    except Exception as e:
        # Fallback ke parameter minimal
        log_message(ui_components, f"Mencoba download dengan parameter minimal: {str(e)}", "warning", "⚠️")
        result = dataset_manager.download_from_roboflow(
            api_key=params['api_key'],
            workspace=params['workspace'],
            project=params['project'],
            version=params['version'],
            output_dir=params['output_dir']
        )
        
        # Update progress setelah download selesai
        update_progress(ui_components, 90, "Download selesai dengan parameter minimal...")
        
        return result

def process_download_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Proses hasil download dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil download dari dataset service
    """
    # Cek apakah result adalah None atau tidak valid
    if result is None:
        error_msg = "Hasil download tidak valid (None)"
        log_message(ui_components, error_msg, "error", "❌")
        
        # Reset UI
        reset_ui_after_download(ui_components)
        return
    
    # Ekstrak informasi dari result
    success = result.get('success', False) or result.get('status') == 'success'
    message = result.get('message', '')
    
    if success:
        # Download berhasil
        log_message(ui_components, f"Download dataset berhasil: {message}", "success", "✅")
        
        # Gunakan update_progress sebagai notifikasi
        update_progress(ui_components, 100, "Download selesai")
        
        # Simpan konfigurasi setelah download berhasil
        try:
            from smartcash.ui.dataset.download.handlers.save_handler import handle_save_config
            handle_save_config(ui_components)
        except Exception as e:
            log_message(ui_components, f"Gagal menyimpan konfigurasi: {str(e)}", "warning", "⚠️")
    else:
        # Download gagal
        log_message(ui_components, f"Download dataset gagal: {message}", "error", "❌")
    
    # Reset UI setelah proses selesai
    reset_ui_after_download(ui_components) 