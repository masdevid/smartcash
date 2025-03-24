"""
File: smartcash/ui/dataset/handlers/confirmation_handler.py
Deskripsi: Handler untuk konfirmasi sebelum download dataset untuk mencegah overwrite
"""

from typing import Dict, Any, Callable, Optional, Tuple
from pathlib import Path
from IPython.display import display

def check_existing_dataset(output_dir: str) -> bool:
    """
    Periksa apakah dataset sudah ada di lokasi output.
    
    Args:
        output_dir: Direktori output dataset
        
    Returns:
        Boolean menunjukkan apakah dataset sudah ada
    """
    # Cek berdasarkan struktur folder standar YOLO
    path = Path(output_dir)
    standard_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    return path.exists() and any((path / subdir).exists() for subdir in standard_dirs)

def create_dataset_confirmation(
    ui_components: Dict[str, Any],
    callback_proceed: Callable,
    callback_cancel: Optional[Callable] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Buat dan tampilkan dialog konfirmasi jika dataset sudah ada.
    
    Args:
        ui_components: Dictionary komponen UI
        callback_proceed: Callback jika user memilih melanjutkan
        callback_cancel: Callback jika user membatalkan (opsional)
        output_dir: Direktori output dataset (opsional, default dari ui_components)
    """
    # Dapatkan output dir dari UI jika tidak disediakan
    if output_dir is None and 'output_dir' in ui_components:
        output_dir = ui_components['output_dir'].value
    
    # Cek apakah dataset sudah ada
    data_exists = check_existing_dataset(output_dir)
    
    if data_exists:
        # Import komponen konfirmasi
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        
        # Pesan konfirmasi
        message = (f"Dataset sudah ada di {output_dir}. "
                  f"Melanjutkan operasi akan menimpa data yang ada. "
                  f"Apakah Anda yakin ingin melanjutkan?")
        
        # Buat dialog konfirmasi
        dialog = create_confirmation_dialog(
            message=message,
            on_confirm=callback_proceed,
            on_cancel=callback_cancel,
            title="Dataset Sudah Ada",
            confirm_label="Ya, Lanjutkan",
            cancel_label="Batal"
        )
        
        # Tampilkan dialog
        status_output = ui_components.get('status')
        if status_output:
            with status_output:
                display(dialog)
        else:
            display(dialog)
        
        # Log konfirmasi ke logger
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Dataset sudah ada di {output_dir}, menunggu konfirmasi")
    else:
        # Jika dataset belum ada, langsung jalankan callback
        callback_proceed()

def prepare_dataset_confirmation(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    handler_func: Callable
) -> Tuple[Callable, Optional[Callable]]:
    """
    Siapkan fungsi callback untuk konfirmasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download
        handler_func: Fungsi handler yang akan dijalankan setelah konfirmasi
        
    Returns:
        Tuple (proceed_callback, cancel_callback)
    """
    # Definisikan callback proceed (dengan closure untuk akses config)
    def proceed_callback():
        # Tambahkan flag backup_existing ke config
        config['backup_existing'] = True
        
        # Jalankan handler function dengan config yang diupdate
        handler_func(config)
    
    # Definisikan callback cancel
    def cancel_callback():
        # Enable kembali tombol
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, "Download dibatalkan", "warning", "⚠️")
        
        # Aktifkan kembali tombol
        for button_key in ['download_button', 'check_button']:
            if button_key in ui_components:
                ui_components[button_key].disabled = False
    
    return proceed_callback, cancel_callback