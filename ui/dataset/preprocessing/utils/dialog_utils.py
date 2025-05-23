"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_utils.py
Deskripsi: Utilitas untuk dialog konfirmasi preprocessing dengan existing data check
"""

from typing import Dict, Any, Callable, Optional
from IPython.display import display

from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.preprocessing.utils.drive_utils import check_existing_preprocessing_data
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_preprocessing_confirmation_dialog(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    on_confirm: Callable,
    on_cancel: Callable
) -> None:
    """
    Buat dialog konfirmasi preprocessing dengan pengecekan data existing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi preprocessing
        on_confirm: Callback saat dikonfirmasi
        on_cancel: Callback saat dibatalkan
    """
    # Cek existing data
    existing_data = check_existing_preprocessing_data(ui_components)
    
    # Format konfigurasi untuk ditampilkan
    config_summary = _format_config_summary(config)
    
    # Buat pesan konfirmasi
    message = f"Anda akan menjalankan preprocessing dataset dengan konfigurasi:\n\n{config_summary}\n\n"
    
    # Tambahkan informasi data existing jika ada
    if existing_data['exists']:
        data_info = _format_existing_data_info(existing_data)
        message += f"⚠️ **Data preprocessing sudah ada:**\n{data_info}\n\n"
        
        if config.get('force_reprocess', False):
            message += "✅ **Force reprocess aktif** - semua data akan diproses ulang\n\n"
        else:
            message += "ℹ️ **Incremental processing** - hanya data baru yang diproses\n\n"
    
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Tentukan danger mode berdasarkan existing data
    danger_mode = existing_data['exists'] and config.get('force_reprocess', False)
    
    # Buat dan tampilkan dialog
    _show_confirmation_dialog(
        ui_components,
        title="Konfirmasi Preprocessing Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        danger_mode=danger_mode
    )

def create_cleanup_confirmation_dialog(
    ui_components: Dict[str, Any],
    on_confirm: Callable,
    on_cancel: Callable
) -> None:
    """
    Buat dialog konfirmasi cleanup data preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        on_confirm: Callback saat dikonfirmasi
        on_cancel: Callback saat dibatalkan
    """
    # Cek existing data
    existing_data = check_existing_preprocessing_data(ui_components)
    
    if not existing_data['exists']:
        message = "Tidak ada data preprocessing yang perlu dibersihkan."
        _show_info_dialog(ui_components, "Info Cleanup", message)
        return
    
    # Format informasi data yang akan dihapus
    data_info = _format_existing_data_info(existing_data)
    
    message = f"Anda akan menghapus semua data preprocessing:\n\n{data_info}\n\n"
    message += "⚠️ **Perhatian:**\n"
    message += "• Symlink augmentasi akan dipertahankan\n"
    message += "• Data di Google Drive akan tetap aman jika menggunakan symlink\n"
    message += "• Tindakan ini tidak dapat dibatalkan untuk data lokal\n\n"
    message += "Apakah Anda yakin ingin melanjutkan pembersihan?"
    
    # Tampilkan dialog dengan danger mode
    _show_confirmation_dialog(
        ui_components,
        title="Konfirmasi Hapus Data Preprocessing",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        danger_mode=True,
        confirm_text="Ya, Hapus Data"
    )
def show_cleanup_confirmation(
    ui_components: Dict[str, Any],
    message: str,
    cleanup_info: Dict[str, Any],
    on_confirm: Callable,
    on_cancel: Callable
) -> None:
    """
    Tampilkan dialog konfirmasi khusus untuk cleanup.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan konfirmasi
        cleanup_info: Informasi cleanup
        on_confirm: Callback konfirmasi
        on_cancel: Callback pembatalan
    """
    # Use the existing confirmation dialog function
    create_cleanup_confirmation_dialog(
        ui_components,
        on_confirm,
        on_cancel
    )
def _format_config_summary(config: Dict[str, Any]) -> str:
    """
    Format summary konfigurasi untuk ditampilkan.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        str: Summary konfigurasi yang diformat
    """
    # Extract informasi konfigurasi
    resolution = config.get('resolution', (640, 640))
    if isinstance(resolution, (list, tuple)):
        resolution_str = f"{resolution[0]}x{resolution[1]}"
    else:
        resolution_str = str(resolution)
    
    normalization = config.get('normalization', 'minmax')
    augmentation = "Ya" if config.get('augmentation', False) else "Tidak"
    preserve_ratio = "Ya" if config.get('preserve_aspect_ratio', True) else "Tidak"
    force_reprocess = "Ya" if config.get('force_reprocess', False) else "Tidak"
    
    split = config.get('split', 'all')
    split_map = {
        'train': 'Training',
        'val': 'Validasi', 
        'test': 'Testing',
        'all': 'Semua Split'
    }
    split_str = split_map.get(split, split.title())
    
    num_workers = config.get('num_workers', 1)
    
    summary = f"📊 **Konfigurasi Preprocessing:**\n"
    summary += f"• Resolusi: {resolution_str}\n"
    summary += f"• Normalisasi: {normalization}\n"
    summary += f"• Augmentasi: {augmentation}\n"
    summary += f"• Preserve Aspect Ratio: {preserve_ratio}\n"
    summary += f"• Force Reprocess: {force_reprocess}\n"
    summary += f"• Target Split: {split_str}\n"
    summary += f"• Workers: {num_workers}"
    
    return summary

def _format_existing_data_info(existing_data: Dict[str, Any]) -> str:
    """
    Format informasi data existing untuk ditampilkan.
    
    Args:
        existing_data: Data existing dari check_existing_preprocessing_data
        
    Returns:
        str: Informasi data yang diformat
    """
    if not existing_data['exists']:
        return "Tidak ada data preprocessing"
    
    info = f"📁 **Data Preprocessing Existing:**\n"
    info += f"• Total File: {existing_data['total_files']:,}\n"
    info += f"• Total Size: {existing_data['size_mb']:.1f} MB\n"
    
    if existing_data['symlink_active']:
        info += f"• Storage: Google Drive (Symlink) 🔗\n"
    else:
        info += f"• Storage: Lokal 💾\n"
    
    # Detail per split
    if existing_data['splits']:
        info += f"• **Per Split:**\n"
        for split_name, split_data in existing_data['splits'].items():
            info += f"  - {split_name}: {split_data['files']:,} file ({split_data['size_mb']:.1f} MB)\n"
    
    return info

def _show_confirmation_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_confirm: Callable,
    on_cancel: Callable,
    danger_mode: bool = False,
    confirm_text: str = "Ya, Lanjutkan",
    cancel_text: str = "Batal"
) -> None:
    """
    Tampilkan dialog konfirmasi dengan area yang tepat.
    
    Args:
        ui_components: Dictionary komponen UI
        title: Judul dialog
        message: Pesan konfirmasi
        on_confirm: Callback konfirmasi
        on_cancel: Callback pembatalan
        danger_mode: Mode berbahaya (red button)
        confirm_text: Text tombol konfirmasi
        cancel_text: Text tombol batal
    """
    # Pastikan area konfirmasi tersedia
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
        logger.warning("⚠️ Area konfirmasi dibuat otomatis")
    
    # Bersihkan area konfirmasi
    ui_components['confirmation_area'].clear_output(wait=True)
    
    # Pastikan area visible
    if hasattr(ui_components['confirmation_area'], 'layout'):
        ui_components['confirmation_area'].layout.display = 'block'
        ui_components['confirmation_area'].layout.visibility = 'visible'
    
    # Wrapper untuk callback yang membersihkan dialog
    def wrapped_confirm(*args):
        try:
            ui_components['confirmation_area'].clear_output(wait=True)
            on_confirm()
        except Exception as e:
            logger.error(f"❌ Error confirm callback: {str(e)}")
    
    def wrapped_cancel(*args):
        try:
            ui_components['confirmation_area'].clear_output(wait=True)
            on_cancel()
        except Exception as e:
            logger.error(f"❌ Error cancel callback: {str(e)}")
    
    # Buat dan tampilkan dialog
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=wrapped_confirm,
            on_cancel=wrapped_cancel,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
        display(dialog)

def _show_info_dialog(ui_components: Dict[str, Any], title: str, message: str) -> None:
    """
    Tampilkan dialog informasi sederhana.
    
    Args:
        ui_components: Dictionary komponen UI
        title: Judul dialog
        message: Pesan informasi
    """
    def close_dialog():
        ui_components['confirmation_area'].clear_output(wait=True)
    
    _show_confirmation_dialog(
        ui_components,
        title=title,
        message=message,
        on_confirm=close_dialog,
        on_cancel=close_dialog,
        confirm_text="OK",
        cancel_text="Tutup"
    )

def create_drive_setup_dialog(
    ui_components: Dict[str, Any],
    on_setup_drive: Callable,
    on_use_local: Callable,
    on_cancel: Callable
) -> None:
    """
    Buat dialog untuk setup penyimpanan Drive atau lokal.
    
    Args:
        ui_components: Dictionary komponen UI
        on_setup_drive: Callback untuk setup Drive
        on_use_local: Callback untuk gunakan lokal
        on_cancel: Callback untuk batal
    """
    message = """🔗 **Setup Penyimpanan Preprocessing**

Anda dapat memilih lokasi penyimpanan data preprocessing:

**📁 Google Drive (Recommended)**
• Data tersimpan permanen di Drive
• Akses dari session berbeda
• Symlink otomatis ke direktori lokal
• Sinkronisasi real-time

**💾 Penyimpanan Lokal**
• Data hanya tersimpan di session ini
• Akan hilang jika runtime restart
• Performa lebih cepat
• Tidak perlu koneksi Drive

Pilih penyimpanan yang Anda inginkan:"""
    
    # Pastikan area konfirmasi tersedia
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
    
    # Bersihkan dan tampilkan area
    ui_components['confirmation_area'].clear_output(wait=True)
    if hasattr(ui_components['confirmation_area'], 'layout'):
        ui_components['confirmation_area'].layout.display = 'block'
    
    # Wrapper callbacks
    def wrapped_setup_drive(*args):
        ui_components['confirmation_area'].clear_output(wait=True)
        on_setup_drive()
    
    def wrapped_use_local(*args):
        ui_components['confirmation_area'].clear_output(wait=True)
        on_use_local()
    
    def wrapped_cancel(*args):
        ui_components['confirmation_area'].clear_output(wait=True)
        on_cancel()
    
    # Tampilkan dialog dengan 3 tombol
    with ui_components['confirmation_area']:
        import ipywidgets as widgets
        from smartcash.ui.utils.constants import COLORS
        
        # HTML message
        message_html = widgets.HTML(
            value=f"""
            <div style="padding: 20px; background-color: white; border-radius: 8px; 
                       box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h3 style="margin-top: 0; color: {COLORS['primary']};">Setup Penyimpanan</h3>
                <div style="white-space: pre-line; line-height: 1.6;">{message}</div>
            </div>
            """
        )
        
        # Tombol-tombol
        drive_button = widgets.Button(
            description="Setup Google Drive",
            button_style='success',
            icon='cloud',
            layout=widgets.Layout(width='auto', margin='0 5px')
        )
        
        local_button = widgets.Button(
            description="Gunakan Lokal",
            button_style='info', 
            icon='folder',
            layout=widgets.Layout(width='auto', margin='0 5px')
        )
        
        cancel_button = widgets.Button(
            description="Batal",
            button_style='',
            icon='times',
            layout=widgets.Layout(width='auto', margin='0 5px')
        )
        
        # Event handlers
        drive_button.on_click(wrapped_setup_drive)
        local_button.on_click(wrapped_use_local)
        cancel_button.on_click(wrapped_cancel)
        
        # Container
        button_container = widgets.HBox(
            [drive_button, local_button, cancel_button],
            layout=widgets.Layout(
                justify_content='center',
                margin='15px 0'
            )
        )
        
        dialog = widgets.VBox([message_html, button_container])
        display(dialog)