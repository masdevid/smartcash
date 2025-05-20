"""
File: smartcash/ui/dataset/augmentation/handlers/button_handler.py
Deskripsi: Handler tombol untuk modul augmentasi dataset
"""

from typing import Dict, Any, Optional, Callable, Union, List
import os
import time
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.utils.notification_manager import get_notification_manager

logger = get_logger()

def notify_process_stop(ui_components: Dict[str, Any], display_info: str = "") -> None:
    """Notifikasi observer bahwa proses telah dihentikan oleh pengguna."""
    logger = ui_components.get('logger')
    if logger: logger.warning(f"{ICONS['stop']} Proses augmentasi dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop']("augmentation", {
            'display_info': display_info
        })

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan."""
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', 'augmentation_button', 'save_button'
    ]
    
    # Tambahan komponen untuk augmentasi
    if 'augmentation_options' in ui_components:
        disable_components.append('augmentation_options')
    if 'rotation_range' in ui_components:
        disable_components.append('rotation_range')
    if 'width_shift_range' in ui_components:
        disable_components.append('width_shift_range')
    if 'height_shift_range' in ui_components:
        disable_components.append('height_shift_range')
    if 'zoom_range' in ui_components:
        disable_components.append('zoom_range')
    if 'horizontal_flip' in ui_components:
        disable_components.append('horizontal_flip')
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components and hasattr(ui_components[component], 'disabled'):
            ui_components[component].disabled = disable

def execute_augmentation(ui_components: Dict[str, Any], split: str, split_info: str) -> Dict[str, Any]:
    """Eksekusi augmentasi dengan parameter dari UI.

    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset yang akan diaugmentasi
        split_info: Informasi tambahan tentang split

    Returns:
        Dictionary hasil augmentasi
    """
    logger = ui_components.get('logger')
    
    # Dapatkan parameter dari UI
    try:
        # Dapatkan augmentation service
        augmentation_service = ui_components.get('augmentation_service')
        
        # Dapatkan parameter dari UI
        data_dir = ui_components.get('data_dir', '')
        if not data_dir:
            data_dir = ui_components.get('config', {}).get('dataset', {}).get('path', '')
        
        # Dapatkan direktori preprocessed dan augmented
        preprocessed_dir = os.path.join(data_dir, 'preprocessed')
        augmented_dir = os.path.join(data_dir, 'augmented')
        
        # Dapatkan parameter augmentasi dari UI
        aug_types = []
        
        # Cek apakah augmentation_options ada dan memiliki struktur yang benar
        if 'augmentation_options' in ui_components and hasattr(ui_components['augmentation_options'], 'children'):
            # Struktur tab augmentation_options:
            # Tab 0: Basic Options (jumlah variasi, target count, workers, prefix)
            # Tab 1: Split Section (target split)
            # Tab 2: Augmentation Types (jenis augmentasi, balance classes)
            
            # Dapatkan jenis augmentasi dari tab 2
            if len(ui_components['augmentation_options'].children) > 2:
                aug_types_widget = ui_components['augmentation_options'].children[2].children[0]
                if hasattr(aug_types_widget, 'value'):
                    aug_types = list(aug_types_widget.value)
        
        # Fallback jika aug_types kosong
        if not aug_types:
            aug_types = ['combined']
            if logger: logger.warning(f"{ICONS['warning']} Jenis augmentasi tidak ditemukan, menggunakan default: {aug_types}")
        
        # Dapatkan parameter lainnya dengan fallback ke nilai default
        aug_prefix = 'aug'
        aug_factor = 2
        balance_classes = False
        num_workers = 4
        
        # Cek parameter dari UI jika tersedia
        if 'aug_prefix' in ui_components and hasattr(ui_components['aug_prefix'], 'value'):
            aug_prefix = ui_components['aug_prefix'].value
        
        if 'aug_factor' in ui_components and hasattr(ui_components['aug_factor'], 'value'):
            aug_factor = ui_components['aug_factor'].value
        
        if 'balance_classes' in ui_components and hasattr(ui_components['balance_classes'], 'value'):
            balance_classes = ui_components['balance_classes'].value
        
        if 'num_workers' in ui_components and hasattr(ui_components['num_workers'], 'value'):
            num_workers = ui_components['num_workers'].value
        
        # Dapatkan/buat AugmentationService jika belum ada
        if not augmentation_service:
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Buat instance service
            augmentation_service = AugmentationService(
                config=ui_components.get('config', {}), 
                data_dir=data_dir,
                logger=logger,
                num_workers=num_workers
            )
            
            # Simpan ke ui_components
            ui_components['augmentation_manager'] = augmentation_service
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](augmentation_service)
        
        # Log awal augmentasi
        if logger: logger.info(f"{ICONS['start']} Memulai augmentasi {split_info}")
        
        # Jalankan augmentasi dengan parameter yang benar
        augment_result = augmentation_service.augment_dataset(
            split=split,
            augmentation_types=aug_types,
            num_variations=aug_factor,
            output_prefix=aug_prefix,
            target_balance=balance_classes,
            target_count=1000,  # Nilai default untuk target jumlah gambar
            move_to_preprocessed=True,
            validate_results=True
        )
        
        # Tambahkan path output jika tidak ada
        if 'output_dir' not in augment_result:
            augment_result['output_dir'] = augmented_dir
        
        # Setelah selesai, update UI dengan status sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS['success']} Augmentasi {split_info} selesai"))
        
        # Update status panel
        update_status_panel(ui_components, "success", 
                           f"{ICONS['success']} Augmentasi dataset berhasil diselesaikan")
        
        # Update UI state - tampilkan summary dan visualisasi
        for component in ['visualization_container', 'summary_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'block'
        
        # Tampilkan tombol visualisasi dan cleanup
        if 'visualization_buttons' in ui_components:
            ui_components['visualization_buttons'].layout.display = 'flex'
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].layout.display = 'block'
        
        # Update summary dengan hasil augmentasi
        if 'generate_summary' in ui_components and callable(ui_components['generate_summary']):
            ui_components['generate_summary'](ui_components, preprocessed_dir, augmented_dir)
        
        # Notifikasi observer tentang selesai
        notify_process_complete(ui_components, augment_result, split_info)
            
    except Exception as e:
        # Handle error dengan notifikasi
        with ui_components['status']: 
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Update status panel
        update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal: {str(e)}")
        
        # Notifikasi observer tentang error
        notify_process_error(ui_components, str(e))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error saat augmentasi dataset: {str(e)}")
        if logger: logger.error(traceback.format_exc())
    
    finally:
        # Tandai augmentasi selesai
        ui_components['augmentation_running'] = False
        
        # Restore UI
        if 'cleanup_ui' in ui_components and callable(ui_components['cleanup_ui']):
            ui_components['cleanup_ui'](ui_components)
        else:
            disable_ui_during_processing(ui_components, False)

def setup_augmentation_button_handlers(
    ui_components: Dict[str, Any], 
    module_type: str = 'augmentation',
    config: Dict[str, Any] = None, 
    env = None
) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul (default: 'augmentation')
        config: Konfigurasi modul (opsional)
        env: Environment (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger', get_logger())
    
    # Handler tombol primary dengan dukungan progress tracking
    def on_primary_click(b):
        # Cek apakah sudah running
        if ui_components.get('augmentation_running', False):
            with ui_components['status']:
                display(create_status_indicator("warning", f"{ICONS['warning']} Augmentasi sudah berjalan"))
            return
        
        # Set flag running
        ui_components['augmentation_running'] = True
        
        # Disable UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Tampilkan tombol stop jika tersedia
        if 'augmentation_button' in ui_components and hasattr(ui_components['augmentation_button'], 'layout'):
            ui_components['augmentation_button'].layout.display = 'none'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'block'
        
        # Expand log accordion jika tersedia
        if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # Dapatkan split yang dipilih
        split = None
        if 'split_selector' in ui_components:
            split = ui_components['split_selector'].value
        
        # Validasi split
        if not split:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Error: Split tidak dipilih"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal: Split tidak dipilih")
            
            # Reset UI
            ui_components['augmentation_running'] = False
            disable_ui_during_processing(ui_components, False)
            return
        
        # Dapatkan info split untuk display
        split_info = f"untuk split '{split}'"
        
        # Update status
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi {split_info}..."))
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai augmentasi {split_info}...")
        
        # Notifikasi observer bahwa proses dimulai
        notify_process_start(ui_components, "augmentasi", split_info, split)
        
        # Jalankan augmentasi dengan parameter dari UI
        execute_augmentation(ui_components, split, split_info)
    
    # Handler untuk menghentikan proses
    def on_stop_click(b):
        # Set flag untuk menghentikan proses
        ui_components['augmentation_running'] = False
        
        # Update status
        with ui_components['status']:
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Menghentikan augmentasi...")
        
        # Notifikasi observer bahwa proses dihentikan
        notify_process_stop(ui_components)
    
    # Reset UI dan konfigurasi ke default
    def on_reset_click(b):
        # Konfirmasi reset
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("warning", f"{ICONS['warning']} Reset konfigurasi augmentasi..."))
        
        # Reset UI ke kondisi awal
        _reset_ui(ui_components)
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['info']} Konfigurasi augmentasi direset ke default")
    
    # Reset UI ke kondisi awal
    def _reset_ui(ui_components: Dict[str, Any]):
        # Reset flag
        ui_components['augmentation_running'] = False
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # Reset UI components ke default
        default_values = {
            'aug_factor': 2,
            'balance_classes': False,
            'num_workers': 4,
            'aug_prefix': 'aug'
        }
        
        # Update UI components dengan nilai default
        for key, value in default_values.items():
            if key in ui_components and hasattr(ui_components[key], 'value'):
                ui_components[key].value = value
        
        # Reset jenis augmentasi ke default jika tersedia
        if 'augmentation_options' in ui_components and hasattr(ui_components['augmentation_options'], 'children'):
            if len(ui_components['augmentation_options'].children) > 2:
                aug_types_widget = ui_components['augmentation_options'].children[2].children[0]
                if hasattr(aug_types_widget, 'value'):
                    aug_types_widget.value = ('combined',)
        
        # Aktifkan kembali UI
        disable_ui_during_processing(ui_components, False)
        
        # Tampilkan tombol primary dan sembunyikan tombol stop
        if 'augmentation_button' in ui_components and hasattr(ui_components['augmentation_button'], 'layout'):
            ui_components['augmentation_button'].layout.display = 'block'
        
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
    
    # Register handler untuk tombol
    if 'augmentation_button' in ui_components:
        ui_components['augmentation_button'].on_click(on_primary_click)
    
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(on_stop_click)
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(on_reset_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_primary_click': on_primary_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'execute_augmentation': execute_augmentation,
        'disable_ui_during_processing': disable_ui_during_processing,
        'notify_process_stop': notify_process_stop
    })
    
    # Pastikan NotificationManager tersedia
    notification_manager = get_notification_manager(ui_components)
    
    # Tambahkan fungsi notifikasi dari NotificationManager
    ui_components.update({
        'notify_process_start': notification_manager.notify_process_start,
        'notify_process_complete': notification_manager.notify_process_complete,
        'notify_process_error': notification_manager.notify_process_error
    })
    
    # Set flag running ke False
    ui_components['augmentation_running'] = False
    
    return ui_components
