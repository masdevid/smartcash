"""
File: smartcash/ui/handlers/processing_button_handler.py
Deskripsi: Handler tombol bersama untuk modul preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable, Union
from IPython.display import display, clear_output
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_processing_button_handlers(
    ui_components: Dict[str, Any], 
    module_type: str = 'preprocessing',
    config: Dict[str, Any] = None, 
    env = None
) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI processing (preprocessing/augmentation).
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Setup nama tombol dan fungsi berdasarkan module_type
    primary_button_key = f"{module_type}_button"
    process_name = "preprocessing" if module_type == 'preprocessing' else "augmentasi"
    running_flag_key = f"{module_type}_running"
    
    # Menonaktifkan error panel dan UI selama proses
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_primary_click(b):
        """Handler tombol primary dengan dukungan progress tracking."""
        # Mendapatkan opsi spesifik berdasarkan module_type
        if module_type == 'preprocessing':
            # Dapatkan split dari UI
            split_option = ui_components['split_selector'].value
            split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
            split = split_map.get(split_option)
            display_info = f"Split {split}" if split else "Semua split"
        else:  # augmentation
            # Dapatkan target split dari UI untuk augmentasi
            split_option = ui_components['aug_options'].children[4].value
            split = split_option
            display_info = f"Split {split_option}"
        
        # Update UI: menampilkan proses dimulai
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai {process_name} dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Disable semua UI input
        _disable_ui_during_processing(ui_components, True)
        
        # Update tombol untuk mode processing
        ui_components[primary_button_key].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Update konfigurasi dari UI dan simpan
        try:
            updated_config = ui_components['update_config_from_ui'](ui_components, config)
            ui_components[f'save_{module_type}_config'](updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi {process_name} berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Set flag processing sedang berjalan
        ui_components[running_flag_key] = True
        
        # Update status panel
        _update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai {process_name} {display_info}...")
        
        # Verifikasi direktori dataset
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # PERBAIKAN: Cek apakah path identik (hanya untuk preprocessing)
        if module_type == 'preprocessing' and os.path.realpath(data_dir) == os.path.realpath(preprocessed_dir):
            error_msg = f"Path data input dan output sama: {data_dir}, ini akan menyebabkan masalah"
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} {error_msg}"))
            _update_status_panel(ui_components, "error", f"{ICONS['error']} {error_msg}")
            _cleanup_ui(ui_components)
            return
        
        # Notifikasi observer tentang mulai proses
        _notify_process_start(ui_components, module_type, process_name, display_info, split)
        
        # Jika augmentation, dapatkan parameter tambahan
        if module_type == 'augmentation':
            aug_options = ui_components.get('aug_options')
            if aug_options and hasattr(aug_options, 'children') and len(aug_options.children) >= 7:
                aug_types = aug_options.children[0].value
                aug_prefix = aug_options.children[2].value
                aug_factor = aug_options.children[3].value
                balance_classes = aug_options.children[5].value
                num_workers = aug_options.children[6].value
                
                # Update params di ui_components
                ui_components.update({
                    'aug_types': aug_types,
                    'aug_prefix': aug_prefix,
                    'aug_factor': aug_factor,
                    'balance_classes': balance_classes,
                    'num_workers': num_workers
                })
        
        # Execute processing (preprocessing atau augmentation)
        if module_type == 'preprocessing':
            _execute_preprocessing(ui_components, split, display_info)
        else:
            _execute_augmentation(ui_components, split, display_info)
    
    def on_stop_click(b):
        """Handler untuk menghentikan proses."""
        # Set flag untuk menghentikan proses
        ui_components[running_flag_key] = False
        
        # Tampilkan pesan di status
        with ui_components['status']: 
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan {process_name}..."))
        
        # Update status panel
        _update_status_panel(ui_components, "warning", f"{ICONS['warning']} {process_name.capitalize()} dihentikan oleh pengguna")
        
        # Notifikasi observer
        _notify_process_stop(ui_components, module_type)
        
        # Reset UI
        _cleanup_ui(ui_components)
    
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Reset UI
        _reset_ui(ui_components)
        
        # Load konfigurasi default dan update UI
        try:
            # Dynamic resolve config functions
            load_config_func = ui_components[f'load_{module_type}_config']
            update_ui_func = ui_components['update_ui_from_config']
            
            default_config = load_config_func()
            update_ui_func(ui_components, default_config)
            
            # Deteksi state modul
            detect_state_func = ui_components.get(f'detect_{module_type}_state')
            if callable(detect_state_func):
                detect_state_func(ui_components)
            
            # Tampilkan pesan sukses
            with ui_components['status']: 
                display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
            
            # Update status panel
            _update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            
            # Log success
            if logger: logger.success(f"{ICONS['success']} Konfigurasi {process_name} berhasil direset ke nilai default")
        except Exception as e:
            # Jika gagal reset konfigurasi, tampilkan pesan error
            with ui_components['status']: 
                display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            
            # Update status panel
            _update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Register handlers untuk tombol-tombol
    button_handlers = {
        primary_button_key: on_primary_click,
        'stop_button': on_stop_click,
        'reset_button': on_reset_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in button_handlers.items() if button in ui_components]
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        f'on_{module_type}_click': on_primary_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        running_flag_key: False,
        'disable_ui_during_processing': _disable_ui_during_processing,
        'cleanup_ui': _cleanup_ui,
        'reset_ui': _reset_ui
    })
    
    return ui_components

def _execute_preprocessing(ui_components: Dict[str, Any], split, split_info: str) -> None:
    """Eksekusi preprocessing dengan parameter dari UI."""
    logger = ui_components.get('logger')
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    # Dapatkan dataset manager
    dataset_manager = _get_dataset_manager(ui_components, "preprocessing")
    if not dataset_manager:
        with ui_components['status']: 
            display(create_status_indicator("error", f"{ICONS['error']} Dataset Manager tidak tersedia"))
        _cleanup_ui(ui_components)
        return
    
    # Dapatkan opsi preprocessing dari UI
    options = ui_components.get('preprocess_options')
    normalize, preserve_aspect_ratio = False, True
    if options and hasattr(options, 'children') and len(options.children) >= 3:
        normalize, preserve_aspect_ratio = options.children[1].value, options.children[2].value
    
    # Jalankan preprocessing
    try:
        # Update konfigurasi dataset manager dengan path
        if hasattr(dataset_manager, 'config'): 
            dataset_manager.config['dataset_dir'] = data_dir
            if 'preprocessing' in dataset_manager.config:
                dataset_manager.config['preprocessing']['output_dir'] = preprocessed_dir
        
        # Update konfigurasi preproc jika ada
        if hasattr(dataset_manager, 'preprocess_config'):
            dataset_manager.preprocess_config.preprocessed_dir = preprocessed_dir
            dataset_manager.preprocess_config.raw_dataset_dir = data_dir
        
        # Log awal preprocessing
        if logger: logger.info(f"{ICONS['start']} Memulai preprocessing {split_info}")
        
        # PERBAIKAN: Pastikan data_dir dan preprocessed_dir adalah string
        data_dir_str = str(data_dir) if isinstance(data_dir, Path) else data_dir
        preprocessed_dir_str = str(preprocessed_dir) if isinstance(preprocessed_dir, Path) else preprocessed_dir
        
        # Jalankan preprocessing
        preprocess_result = dataset_manager.preprocess_dataset(
            split=split, 
            force_reprocess=True,
            normalize=normalize,
            preserve_aspect_ratio=preserve_aspect_ratio,
            # Parameter eksplisit untuk mencegah error
            raw_dataset_dir=data_dir_str,
            preprocessed_dir=preprocessed_dir_str
        )
        
        # Tambahkan path output jika tidak ada
        if 'output_dir' not in preprocess_result: 
            preprocess_result['output_dir'] = preprocessed_dir
        
        # Setelah selesai, update UI dengan status sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS['success']} Preprocessing {split_info} selesai"))
        
        # Update status panel
        _update_status_panel(ui_components, "success", 
                           f"{ICONS['success']} Preprocessing dataset berhasil diselesaikan")
        
        # Update UI state - tampilkan summary dan visualisasi
        for component in ['visualization_container', 'summary_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'block'
        
        # Tampilkan tombol visualisasi dan cleanup
        ui_components['visualization_buttons'].layout.display = 'flex'
        ui_components['cleanup_button'].layout.display = 'block'
        
        # Update summary dengan hasil preprocessing
        if 'generate_summary' in ui_components and callable(ui_components['generate_summary']):
            ui_components['generate_summary'](ui_components, preprocessed_dir)
        
        # Notifikasi observer tentang selesai
        _notify_process_complete(ui_components, "preprocessing", preprocess_result, split_info)
            
    except Exception as e:
        # Handle error dengan notifikasi
        with ui_components['status']: 
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Update status panel
        _update_status_panel(ui_components, "error", f"{ICONS['error']} Preprocessing gagal: {str(e)}")
        
        # Notifikasi observer tentang error
        _notify_process_error(ui_components, "preprocessing", str(e))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error saat preprocessing dataset: {str(e)}")
    
    finally:
        # Tandai preprocessing selesai
        ui_components['preprocessing_running'] = False
        
        # Restore UI
        _cleanup_ui(ui_components)
def _execute_augmentation(ui_components: Dict[str, Any], split: str, split_info: str) -> None:
    """Eksekusi augmentasi dengan parameter dari UI."""
    logger = ui_components.get('logger')
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    # Dapatkan parameter augmentasi
    aug_types = ui_components.get('aug_types', [])
    aug_prefix = ui_components.get('aug_prefix', 'aug')
    aug_factor = ui_components.get('aug_factor', 2)
    balance_classes = ui_components.get('balance_classes', True)
    num_workers = ui_components.get('num_workers', 4)
    
    # Dapatkan/buat AugmentationService
    try:
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
        
        # Jalankan augmentasi
        augment_result = augmentation_service.augment_dataset(
            split=split,
            augmentation_types=aug_types,
            augmentation_factor=aug_factor,
            target_dir=preprocessed_dir,
            output_dir=augmented_dir,
            prefix=aug_prefix,
            balance_classes=balance_classes
        )
        
        # Tambahkan path output jika tidak ada
        if 'output_dir' not in augment_result:
            augment_result['output_dir'] = augmented_dir
        
        # Setelah selesai, update UI dengan status sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS['success']} Augmentasi {split_info} selesai"))
        
        # Update status panel
        _update_status_panel(ui_components, "success", 
                           f"{ICONS['success']} Augmentasi dataset berhasil diselesaikan")
        
        # Update UI state - tampilkan summary dan visualisasi
        for component in ['visualization_container', 'summary_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'block'
        
        # Tampilkan tombol visualisasi dan cleanup
        ui_components['visualization_buttons'].layout.display = 'flex'
        ui_components['cleanup_button'].layout.display = 'block'
        
        # Update summary dengan hasil augmentasi
        if 'generate_summary' in ui_components and callable(ui_components['generate_summary']):
            ui_components['generate_summary'](ui_components, preprocessed_dir, augmented_dir)
        
        # Notifikasi observer tentang selesai
        _notify_process_complete(ui_components, "augmentation", augment_result, split_info)
            
    except Exception as e:
        # Handle error dengan notifikasi
        with ui_components['status']: 
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
        
        # Update status panel
        _update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal: {str(e)}")
        
        # Notifikasi observer tentang error
        _notify_process_error(ui_components, "augmentation", str(e))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error saat augmentasi dataset: {str(e)}")
    
    finally:
        # Tandai augmentasi selesai
        ui_components['augmentation_running'] = False
        
        # Restore UI
        _cleanup_ui(ui_components)