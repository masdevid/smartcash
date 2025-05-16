"""
File: smartcash/ui/handlers/processing_button_handler.py
Deskripsi: Handler tombol bersama untuk modul preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable, Union, List
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

# Import status handler yang konsisten
try:
    from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel as _update_status_panel
except ImportError:
    # Fallback jika tidak ditemukan
    from smartcash.ui.handlers.processing_cleanup_handler import _update_status_panel

# Fungsi notifikasi untuk observer pattern
def _notify_process_start(ui_components: Dict[str, Any], module_type: str, process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """Notifikasi observer bahwa proses telah dimulai."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start'](module_type, {
            'split': split,
            'display_info': display_info
        })

def _notify_process_complete(ui_components: Dict[str, Any], module_type: str, result: Dict[str, Any], display_info: str) -> None:
    """Notifikasi observer bahwa proses telah selesai dengan sukses."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['success']} {module_type.capitalize()} {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete'](module_type, result)

def _notify_process_error(ui_components: Dict[str, Any], module_type: str, error_message: str) -> None:
    """Notifikasi observer bahwa proses mengalami error."""
    logger = ui_components.get('logger')
    if logger: logger.error(f"{ICONS['error']} Error pada {module_type}: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error'](module_type, error_message)

def _notify_process_stop(ui_components: Dict[str, Any], module_type: str, display_info: str = "") -> None:
    """Notifikasi observer bahwa proses telah dihentikan oleh pengguna."""
    logger = ui_components.get('logger')
    if logger: logger.warning(f"{ICONS['stop']} Proses {module_type} dihentikan oleh pengguna")
    
    # Panggil callback jika tersedia
    if 'on_process_stop' in ui_components and callable(ui_components['on_process_stop']):
        ui_components['on_process_stop'](module_type, {
            'display_info': display_info
        })

def _disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True, module_type: str = 'preprocessing') -> None:
    """Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan."""
    # Daftar komponen yang perlu dinonaktifkan
    primary_button_key = f"{module_type}_button"
    disable_components = [
        'split_selector', 'config_accordion', 'options_accordion',
        'reset_button', primary_button_key, 'save_button'
    ]
    
    # Tambahan komponen untuk augmentation
    if module_type == 'augmentation':
        disable_components.extend(['aug_options', 'aug_factor_slider'])
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components and hasattr(ui_components[component], 'disabled'):
            ui_components[component].disabled = disable

def _cleanup_ui(ui_components: Dict[str, Any], module_type: str = 'preprocessing') -> None:
    """Membersihkan UI setelah proses selesai."""
    # Aktifkan kembali UI
    _disable_ui_during_processing(ui_components, False, module_type)
    
    # Tampilkan tombol utama, sembunyikan tombol stop
    primary_button_key = f"{module_type}_button"
    # Periksa keberadaan tombol primary sebelum mengakses
    if primary_button_key in ui_components:
        ui_components[primary_button_key].layout.display = 'block'
    elif 'augment_button' in ui_components and module_type == 'augmentation':
        # Fallback untuk augmentation module
        ui_components['augment_button'].layout.display = 'block'
    
    # Sembunyikan tombol stop jika ada
    if 'stop_button' in ui_components:
        ui_components['stop_button'].layout.display = 'none'
    
    # Reset progress bar
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()

def _get_dataset_manager(ui_components: Dict[str, Any], module_type: str) -> Any:
    """Mendapatkan atau membuat dataset manager dari ui_components.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dataset manager instance atau None jika gagal
    """
    logger = ui_components.get('logger')
    
    # Cek apakah manager sudah ada di ui_components
    manager_key = f"{module_type}_manager"
    if manager_key in ui_components and ui_components[manager_key] is not None:
        return ui_components[manager_key]
    
    # Jika belum ada, coba buat baru
    try:
        if module_type == 'preprocessing':
            # Import dan buat preprocessing manager
            from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
            
            # Dapatkan parameter dari ui_components
            data_dir = ui_components.get('data_dir', 'data')
            config = ui_components.get('config', {})
            
            # Buat konfigurasi preprocessing
            preproc_config = config.copy()
            if 'preprocessing' not in preproc_config:
                preproc_config['preprocessing'] = {}
            preproc_config['preprocessing']['output_dir'] = ui_components.get('preprocessed_dir', 'data/preprocessed')
            preproc_config['dataset_dir'] = data_dir
            
            # Buat instance service
            manager = DatasetPreprocessor(
                config=preproc_config,
                logger=logger
            )
            
        elif module_type == 'augmentation':
            # Import dan buat augmentation manager
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Dapatkan parameter dari ui_components
            data_dir = ui_components.get('data_dir', 'data')
            config = ui_components.get('config', {})
            num_workers = ui_components.get('num_workers', 4)
            
            # Buat instance service
            manager = AugmentationService(
                config=config,
                data_dir=data_dir,
                logger=logger,
                num_workers=num_workers
            )
            
        else:
            # Tipe modul tidak didukung
            if logger: logger.warning(f"{ICONS['warning']} Tipe modul tidak didukung: {module_type}")
            return None
        
        # Simpan manager ke ui_components
        ui_components[manager_key] = manager
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](manager)
        
        return manager
        
    except Exception as e:
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error saat membuat {module_type} manager: {str(e)}")
        return None

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
            # Gunakan augmentation_options yang merupakan nama komponen yang benar
            augmentation_options = ui_components.get('augmentation_options')
            
            if augmentation_options and hasattr(augmentation_options, 'children') and len(augmentation_options.children) > 0:
                # Struktur: Tab -> children[1] adalah split section
                tabs = augmentation_options.children[0]
                if hasattr(tabs, 'children') and len(tabs.children) > 1:
                    split_section = tabs.children[1]
                    if hasattr(split_section, 'children') and len(split_section.children) > 1:
                        # Dropdown split adalah children[1] dari split_section
                        split_option = split_section.children[1].value
                        split = split_option
                        display_info = f"Split {split_option}"
                    else:
                        # Fallback jika tidak bisa mendapatkan split dari UI
                        split = 'train'
                        display_info = "Split train (default)"
                else:
                    # Fallback jika tidak bisa mendapatkan split dari UI
                    split = 'train'
                    display_info = "Split train (default)"
            else:
                # Fallback jika komponen augmentation_options tidak ditemukan
                split = 'train'
                display_info = "Split train (default)"
        
        # Update UI: menampilkan proses dimulai
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai {process_name} dataset..."))
        
        # Tampilkan log panel dan progress bar jika tersedia
        if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Tampilkan progress bar dan label jika tersedia
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components and hasattr(ui_components[element], 'layout'):
                ui_components[element].layout.visibility = 'visible'
        
        # Disable semua UI input
        _disable_ui_during_processing(ui_components, True, module_type)
        
        # Update tombol untuk mode processing
        if primary_button_key in ui_components and hasattr(ui_components[primary_button_key], 'layout'):
            ui_components[primary_button_key].layout.display = 'none'
        
        # Tampilkan tombol stop jika tersedia
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
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
        
        # Jika augmentasi, dapatkan parameter tambahan
        if module_type == 'augmentation':
            # Dapatkan parameter augmentasi dari UI
            augmentation_options = ui_components.get('augmentation_options')
            aug_types = ['combined']
            aug_prefix = 'aug'
            aug_factor = 2
            balance_classes = False
            num_workers = 4
            
            # Ekstrak nilai dari UI jika tersedia
            if augmentation_options and hasattr(augmentation_options, 'children') and len(augmentation_options.children) > 0:
                tabs = augmentation_options.children[0]
                
                if hasattr(tabs, 'children') and len(tabs.children) >= 3:
                    # Tab 0: Basic Options
                    basic_tab = tabs.children[0]
                    if hasattr(basic_tab, 'children') and len(basic_tab.children) >= 4:
                        # Num variations (factor)
                        aug_factor = basic_tab.children[0].value
                        # Output prefix
                        aug_prefix = basic_tab.children[3].value
                        # Num workers
                        num_workers = basic_tab.children[2].value
                    
                    # Tab 2: Augmentation Types
                    aug_types_tab = tabs.children[2]
                    if hasattr(aug_types_tab, 'children') and len(aug_types_tab.children) >= 5:
                        # Aug types
                        aug_types = aug_types_tab.children[1].value
                        # Balance classes
                        balance_classes = aug_types_tab.children[3].children[1].value if hasattr(aug_types_tab.children[3], 'children') else False
                
                # Update params di ui_components
                ui_components.update({
                    'aug_types': aug_types,
                    'aug_prefix': aug_prefix,
                    'aug_factor': aug_factor,
                    'balance_classes': balance_classes,
                    'num_workers': num_workers
                })
                
                # Log parameter augmentasi yang digunakan
                if logger:
                    logger.debug(f"🔧 Parameter augmentasi: types={aug_types}, prefix={aug_prefix}, factor={aug_factor}, balance={balance_classes}, workers={num_workers}")
        
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
        _notify_process_stop(ui_components, module_type, display_info="")
        
        # Reset UI
        _cleanup_ui(ui_components)
    
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Validasi ui_components
        if ui_components is None:
            if logger: logger.error(f"{ICONS['error']} ui_components adalah None saat reset {module_type}")
            return
            
        # Reset UI dengan validasi
        try:
            _reset_ui(ui_components)
        except Exception as reset_error:
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset UI: {str(reset_error)}")
        
        # Load konfigurasi default dan update UI
        try:
            # Dynamic resolve config functions dengan validasi
            load_config_key = f'load_{module_type}_config'
            if load_config_key not in ui_components or not callable(ui_components.get(load_config_key)):
                if logger: logger.warning(f"{ICONS['warning']} Fungsi {load_config_key} tidak tersedia")
                return
                
            if 'update_ui_from_config' not in ui_components or not callable(ui_components.get('update_ui_from_config')):
                if logger: logger.warning(f"{ICONS['warning']} Fungsi update_ui_from_config tidak tersedia")
                return
                
            load_config_func = ui_components[load_config_key]
            update_ui_func = ui_components['update_ui_from_config']
            
            # Load config dengan validasi
            try:
                default_config = load_config_func()
                if default_config is None:
                    default_config = {}
                    if logger: logger.warning(f"{ICONS['warning']} Config default adalah None, menggunakan dict kosong")
            except Exception as config_error:
                if logger: logger.warning(f"{ICONS['warning']} Error saat load config: {str(config_error)}")
                default_config = {}
            
            # Update UI dari config
            try:
                update_ui_func(ui_components, default_config)
            except Exception as update_error:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update UI: {str(update_error)}")
            
            # Deteksi state modul dengan validasi
            detect_state_key = f'detect_{module_type}_state'
            if detect_state_key in ui_components and callable(ui_components.get(detect_state_key)):
                try:
                    ui_components[detect_state_key](ui_components)
                except Exception as state_error:
                    if logger: logger.warning(f"{ICONS['warning']} Error saat detect state: {str(state_error)}")
            
            # Tampilkan pesan sukses dengan validasi
            if 'status' in ui_components and ui_components['status'] is not None:
                try:
                    with ui_components['status']: 
                        display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
                except Exception as status_error:
                    if logger: logger.warning(f"{ICONS['warning']} Error saat update status: {str(status_error)}")
            
            # Update status panel dengan validasi
            try:
                _update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            except Exception as panel_error:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update status panel: {str(panel_error)}")
            
            # Log success
            if logger: logger.success(f"{ICONS['success']} Konfigurasi {process_name} berhasil direset ke nilai default")
        except Exception as e:
            # Jika gagal reset konfigurasi, tampilkan pesan error dengan validasi
            if 'status' in ui_components and ui_components['status'] is not None:
                try:
                    with ui_components['status']: 
                        display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
                except Exception:
                    pass
            
            # Update status panel dengan validasi
            try:
                _update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            except Exception:
                pass
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Gunakan fungsi helper global
    
    def _reset_ui(ui_components: Dict[str, Any]) -> None:
        """Reset UI ke kondisi awal."""
        # Validasi ui_components untuk mencegah KeyError
        if ui_components is None:
            logger.error(f"{ICONS['error']} ui_components adalah None saat reset UI")
            return
            
        # Bersihkan UI dengan validasi
        try:
            _cleanup_ui(ui_components, module_type)
        except Exception as e:
            logger.warning(f"{ICONS['warning']} Error saat cleanup UI: {str(e)}")
        
        # Sembunyikan visualisasi dan summary dengan validasi
        for component in ['visualization_container', 'summary_container', 'visualization_buttons']:
            if component in ui_components and ui_components[component] is not None:
                try:
                    ui_components[component].layout.display = 'none'
                except Exception as e:
                    logger.debug(f"{ICONS['warning']} Error saat menyembunyikan {component}: {str(e)}")
        
        # Reset status panel dengan validasi
        if 'status' in ui_components and ui_components['status'] is not None:
            try:
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("info", f"{ICONS['info']} Siap untuk memulai {module_type}"))
            except Exception as e:
                logger.warning(f"{ICONS['warning']} Error saat reset status panel: {str(e)}")
        else:
            logger.debug(f"{ICONS['info']} Status panel tidak tersedia saat reset UI")
    
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
        _cleanup_ui(ui_components, "preprocessing")
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
        _cleanup_ui(ui_components, "preprocessing")
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
        _cleanup_ui(ui_components, "augmentation")