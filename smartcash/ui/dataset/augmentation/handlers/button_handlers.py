"""
File: smartcash/ui/dataset/augmentation/handlers/button_handlers.py
Deskripsi: Handler tombol untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel

# Fungsi notifikasi untuk observer pattern
def _notify_process_start(ui_components: Dict[str, Any], process_name: str, display_info: str, split: Optional[str] = None) -> None:
    """Notifikasi observer bahwa proses telah dimulai."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['start']} Memulai {process_name} {display_info}")
    
    # Panggil callback jika tersedia
    if 'on_process_start' in ui_components and callable(ui_components['on_process_start']):
        ui_components['on_process_start']("augmentation", {
            'split': split,
            'display_info': display_info
        })

def _notify_process_complete(ui_components: Dict[str, Any], result: Dict[str, Any], display_info: str) -> None:
    """Notifikasi observer bahwa proses telah selesai dengan sukses."""
    logger = ui_components.get('logger')
    if logger: logger.info(f"{ICONS['success']} Augmentasi {display_info} selesai")
    
    # Panggil callback jika tersedia
    if 'on_process_complete' in ui_components and callable(ui_components['on_process_complete']):
        ui_components['on_process_complete']("augmentation", result)

def _notify_process_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """Notifikasi observer bahwa proses mengalami error."""
    logger = ui_components.get('logger')
    if logger: logger.error(f"{ICONS['error']} Error pada augmentasi: {error_message}")
    
    # Panggil callback jika tersedia
    if 'on_process_error' in ui_components and callable(ui_components['on_process_error']):
        ui_components['on_process_error']("augmentation", error_message)

def setup_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI augmentasi dengan sistem progress yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Penanganan error dengan decorator standar
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_augment_click(b):
        """Handler tombol augmentasi dengan dukungan progress tracking yang dioptimalkan."""
        # Target split selalu 'train'
        split_option = 'train'
        
        # Update UI: menampilkan proses dimulai
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        # Tampilkan progress bar dan label
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Disable semua komponen UI selama proses
        disable_ui_during_processing(ui_components, True)
        
        # Update UI tombol
        ui_components['augment_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Update konfigurasi dari UI dan simpan
        try:
            updated_config = ui_components['update_config_from_ui'](ui_components, config)
            ui_components['save_augmentation_config'](updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Tandai augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Update status panel dengan informasi awal
        split_info = f"Split {split_option}"
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai augmentasi {split_info}...")
        
        # Dapatkan opsi augmentasi dari UI dengan validasi yang lebih kuat
        # Inisialisasi dengan nilai default yang aman
        aug_types = ['Combined (Recommended)']  # Nilai tetap
        aug_prefix = 'aug'
        aug_factor = 2
        balance_classes = True
        num_workers = 4
        
        try:
            # Ekstrak parameter dari UI dengan validasi
            # Prefix (sekarang di posisi 1)
            if hasattr(ui_components['aug_options'].children[1], 'value'):
                extracted_prefix = ui_components['aug_options'].children[1].value
                if extracted_prefix is not None and isinstance(extracted_prefix, str) and extracted_prefix.strip():
                    aug_prefix = extracted_prefix
            
            # Factor (sekarang di posisi 2)
            if hasattr(ui_components['aug_options'].children[2], 'value'):
                extracted_factor = ui_components['aug_options'].children[2].value
                if extracted_factor is not None and isinstance(extracted_factor, (int, float)) and extracted_factor > 0:
                    aug_factor = extracted_factor
            
            # Balance classes (sekarang di posisi 4)
            if hasattr(ui_components['aug_options'].children[4], 'value'):
                extracted_balance = ui_components['aug_options'].children[4].value
                if extracted_balance is not None:
                    balance_classes = bool(extracted_balance)
            
            # Num workers (sekarang di posisi 5)
            if hasattr(ui_components['aug_options'].children[5], 'value'):
                extracted_workers = ui_components['aug_options'].children[5].value
                if extracted_workers is not None and isinstance(extracted_workers, int) and extracted_workers > 0:
                    num_workers = extracted_workers
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat mendapatkan parameter augmentasi: {str(e)}, menggunakan nilai default")
            # Nilai default sudah diinisialisasi di awal
        
        # Verifikasi direktori dataset
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Notifikasi observer tentang mulai augmentasi
        _notify_process_start(ui_components, "augmentasi", split_info, split_option)
        
        # Setup augmentation service
        try:
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Buat instance AugmentationService
            augmentation_service = AugmentationService(
                config, 
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
            
            # Import utilitas persistensi
            from smartcash.common.config.manager import get_config_manager
            from smartcash.ui.utils.persistence_utils import validate_ui_param
            
            # Dapatkan config manager
            config_manager = get_config_manager()
            
            # Pastikan nilai tetap selalu valid
            aug_types = ['Combined (Recommended)']  # Nilai tetap
            
            # Validasi parameter lain dengan utilitas validasi
            try:
                split_option = validate_ui_param(
                    split_option, 
                    'train', 
                    str,
                    ['train', 'val', 'test', 'all'],
                    logger
                )
            except Exception as e:
                split_option = 'train'  # Nilai default
                if logger: logger.warning(f"{ICONS['warning']} Error saat validasi split_option: {str(e)}, menggunakan default: {split_option}")
            
            try:
                aug_prefix = validate_ui_param(
                    aug_prefix, 
                    'aug', 
                    str,
                    None,
                    logger
                )
            except Exception as e:
                aug_prefix = 'aug'  # Nilai default
                if logger: logger.warning(f"{ICONS['warning']} Error saat validasi aug_prefix: {str(e)}, menggunakan default: {aug_prefix}")
            
            try:
                aug_factor = validate_ui_param(
                    aug_factor, 
                    2, 
                    (int, float),
                    None,
                    logger
                )
            except Exception as e:
                aug_factor = 2  # Nilai default
                if logger: logger.warning(f"{ICONS['warning']} Error saat validasi aug_factor: {str(e)}, menggunakan default: {aug_factor}")
            
            try:
                num_workers = validate_ui_param(
                    num_workers, 
                    4, 
                    int,
                    None,
                    logger
                )
            except Exception as e:
                num_workers = 4  # Nilai default
                if logger: logger.warning(f"{ICONS['warning']} Error saat validasi num_workers: {str(e)}, menggunakan default: {num_workers}")
            
            # Pastikan UI components terdaftar untuk persistensi
            try:
                from smartcash.ui.utils.persistence_utils import ensure_ui_persistence
                ensure_ui_persistence(ui_components, 'augmentation', logger)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat memastikan persistensi UI: {str(e)}")
            
            # Simpan konfigurasi augmentasi terbaru
            try:
                aug_config = {
                    'augmentation': {
                        'types': aug_types,
                        'split': split_option,
                        'prefix': aug_prefix,
                        'factor': aug_factor,
                        'balance_classes': balance_classes,
                        'num_workers': num_workers
                    }
                }
                config_manager.save_module_config('augmentation', aug_config)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat menyimpan konfigurasi: {str(e)}")
            
            # Log parameter final yang akan digunakan
            if logger: logger.info(f"{ICONS['info']} Parameter augmentasi final: aug_types=Combined (Recommended), split=train, factor={aug_factor}, prefix={aug_prefix}, balance={balance_classes}")
            
            # Lakukan augmentasi dengan parameter yang sudah divalidasi dan error handling yang lebih baik
            try:
                # Log parameter yang akan digunakan untuk augmentasi
                if logger: logger.info(f"{ICONS['info']} Menjalankan augmentasi dengan parameter: split=train, types=Combined, factor={aug_factor}, prefix={aug_prefix}, balance={balance_classes}, workers={num_workers}")
                
                # Pastikan input dan output direktori ada
                import os
                from pathlib import Path
                
                # Validasi direktori input
                input_dir = Path(preprocessed_dir) / 'train' / 'images'
                if not input_dir.exists():
                    input_dir = Path(preprocessed_dir) / 'images'
                    if not input_dir.exists():
                        input_dir = Path(preprocessed_dir)
                        if not input_dir.exists():
                            os.makedirs(str(input_dir), exist_ok=True)
                            if logger: logger.warning(f"{ICONS['warning']} Membuat direktori input: {input_dir}")
                
                # Validasi direktori output
                output_dir = Path(augmented_dir)
                if not output_dir.exists():
                    os.makedirs(str(output_dir), exist_ok=True)
                    if logger: logger.info(f"{ICONS['info']} Membuat direktori output: {output_dir}")
                
                # Update status dengan informasi direktori
                with ui_components['status']:
                    display(create_status_indicator("info", f"{ICONS['processing']} Augmentasi dari {input_dir} ke {output_dir}..."))
                
                # Jalankan augmentasi dengan try-except yang lebih detail
                try:
                    result = augmentation_service.augment_dataset(
                        split='train',  # Nilai tetap
                        augmentation_types=['Combined (Recommended)'],  # Nilai tetap
                        num_variations=aug_factor,
                        output_prefix=aug_prefix,
                        target_balance=balance_classes,
                        num_workers=num_workers
                    )
                    
                    if logger: logger.success(f"{ICONS['success']} Augmentasi berhasil dijalankan")
                except AttributeError as attr_error:
                    if logger: logger.error(f"{ICONS['error']} AttributeError saat augmentasi: {str(attr_error)}")
                    # Coba jalankan dengan cara alternatif
                    if logger: logger.info(f"{ICONS['refresh']} Mencoba metode alternatif untuk augmentasi...")
                    
                    # Import langsung dari modul
                    from smartcash.dataset.services.augmentor.augmentor import augment_dataset as direct_augment
                    
                    result = direct_augment(
                        input_dir=str(input_dir),
                        output_dir=str(output_dir),
                        augmentation_types=['Combined (Recommended)'],
                        num_variations=aug_factor,
                        output_prefix=aug_prefix,
                        target_balance=balance_classes,
                        num_workers=num_workers,
                        logger=logger
                    )
                    
                    if logger: logger.success(f"{ICONS['success']} Augmentasi berhasil dengan metode alternatif")
            except Exception as e:
                if logger: logger.error(f"{ICONS['error']} Error saat melakukan augmentasi: {str(e)}")
                # Tampilkan detail error untuk debugging
                import traceback
                if logger: logger.debug(f"Traceback: {traceback.format_exc()}")
                raise  # Re-raise exception untuk ditangkap oleh blok except di luar
            
            # Tambahkan path output jika tidak ada
            try:
                if 'output_dir' not in result:
                    result['output_dir'] = augmented_dir
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat menambahkan path output: {str(e)}")
            
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
            ui_components['visualization_buttons'].layout.display = 'flex'
            ui_components['cleanup_button'].layout.display = 'block'
            
            # Generate dan tampilkan ringkasan augmentasi
            try:
                from smartcash.ui.dataset.augmentation.handlers.state_handler import generate_augmentation_summary
                generate_augmentation_summary(ui_components, preprocessed_dir, augmented_dir)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat generate ringkasan augmentasi: {str(e)}")
            
            # Notifikasi observer tentang selesai
            try:
                _notify_process_complete(ui_components, result, split_info)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat notifikasi proses selesai: {str(e)}")
            
        except Exception as e:
            # Handle error dengan notifikasi
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal: {str(e)}")
            
            # Notifikasi observer tentang error
            _notify_process_error(ui_components, str(e))
            
            # Log error
            if logger: logger.error(f"{ICONS['error']} Error saat augmentasi dataset: {str(e)}")
        
        finally:
            # Tandai augmentasi selesai
            ui_components['augmentation_running'] = False
            
            # Restore UI
            cleanup_ui(ui_components)
    
    # Handler untuk tombol stop dengan notifikasi yang ditingkatkan
    def on_stop_click(b):
        """Handler untuk menghentikan augmentasi."""
        # Set flag untuk menghentikan augmentasi
        ui_components['augmentation_running'] = False
        
        # Tampilkan pesan di status
        with ui_components['status']: 
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        
        # Update status panel
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
        
        # Notifikasi observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.AUGMENTATION_END,
                sender="augmentation_handler",
                message=f"Augmentasi dihentikan oleh pengguna",
                status="cancelled"
            )
        except ImportError:
            pass
        
        # Reset UI
        cleanup_ui(ui_components)
    
    # Reset handler untuk reset button
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Validasi ui_components
        if ui_components is None:
            if logger: logger.error(f"{ICONS['error']} ui_components adalah None saat reset augmentation")
            return
            
        # Reset UI
        reset_ui(ui_components)
        
        # Load konfigurasi default dan update UI
        try:
            # Pastikan fungsi load_augmentation_config tersedia
            if 'load_augmentation_config' not in ui_components or not callable(ui_components.get('load_augmentation_config')):
                # Jika tidak tersedia, import langsung
                from smartcash.ui.dataset.augmentation.handlers.config_handlers import load_augmentation_config
                default_config = load_augmentation_config(ui_components=ui_components)
            else:
                # Pastikan parameter ui_components diteruskan ke load_augmentation_config
                default_config = ui_components['load_augmentation_config'](ui_components=ui_components)
            
            # Pastikan fungsi update_ui_from_config tersedia
            if 'update_ui_from_config' not in ui_components or not callable(ui_components.get('update_ui_from_config')):
                # Jika tidak tersedia, import langsung
                from smartcash.ui.dataset.augmentation.handlers.config_handlers import update_ui_from_config
                update_ui_from_config(ui_components, default_config)
            else:
                ui_components['update_ui_from_config'](ui_components, default_config)
            
            # Deteksi state augmentation
            from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
            detect_augmentation_state(ui_components)
            
            # Tampilkan pesan sukses dengan validasi
            if 'status' in ui_components and ui_components['status'] is not None:
                try:
                    with ui_components['status']: 
                        display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
                except Exception as status_error:
                    if logger: logger.warning(f"{ICONS['warning']} Error saat update status: {str(status_error)}")
            
            # Update status panel dengan validasi
            try:
                update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            except Exception as panel_error:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update status panel: {str(panel_error)}")
            
            # Log success
            if logger: logger.success(f"{ICONS['success']} Konfigurasi augmentasi berhasil direset ke nilai default")
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
                update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            except Exception:
                pass
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Register handlers untuk tombol-tombol dengan one-liner
    button_handlers = {
        'augment_button': on_augment_click,
        'stop_button': on_stop_click,
        'reset_button': on_reset_click
    }
    
    [ui_components[button].on_click(handler) for button, handler in button_handlers.items() if button in ui_components]
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_augment_click': on_augment_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'augmentation_running': False
    })
    
    return ui_components

def disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Enable/disable komponen UI saat augmentasi berjalan.
    
    Args:
        ui_components: Dictionary komponen UI 
        disable: Apakah perlu di-disable
    """
    # Disable komponen utama
    aug_options = ui_components.get('aug_options')
    if aug_options is not None and hasattr(aug_options, 'children'):
        for child in aug_options.children:
            if hasattr(child, 'disabled'):
                child.disabled = disable
    
    # Disable tombol-tombol
    for btn in ['save_button', 'reset_button', 'cleanup_button']:
        if btn in ui_components:
            ui_components[btn].disabled = disable

def cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan UI ke kondisi awal setelah augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Enable kembali semua UI component
    disable_ui_during_processing(ui_components, False)
    
    # Kembalikan tampilan tombol
    ui_components['augment_button'].layout.display = 'block'
    ui_components['stop_button'].layout.display = 'none'
    
    # Reset progress bar
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()
    else:
        # Fallback jika fungsi reset tidak tersedia
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                if hasattr(ui_components[element], 'layout') and hasattr(ui_components[element].layout, 'visibility'):
                    ui_components[element].layout.visibility = 'hidden'
                if hasattr(ui_components[element], 'value') and element in ['progress_bar', 'current_progress']:
                    ui_components[element].value = 0

def reset_ui(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI ke kondisi default.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset tombol dan progress
    cleanup_ui(ui_components)
    
    # Hide containers
    for component in ['visualization_container', 'summary_container']:
        if component in ui_components:
            ui_components[component].layout.display = 'none'
            with ui_components[component]: 
                clear_output()
    
    # Hide buttons
    for btn in ['visualization_buttons', 'cleanup_button']:
        if btn in ui_components:
            ui_components[btn].layout.display = 'none'
    
    # Reset logs
    if 'status' in ui_components:
        with ui_components['status']: 
            clear_output()
    
    # Reset accordion
    if 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = None

def get_dataset_manager(ui_components: Dict[str, Any], config: Dict[str, Any] = None, logger=None):
    """
    Dapatkan dataset manager dengan fallback yang terstandarisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
        
    Returns:
        Dataset manager instance
    """
    # Cek apakah sudah ada di ui_components
    if 'augmentation_manager' in ui_components and ui_components['augmentation_manager']:
        return ui_components['augmentation_manager']
    
    # Gunakan fallback_utils untuk konsistensi
    try:
        from smartcash.ui.utils.fallback_utils import get_augmentation_manager
        augmentation_manager = get_augmentation_manager(config, logger)
        
        # Register progress callback jika tersedia
        try:
            if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
                ui_components['register_progress_callback'](augmentation_manager)
        except Exception:
            pass
        
        # Simpan ke ui_components
        ui_components['augmentation_manager'] = augmentation_manager
        
        return augmentation_manager
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Gagal membuat augmentation manager: {str(e)}")
        return None