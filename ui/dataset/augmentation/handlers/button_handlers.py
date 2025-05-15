"""
File: smartcash/ui/dataset/augmentation/handlers/button_handlers.py
Deskripsi: Handler tombol untuk augmentasi dataset dengan pendekatan SRP
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import os
import traceback
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel

# Import handler terpisah untuk SRP
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence, save_augmentation_config, get_augmentation_config
)
from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_mapper import (
    map_ui_to_config, map_config_to_ui, extract_augmentation_params
)
from smartcash.ui.dataset.augmentation.handlers.observer_handler import (
    notify_process_start, notify_process_complete, notify_process_error
)

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
    # Pastikan persistensi UI components
    ensure_ui_persistence(ui_components)
    logger = ui_components.get('logger')
    
    # Import handler terpisah untuk SRP
    from smartcash.ui.dataset.augmentation.handlers.progress_handler import (
        register_progress_callback, reset_progress_tracking
    )
    from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import (
        get_augmentation_service, execute_augmentation, stop_augmentation
    )
    
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
        
        # Update konfigurasi dari UI dan simpan dengan ConfigManager
        try:
            # Gunakan config mapper untuk update konfigurasi dari UI
            updated_config = map_ui_to_config(ui_components, config)
            success = save_augmentation_config(updated_config)
            if success and logger:
                logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
            else:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi augmentasi")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Tandai augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Update status panel dengan informasi awal
        split_info = f"Split {split_option}"
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai augmentasi {split_info}...")
        
        # Notifikasi observer tentang mulai augmentasi
        notify_process_start(ui_components, "augmentasi", split_info, split_option)
        
        # Verifikasi direktori dataset
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Ekstrak parameter augmentasi dari UI menggunakan config mapper
        augmentation_params = extract_augmentation_params(ui_components)
        
        # Log parameter yang akan digunakan
        if logger:
            logger.info(f"{ICONS['info']} Parameter augmentasi: ")
            logger.info(f"  - Split: {augmentation_params['split']}")
            logger.info(f"  - Jenis: {augmentation_params['augmentation_types']}")
            logger.info(f"  - Jumlah variasi: {augmentation_params['num_variations']}")
            logger.info(f"  - Prefix: {augmentation_params['output_prefix']}")
            logger.info(f"  - Balance kelas: {augmentation_params['target_balance']}")
            logger.info(f"  - Target count: {augmentation_params['target_count']}")
            logger.info(f"  - Workers: {augmentation_params['num_workers']}")
            logger.info(f"  - Pindahkan ke preprocessed: {augmentation_params['move_to_preprocessed']}")
        
        # Pastikan direktori input dan output ada
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
        
        # Setup augmentation service menggunakan handler terpisah
        try:
            # Dapatkan augmentation service dari handler
            augmentation_service = get_augmentation_service(ui_components, config)
            
            # Register progress callback dari handler terpisah
            register_progress_callback(augmentation_service, ui_components)
            
            # Log awal augmentasi
            if logger: logger.info(f"{ICONS['start']} Memulai augmentasi {split_info}")
            
            # Eksekusi augmentasi dengan parameter dari config mapper
            result = execute_augmentation(augmentation_service, augmentation_params)
            
            # Cek hasil augmentasi
            if result['status'] == 'success':
                if logger: logger.success(f"{ICONS['success']} Augmentasi berhasil dijalankan")
            elif result['status'] == 'error':
                error_message = result.get('message', 'Terjadi kesalahan saat augmentasi')
                if logger: logger.error(f"{ICONS['error']} {error_message}")
                raise Exception(error_message)
            
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
                notify_process_complete(ui_components, result, split_info)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat notifikasi proses selesai: {str(e)}")
            
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
        
        # Hentikan augmentasi menggunakan handler terpisah
        try:
            # Dapatkan augmentation service dari ui_components
            augmentation_service = ui_components.get('augmentation_service')
            if augmentation_service:
                # Gunakan handler untuk menghentikan augmentasi
                stop_augmentation(augmentation_service)
                if logger: logger.info(f"{ICONS['stop']} Augmentasi dihentikan oleh pengguna")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat menghentikan augmentasi: {str(e)}")
        
        # Notifikasi observer tentang pembatalan
        notify_process_error(ui_components, "Augmentasi dihentikan oleh pengguna")
        
        # Reset UI
        cleanup_ui(ui_components)
    
    # Reset handler untuk reset button
    @try_except_decorator(ui_components.get('status'))
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Tampilkan status
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Mereset konfigurasi ke default..."))
        
        try:
            # Definisikan default config
            default_config = {
                'augmentation': {
                    'prefix': 'aug',
                    'factor': 2,
                    'balance_classes': True,
                    'num_workers': 4,
                    'techniques': {
                        'flip': True,
                        'rotate': True,
                        'blur': False,
                        'noise': False,
                        'contrast': False,
                        'brightness': False,
                        'saturation': False,
                        'hue': False,
                        'cutout': False
                    },
                    'advanced': {
                        'rotate_range': 15,
                        'blur_limit': 7,
                        'noise_var': 25,
                        'contrast_limit': 0.2,
                        'brightness_limit': 0.2,
                        'saturation_limit': 0.2,
                        'hue_shift_limit': 20,
                        'cutout_size': 0.1,
                        'cutout_count': 4
                    }
                }
            }
            
            # Log informasi konfigurasi default
            if logger: logger.info(f"{ICONS['info']} Menggunakan konfigurasi default: {default_config['augmentation']}")
            
            # Simpan default config ke ConfigManager menggunakan handler terpisah
            try:
                # Validasi konfigurasi default
                validated_config = validate_augmentation_config(default_config)
                
                # Simpan konfigurasi yang sudah divalidasi
                success = save_augmentation_config(validated_config)
                
                if success:
                    if logger: logger.info(f"{ICONS['success']} Konfigurasi default berhasil disimpan")
                else:
                    if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi default")
            except Exception as e:
                if logger: logger.error(f"{ICONS['error']} Error saat menyimpan konfigurasi default: {str(e)}")

            # Update UI dari konfigurasi default menggunakan handler terpisah
            try:
                map_config_to_ui(ui_components, default_config)
                if logger: logger.info(f"{ICONS['success']} UI berhasil diupdate dari konfigurasi default")
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Error saat update UI dari konfigurasi default: {str(e)}")
            
            # Update status panel
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi berhasil direset ke default"))
            
            update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi augmentasi berhasil direset ke default")
            
        except Exception as e:
            # Update status
            with ui_components['status']: 
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error saat mereset konfigurasi: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error saat mereset konfigurasi: {str(e)}")
            
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
        # Disable tab container (children[1])
        if len(aug_options.children) >= 2 and hasattr(aug_options.children[1], 'children'):
            tab = aug_options.children[1]
            # Disable semua tab
            for tab_child in tab.children:
                if hasattr(tab_child, 'children'):
                    for child in tab_child.children:
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
    
    # Reset progress bar menggunakan handler terpisah
    try:
        from smartcash.ui.dataset.augmentation.handlers.progress_handler import reset_progress_tracking
        reset_progress_tracking(ui_components)
    except Exception as e:
        # Fallback jika import gagal
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
# Fungsi get_dataset_manager telah dipindahkan ke augmentation_service_handler.py
# Gunakan get_augmentation_service dari augmentation_service_handler.py sebagai gantinya