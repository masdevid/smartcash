"""
File: smartcash/ui/dataset/augmentation/handlers/button_handlers.py
Deskripsi: Handler tombol untuk augmentasi dataset
"""

from typing import Dict, Any, Optional, List
import os
from pathlib import Path
import time
from IPython.display import display, clear_output
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.shared.status_panel import update_status_panel

def setup_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol UI augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Helper untuk disable/enable UI selama proses
    def toggle_ui_enabled(disabled: bool = True) -> None:
        """
        Toggle disabled state untuk komponen UI.
        
        Args:
            disabled: Flag untuk disable/enable
        """
        # Disable/enable ui options
        if 'aug_options' in ui_components:
            ui_components['aug_options'].disabled = disabled
        
        # Disable/enable tombol-tombol
        for btn in ['save_button', 'reset_button', 'cleanup_button', 'visualize_button', 
                   'compare_button', 'distribution_button']:
            if btn in ui_components:
                ui_components[btn].disabled = disabled
                
        # Toggle tombol augment dan stop
        if 'augment_button' in ui_components:
            ui_components['augment_button'].layout.display = 'none' if disabled else 'block'
        if 'stop_button' in ui_components:
            ui_components['stop_button'].layout.display = 'block' if disabled else 'none'
    
    # Handler untuk tombol augmentasi
    def on_augment_click(b):
        """Handler untuk tombol augmentasi dataset."""
        # Tampilkan status dan nonaktifkan UI
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi dataset..."))
        
        # Tampilkan log panel dan progress bar
        ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Tampilkan progress bar
        for element in ['progress_bar', 'current_progress', 'overall_message', 'step_message']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Tandai augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Disable UI selama proses
        toggle_ui_enabled(True)
        
        # Update konfigurasi dari UI dan simpan
        try:
            updated_config = ui_components['update_config_from_ui'](ui_components, config)
            ui_components['save_augmentation_config'](updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Memulai augmentasi dataset...")
        
        # Cek paths yang diperlukan
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        # Validasi path preprocessed ada dan berisi data
        if not os.path.exists(preprocessed_dir):
            error_msg = f"Direktori preprocessing tidak ditemukan: {preprocessed_dir}"
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} {error_msg}"))
            update_status_panel(ui_components, "error", f"{ICONS['error']} {error_msg}")
            cleanup_ui()
            return
        
        # Notifikasi observer tentang mulai augmentasi
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.AUGMENTATION_START,
                sender="augmentation_handler",
                message="Memulai augmentasi dataset"
            )
        except ImportError:
            pass
        
        # Create AugmentationManager jika belum ada
        augmentation_manager = get_augmentation_manager(ui_components, logger)
        if not augmentation_manager:
            error_msg = "Tidak dapat membuat Augmentation Manager"
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} {error_msg}"))
            update_status_panel(ui_components, "error", f"{ICONS['error']} {error_msg}")
            cleanup_ui()
            return
        
        # Ekstrak opsi dari UI dengan one-liner
        types, variations, prefix, process_bboxes, validate_results, workers, balance = [
            ui_components['aug_options'].children[i].value for i in range(7)
        ]
        
        # Map UI types ke config types
        type_map = {
            'Combined (Recommended)': 'combined', 
            'Position Variations': 'position', 
            'Lighting Variations': 'lighting', 
            'Extreme Rotation': 'extreme_rotation'
        }
        aug_types = [type_map.get(t, 'combined') for t in types]
        
        # Jalankan augmentasi dengan threading
        def run_augmentation():
            try:
                # Log awal augmentasi
                if logger: logger.info(f"{ICONS['start']} Memulai augmentasi dataset dengan tipe: {', '.join(aug_types)}")
                
                # Jalankan augmentasi
                start_time = time.time()
                
                # Update path pada manager jika perlu
                augmentation_manager.set_paths(preprocessed_dir, augmented_dir)
                
                # Jalankan augmentasi dengan parameter dari UI
                result = augmentation_manager.run_augmentation(
                    aug_types=aug_types,
                    variations=variations,
                    prefix=prefix,
                    process_bboxes=process_bboxes,
                    validate=validate_results,
                    num_workers=workers,
                    balance=balance
                )
                
                # Hitung waktu proses
                processing_time = time.time() - start_time
                
                # Update result dengan informasi tambahan
                result['duration'] = processing_time
                result['augmentation_types'] = aug_types
                result['output_dir'] = augmented_dir
                
                # Tampilkan status sukses
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator(
                        "success", 
                        f"{ICONS['success']} Augmentasi selesai dalam {processing_time:.2f} detik. "
                        f"Total {result.get('total_files', 0)} file."
                    ))
                
                # Update status panel
                update_status_panel(ui_components, "success", 
                                  f"{ICONS['success']} Augmentasi berhasil diselesaikan")
                
                # Tampilkan summary
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    ui_components['update_summary'](result)
                    ui_components['summary_container'].layout.display = 'block'
                
                # Tampilkan tombol visualisasi dan cleanup
                ui_components['visualization_buttons'].layout.display = 'flex'
                ui_components['cleanup_button'].layout.display = 'block'
                
                # Notifikasi observer
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.AUGMENTATION_END,
                        sender="augmentation_handler",
                        message="Augmentasi dataset selesai",
                        result=result,
                        duration=processing_time
                    )
                except ImportError:
                    pass
                
            except Exception as e:
                # Handle error
                error_message = f"Error saat augmentasi: {str(e)}"
                
                with ui_components['status']: 
                    display(create_status_indicator("error", f"{ICONS['error']} {error_message}"))
                
                update_status_panel(ui_components, "error", f"{ICONS['error']} {error_message}")
                
                # Notifikasi observer
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.AUGMENTATION_ERROR,
                        sender="augmentation_handler",
                        message=error_message,
                        error=str(e)
                    )
                except ImportError:
                    pass
                
                if logger: logger.error(f"{ICONS['error']} {error_message}")
            
            finally:
                # Selalu cleanup UI dan tandai telah selesai
                ui_components['augmentation_running'] = False
                cleanup_ui()
        
        # Jalankan augmentasi dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_augmentation)
    
    # Handler untuk tombol stop
    def on_stop_click(b):
        """Handler untuk menghentikan augmentasi."""
        # Set flag stop
        ui_components['augmentation_running'] = False
        
        # Tampilkan status
        with ui_components['status']: 
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
        
        # Notifikasi observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.AUGMENTATION_END, 
                sender="augmentation_handler",
                message="Augmentasi dihentikan oleh pengguna",
                status="cancelled"
            )
        except ImportError:
            pass
        
        # Cleanup UI
        cleanup_ui()
    
    # Handler untuk tombol reset
    def on_reset_click(b):
        """Reset UI dan konfigurasi."""
        # Reset UI
        reset_ui()
        
        # Load konfigurasi default dan update UI
        try:
            default_config = ui_components['load_augmentation_config']()
            ui_components['update_ui_from_config'](ui_components, default_config)
            
            # Detect state augmentation
            from smartcash.ui.dataset.shared.setup_utils import detect_module_state
            detect_module_state(ui_components, 'augmentation')
            
            # Tampilkan pesan sukses
            with ui_components['status']: 
                display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset"))
            
            # Update status panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            
            if logger: logger.success(f"{ICONS['success']} Konfigurasi augmentation berhasil direset")
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            
            update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset UI sebagian: {str(e)}")
            
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Fungsi untuk cleanup UI setelah augmentasi selesai
    def cleanup_ui():
        """Kembalikan UI ke kondisi operasional."""
        # Aktifkan kembali komponen UI
        toggle_ui_enabled(False)
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback manual
            for element in ['progress_bar', 'current_progress', 'overall_message', 'step_message']:
                if element in ui_components:
                    if hasattr(ui_components[element], 'value'):
                        ui_components[element].value = 0
                    if hasattr(ui_components[element], 'layout') and hasattr(ui_components[element].layout, 'visibility'):
                        ui_components[element].layout.visibility = 'hidden'
    
    # Fungsi untuk reset UI ke kondisi default
    def reset_ui():
        """Reset semua komponen UI ke kondisi default."""
        # Reset tombol dan progress
        cleanup_ui()
        
        # Hide containers
        for component in ['visualization_container', 'summary_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'none'
                with ui_components[component]: 
                    clear_output()
        
        # Hide tombol
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
    
    # Helper untuk mendapatkan AugmentationManager
    def get_augmentation_manager(ui_components: Dict[str, Any], logger) -> Any:
        """
        Dapatkan AugmentationManager dengan fallback.
        
        Args:
            ui_components: Dictionary komponen UI
            logger: Logger untuk logging
            
        Returns:
            AugmentationManager atau None jika gagal
        """
        # Cek jika sudah ada
        if 'augmentation_manager' in ui_components and ui_components['augmentation_manager']:
            return ui_components['augmentation_manager']
        
        try:
            # Coba import AugmentationService
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Dapatkan konfigurasi
            config = ui_components.get('config', {})
            
            # Dapatkan direktori data
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Dapatkan num_workers dari UI
            num_workers = 4  # Default
            if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 5:
                num_workers = ui_components['aug_options'].children[5].value
            
            # Buat instance
            augmentation_manager = AugmentationService(config, preprocessed_dir, logger, num_workers)
            
            # Register progress callback jika tersedia
            if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
                ui_components['register_progress_callback'](augmentation_manager)
            
            # Simpan ke ui_components
            ui_components['augmentation_manager'] = augmentation_manager
            
            return augmentation_manager
        except Exception as e:
            if logger: 
                logger.error(f"{ICONS['error']} Gagal membuat Augmentation Manager: {str(e)}")
            return None
    
    # Register handlers untuk tombol-tombol
    ui_components['augment_button'].on_click(on_augment_click)
    ui_components['stop_button'].on_click(on_stop_click)
    ui_components['reset_button'].on_click(on_reset_click)
    
    # Tambahkan fungsi-fungsi ke ui_components
    ui_components.update({
        'on_augment_click': on_augment_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'toggle_ui_enabled': toggle_ui_enabled,
        'cleanup_ui': cleanup_ui,
        'reset_ui': reset_ui,
        'get_augmentation_manager': get_augmentation_manager
    })
    
    return ui_components