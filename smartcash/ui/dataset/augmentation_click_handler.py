"""
File: smartcash/ui/dataset/augmentation_click_handler.py
Deskripsi: Handler tombol dan interaksi UI untuk augmentasi dataset dengan pendekatan DRY dan one-liner
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol UI augmentasi dengan pendekatan DRY."""
    
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Penanganan error dengan decorator
    from smartcash.ui.handlers.error_handler import try_except_decorator

    @try_except_decorator(ui_components.get('status'))
    def on_augment_click(b):
        """Handler tombol augmentasi dengan alur augmentasi yang menggunakan pendekatan one-liner."""
        # Dapatkan augmentation types dari UI dengan one-liner
        aug_types_widgets = ui_components['aug_options'].children[0].value
        
        # Persiapkan augmentasi dengan utilitas UI standar
        from smartcash.ui.dataset.augmentation_initialization import update_status_panel
        from smartcash.ui.utils.alert_utils import create_status_indicator

        # Update UI untuk menunjukkan proses dimulai dengan status indicator
        with ui_components['status']: clear_output(wait=True); display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi dataset..."))
    
        # Tampilkan log panel, progress bar dan perbarui UI tombol
        ui_components['log_accordion'].selected_index = 0
        ui_components['augment_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        ui_components['progress_bar'].layout.visibility = ui_components['current_progress'].layout.visibility = 'visible'
        
        # Nonaktifkan tombol lain selama proses berjalan dengan one-liner untuk tombol-tombol standard
        [setattr(ui_components[btn], 'disabled', True) for btn in ['reset_button', 'save_button', 'cleanup_button'] if btn in ui_components]
        
        # Update konfigurasi dari UI secara efisien
        try:
            from smartcash.ui.dataset.augmentation_config_handler import update_config_from_ui, save_augmentation_config
            updated_config = update_config_from_ui(ui_components, config); save_augmentation_config(updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Map UI types to config format dengan dictionary comprehension
        type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 
                   'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
        aug_types = [type_map.get(t, 'combined') for t in aug_types_widgets]
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{ICONS['processing']} Augmentasi dataset dengan jenis: {', '.join(aug_types)}...")
        
        # Notifikasi observer tentang mulai augmentasi
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(event_type=EventTopics.AUGMENTATION_START, sender="augmentation_handler", 
                  message=f"Memulai augmentasi dataset dengan jenis: {', '.join(aug_types)}")
        except ImportError: pass
        
        # Dapatkan augmentation manager dan return jika tidak tersedia
        if not (augmentation_manager := ui_components.get('augmentation_manager')):
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} AugmentationManager tidak tersedia"))
            cleanup_ui(); return
        
        # Ambil semua parameter dari UI dan set ke manager
        variations, prefix, process_bboxes, validate = [ui_components['aug_options'].children[i].value for i in range(1, 5)]
        num_workers = ui_components['aug_options'].children[5].value if len(ui_components['aug_options'].children) > 5 else 4
        target_balance = ui_components['aug_options'].children[6].value if len(ui_components['aug_options'].children) > 6 else False
        
        # Tetapkan num_workers ke augmentation_manager
        augmentation_manager.num_workers = num_workers
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](augmentation_manager)
        
        # Tandai augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Jalankan augmentasi dengan alur baru
        try:
            # Jalankan augmentasi dengan one-liner parameter
            result = augmentation_manager.augment_dataset(split='train', augmentation_types=aug_types,
                                                         num_variations=variations, output_prefix=prefix, 
                                                         validate_results=validate, resume=False, 
                                                         process_bboxes=process_bboxes, target_balance=target_balance,
                                                         num_workers=num_workers, move_to_preprocessed=True)
            
            # Proses hasil sukses
            if result.get("status") != "error":
                # Update UI dengan hasil sukses
                with ui_components['status']: clear_output(wait=True); display(create_status_indicator("success", f"{ICONS['success']} Augmentasi dataset selesai"))
                
                # Update summary dan status panel
                if 'update_summary' in ui_components and callable(ui_components['update_summary']): ui_components['update_summary'](result)
                update_status_panel(ui_components, "success", 
                    f"{ICONS['success']} Augmentasi dataset berhasil dengan {result.get('generated', 0)} gambar baru, dipindahkan ke {result.get('final_output_dir', 'preprocessed')}")
                
                # Tampilkan tombol visualisasi dan cleanup
                ui_components['visualization_buttons'].layout.display = 'flex'
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
                # Notifikasi observer tentang selesai augmentasi
                try:
                    from smartcash.components.observer import notify
                    notify(event_type=EventTopics.AUGMENTATION_END, sender="augmentation_handler",
                         message=f"Augmentasi dataset selesai dengan {result.get('generated', 0)} gambar baru", result=result)
                except ImportError: pass
            else:
                # Tangani error dengan menampilkan pesan error
                with ui_components['status']: clear_output(wait=True); display(create_status_indicator("error", f"{ICONS['error']} Error: {result.get('message', 'Unknown error')}"))
                update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal: {result.get('message', 'Unknown error')}")
                
                # Notifikasi observer tentang error
                try:
                    from smartcash.components.observer import notify
                    notify(event_type=EventTopics.AUGMENTATION_ERROR, sender="augmentation_handler",
                         message=f"Error saat augmentasi: {result.get('message', 'Unknown error')}")
                except ImportError: pass
        except Exception as e:
            # Handle other errors dengan menampilkan pesan error
            with ui_components['status']: display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            update_status_panel(ui_components, "error", f"{ICONS['error']} Augmentasi gagal dengan error: {str(e)}")
            
            # Notifikasi observer tentang error
            try:
                from smartcash.components.observer import notify
                notify(event_type=EventTopics.AUGMENTATION_ERROR, sender="augmentation_handler", message=f"Error saat augmentasi: {str(e)}")
            except ImportError: pass
        finally:
            # Tandai augmentasi selesai dan restore UI
            ui_components['augmentation_running'] = False
            cleanup_ui()
    
    # Handler untuk tombol stop
    def on_stop_click(b):
        """Handler untuk menghentikan augmentasi dengan pendekatan one-liner."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.augmentation_initialization import update_status_panel
        
        # Set flag berhenti, tampilkan pesan dan update status
        ui_components['augmentation_running'] = False
        with ui_components['status']: clear_output(wait=True); display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        update_status_panel(ui_components, "warning", f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna")
        
        # Notifikasi observer dan reset UI
        try:
            from smartcash.components.observer import notify
            notify(event_type=EventTopics.AUGMENTATION_END, sender="augmentation_handler", message=f"Augmentasi dihentikan oleh pengguna")
        except ImportError: pass
        cleanup_ui()
    
    # Handler untuk tombol reset
    def on_reset_click(b):
        """Handler untuk reset UI ke kondisi awal dengan pendekatan one-liner."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.augmentation_initialization import detect_augmentation_state

        # Reset UI dan load konfigurasi default
        reset_ui()
        try:
            # Load konfigurasi default dan update UI dari konfigurasi default
            from smartcash.ui.dataset.augmentation_config_handler import load_default_augmentation_config, update_ui_from_config
            update_ui_from_config(ui_components, load_default_augmentation_config())
            
            # Re-detect state dan tampilkan pesan sukses
            detect_augmentation_state(ui_components, env, config)
            with ui_components['status']: clear_output(wait=True); display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
            
            # Update status panel
            from smartcash.ui.dataset.augmentation_initialization import update_status_panel
            update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
            
            # Log success jika logger tersedia
            if logger: logger.success(f"{ICONS['success']} Konfigurasi augmentasi berhasil direset ke nilai default")
        except Exception as e:
            with ui_components['status']: clear_output(wait=True); display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Function untuk cleanup UI setelah augmentasi
    def cleanup_ui():
        """Kembalikan UI ke kondisi operasional setelah augmentasi dengan pendekatan one-liner."""
        # Reset tombol dan progress dengan one-liner
        ui_components['augment_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        [setattr(ui_components[btn], 'disabled', False) for btn in ['reset_button', 'save_button', 'cleanup_button'] if btn in ui_components]
        
        # Reset progress bar dengan one-liner jika tersedia
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback jika fungsi reset tidak tersedia
            ui_components['progress_bar'].value = ui_components['current_progress'].value = 0
            ui_components['progress_bar'].layout.visibility = ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Function untuk reset komplet UI
    def reset_ui():
        """Reset UI ke kondisi default total dengan pendekatan one-liner."""
        # Reset tombol dan progress
        cleanup_ui()
        
        # Reset components dengan one-liner
        for component in ['summary_container', 'visualization_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'none'
                with ui_components[component]: clear_output()
        
        # Hide buttons dengan one-liner
        for btn in ['visualization_buttons', 'cleanup_button']:
            if btn in ui_components: ui_components[btn].layout.display = 'none'
        
        # Reset logs dan accordion
        if 'status' in ui_components:
            with ui_components['status']: clear_output()
        if 'log_accordion' in ui_components:
            ui_components['log_accordion'].selected_index = None
    
    # Register handlers untuk tombol-tombol dengan one-liner
    for button, handler in [('augment_button', on_augment_click), ('stop_button', on_stop_click), ('reset_button', on_reset_click)]:
        if button in ui_components: ui_components[button].on_click(handler)
    
    # Tambahkan referensi ke handlers di ui_components
    ui_components.update({
        'on_augment_click': on_augment_click,
        'on_stop_click': on_stop_click,
        'on_reset_click': on_reset_click,
        'cleanup_ui': cleanup_ui,
        'reset_ui': reset_ui,
        'augmentation_running': False
    })
    
    return ui_components