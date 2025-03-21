"""
File: smartcash/ui/dataset/augmentation_click_handler.py
Deskripsi: Handler tombol dan interaksi UI untuk augmentasi dataset dengan dukungan balancing kelas dan sumber dari preprocessed
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol UI augmentasi dengan pengelompokan yang lebih baik."""
    
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Penanganan error dengan decorator
    from smartcash.ui.handlers.error_handler import try_except_decorator
    
    @try_except_decorator(ui_components.get('status'))
    def on_augment_click(b):
        """Handler tombol augmentasi dengan error handling standar."""
        # Dapatkan augmentation types dari UI
        aug_types_widgets = ui_components['aug_options'].children[0].value
        
        # Persiapkan augmentasi dengan utilitas UI standar
        from smartcash.ui.dataset.augmentation_initialization import update_status_panel
        from smartcash.ui.utils.alert_utils import create_status_indicator

        # Update UI untuk menunjukkan proses dimulai
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['processing']} Memulai augmentasi dataset..."))
        
        # Tampilkan log panel
        ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Update UI: sembunyikan tombol augment, tampilkan tombol stop
        ui_components['augment_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        
        # Nonaktifkan tombol lain selama proses berjalan
        ui_components['reset_button'].disabled = True
        ui_components['save_button'].disabled = True
        ui_components['cleanup_button'].disabled = True
        
        # Tampilkan progress bar
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['current_progress'].layout.visibility = 'visible'
        
        # Update konfigurasi dari UI
        try:
            from smartcash.ui.dataset.augmentation_config_handler import update_config_from_ui, save_augmentation_config
            updated_config = update_config_from_ui(ui_components, config)
            save_augmentation_config(updated_config)
            if logger: logger.info(f"{ICONS['success']} Konfigurasi augmentasi berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Map UI types to config format
        type_map = {
            'Combined (Recommended)': 'combined',
            'Position Variations': 'position',
            'Lighting Variations': 'lighting',
            'Extreme Rotation': 'extreme_rotation'
        }
        aug_types = [type_map.get(t, 'combined') for t in aug_types_widgets]
        
        # Update status panel
        update_status_panel(
            ui_components, 
            "info", 
            f"{ICONS['processing']} Augmentasi dataset dengan jenis: {', '.join(aug_types)}..."
        )
        
        # Notifikasi observer tentang mulai augmentasi
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.AUGMENTATION_START,
                sender="augmentation_handler",
                message=f"Memulai augmentasi dataset dengan jenis: {', '.join(aug_types)}"
            )
        except ImportError:
            pass
        
        # Dapatkan augmentation manager
        augmentation_manager = ui_components.get('augmentation_manager')
        if not augmentation_manager:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} AugmentationManager tidak tersedia"))
            cleanup_ui()
            return
        
        # Ambil jumlah workers dari UI jika tersedia
        num_workers = 4  # Default value
        if len(ui_components['aug_options'].children) > 5:
            num_workers = ui_components['aug_options'].children[5].value
            # Update num_workers pada augmentation_manager
            augmentation_manager.num_workers = num_workers
            
        # Dapatkan opsi dari UI
        variations = ui_components['aug_options'].children[1].value
        prefix = ui_components['aug_options'].children[2].value
        process_bboxes = ui_components['aug_options'].children[3].value
        validate = ui_components['aug_options'].children[4].value
        
        # Cek opsi balancing kelas (opsi baru)
        target_balance = False
        if len(ui_components['aug_options'].children) > 6:
            target_balance = ui_components['aug_options'].children[6].value
        
        # Register progress callback jika tersedia
        if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
            ui_components['register_progress_callback'](augmentation_manager)
        
        # Tandai augmentasi sedang berjalan
        ui_components['augmentation_running'] = True
        
        # Jalankan augmentasi langsung
        try:
            # Gunakan sumber data preprocessed dan balancing kelas
            result = augmentation_manager.augment_dataset(
                split='train',  # Augmentasi untuk train split
                augmentation_types=aug_types,
                num_variations=variations,
                output_prefix=prefix,
                validate_results=validate,
                resume=False,  # Tidak menggunakan resume
                process_bboxes=process_bboxes,
                target_balance=target_balance,  # Aktifkan balancing
                num_workers=num_workers  # Gunakan jumlah workers dari UI
            )
            
            # Proses hasil sukses
            if result.get("status") != "error":
                # Update UI dengan hasil sukses
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("success", f"{ICONS['success']} Augmentasi dataset selesai"))
                
                # Update summary dengan hasil augmentasi
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    ui_components['update_summary'](result)
                
                # Update status panel
                update_status_panel(
                    ui_components, 
                    "success", 
                    f"{ICONS['success']} Augmentasi dataset berhasil dengan {result.get('generated', 0)} gambar baru"
                )
                
                # Tampilkan tombol visualisasi
                ui_components['visualization_buttons'].layout.display = 'flex'
                
                # Tampilkan tombol cleanup
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
                # Notifikasi observer tentang selesai augmentasi
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.AUGMENTATION_END,
                        sender="augmentation_handler",
                        message=f"Augmentasi dataset selesai dengan {result.get('generated', 0)} gambar baru",
                        result=result
                    )
                except ImportError:
                    pass
            else:
                # Tangani error
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("error", f"{ICONS['error']} Error: {result.get('message', 'Unknown error')}"))
                
                # Update status panel
                update_status_panel(
                    ui_components, 
                    "error", 
                    f"{ICONS['error']} Augmentasi gagal: {result.get('message', 'Unknown error')}"
                )
                
                # Notifikasi observer tentang error
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.AUGMENTATION_ERROR,
                        sender="augmentation_handler",
                        message=f"Error saat augmentasi: {result.get('message', 'Unknown error')}"
                    )
                except ImportError:
                    pass
        except Exception as e:
            # Handle other errors
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
                
            # Update status panel
            update_status_panel(
                ui_components, 
                "error", 
                f"{ICONS['error']} Augmentasi gagal dengan error: {str(e)}"
            )
            
            # Notifikasi observer tentang error
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_ERROR,
                    sender="augmentation_handler",
                    message=f"Error saat augmentasi: {str(e)}"
                )
            except ImportError:
                pass
        finally:
            # Tandai augmentasi selesai
            ui_components['augmentation_running'] = False
            
            # Restore UI
            cleanup_ui()
    
    # Handler untuk tombol stop
    def on_stop_click(b):
        """Handler untuk menghentikan augmentasi."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.augmentation_initialization import update_status_panel
        
        ui_components['augmentation_running'] = False
        
        # Tampilkan pesan di status
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("warning", f"{ICONS['warning']} Menghentikan augmentasi..."))
        
        # Update status panel
        update_status_panel(
            ui_components, 
            "warning", 
            f"{ICONS['warning']} Augmentasi dihentikan oleh pengguna"
        )
        
        # Notifikasi observer
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            notify(
                event_type=EventTopics.AUGMENTATION_END,
                sender="augmentation_handler",
                message=f"Augmentasi dihentikan oleh pengguna"
            )
        except ImportError:
            pass
        
        # Reset UI
        cleanup_ui()
    
    # Handler untuk tombol reset
    def on_reset_click(b):
        """Handler untuk reset UI ke kondisi awal dengan konfigurasi default yang benar."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.augmentation_initialization import detect_augmentation_state

        # Reset UI ke kondisi awal
        reset_ui()
        
        # Reset config ke default - load konfigurasi default
        try:
            # Load konfigurasi default
            from smartcash.ui.dataset.augmentation_config_handler import load_default_augmentation_config, update_ui_from_config
            
            # Dapatkan konfigurasi default dengan memanggil fungsi yang tepat
            default_config = load_default_augmentation_config()
            
            # Update UI dari konfigurasi default
            update_ui_from_config(ui_components, default_config)
            
            # Re-detect state
            detect_augmentation_state(ui_components, env, config)
            
            # Tampilkan pesan sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
                
            # Update status panel
            from smartcash.ui.dataset.augmentation_initialization import update_status_panel
            update_status_panel(
                ui_components, 
                "success", 
                f"{ICONS['success']} Konfigurasi direset ke nilai default"
            )
            
            # Log success jika logger tersedia
            if logger: logger.success(f"{ICONS['success']} Konfigurasi augmentasi berhasil direset ke nilai default")
        except Exception as e:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
                
            # Log error jika logger tersedia
            if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
    
    # Function untuk cleanup UI setelah augmentasi
    def cleanup_ui():
        """Kembalikan UI ke kondisi operasional setelah augmentasi."""
        ui_components['augment_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        
        # Re-aktifkan tombol yang dinonaktifkan
        ui_components['reset_button'].disabled = False
        ui_components['save_button'].disabled = False
        ui_components['cleanup_button'].disabled = False
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback jika fungsi reset tidak tersedia
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'hidden'
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Function untuk reset komplet UI
    def reset_ui():
        """Reset UI ke kondisi default total, termasuk semua panel."""
        # Reset tombol dan progress
        cleanup_ui()
        
        # Reset summary
        if 'summary_container' in ui_components:
            ui_components['summary_container'].layout.display = 'none'
            with ui_components['summary_container']:
                clear_output()
        
        # Reset visualisasi
        if 'visualization_container' in ui_components:
            ui_components['visualization_container'].layout.display = 'none'
            with ui_components['visualization_container']:
                clear_output()
                
        # Sembunyikan tombol visualisasi dan cleanup
        if 'visualization_buttons' in ui_components:
            ui_components['visualization_buttons'].layout.display = 'none'
            
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].layout.display = 'none'
        
        # Reset logs
        if 'status' in ui_components:
            with ui_components['status']:
                clear_output()
            
        # Reset accordion
        if 'log_accordion' in ui_components:
            ui_components['log_accordion'].selected_index = None
    
    # Register handlers untuk tombol-tombol
    if 'augment_button' in ui_components:
        ui_components['augment_button'].on_click(on_augment_click)
    
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(on_stop_click)
        
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(on_reset_click)
    
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