"""
File: smartcash/ui/dataset/augmentation_click_handler.py
Deskripsi: Update handler click augmentasi untuk mendukung alur baru
"""

@try_except_decorator(ui_components.get('status'))
def on_augment_click(b):
    """Handler tombol augmentasi dengan alur augmentasi yang diperbarui."""
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
    
    # Jalankan augmentasi dengan alur baru
    try:
        # Gunakan sumber data preprocessed dengan alur baru
        result = augmentation_manager.augment_dataset(
            split='train',                # Augmentasi untuk train split
            augmentation_types=aug_types,
            num_variations=variations,
            output_prefix=prefix,
            validate_results=validate,
            resume=False,                 # Tidak menggunakan resume
            process_bboxes=process_bboxes,
            target_balance=target_balance,
            num_workers=num_workers,      # Gunakan jumlah workers dari UI
            move_to_preprocessed=True     # ALUR BARU: Pindahkan ke preprocessed setelah augmentasi
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
                f"{ICONS['success']} Augmentasi dataset berhasil dengan {result.get('generated', 0)} gambar baru, dipindahkan ke {result.get('final_output_dir', 'preprocessed')}"
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