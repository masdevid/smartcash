"""
File: smartcash/ui/dataset/augmentation_cleanup_handler.py
Deskripsi: Perbaikan handler cleanup untuk mendukung lokasi augmentasi yang baru
"""

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol cleanup augmentasi data dengan progress tracking."""
    logger = ui_components.get('logger')
    
    def perform_cleanup():
        """Bersihkan folder augmentasi dengan progress tracking."""
        # Dapatkan direktori augmentasi dari konfigurasi
        augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
        
        # Dapatkan direktori preprocessed untuk membersihkan file augmentasi
        preprocessed_dir = config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        
        try:
            # Update status dan tampilkan progress bar
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('cleanup', '🧹')} Mempersiapkan pembersihan..."))
                
            # Pastikan direktori ada
            path = Path(augmented_dir)
            if not path.exists():
                with ui_components['status']:
                    display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Direktori temp tidak ditemukan: {augmented_dir}"))
                
            # Notifikasi observer via observer system
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_START,
                    sender="augmentation_handler",
                    message=f"Mulai membersihkan data augmentasi di {augmented_dir} dan {preprocessed_dir}"
                )
            except (ImportError, AttributeError):
                pass
            
            # Tampilkan progress bar dan status
            if 'progress_bar' in ui_components and 'current_progress' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'visible'
                ui_components['current_progress'].layout.visibility = 'visible'
                ui_components['progress_bar'].description = 'Memproses: 0%'
                ui_components['current_progress'].description = 'Mencari file...'
            
            # Update status panel
            from smartcash.ui.dataset.augmentation_initialization import update_status_panel
            update_status_panel(
                ui_components,
                "info",
                f"{ICONS.get('cleanup', '🧹')} Membersihkan data augmentasi di {augmented_dir} dan {preprocessed_dir}"
            )
            
            # Hapus isi subdirektori images dan labels di folder temp
            subdirs = ['images', 'labels']
            total_files_deleted = 0
            
            # Ambil prefix augmentasi dari UI
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            
            # 1. Bersihkan folder temp augmentasi
            temp_files_to_delete = []
            for subdir in subdirs:
                dir_path = path / subdir
                if not dir_path.exists():
                    continue
                
                # Cari file dengan pola augmentasi di folder temp
                temp_files_to_delete.extend(list(dir_path.glob(f"{aug_prefix}_*.*")))
            
            # 2. Bersihkan file augmentasi di folder preprocessed (untuk setiap split)
            preprocessed_files_to_delete = []
            for split in ['train', 'valid', 'test']:
                for subdir in subdirs:
                    dir_path = Path(preprocessed_dir) / split / subdir
                    if not dir_path.exists():
                        continue
                    
                    # Cari file dengan pola augmentasi di folder preprocessed
                    preprocessed_files_to_delete.extend(list(dir_path.glob(f"{aug_prefix}_*.*")))
            
            # Gabungkan semua file yang perlu dihapus
            all_files_to_delete = temp_files_to_delete + preprocessed_files_to_delete
            
            # Update progress bar dengan total file
            total_files = len(all_files_to_delete)
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = total_files if total_files > 0 else 1
                ui_components['progress_bar'].value = 0
                ui_components['progress_bar'].description = f'Total: {total_files} file'
            
            # Hapus file dengan progress tracking
            for i, file_path in enumerate(all_files_to_delete):
                try:
                    # Hapus file
                    os.remove(file_path)
                    total_files_deleted += 1
                    
                    # Update progress
                    if 'progress_bar' in ui_components and 'current_progress' in ui_components:
                        ui_components['progress_bar'].value = i + 1
                        progress_percent = int((i + 1) / total_files * 100) if total_files > 0 else 100
                        ui_components['current_progress'].value = progress_percent
                        ui_components['current_progress'].description = f'Menghapus: {file_path.name}'
                    
                    # Report progress via observer (lebih jarang untuk mengurangi overhead)
                    if i % max(10, total_files//10) == 0 or i == len(all_files_to_delete) - 1:
                        try:
                            from smartcash.components.observer import notify
                            from smartcash.components.observer.event_topics_observer import EventTopics
                            notify(
                                event_type=EventTopics.AUGMENTATION_CLEANUP_PROGRESS,
                                sender="augmentation_handler",
                                message=f"Menghapus file augmentasi ({i+1}/{total_files})",
                                progress=i+1,
                                total=total_files
                            )
                        except (ImportError, AttributeError):
                            pass
                
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Gagal menghapus {file_path}: {e}")
            
            # Tampilkan status sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", 
                    f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan. {total_files_deleted} file dihapus"))
            
            # Update status panel
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan. {total_files_deleted} file dihapus"
            )
            
            # Notifikasi observer via observer system
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_END,
                    sender="augmentation_handler",
                    message=f"Berhasil membersihkan {total_files_deleted} file augmentasi",
                    files_deleted=total_files_deleted
                )
            except (ImportError, AttributeError):
                pass
            
            # Reset UI
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'none'
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'none'
            if 'summary_container' in ui_components:
                ui_components['summary_container'].layout.display = 'none'
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'none'
            
            # Reset progress bar
            if 'progress_bar' in ui_components and 'current_progress' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'hidden'
                ui_components['current_progress'].layout.visibility = 'hidden'
            
        except Exception as e:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Gagal membersihkan: {str(e)}"))
            
            # Update status panel
            try:
                from smartcash.ui.dataset.augmentation_initialization import update_status_panel
                update_status_panel(
                    ui_components,
                    "error",
                    f"{ICONS.get('error', '❌')} Gagal membersihkan data augmentasi: {str(e)}"
                )
            except ImportError:
                pass
            
            if logger:
                logger.error(f"❌ Error saat membersihkan data: {str(e)}")
            
            # Notifikasi observer via observer system
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_ERROR,
                    sender="augmentation_handler",
                    message=f"Error saat membersihkan data: {str(e)}"
                )
            except (ImportError, AttributeError):
                pass
    
    # Daftarkan handler untuk tombol cleanup
    ui_components['cleanup_button'].on_click(lambda b: perform_cleanup())
    
    return ui_components