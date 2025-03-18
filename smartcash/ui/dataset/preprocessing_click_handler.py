"""
File: smartcash/ui/dataset/preprocessing_click_handler.py
Deskripsi: Handler yang disederhanakan untuk tombol preprocessing dataset
"""

from typing import Dict, Any, List
from IPython.display import display, clear_output
import threading
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.alerts import create_status_indicator

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')

    # Handler untuk tombol preprocessing
    def on_preprocess_click(b):
        # Dapatkan splitter dari UI
        split_option = ui_components['split_selector'].value
        split_map = {
            'All Splits': None,
            'Train Only': 'train',
            'Validation Only': 'valid',
            'Test Only': 'test'
        }
        split = split_map.get(split_option)
        
        # Persiapkan preprocessing
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('processing', '🔄')} Memulai preprocessing dataset..."))
        
        # Update UI
        ui_components['preprocess_button'].layout.display = 'none'
        ui_components['stop_button'].layout.display = 'block'
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['current_progress'].layout.visibility = 'visible'
        ui_components['log_accordion'].selected_index = 0  # Expand log
        
        # Update config dari UI untuk disimpan
        try:
            from smartcash.ui.dataset.preprocessing_config_handler import update_config_from_ui, save_preprocessing_config
            updated_config = update_config_from_ui(ui_components, config)
            save_preprocessing_config(updated_config)
            if logger: logger.info(f"{ICONS.get('success', '✅')} Konfigurasi preprocessing berhasil disimpan")
        except Exception as e:
            if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Jalankan preprocessing dalam thread terpisah
        def run_preprocessing():
            try:
                # Update status panel
                from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
                update_status_panel(ui_components, "info", f"{ICONS.get('processing', '🔄')} Preprocessing dataset...")
                
                # Dapatkan dataset manager
                dataset_manager = ui_components.get('dataset_manager')
                if not dataset_manager:
                    with ui_components['status']:
                        display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Dataset Manager tidak tersedia"))
                    cleanup_ui()
                    return
                
                # Dapatkan opsi preprocessing dari UI
                img_size = ui_components['preprocess_options'].children[0].value
                normalize = ui_components['preprocess_options'].children[1].value
                preserve_aspect_ratio = ui_components['preprocess_options'].children[2].value
                use_cache = ui_components['preprocess_options'].children[3].value
                num_workers = ui_components['preprocess_options'].children[4].value
                
                # Dapatkan opsi validasi dari UI
                validate = ui_components['validation_options'].children[0].value
                fix_issues = ui_components['validation_options'].children[1].value
                move_invalid = ui_components['validation_options'].children[2].value
                
                # Preprocessing dengan dataset manager
                ui_components['preprocessing_running'] = True
                
                # Preprocess dataset
                preprocess_result = dataset_manager.preprocess_dataset(
                    split=split,
                    force_reprocess=True,
                    img_size=[img_size, img_size],
                    normalize=normalize,
                    preserve_aspect_ratio=preserve_aspect_ratio,
                    cache=use_cache,
                    num_workers=num_workers,
                    validate=validate,
                    fix_issues=fix_issues,
                    move_invalid=move_invalid
                )
                
                # Setelah selesai
                ui_components['preprocessing_running'] = False
                
                # Update UI dengan hasil
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Preprocessing dataset selesai"))
                
                # Update summary
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    ui_components['update_summary'](preprocess_result)
                
                # Update status panel
                update_status_panel(ui_components, "success", f"{ICONS.get('success', '✅')} Preprocessing dataset berhasil")
                
                # Show cleanup button
                ui_components['cleanup_button'].layout.display = 'block'
                
            except Exception as e:
                ui_components['preprocessing_running'] = False
                with ui_components['status']:
                    display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error saat preprocessing: {str(e)}"))
                
                # Update status panel
                update_status_panel(ui_components, "error", f"{ICONS.get('error', '❌')} Preprocessing gagal: {str(e)}")
                
                # Log error
                if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat preprocessing dataset: {str(e)}")
            
            finally:
                # Restore UI
                cleanup_ui()
        
        # Jalankan thread preprocessing
        ui_components['preprocessing_running'] = True
        ui_components['preprocessing_thread'] = threading.Thread(target=run_preprocessing)
        ui_components['preprocessing_thread'].daemon = True
        ui_components['preprocessing_thread'].start()
    
    # Handler untuk tombol stop
    def on_stop_click(b):
        if ui_components.get('preprocessing_running', False):
            # Stop preprocessing
            ui_components['preprocessing_running'] = False
            
            with ui_components['status']:
                display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menghentikan preprocessing..."))
            
            # Update status panel
            from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
            update_status_panel(ui_components, "warning", f"{ICONS.get('warning', '⚠️')} Preprocessing dihentikan oleh pengguna")
            
            # Wait for thread to finish
            thread = ui_components.get('preprocessing_thread')
            if thread and thread.is_alive():
                thread.join(timeout=1)
            
            # Restore UI
            cleanup_ui()
    
    # Function to clean up UI after preprocessing
    def cleanup_ui():
        ui_components['preprocess_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        
        # Reset progress bars
        ui_components['progress_bar'].value = 0
        ui_components['current_progress'].value = 0
    
    # Register button click handlers
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    ui_components['stop_button'].on_click(on_stop_click)
    
    # Add reference to the handlers in ui_components
    ui_components['on_preprocess_click'] = on_preprocess_click
    ui_components['on_stop_click'] = on_stop_click
    ui_components['cleanup_ui'] = cleanup_ui
    
    return ui_components