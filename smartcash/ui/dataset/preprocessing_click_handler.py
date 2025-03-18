"""
File: smartcash/ui/dataset/preprocessing_click_handler.py
Deskripsi: Handler untuk tombol preprocessing dataset
"""

from typing import Dict, Any, List
from IPython.display import display, HTML, clear_output
import threading
import time

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
    try:
        from smartcash.ui.components.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        from smartcash.ui.dataset.preprocessing_config_handler import update_config_from_ui, save_preprocessing_config
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def update_status_panel(ui_components, status_type, message):
            pass
            
        ICONS = {
            'processing': '🔄',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }

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
        updated_config = update_config_from_ui(ui_components, config)
        
        # Simpan konfigurasi
        try:
            save_preprocessing_config(updated_config)
            if 'logger' in ui_components:
                ui_components['logger'].info(f"{ICONS.get('success', '✅')} Konfigurasi preprocessing berhasil disimpan")
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi: {str(e)}")
        
        # Jalankan preprocessing dalam thread terpisah
        def run_preprocessing():
            try:
                # Update status panel
                update_status_panel(ui_components, "info", f"{ICONS.get('processing', '🔄')} Preprocessing dataset...")
                
                # Dapatkan dataset manager
                dataset_manager = ui_components.get('dataset_manager')
                if not dataset_manager:
                    with ui_components['status']:
                        display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Dataset Manager tidak tersedia"))
                    cleanup_ui()
                    return
                
                # Dapatkan opsi preprocessing dari UI
                preprocess_options = ui_components.get('preprocess_options')
                if preprocess_options and hasattr(preprocess_options, 'children'):
                    img_size = preprocess_options.children[0].value
                    normalize = preprocess_options.children[1].value
                    preserve_aspect_ratio = preprocess_options.children[2].value
                    use_cache = preprocess_options.children[3].value
                    num_workers = preprocess_options.children[4].value
                else:
                    # Default values
                    img_size = 640
                    normalize = True
                    preserve_aspect_ratio = True
                    use_cache = True
                    num_workers = 4
                
                # Dapatkan opsi validasi dari UI
                validation_options = ui_components.get('validation_options')
                if validation_options and hasattr(validation_options, 'children'):
                    validate = validation_options.children[0].value
                    fix_issues = validation_options.children[1].value
                    move_invalid = validation_options.children[2].value
                else:
                    # Default values
                    validate = True
                    fix_issues = True
                    move_invalid = True
                
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
                update_summary(preprocess_result)
                
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
                if 'logger' in ui_components:
                    ui_components['logger'].error(f"{ICONS.get('error', '❌')} Error saat preprocessing dataset: {str(e)}")
            
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
            update_status_panel(ui_components, "warning", f"{ICONS.get('warning', '⚠️')} Preprocessing dihentikan oleh pengguna")
            
            # Wait for thread to finish
            if 'preprocessing_thread' in ui_components and ui_components['preprocessing_thread'].is_alive():
                ui_components['preprocessing_thread'].join(timeout=1)
            
            # Restore UI
            cleanup_ui()
    
    # Function to clean up UI after preprocessing
    def cleanup_ui():
        ui_components['preprocess_button'].layout.display = 'block'
        ui_components['stop_button'].layout.display = 'none'
        
        # Reset progress bars
        ui_components['progress_bar'].value = 0
        ui_components['current_progress'].value = 0
    
    # Function to update summary after preprocessing
    def update_summary(preprocessing_result):
        summary_container = ui_components.get('summary_container')
        if not summary_container:
            return
            
        # Clear and show summary container
        summary_container.clear_output()
        summary_container.layout.display = 'block'
        
        # Extract stats from preprocessing result
        with summary_container:
            try:
                display(HTML(f"""
                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h4 style="margin-top:0">📊 Preprocessing Summary</h4>
                    <p><strong>📁 Images Processed:</strong> {preprocessing_result.get('images_processed', 'N/A')}</p>
                    <p><strong>✅ Successfully Processed:</strong> {preprocessing_result.get('success_count', 'N/A')}</p>
                    <p><strong>⚠️ Warnings:</strong> {preprocessing_result.get('warning_count', 'N/A')}</p>
                    <p><strong>❌ Errors:</strong> {preprocessing_result.get('error_count', 'N/A')}</p>
                    <p><strong>⏱️ Processing Time:</strong> {preprocessing_result.get('processing_time', 'N/A')} seconds</p>
                </div>
                """))
            except Exception as e:
                display(HTML(f"""
                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h4 style="margin-top:0">📊 Preprocessing Summary</h4>
                    <p><strong>✅ Status:</strong> Completed</p>
                    <p><em>Detailed statistics not available</em></p>
                </div>
                """))
    
    # Register button click handlers
    ui_components['preprocess_button'].on_click(on_preprocess_click)
    ui_components['stop_button'].on_click(on_stop_click)
    
    # Add reference to the handlers in ui_components
    ui_components['on_preprocess_click'] = on_preprocess_click
    ui_components['on_stop_click'] = on_stop_click
    ui_components['update_summary'] = update_summary
    ui_components['cleanup_ui'] = cleanup_ui
    
    return ui_components