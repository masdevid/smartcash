"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py  
Deskripsi: Simplified handlers dengan dual progress tracker
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.utils.ui_utils import clear_outputs, handle_ui_error, show_ui_success
from smartcash.ui.dataset.preprocessing.utils.button_manager import get_button_manager
from smartcash.ui.dataset.preprocessing.utils.progress_utils import create_dual_progress_callback, setup_dual_progress_tracker, complete_progress_tracker, error_progress_tracker
from smartcash.ui.dataset.preprocessing.utils.backend_utils import validate_dataset_ready, check_preprocessed_exists, create_backend_preprocessor, create_backend_checker, create_backend_cleanup_service

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Simplified setup dengan dual progress tracker"""
    
    # Setup dual progress callback
    ui_components['progress_callback'] = create_dual_progress_callback(ui_components)
    
    # Setup config handler
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup handlers
    setup_preprocessing_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    setup_config_handlers(ui_components, config)
    
    return ui_components

def setup_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Simplified preprocessing handler dengan disable ALL buttons"""
    
    def execute_preprocessing(button=None):
        button_manager = get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        clear_outputs(ui_components)
        button_manager.set_processing_mode('preprocess_button')  # Disable ALL buttons
        
        try:
            if logger:
                logger.info("🚀 Memulai preprocessing dataset")
            
            # Validate dataset
            valid, msg = validate_dataset_ready(config, logger)
            if not valid:
                handle_ui_error(ui_components, f"❌ {msg}", button_manager)
                return
            
            # Setup dual progress tracker
            setup_dual_progress_tracker(ui_components, "Dataset Preprocessing")
            
            # Create preprocessor
            preprocessor = create_backend_preprocessor(config, logger)
            if not preprocessor:
                handle_ui_error(ui_components, "❌ Gagal membuat preprocessing service", button_manager)
                return
            
            # Register progress callback
            preprocessor.register_progress_callback(ui_components['progress_callback'])
            
            # Execute preprocessing
            result = preprocessor.preprocess_dataset()
            
            if result and result.get('success', False):
                complete_progress_tracker(ui_components, "Preprocessing selesai")
                stats = result.get('stats', {})
                total_images = stats.get('total_images', 0)
                success_msg = f"Preprocessing berhasil: {total_images:,} gambar"
                show_ui_success(ui_components, success_msg, button_manager)
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'No response from service'
                handle_ui_error(ui_components, error_msg, button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Preprocessing gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error preprocessing: {str(e)}", button_manager)
        
        finally:
            button_manager.reset_from_processing_mode()  # Enable ALL buttons
    
    preprocess_button = ui_components.get('preprocess_button')
    if preprocess_button:
        preprocess_button.on_click(execute_preprocessing)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Simplified check handler dengan disable ALL buttons"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        clear_outputs(ui_components)
        button_manager.set_processing_mode('check_button')  # Disable ALL buttons
        
        try:
            if logger:
                logger.info("🔍 Checking dataset")
            
            setup_dual_progress_tracker(ui_components, "Dataset Check")
            
            valid, source_msg = validate_dataset_ready(config, logger)
            
            if valid:
                if logger:
                    logger.success(f"✅ {source_msg}")
                
                preprocessed_exists, preprocessed_count = check_preprocessed_exists(config)
                
                if preprocessed_exists:
                    if logger:
                        logger.success(f"💾 Preprocessed dataset: {preprocessed_count:,} gambar")
                else:
                    if logger:
                        logger.info("ℹ️ Belum ada preprocessed dataset")
                
                complete_progress_tracker(ui_components, "Dataset check selesai")
                show_ui_success(ui_components, f"Check: {source_msg.split(': ')[1] if ': ' in source_msg else source_msg}", button_manager)
            else:
                handle_ui_error(ui_components, f"❌ {source_msg}", button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Check gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error check: {str(e)}", button_manager)
        
        finally:
            button_manager.reset_from_processing_mode()  # Enable ALL buttons
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Simplified cleanup handler dengan disable ALL buttons"""
    
    def execute_cleanup(button=None):
        button_manager = get_button_manager(ui_components)
        logger = ui_components.get('logger')
        clear_outputs(ui_components)
        
        try:
            has_data, count = check_preprocessed_exists(config)
            if not has_data:
                show_ui_success(ui_components, "ℹ️ Tidak ada preprocessed data untuk dibersihkan")
                return
            
            def confirmed_cleanup():
                button_manager.set_processing_mode('cleanup_button')  # Disable ALL buttons
                
                try:
                    if logger:
                        logger.info("🧹 Memulai cleanup preprocessed data")
                    
                    setup_dual_progress_tracker(ui_components, "Dataset Cleanup")
                    
                    cleanup_service = create_backend_cleanup_service(config, logger)
                    if not cleanup_service:
                        handle_ui_error(ui_components, "❌ Gagal membuat cleanup service", button_manager)
                        return
                    
                    result = cleanup_service.cleanup_preprocessed_data()
                    
                    if result and result.get('success', False):
                        stats = result.get('stats', {})
                        files_removed = stats.get('files_removed', 0)
                        complete_progress_tracker(ui_components, f"Cleanup selesai: {files_removed:,} file")
                        show_ui_success(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file", button_manager)
                    else:
                        error_msg = result.get('message', 'Unknown cleanup error') if result else 'No response from service'
                        handle_ui_error(ui_components, error_msg, button_manager)
                        
                except Exception as e:
                    error_progress_tracker(ui_components, f"Cleanup gagal: {str(e)}")
                    handle_ui_error(ui_components, f"❌ Error cleanup: {str(e)}", button_manager)
                
                finally:
                    button_manager.reset_from_processing_mode()  # Enable ALL buttons
            
            # Show confirmation
            from IPython.display import display, clear_output
            confirmation_area = ui_components.get('confirmation_area')
            if confirmation_area:
                with confirmation_area:
                    clear_output(wait=True)
                    confirm_html = f"""
                    <div style="padding:10px; background:#fff3cd; border:1px solid #ffeaa7; border-radius:4px;">
                        <h4>⚠️ Konfirmasi Cleanup</h4>
                        <p>Akan menghapus {count:,} file preprocessed. Lanjutkan?</p>
                        <button onclick="this.style.display='none'; this.nextElementSibling.click()" 
                                style="background:#dc3545; color:white; padding:5px 15px; border:none; border-radius:3px; margin:5px;">
                            Ya, Hapus
                        </button>
                        <button onclick="this.parentElement.innerHTML='<p>Cleanup dibatalkan</p>'" 
                                style="background:#6c757d; color:white; padding:5px 15px; border:none; border-radius:3px; margin:5px;">
                            Batal
                        </button>
                    </div>
                    """
                    display(widgets.HTML(confirm_html))
                    
                    # Create hidden button for actual cleanup
                    confirm_button = widgets.Button(description="Confirm Cleanup", style={'_view_name': 'none'})
                    confirm_button.on_click(lambda b: confirmed_cleanup())
                    confirm_button.layout.display = 'none'
                    display(confirm_button)
                    
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error cleanup check: {str(e)}")
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Simplified config handlers dengan disable ALL buttons"""
    
    def save_config(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.set_processing_mode('save_button')  # Disable ALL buttons
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.save_config(ui_components)
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error save config: {str(e)}")
        finally:
            button_manager.reset_from_processing_mode()  # Enable ALL buttons
    
    def reset_config(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.set_processing_mode('reset_button')  # Disable ALL buttons
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.reset_config(ui_components)
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset config: {str(e)}")
        finally:
            button_manager.reset_from_processing_mode()  # Enable ALL buttons
    
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config)
    if reset_button:
        reset_button.on_click(reset_config)