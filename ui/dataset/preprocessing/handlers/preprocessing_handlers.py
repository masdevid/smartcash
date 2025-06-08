"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Refactored handlers dengan utils separation dan progress_tracker baru
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_preprocessing_config, display_preprocessing_results, show_preprocessing_success, clear_outputs, handle_ui_error, show_ui_success
from smartcash.ui.dataset.preprocessing.utils.button_manager import get_button_manager
from smartcash.ui.dataset.preprocessing.utils.progress_utils import create_progress_callback, setup_progress_tracker, complete_progress_tracker, error_progress_tracker
from smartcash.ui.dataset.preprocessing.utils.backend_utils import validate_dataset_ready, check_preprocessed_exists, create_backend_preprocessor, create_backend_checker, create_backend_cleanup_service
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup preprocessing handlers dengan utils separation"""
    
    # Setup progress callback
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    # Setup handlers
    setup_preprocessing_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    setup_config_handlers(ui_components, config)
    
    return ui_components

def setup_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup preprocessing handler dengan new progress tracker"""
    
    def execute_preprocessing(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('preprocess_button')
        
        try:
            # Validate dataset ready
            valid, msg = validate_dataset_ready(config, ui_components.get('logger'))
            if not valid:
                handle_ui_error(ui_components, f"❌ {msg}", button_manager)
                return
            
            # Setup progress tracker dengan steps
            setup_progress_tracker(ui_components, "Dataset Preprocessing")
            
            # Extract processing params dengan normalization
            params = _extract_processing_params(ui_components)
            processing_config = {
                **config, 
                'preprocessing': {
                    **config.get('preprocessing', {}), 
                    'normalization': params['normalization'],
                    'target_split': params['target_split']
                },
                'performance': {
                    **config.get('performance', {}),
                    'num_workers': params['num_workers']
                }
            }
            
            # Log config
            log_preprocessing_config(ui_components, processing_config)
            
            # Create dan execute preprocessor
            preprocessor = create_backend_preprocessor(processing_config, ui_components.get('logger'))
            if not preprocessor:
                handle_ui_error(ui_components, "❌ Gagal membuat preprocessing service", button_manager)
                return
            
            preprocessor.register_progress_callback(ui_components['progress_callback'])
            
            result = preprocessor.preprocess_with_uuid_consistency(
                split=params.get('split', 'all'),
                force_reprocess=params.get('force_reprocess', False)
            )
            
            if result and result.get('success', False):
                complete_progress_tracker(ui_components, "Preprocessing selesai")
                display_preprocessing_results(ui_components, result)
                show_preprocessing_success(ui_components, result)
                show_ui_success(ui_components, "Preprocessing berhasil", button_manager)
            else:
                error_msg = result.get('message', 'Unknown preprocessing error') if result else 'No response from service'
                handle_ui_error(ui_components, error_msg, button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Preprocessing gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error preprocessing: {str(e)}", button_manager)
        
        finally:
            button_manager.enable_buttons()
    
    preprocess_button = ui_components.get('preprocess_button')
    if preprocess_button:
        preprocess_button.on_click(execute_preprocessing)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend checker"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            setup_progress_tracker(ui_components, "Dataset Check")
            
            # Check source dataset
            valid, source_msg = validate_dataset_ready(config, ui_components.get('logger'))
            
            if valid:
                logger = ui_components.get('logger')
                if logger:
                    logger.success(f"✅ {source_msg}")
                
                # Check preprocessed data
                preprocessed_exists, preprocessed_count = check_preprocessed_exists(config)
                
                if preprocessed_exists:
                    if logger:
                        logger.success(f"💾 Preprocessed dataset: {preprocessed_count:,} gambar")
                    _show_preprocessed_breakdown(ui_components, config)
                else:
                    if logger:
                        logger.info("ℹ️ Belum ada preprocessed dataset")
                
                complete_progress_tracker(ui_components, "Dataset check selesai")
                show_ui_success(ui_components, f"Dataset siap: {source_msg.split(': ')[1] if ': ' in source_msg else source_msg}", button_manager)
            else:
                handle_ui_error(ui_components, f"❌ {source_msg}", button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Check gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error check: {str(e)}", button_manager)
        
        finally:
            button_manager.enable_buttons()
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan confirmation"""
    
    def execute_cleanup(button=None):
        clear_outputs(ui_components)
        
        # Check existing preprocessed data
        has_data, count = check_preprocessed_exists(config)
        if not has_data:
            show_ui_success(ui_components, "ℹ️ Tidak ada data preprocessed untuk dibersihkan")
            return
        
        def confirmed_cleanup():
            button_manager = get_button_manager(ui_components)
            button_manager.disable_buttons('cleanup_button')
            
            try:
                setup_progress_tracker(ui_components, "Dataset Cleanup")
                
                cleanup_service = create_backend_cleanup_service(config, ui_components.get('logger'))
                if not cleanup_service:
                    handle_ui_error(ui_components, "❌ Gagal membuat cleanup service", button_manager)
                    return
                
                cleanup_service.register_progress_callback(ui_components['progress_callback'])
                
                result = cleanup_service.cleanup_preprocessed_data(safe_mode=True)
                
                if result and result.get('success', False):
                    stats = result.get('stats', {})
                    files_removed = stats.get('files_removed', 0)
                    
                    complete_progress_tracker(ui_components, f"Cleanup selesai: {files_removed:,} file dihapus")
                    show_ui_success(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file", button_manager)
                else:
                    error_msg = result.get('message', 'Unknown cleanup error') if result else 'No response from service'
                    handle_ui_error(ui_components, error_msg, button_manager)
                    
            except Exception as e:
                error_progress_tracker(ui_components, f"Cleanup gagal: {str(e)}")
                handle_ui_error(ui_components, f"❌ Error cleanup: {str(e)}", button_manager)
            
            finally:
                button_manager.enable_buttons()
        
        # Show confirmation dialog
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area:
            from IPython.display import display, clear_output
            with confirmation_area:
                clear_output(wait=True)
                dialog = create_destructive_confirmation(
                    title="⚠️ Konfirmasi Cleanup Dataset",
                    message=f"Operasi ini akan menghapus {count:,} file preprocessed.\n\nData asli tetap aman. Lanjutkan?",
                    on_confirm=lambda b: (confirmed_cleanup(), clear_outputs(ui_components)),
                    on_cancel=lambda b: clear_outputs(ui_components),
                    item_name="data preprocessed"
                )
                display(dialog)
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config save/reset handlers"""
    
    def save_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                success = config_handler.save_config(ui_components)
                if success:
                    show_ui_success(ui_components, "✅ Konfigurasi tersimpan")
                else:
                    handle_ui_error(ui_components, "❌ Gagal simpan konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error save: {str(e)}")
    
    def reset_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                success = config_handler.reset_config(ui_components)
                if success:
                    show_ui_success(ui_components, "🔄 Konfigurasi direset")
                else:
                    handle_ui_error(ui_components, "❌ Gagal reset konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset: {str(e)}")
    
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config)
    if reset_button:
        reset_button.on_click(reset_config)

def _extract_processing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters dari UI components"""
    resolution = getattr(ui_components.get('resolution_dropdown'), 'value', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    normalization_method = getattr(ui_components.get('normalization_dropdown'), 'value', 'minmax')
    
    return {
        'target_size': [width, height],
        'normalization': {
            'enabled': normalization_method != 'none',
            'method': normalization_method
        },
        'num_workers': getattr(ui_components.get('worker_slider'), 'value', 8),
        'target_split': getattr(ui_components.get('split_dropdown'), 'value', 'all'),
        'force_reprocess': False
    }

def _show_preprocessed_breakdown(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Show detailed preprocessed breakdown"""
    from pathlib import Path
    from IPython.display import display, HTML
    
    preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    breakdown = {}
    total_images = 0
    
    for split in ['train', 'valid', 'test']:
        split_images_dir = preprocessed_dir / split / 'images'
        if split_images_dir.exists():
            split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
            count = len(split_files)
            if count > 0:
                breakdown[split] = count
                total_images += count
                logger = ui_components.get('logger')
                if logger:
                    logger.info(f"📂 {split}: {count:,} gambar preprocessed")
    
    # Display detailed report
    if ui_components.get('log_output') and breakdown:
        report_html = f"""
        <div style="background:#f0f8ff;padding:10px;border-radius:5px;margin:10px 0;">
            <strong>📊 Preprocessed Dataset Breakdown:</strong><br>
            <ul style="margin:8px 0;padding-left:20px;">
        """
        
        for split, count in breakdown.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            report_html += f"<li><strong>{split}:</strong> {count:,} gambar ({percentage:.1f}%)</li>"
        
        report_html += f"""
            </ul>
            <div style="margin-top:8px;padding:6px;background:#e7f3ff;border-radius:3px;">
                <strong>Total:</strong> {total_images:,} gambar preprocessed siap untuk training
            </div>
        </div>
        """
        
        with ui_components['log_output']:
            display(HTML(report_html))