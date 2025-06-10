"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Enhanced handlers dengan progress tracker kompatibel dan multi-split support
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_preprocessing_config, display_preprocessing_results, show_preprocessing_success, clear_outputs, handle_ui_error, show_ui_success
from smartcash.ui.dataset.preprocessing.utils.button_manager import get_button_manager
from smartcash.ui.dataset.preprocessing.utils.progress_utils import create_enhanced_progress_callback, setup_enhanced_progress_tracker, complete_progress_tracker, error_progress_tracker
from smartcash.ui.dataset.preprocessing.utils.backend_utils import validate_dataset_ready, check_preprocessed_exists, create_backend_preprocessor, create_backend_checker, create_backend_cleanup_service
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Enhanced setup dengan progress tracker kompatibel dan multi-split support"""
    
    # Setup enhanced progress callback
    ui_components['progress_callback'] = create_enhanced_progress_callback(ui_components)
    
    # Setup config handler dengan UI logger integration
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup enhanced handlers
    setup_enhanced_preprocessing_handler(ui_components, config)
    setup_enhanced_check_handler(ui_components, config)
    setup_enhanced_cleanup_handler(ui_components, config)
    setup_config_handlers_enhanced(ui_components, config)
    
    return ui_components

def setup_config_handlers_enhanced(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Enhanced config handlers dengan proper UI logging"""
    
    def save_config_enhanced(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.save_config(ui_components)
            # Logger sudah handle di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error save enhanced config: {str(e)}")
    
    def reset_config_enhanced(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.reset_config(ui_components)
            # Logger sudah handle di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error reset enhanced config: {str(e)}")
    
    # Bind enhanced handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_enhanced)
    if reset_button:
        reset_button.on_click(reset_config_enhanced)

def setup_enhanced_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Enhanced preprocessing handler dengan multi-split dan progress tracker kompatibel"""
    
    def execute_enhanced_preprocessing(button=None):
        button_manager = get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('preprocess_button')
        
        try:
            if logger:
                logger.info("🚀 Memulai enhanced preprocessing dataset")
            
            # Validate dataset
            valid, msg = validate_dataset_ready(config, logger)
            if not valid:
                handle_ui_error(ui_components, f"❌ {msg}", button_manager)
                return
            
            # Setup enhanced progress tracker
            setup_enhanced_progress_tracker(ui_components, "Enhanced Dataset Preprocessing")
            
            # Extract enhanced params
            params = _extract_enhanced_processing_params(ui_components)
            
            # Build enhanced processing config
            processing_config = {
                **config,
                'preprocessing': {
                    **config.get('preprocessing', {}),
                    'target_splits': params['target_splits'],
                    'normalization': {
                        **config.get('preprocessing', {}).get('normalization', {}),
                        'enabled': params['normalization']['enabled'],
                        'method': params['normalization']['method'],
                        'target_size': params['target_size'],
                        'preserve_aspect_ratio': params['preserve_aspect_ratio']
                    },
                    'validation': {
                        **config.get('preprocessing', {}).get('validation', {}),
                        'enabled': params['validation_enabled'],
                        'move_invalid': params['move_invalid'],
                        'invalid_dir': params['invalid_dir']
                    }
                },
                'performance': {
                    **config.get('performance', {}),
                    'batch_size': params['batch_size']
                }
            }
            
            # Log enhanced config
            log_preprocessing_config(ui_components, processing_config)
            
            # Create enhanced preprocessor
            preprocessor = create_backend_preprocessor(processing_config, logger)
            if not preprocessor:
                handle_ui_error(ui_components, "❌ Gagal membuat enhanced preprocessing service", button_manager)
                return
            
            # Register enhanced progress callback
            preprocessor.register_progress_callback(ui_components['progress_callback'])
            
            # Execute enhanced preprocessing dengan multi-split support
            result = preprocessor.preprocess_enhanced_with_multi_splits(
                target_splits=params['target_splits'],
                preserve_aspect_ratio=params['preserve_aspect_ratio'],
                validation_config=params.get('validation_config', {}),
                force_reprocess=params.get('force_reprocess', False)
            )
            
            if result and result.get('success', False):
                complete_progress_tracker(ui_components, "Enhanced preprocessing selesai")
                display_preprocessing_results(ui_components, result)
                show_preprocessing_success(ui_components, result)
                
                # Enhanced success message
                stats = result.get('stats', {})
                splits_processed = len(stats.get('splits', {}))
                total_images = stats.get('total_images', 0)
                
                success_msg = f"Enhanced preprocessing berhasil: {total_images:,} gambar dalam {splits_processed} split"
                show_ui_success(ui_components, success_msg, button_manager)
            else:
                error_msg = result.get('message', 'Unknown enhanced preprocessing error') if result else 'No response from enhanced service'
                handle_ui_error(ui_components, error_msg, button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Enhanced preprocessing gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error enhanced preprocessing: {str(e)}", button_manager)
        
        finally:
            button_manager.enable_buttons()
    
    preprocess_button = ui_components.get('preprocess_button')
    if preprocess_button:
        preprocess_button.on_click(execute_enhanced_preprocessing)

def setup_enhanced_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Enhanced check handler dengan multi-split validation"""
    
    def execute_enhanced_check(button=None):
        button_manager = get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            if logger:
                logger.info("🔍 Memulai enhanced dataset check")
            
            setup_enhanced_progress_tracker(ui_components, "Enhanced Dataset Check")
            
            # Enhanced validation
            valid, source_msg = validate_dataset_ready(config, logger)
            
            if valid:
                if logger:
                    logger.success(f"✅ {source_msg}")
                
                # Enhanced preprocessed check dengan split breakdown
                preprocessed_exists, preprocessed_count = check_preprocessed_exists(config)
                
                if preprocessed_exists:
                    if logger:
                        logger.success(f"💾 Enhanced preprocessed dataset: {preprocessed_count:,} gambar")
                    _show_enhanced_preprocessed_breakdown(ui_components, config)
                else:
                    if logger:
                        logger.info("ℹ️ Belum ada enhanced preprocessed dataset")
                
                # Enhanced multi-split validation
                _validate_multi_split_readiness(ui_components, config, logger)
                
                complete_progress_tracker(ui_components, "Enhanced dataset check selesai")
                show_ui_success(ui_components, f"Enhanced check: {source_msg.split(': ')[1] if ': ' in source_msg else source_msg}", button_manager)
            else:
                handle_ui_error(ui_components, f"❌ {source_msg}", button_manager)
                
        except Exception as e:
            error_progress_tracker(ui_components, f"Enhanced check gagal: {str(e)}")
            handle_ui_error(ui_components, f"❌ Error enhanced check: {str(e)}", button_manager)
        
        finally:
            button_manager.enable_buttons()
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_enhanced_check)

def setup_enhanced_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Enhanced cleanup handler dengan multi-split awareness"""
    
    def execute_enhanced_cleanup(button=None):
        logger = ui_components.get('logger')
        
        clear_outputs(ui_components)
        
        try:
            if logger:
                logger.info("🧹 Checking enhanced cleanup targets")
            
            has_data, count = check_preprocessed_exists(config)
            if not has_data:
                show_ui_success(ui_components, "ℹ️ Tidak ada enhanced preprocessed data untuk dibersihkan")
                return
            
            def confirmed_enhanced_cleanup():
                button_manager = get_button_manager(ui_components)
                button_manager.disable_buttons('cleanup_button')
                
                try:
                    if logger:
                        logger.info("🧹 Memulai enhanced cleanup preprocessed data")
                    
                    setup_enhanced_progress_tracker(ui_components, "Enhanced Dataset Cleanup")
                    
                    cleanup_service = create_backend_cleanup_service(config, logger)
                    if not cleanup_service:
                        handle_ui_error(ui_components, "❌ Gagal membuat enhanced cleanup service", button_manager)
                        return
                    
                    cleanup_service.register_progress_callback(ui_components['progress_callback'])
                    
                    # Enhanced cleanup dengan multi-split awareness
                    result = cleanup_service.cleanup_enhanced_preprocessed_data(
                        safe_mode=True,
                        preserve_splits=True
                    )
                    
                    if result and result.get('success', False):
                        stats = result.get('stats', {})
                        files_removed = stats.get('files_removed', 0)
                        splits_cleaned = len(stats.get('splits_cleaned', []))
                        
                        complete_progress_tracker(ui_components, f"Enhanced cleanup selesai: {files_removed:,} file dari {splits_cleaned} split")
                        show_ui_success(ui_components, f"🧹 Enhanced cleanup berhasil: {files_removed:,} file", button_manager)
                    else:
                        error_msg = result.get('message', 'Unknown enhanced cleanup error') if result else 'No response from enhanced service'
                        handle_ui_error(ui_components, error_msg, button_manager)
                        
                except Exception as e:
                    error_progress_tracker(ui_components, f"Enhanced cleanup gagal: {str(e)}")
                    handle_ui_error(ui_components, f"❌ Error enhanced cleanup: {str(e)}", button_manager)
                
                finally:
                    button_manager.enable_buttons()
            
            # Enhanced confirmation dialog
            confirmation_area = ui_components.get('confirmation_area')
            if confirmation_area:
                from IPython.display import display, clear_output
                with confirmation_area:
                    clear_output(wait=True)
                    dialog = create_destructive_confirmation(
                        title="⚠️ Konfirmasi Enhanced Cleanup",
                        message=f"Operasi ini akan menghapus {count:,} file preprocessed dari semua split.\n\nData asli tetap aman. Lanjutkan enhanced cleanup?",
                        on_confirm=lambda b: (confirmed_enhanced_cleanup(), clear_outputs(ui_components)),
                        on_cancel=lambda b: clear_outputs(ui_components),
                        item_name="enhanced preprocessed data"
                    )
                    display(dialog)
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error enhanced cleanup check: {str(e)}")
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_enhanced_cleanup)

def _extract_enhanced_processing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract enhanced parameters dari UI components"""
    # Resolution extraction
    resolution = getattr(ui_components.get('resolution_dropdown'), 'value', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    # Normalization extraction
    normalization_method = getattr(ui_components.get('normalization_dropdown'), 'value', 'minmax')
    
    # Multi-select target splits extraction
    target_splits_widget = ui_components.get('target_splits_select')
    target_splits = list(target_splits_widget.value) if target_splits_widget and hasattr(target_splits_widget, 'value') and target_splits_widget.value else ['train', 'valid']
    
    # Enhanced parameters extraction
    batch_size = getattr(ui_components.get('batch_size_input'), 'value', 32)
    preserve_aspect_ratio = getattr(ui_components.get('preserve_aspect_checkbox'), 'value', True)
    validation_enabled = getattr(ui_components.get('validation_checkbox'), 'value', True)
    move_invalid = getattr(ui_components.get('move_invalid_checkbox'), 'value', True)
    invalid_dir = getattr(ui_components.get('invalid_dir_input'), 'value', 'data/invalid').strip() or 'data/invalid'
    
    return {
        'target_size': [width, height],
        'target_splits': target_splits,
        'normalization': {
            'enabled': normalization_method != 'none',
            'method': normalization_method if normalization_method != 'none' else 'minmax'
        },
        'batch_size': max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32,
        'preserve_aspect_ratio': preserve_aspect_ratio,
        'validation_enabled': validation_enabled,
        'move_invalid': move_invalid,
        'invalid_dir': invalid_dir,
        'validation_config': {
            'enabled': validation_enabled,
            'move_invalid': move_invalid,
            'invalid_dir': invalid_dir
        }
    }

def _show_enhanced_preprocessed_breakdown(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Show enhanced preprocessed breakdown dengan split details"""
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
    
    if ui_components.get('log_output') and breakdown:
        report_html = f"""
        <div style="background:#f0f8ff;padding:12px;border-radius:6px;margin:10px 0;border-left:4px solid #28a745;">
            <strong style="color:#495057;">📊 Enhanced Preprocessed Dataset Breakdown:</strong><br>
            <div style="margin:10px 0;">
                <ul style="margin:8px 0;padding-left:20px;list-style-type:none;">
        """
        
        for split, count in breakdown.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            split_emoji = {'train': '🏋️', 'valid': '✅', 'test': '🧪'}.get(split, '📁')
            report_html += f"""
                <li style="margin:4px 0;">
                    <strong>{split_emoji} {split.title()}:</strong> 
                    <span style="color:#28a745;">{count:,} gambar</span> 
                    <span style="color:#6c757d;">({percentage:.1f}%)</span>
                </li>
            """
        
        report_html += f"""
                </ul>
            </div>
            <div style="margin-top:10px;padding:8px;background:#e7f3ff;border-radius:4px;border-left:3px solid #007bff;">
                <strong style="color:#0056b3;">🎯 Total Enhanced:</strong> 
                <span style="color:#28a745;font-size:16px;">{total_images:,}</span> 
                gambar siap untuk training
            </div>
        </div>
        """
        
        with ui_components['log_output']:
            display(HTML(report_html))

def _validate_multi_split_readiness(ui_components: Dict[str, Any], config: Dict[str, Any], logger):
    """Validate readiness untuk multi-split processing"""
    from pathlib import Path
    
    data_dir = Path(config.get('data', {}).get('dir', 'data'))
    required_splits = ['train', 'valid', 'test']
    available_splits = []
    
    for split in required_splits:
        split_path = data_dir / split
        if split_path.exists():
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len([f for f in images_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                if image_count > 0:
                    available_splits.append(split)
    
    if logger:
        if len(available_splits) > 1:
            logger.success(f"🎯 Multi-split ready: {', '.join(available_splits)}")
        elif len(available_splits) == 1:
            logger.info(f"📂 Single split available: {available_splits[0]}")
        else:
            logger.warning("⚠️ No valid splits found untuk multi-split processing")