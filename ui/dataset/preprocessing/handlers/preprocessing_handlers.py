"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Enhanced handlers dengan preprocessing API integration dan progress tracking
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan API integration dan UI logging"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup config handlers dengan UI integration
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers dengan API baru
        _setup_operation_handlers(ui_components)
        
        logger.info("✅ Preprocessing handlers dengan API integration berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup handlers: {str(e)}")
        return ui_components

# === CONFIG HANDLERS ===

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers dengan UI logging integration"""
    
    def save_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error save config: {str(e)}", "error")
    
    def reset_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            # Set UI components untuk logging  
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error reset config: {str(e)}", "error")
    
    # Bind handlers dengan safety check
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)

# === OPERATION HANDLERS ===

def _setup_operation_handlers(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan API integration"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_operation(ui_components)
    
    def check_handler(button=None):
        return _handle_check_operation(ui_components)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_operation(ui_components)
    
    # Bind handlers dengan safety check
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

# === OPERATION IMPLEMENTATIONS ===

def _handle_preprocessing_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle preprocessing dengan confirmation dan API baru"""
    try:
        _clear_outputs(ui_components)
        
        # Check confirmation flag dan execute jika dikonfirmasi
        if _should_execute_preprocessing(ui_components):
            return _execute_preprocessing_with_api(ui_components)
        
        # Show confirmation dialog jika belum
        if not _is_confirmation_pending(ui_components):
            _show_preprocessing_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Error preprocessing operation: {str(e)}")
        return False

def _handle_check_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle dataset check dengan preprocessing API"""
    try:
        _clear_outputs(ui_components)
        _disable_buttons(ui_components)
        
        _setup_progress(ui_components, "🔍 Memeriksa dataset dengan API baru...")
        
        # Get preprocessing status menggunakan API baru
        from smartcash.dataset.preprocessor import get_preprocessing_status
        
        config = _extract_config(ui_components)
        _log_to_ui(ui_components, "📊 Checking dataset status dengan preprocessing API...", "info")
        
        status_result = get_preprocessing_status(config=config)
        
        # Show results based on API response
        if status_result.get('success', False):
            service_ready = status_result.get('service_ready', False)
            file_stats = status_result.get('file_statistics', {})
            output_directory = status_result.get('output_directory', {})
            configuration = status_result.get('configuration', {})

            if output_directory.get('exists', False):
                _log_to_ui(ui_components, f"📂 Output directory: {output_directory['path']}", "info")
            else:
                _log_to_ui(ui_components, "❌ Output directory tidak ditemukan", "error")
            
            _log_to_ui(ui_components, f"🎯 Configuration: {configuration}", "info")
            # Format results
            raw_images = file_stats.get('raw_images', 0)
            preprocessed_files = file_stats.get('preprocessed_files', 0)
            service_msg = "✅ Siap" if service_ready else "⚠️ Belum siap"
            
            final_msg = f"Dataset: {raw_images:,} raw images | Preprocessed: {preprocessed_files:,} files | Service: {service_msg}"
            
            _complete_progress(ui_components, final_msg)
            _log_to_ui(ui_components, final_msg, "success")
            
            # Log layer analysis jika ada
            if 'layer_analysis' in file_stats:
                layer_info = file_stats['layer_analysis']
                main_objects = layer_info.get('l1_main', {}).get('objects', 0)
                _log_to_ui(ui_components, f"🏦 Main banknotes detected: {main_objects:,} objects", "info")
                
        else:
            error_msg = status_result.get('message', 'Status check failed')
            _error_progress(ui_components, error_msg)
        
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Error check operation: {str(e)}")
        return False

def _handle_cleanup_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle cleanup dengan confirmation dan API baru"""
    try:
        _clear_outputs(ui_components)
        
        # Check confirmation flag dan execute jika dikonfirmasi
        if _should_execute_cleanup(ui_components):
            return _execute_cleanup_with_api(ui_components)
        
        # Show confirmation dialog jika belum
        if not _is_confirmation_pending(ui_components):
            _show_cleanup_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Error cleanup operation: {str(e)}")
        return False

# === API EXECUTION FUNCTIONS ===

def _execute_preprocessing_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute preprocessing menggunakan preprocessing API baru"""
    try:
        _disable_buttons(ui_components)
        _setup_progress(ui_components, "🚀 Memulai preprocessing dengan API baru...")
        
        # Import preprocessing API
        from smartcash.dataset.preprocessor import preprocess_dataset
        
        config = _extract_config(ui_components)
        progress_callback = _create_progress_callback(ui_components)
        
        _log_to_ui(ui_components, "🔧 Starting YOLO preprocessing pipeline...", "info")
        
        # Execute preprocessing dengan API baru
        result = preprocess_dataset(
            config=config,
            progress_callback=progress_callback
        )
        
        # Handle results
        if result.get('success', False):
            stats = result.get('stats', {})
            processing_time = result.get('processing_time', 0)
            
            # Extract processing statistics
            overview = stats.get('overview', {})
            processed_count = overview.get('total_files', 0)
            success_rate = overview.get('success_rate', '100%')
            
            success_msg = f"✅ Preprocessing berhasil: {processed_count:,} files diproses dalam {processing_time:.1f}s (Success rate: {success_rate})"
            
            _complete_progress(ui_components, success_msg)
            _log_to_ui(ui_components, success_msg, "success")
            
            # Log banknote analysis jika ada
            if 'main_banknotes' in stats:
                banknote_stats = stats['main_banknotes']
                total_objects = banknote_stats.get('total_objects', 0)
                _log_to_ui(ui_components, f"🏦 Main banknotes processed: {total_objects:,} objects", "info")
            
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            _error_progress(ui_components, error_msg)
            _log_to_ui(ui_components, f"❌ {error_msg}", "error")
            return False
            
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ API preprocessing error: {str(e)}")
        return False

def _execute_cleanup_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute cleanup menggunakan preprocessing API baru"""
    try:
        _disable_buttons(ui_components)
        _setup_progress(ui_components, "🗑️ Memulai cleanup dengan API baru...")
        
        # Import cleanup API  
        from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files
        
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        data_dir = config.get('data', {}).get('dir', 'data')
        
        progress_callback = _create_progress_callback(ui_components)
        
        _log_to_ui(ui_components, f"🧹 Cleaning up {cleanup_target} files...", "info")
        
        # Execute cleanup dengan API baru
        result = cleanup_preprocessing_files(
            data_dir=data_dir,
            target=cleanup_target,
            splits=target_splits,
            confirm=True,  # Already confirmed via dialog
            progress_callback=progress_callback
        )
        
        # Handle results
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            splits_cleaned = stats.get('splits_cleaned', [])
            
            success_msg = f"✅ Cleanup berhasil: {files_removed:,} files dihapus dari {len(splits_cleaned)} splits"
            
            _complete_progress(ui_components, success_msg)
            _log_to_ui(ui_components, success_msg, "success")
            
            # Log detail per split jika ada
            if 'split_stats' in stats:
                for split, split_stat in stats['split_stats'].items():
                    removed = split_stat.get('files_removed', 0)
                    _log_to_ui(ui_components, f"  📁 {split}: {removed:,} files removed", "info")
            
        else:
            error_msg = result.get('message', 'Cleanup failed')
            _error_progress(ui_components, error_msg)
            _log_to_ui(ui_components, f"❌ {error_msg}", "error")
            return False
            
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ API cleanup error: {str(e)}")
        return False

# === PROGRESS CALLBACK ===

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk preprocessing API - hanya update progress tracker"""
    def progress_callback(level: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Calculate percentage
            progress_percent = int((current / total) * 100) if total > 0 else 0
            
            # Map API level ke tracker method sesuai dokumentasi
            if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(progress_percent, message)
            elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                progress_tracker.update_current(progress_percent, message)
            elif level in ['step', 'batch'] and hasattr(progress_tracker, 'update_current'):
                # Map step/batch ke current untuk dual progress tracker
                progress_tracker.update_current(progress_percent, message)
            
            # Tidak log ke UI untuk menghindari flooding
        except Exception:
            pass  # Silent fail untuk menghindari interrupt processing
    
    return progress_callback

# === CONFIRMATION HANDLERS ===

def _show_preprocessing_confirmation(ui_components: Dict[str, Any]):
    """Show preprocessing confirmation dengan API info"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        show_confirmation_dialog(
            ui_components,
            title="🚀 Konfirmasi Preprocessing",
            message="Mulai preprocessing dataset dengan API baru?<br><br>✅ <strong>YOLO normalization</strong> dengan aspect ratio preservation<br>📊 <strong>Real-time progress tracking</strong><br>🔍 <strong>Minimal validation</strong> untuk performa optimal<br>🎯 <strong>Main banknotes analysis</strong> (7 classes)",
            on_confirm=lambda: _set_preprocessing_confirmed(ui_components),
            on_cancel=lambda: _handle_preprocessing_cancel(ui_components),
            confirm_text="🚀 Mulai Preprocessing",
            cancel_text="❌ Batal"
        )
    except ImportError:
        _log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _set_preprocessing_confirmed(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error showing confirmation: {str(e)}", "warning")

def _show_cleanup_confirmation(ui_components: Dict[str, Any]):
    """Show cleanup confirmation dengan preview info"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        # Target descriptions
        target_descriptions = {
            'preprocessed': '<strong>preprocessing files</strong> (pre_*.npy + pre_*.txt)',
            'samples': '<strong>sample images</strong> (sample_*.jpg)',
            'both': '<strong>preprocessing files dan sample images</strong>'
        }
        
        target_desc = target_descriptions.get(cleanup_target, cleanup_target)
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Cleanup",
            message=f"Hapus {target_desc} dari dataset?<br><br><span style='color:#dc3545;'>⚠️ <strong>Tindakan ini tidak dapat dibatalkan!</strong></span><br><br>🗑️ Files akan dihapus permanent<br>📊 Progress tracking tersedia",
            on_confirm=lambda: _set_cleanup_confirmed(ui_components),
            on_cancel=lambda: _handle_cleanup_cancel(ui_components),
            confirm_text="🗑️ Ya, Hapus",
            cancel_text="❌ Batal",
            danger_mode=True
        )
    except ImportError:
        _log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _set_cleanup_confirmed(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error showing cleanup confirmation: {str(e)}", "warning")

# === CONFIRMATION STATE MANAGEMENT ===

def _set_preprocessing_confirmed(ui_components: Dict[str, Any]):
    """Set preprocessing confirmation flag dan trigger execution"""
    ui_components['_preprocessing_confirmed'] = True
    _log_to_ui(ui_components, "✅ Preprocessing dikonfirmasi, memulai...", "success")
    _hide_confirmation_area(ui_components)
    _execute_preprocessing_with_api(ui_components)

def _set_cleanup_confirmed(ui_components: Dict[str, Any]):
    """Set cleanup confirmation flag dan trigger execution"""
    ui_components['_cleanup_confirmed'] = True
    _log_to_ui(ui_components, "✅ Cleanup dikonfirmasi, memulai...", "success")
    _hide_confirmation_area(ui_components)
    _execute_cleanup_with_api(ui_components)

def _handle_preprocessing_cancel(ui_components: Dict[str, Any]):
    """Handle preprocessing cancellation"""
    _clear_outputs(ui_components)
    _log_to_ui(ui_components, "🚫 Preprocessing dibatalkan oleh user", "info")

def _handle_cleanup_cancel(ui_components: Dict[str, Any]):
    """Handle cleanup cancellation"""
    _clear_outputs(ui_components)
    _log_to_ui(ui_components, "🚫 Cleanup dibatalkan oleh user", "info")

def _should_execute_preprocessing(ui_components: Dict[str, Any]) -> bool:
    """Check if preprocessing should execute (consume confirmation flag)"""
    return ui_components.pop('_preprocessing_confirmed', False)

def _should_execute_cleanup(ui_components: Dict[str, Any]) -> bool:
    """Check if cleanup should execute (consume confirmation flag)"""
    return ui_components.pop('_cleanup_confirmed', False)

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check if confirmation dialog is pending"""
    try:
        from smartcash.ui.components.dialog import is_dialog_visible
        return is_dialog_visible(ui_components)
    except ImportError:
        return ui_components.get('_dialog_visible', False)
    except Exception:
        return False

# === UTILITY FUNCTIONS ===

def _extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        return extract_preprocessing_config(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error extracting config: {str(e)}", "warning")
        return {'data': {'dir': 'data'}, 'preprocessing': {'target_splits': ['train', 'valid']}}

def _setup_progress(ui_components: Dict[str, Any], message: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        if hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, message)

def _complete_progress(ui_components: Dict[str, Any], message: str):
    """Complete progress tracker dengan success message"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(100, message)

def _error_progress(ui_components: Dict[str, Any], message: str):
    """Set error state pada progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, f"❌ {message}")

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log message ke UI dengan timestamp dan styling"""
    try:
        log_output = ui_components.get('log_output')
        if not log_output:
            return
            
        from IPython.display import display, HTML
        import datetime
        
        # Color mapping untuk different levels
        colors = {
            'info': '#2196F3', 'success': '#4CAF50', 
            'warning': '#FF9800', 'error': '#F44336'
        }
        color = colors.get(level, '#2196F3')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Clean message dari duplicate emoji
        clean_msg = message.strip()
        
        html = f"""
        <div style='margin: 2px 0; padding: 4px 8px; border-left: 3px solid {color}; 
                    background-color: rgba(248,249,250,0.8); border-radius: 4px;'>
            <span style='color: #666; font-size: 11px;'>[{timestamp}]</span>
            <span style='color: {color}; margin-left: 4px;'>{clean_msg}</span>
        </div>
        """
        
        with log_output:
            display(HTML(html))
            
    except Exception:
        # Fallback ke print jika UI logging gagal
        print(f"[{level.upper()}] {message}")

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area dengan visibility control"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = '0px'

def _show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear outputs dengan hiding confirmation area"""
    _hide_confirmation_area(ui_components)

def _disable_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons during processing"""
    for btn_key in ['preprocess_button', 'check_button', 'cleanup_button']:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons after processing"""
    for btn_key in ['preprocess_button', 'check_button', 'cleanup_button']:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = False

def _handle_error(ui_components: Dict[str, Any], error_msg: str):
    """Handle error dengan logging dan cleanup"""
    _log_to_ui(ui_components, error_msg, "error")
    _error_progress(ui_components, error_msg)
    _enable_buttons(ui_components)