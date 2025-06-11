"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Handlers dengan API integration dan progress tracking
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan API integration"""
    logger = get_logger('preprocessing_handlers')
    
    try:
        # Setup config handlers
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers dengan API
        _setup_operation_handlers(ui_components)
        
        logger.info("✅ Preprocessing handlers berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup handlers: {str(e)}")
        return ui_components

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers"""
    
    def save_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.save_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error save: {str(e)}", "error")
    
    def reset_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                config_handler.reset_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error reset: {str(e)}", "error")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)

def _setup_operation_handlers(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan API integration"""
    
    def preprocessing_handler(button=None):
        return _handle_preprocessing_operation(ui_components)
    
    def check_handler(button=None):
        return _handle_check_operation(ui_components)
    
    def cleanup_handler(button=None):
        return _handle_cleanup_operation(ui_components)
    
    # Bind operation handlers
    if preprocess_button := ui_components.get('preprocess_button'):
        preprocess_button.on_click(preprocessing_handler)
    if check_button := ui_components.get('check_button'):
        check_button.on_click(check_handler)
    if cleanup_button := ui_components.get('cleanup_button'):
        cleanup_button.on_click(cleanup_handler)

def _handle_preprocessing_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle preprocessing dengan confirmation dan API"""
    try:
        # Clear outputs
        _clear_outputs(ui_components)
        
        # Check if should execute (setelah confirmation)
        if _should_execute_preprocessing(ui_components):
            return _execute_preprocessing_with_api(ui_components)
        
        # Show confirmation jika belum
        if not _is_confirmation_pending(ui_components):
            _show_preprocessing_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"Error preprocessing: {str(e)}")
        return False

def _handle_check_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle dataset check dengan API"""
    try:
        _clear_outputs(ui_components)
        _disable_buttons(ui_components)
        
        # Setup progress
        _setup_progress(ui_components, "🔍 Memeriksa dataset...")
        
        # Execute check menggunakan API
        from smartcash.dataset.preprocessor import validate_dataset, get_preprocessing_status
        
        config = _extract_config(ui_components)
        progress_callback = _create_progress_callback(ui_components)
        
        # Validate source
        _log_to_ui(ui_components, "🔍 Validating source dataset...", "info")
        target_split = config.get('preprocessing', {}).get('target_splits', ['train'])[0]
        validation_result = validate_dataset(config=config, target_split=target_split)
        
        # Check preprocessed status
        _log_to_ui(ui_components, "💾 Checking preprocessed status...", "info")
        status_result = get_preprocessing_status(config=config)
        
        # Show results
        if validation_result.get('success', False):
            summary = validation_result.get('summary', {})
            preprocessed_info = status_result.get('preprocessed_data', {}) if status_result.get('success') else {}
            
            source_msg = f"Dataset sumber: {summary.get('total_images', 0):,} gambar"
            preprocessed_msg = f"Preprocessed: {preprocessed_info.get('total_files', 0):,} files" if preprocessed_info.get('exists') else "Belum ada data preprocessed"
            
            final_msg = f"✅ {source_msg} | {preprocessed_msg}"
            
            _complete_progress(ui_components, final_msg)
            _log_to_ui(ui_components, final_msg, "success")
        else:
            error_msg = validation_result.get('message', 'Validation failed')
            _error_progress(ui_components, error_msg)
        
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"Error check: {str(e)}")
        return False

def _handle_cleanup_operation(ui_components: Dict[str, Any]) -> bool:
    """Handle cleanup dengan confirmation dan API"""
    try:
        _clear_outputs(ui_components)
        
        # Check if should execute
        if _should_execute_cleanup(ui_components):
            return _execute_cleanup_with_api(ui_components)
        
        # Show confirmation jika belum
        if not _is_confirmation_pending(ui_components):
            _show_cleanup_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"Error cleanup: {str(e)}")
        return False

def _execute_preprocessing_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute preprocessing menggunakan API"""
    try:
        from smartcash.dataset.preprocessor import preprocess_dataset
        
        _disable_buttons(ui_components)
        _setup_progress(ui_components, "🚀 Memulai preprocessing...")
        
        config = _extract_config(ui_components)
        progress_callback = _create_progress_callback(ui_components)
        
        _log_to_ui(ui_components, "🏗️ Starting preprocessing pipeline...", "info")
        
        result = preprocess_dataset(
            config=config,
            ui_components=ui_components,
            progress_callback=progress_callback
        )
        
        if result.get('success', False):
            stats = result.get('stats', {})
            processed_count = stats.get('output', {}).get('total_processed', 0)
            processing_time = result.get('processing_time', 0)
            
            success_msg = f"Preprocessing berhasil: {processed_count:,} gambar diproses dalam {processing_time:.1f}s"
            
            _complete_progress(ui_components, success_msg)
            _log_to_ui(ui_components, success_msg, "success")
            
            # Log stats detail
            if 'output' in stats:
                output_stats = stats['output']
                success_rate = output_stats.get('success_rate', '100%')
                _log_to_ui(ui_components, f"📈 Success rate: {success_rate}", "info")
            
            _enable_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            _error_progress(ui_components, error_msg)
            return False
            
    except Exception as e:
        _handle_error(ui_components, f"API preprocessing error: {str(e)}")
        return False

def _execute_cleanup_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute cleanup menggunakan API"""
    try:
        from smartcash.dataset.preprocessor import cleanup_preprocessed_data
        
        _disable_buttons(ui_components)
        _setup_progress(ui_components, "🗑️ Memulai cleanup...")
        
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        _log_to_ui(ui_components, f"🧹 Cleaning up {cleanup_target}...", "info")
        
        result = cleanup_preprocessed_data(
            config=config,
            target_split=None,  # Cleanup semua splits
            ui_components=ui_components
        )
        
        if result.get('success', False):
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            
            success_msg = f"Cleanup berhasil: {files_removed:,} files dihapus"
            
            _complete_progress(ui_components, success_msg)
            _log_to_ui(ui_components, success_msg, "success")
            
            _enable_buttons(ui_components)
            return True
        else:
            error_msg = result.get('message', 'Cleanup failed')
            _error_progress(ui_components, error_msg)
            return False
            
    except Exception as e:
        _handle_error(ui_components, f"API cleanup error: {str(e)}")
        return False

# === UTILITY FUNCTIONS ===

def _extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        return extract_preprocessing_config(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error extracting config: {str(e)}", "warning")
        return {}

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk API"""
    def progress_callback(level: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                if level in ['overall', 'primary'] and hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
                elif level in ['step', 'current', 'batch'] and hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(current, message)
            
            # Log milestone progress
            if _is_milestone(current, total):
                _log_to_ui(ui_components, f"📊 {message} ({current}/{total})", "info")
        except Exception:
            pass
    
    return progress_callback

def _is_milestone(current: int, total: int) -> bool:
    """Check if progress adalah milestone"""
    if total <= 10:
        return True
    milestones = [0, 25, 50, 75, 100]
    progress_pct = (current / total) * 100 if total > 0 else 0
    return any(abs(progress_pct - milestone) < 2 for milestone in milestones) or current == total

def _setup_progress(ui_components: Dict[str, Any], message: str):
    """Setup progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        if hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, message)

def _complete_progress(ui_components: Dict[str, Any], message: str):
    """Complete progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(100, f"✅ {message}")

def _error_progress(ui_components: Dict[str, Any], message: str):
    """Set error state pada progress"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, f"❌ {message}")

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log message ke UI"""
    try:
        log_output = ui_components.get('log_output')
        if log_output:
            from IPython.display import display, HTML
            import datetime
            
            colors = {'info': '#2196F3', 'success': '#4CAF50', 'warning': '#FF9800', 'error': '#F44336'}
            color = colors.get(level, '#2196F3')
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            html = f"""
            <div style='margin: 2px 0; padding: 4px; border-left: 3px solid {color};'>
                <span style='color: #666; font-size: 11px;'>[{timestamp}]</span>
                <span style='color: {color}; margin-left: 4px;'>{message}</span>
            </div>
            """
            
            with log_output:
                display(HTML(html))
    except Exception:
        print(f"[{level.upper()}] {message}")

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI outputs"""
    try:
        from smartcash.ui.components.dialog import clear_dialog_area
        clear_dialog_area(ui_components)
    except ImportError:
        # Fallback manual clear
        if confirmation_area := ui_components.get('confirmation_area'):
            with confirmation_area:
                from IPython.display import clear_output
                clear_output(wait=True)
    except Exception:
        pass

def _disable_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons"""
    for btn_key in ['preprocess_button', 'check_button', 'cleanup_button']:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons"""
    for btn_key in ['preprocess_button', 'check_button', 'cleanup_button']:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = False

def _handle_error(ui_components: Dict[str, Any], error_msg: str):
    """Handle error dengan cleanup"""
    _log_to_ui(ui_components, error_msg, "error")
    _error_progress(ui_components, error_msg)
    _enable_buttons(ui_components)

# === CONFIRMATION HANDLERS ===

def _show_preprocessing_confirmation(ui_components: Dict[str, Any]):
    """Show preprocessing confirmation using dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        show_confirmation_dialog(
            ui_components,
            title="🚀 Konfirmasi Preprocessing",
            message="Apakah Anda yakin ingin memulai preprocessing dataset dengan API baru?<br><br>✅ YOLO normalization<br>📊 Real-time progress tracking<br>🔍 Enhanced validation",
            on_confirm=lambda: _set_preprocessing_confirmed(ui_components),
            on_cancel=lambda: _log_to_ui(ui_components, "🚫 Preprocessing dibatalkan", "info"),
            confirm_text="Ya, Mulai",
            cancel_text="Batal"
        )
    except ImportError:
        # Fallback jika dialog components tidak tersedia
        _log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute preprocessing", "warning")
        _set_preprocessing_confirmed(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error showing confirmation: {str(e)}", "warning")

def _show_cleanup_confirmation(ui_components: Dict[str, Any]):
    """Show cleanup confirmation using dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        target_display = {
            'preprocessed': 'data preprocessed (.npy files)',
            'samples': 'sample images (.jpg files)', 
            'both': 'preprocessed data dan sample images'
        }.get(cleanup_target, cleanup_target)
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Cleanup",
            message=f"Anda akan menghapus <strong>{target_display}</strong>.<br><br><span style='color:#dc3545;'>⚠️ Tindakan ini tidak dapat dibatalkan!</span>",
            on_confirm=lambda: _set_cleanup_confirmed(ui_components),
            on_cancel=lambda: _log_to_ui(ui_components, "🚫 Cleanup dibatalkan", "info"),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
    except ImportError:
        # Fallback jika dialog components tidak tersedia
        _log_to_ui(ui_components, "⚠️ Dialog tidak tersedia, langsung execute cleanup", "warning")
        _set_cleanup_confirmed(ui_components)
    except Exception as e:
        _log_to_ui(ui_components, f"⚠️ Error showing cleanup confirmation: {str(e)}", "warning")

def _set_preprocessing_confirmed(ui_components: Dict[str, Any]):
    """Set preprocessing confirmation flag"""
    ui_components['_preprocessing_confirmed'] = True
    _log_to_ui(ui_components, "✅ Preprocessing dikonfirmasi", "success")

def _set_cleanup_confirmed(ui_components: Dict[str, Any]):
    """Set cleanup confirmation flag"""
    ui_components['_cleanup_confirmed'] = True
    _log_to_ui(ui_components, "✅ Cleanup dikonfirmasi", "success")

def _should_execute_preprocessing(ui_components: Dict[str, Any]) -> bool:
    """Check if preprocessing should execute"""
    return ui_components.pop('_preprocessing_confirmed', False)

def _should_execute_cleanup(ui_components: Dict[str, Any]) -> bool:
    """Check if cleanup should execute"""
    return ui_components.pop('_cleanup_confirmed', False)

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check if confirmation dialog is pending"""
    try:
        from smartcash.ui.components.dialog import is_dialog_visible
        return is_dialog_visible(ui_components)
    except ImportError:
        # Fallback check
        return ui_components.get('_dialog_visible', False)
    except Exception:
        return False