"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Handlers untuk preprocessing dengan proper error handling dan button state management
"""

from typing import Dict, Any, Optional
from functools import wraps

from smartcash.common.exceptions import ErrorContext, UIError
from smartcash.ui.utils import (
    ErrorHandler,
    create_error_context,
    with_error_handling
)
from smartcash.dataset.preprocessor.api import get_preprocessing_status

# Import UI utility functions under a single namespace
from smartcash.ui.dataset.preprocessing import utils as ui_utils

# Initialize error handler
error_handler = ErrorHandler()

# Import shared dialog components
from smartcash.ui.components.dialog.confirmation_dialog import (
    show_confirmation_dialog,
    clear_dialog_area
)

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan API integration dan proper error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi preprocessing
        env: Environment (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate dengan handlers
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
    
    try:
        # Setup config handlers dengan UI integration
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers dengan API baru
        _setup_operation_handlers(ui_components)
        
        logger_bridge.info("✅ Preprocessing handlers dengan API integration berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger_bridge.error(f"❌ Error setup handlers: {str(e)}")
        return ui_components

# === CONFIG HANDLERS ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="setup_config_handlers"
)
def _setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup save/reset handlers dengan UI logging integration
    
    Args:
        ui_components: Dictionary containing UI components including buttons and logger
    """
    logger = ui_components.get('logger_bridge')
    if not logger:
        raise ValueError("Logger bridge not initialized. Make sure to use PreprocessingInitializer")
    
    # Get buttons from ui_components
    save_btn = ui_components.get('save_button')
    reset_btn = ui_components.get('reset_button')
    
    if not save_btn or not reset_btn:
        logger.warning("Save/Reset buttons not found in UI components")
        return
    
    # Setup save handler
    @save_btn.on_click
    @with_error_handling(
        error_handler=error_handler,
        component="preprocessing_handlers",
        operation="save_config"
    )
    def on_save():
        """Handle save button click with config validation and saving"""
        if logger:
            logger.info("Saving preprocessing configuration...")
        
        # Get and validate config
        config = _extract_config(ui_components)
        if not config:
            raise ValueError("Invalid configuration. Please check your inputs.")
        
        # Save config
        config_handler = PreprocessingConfigHandler()
        config_handler.ui_components = ui_components
        config_handler.logger_bridge = logger
        config_handler.save_config(config)
        
        if logger:
            logger.success("Configuration saved successfully!")
    
    # Setup reset handler
    @reset_btn.on_click
    @with_error_handling(
        error_handler=error_handler,
        component="preprocessing_handlers",
        operation="reset_config"
    )
    def on_reset():
        """Handle reset button click with confirmation"""
        if logger:
            logger.info("Resetting configuration to defaults...")
        
        # Reset UI to default values
        config_handler = PreprocessingConfigHandler()
        config_handler.ui_components = ui_components
        config_handler.logger_bridge = logger
        config_handler.reset_ui()
        
        if logger:
            logger.success("Configuration reset to defaults")

# === OPERATION HANDLERS ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="setup_operation_handlers"
)
def _setup_operation_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup operation handlers for preprocessing UI components.
    
    Args:
        ui_components: Dictionary containing UI components including buttons and logger
    """
    logger = ui_components.get('logger_bridge')
    if not logger:
        raise ValueError("Logger bridge not initialized. Make sure to use PreprocessingInitializer")
    
    # Get action buttons from ui_components
    action_components = ui_components.get('action_buttons', {})
    if not action_components:
        logger.warning("Action buttons not found in UI components")
        return
    
    # Get individual buttons
    preprocess_btn = action_components.get('preprocess_btn')
    check_btn = action_components.get('check_btn')
    cleanup_btn = action_components.get('cleanup_btn')
    
    # Clear any existing handlers using a safer approach
    def clear_button_handlers(button):
        """Safely clear all click handlers from a button.
        
        Args:
            button: The button widget to clear handlers from
        """
        if not button:
            return
            
        # Try different methods to clear handlers based on widget type
        if hasattr(button, 'on_click') and callable(getattr(button, 'on_click', None)):
            # For ipywidgets buttons
            button.on_click = None
        elif hasattr(button, 'click') and hasattr(button.click, 'clear'):
            # For buttons with a click manager
            button.click.clear()
        elif hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
            # Fallback to direct access (less preferred)
            button._click_handlers.callbacks.clear()
    
    # Clear handlers for all buttons
    for btn in [preprocess_btn, check_btn, cleanup_btn]:
        clear_button_handlers(btn)
    
    # Setup preprocessing handler
    if preprocess_btn:
        @preprocess_btn.on_click
        def on_preprocess(_):
            _handle_preprocessing_operation(ui_components)
    else:
        logger.warning("Preprocess button not found in UI components")
    
    # Setup check handler
    if check_btn:
        @check_btn.on_click
        def on_check(_):
            _handle_check_operation(ui_components)
    else:
        logger.warning("Check button not found in UI components")
    
    # Setup cleanup handler
    if cleanup_btn:
        @cleanup_btn.on_click
        def on_cleanup(_):
            _handle_cleanup_operation(ui_components)
    else:
        logger.warning("Cleanup button not found in UI components")
    
    logger.debug("Operation handlers setup completed")
    
    if logger:
        logger.debug("Operation handlers setup completed")

# === OPERATION IMPLEMENTATIONS ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="handle_preprocessing"
)
def _handle_preprocessing_operation(ui_components: Dict[str, Any]) -> None:
    """Handle preprocessing operation with confirmation and proper error handling.
    
    Args:
        ui_components: Dictionary containing UI components including logger and buttons
    """
    logger = ui_components.get('logger_bridge')
    if logger:
        logger.debug("Starting preprocessing operation...")
    
    # Check if there's already a pending confirmation
    if _is_confirmation_pending(ui_components):
        if logger:
            logger.warning("Ada operasi konfirmasi yang sedang menunggu")
        return
    
    # Show confirmation dialog
    _show_preprocessing_confirmation(ui_components)
    
    # If confirmed, execute preprocessing
    if _should_execute_preprocessing(ui_components):
        _execute_preprocessing_with_api(ui_components)

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="handle_check_operation"
)
def _handle_check_operation(ui_components: Dict[str, Any]) -> None:
    """Handle dataset check operation with proper error handling and logging.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    logger = ui_components.get('logger_bridge')
    if logger:
        logger.info("🔍 Memulai pemeriksaan dataset...")
    
    # Clear previous outputs
    ui_utils.clear_outputs(ui_components)
    
    # Get config with error handling
    config = _extract_config(ui_components)
    if not config:
        raise ValueError("Konfigurasi tidak valid atau kosong")
    
    # Setup progress tracking
    progress_callback = _create_progress_callback(ui_components)
    
    # Call the API with proper error handling
    result = get_preprocessing_status(
        config=config
    )
    # Add progress callback if supported in the future
    if progress_callback:
        progress_callback(1, 1, "Pemeriksaan selesai")
    
    # Process and display results
    _process_status_result(ui_components, result)
    
    # Log success
    if logger:
        logger.success("✅ Pemeriksaan dataset selesai")
    
    # Update status panel
    status_panel = ui_components.get('status_panel')
    if status_panel and hasattr(status_panel, 'update_status'):
        status_panel.update_status("Pemeriksaan dataset selesai", 'success')

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="handle_cleanup_operation"
)
def _handle_cleanup_operation(ui_components: Dict[str, Any]) -> None:
    """Handle cleanup operation with proper error handling and logging.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    logger = ui_components.get('logger_bridge')
    if logger:
        logger.debug("Memulai operasi cleanup...")
    
    # Clear outputs before starting
    ui_utils.clear_outputs(ui_components)
    
    # Check if there's already a pending confirmation
    if _is_confirmation_pending(ui_components):
        if logger:
            logger.warning("Ada operasi konfirmasi yang sedang menunggu")
        return
    
    # Show confirmation dialog
    _show_cleanup_confirmation(ui_components)
    
    # If confirmed, execute cleanup
    if _should_execute_cleanup(ui_components):
        _execute_cleanup_with_api(ui_components)

# === API EXECUTION FUNCTIONS ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="execute_preprocessing"
)
def _execute_preprocessing_with_api(ui_components: Dict[str, Any]) -> None:
    """Execute preprocessing with proper error handling and progress tracking.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info("🚀 Starting preprocessing pipeline...")
    
    # Disable UI buttons during processing
    ui_utils.disable_buttons(ui_components)
    
    # Setup progress tracking
    ui_utils.setup_progress(ui_components, "🚀 Starting preprocessing...")
    
    try:
        # Get configuration
        config = _extract_config(ui_components)
        if not config:
            raise ValueError("Invalid or empty configuration")
        
        # Create progress callback
        progress_callback = _create_progress_callback(ui_components)
        
        # Log start of preprocessing
        if logger_bridge:
            logger_bridge.info("🔧 Starting YOLO preprocessing pipeline...")
        
        # Update status panel
        status_panel = ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'update_status'):
            status_panel.update_status("Starting preprocessing...", 'info')
        
        # Execute preprocessing
        result = preprocess_dataset(
            config=config,
            progress_callback=progress_callback,
            ui_components=ui_components
        )
        
        # Handle results
        if result.get('success', False):
            _process_success_result(ui_components, result)
            if logger_bridge:
                logger_bridge.success("✅ Preprocessing completed successfully")
            if status_panel and hasattr(status_panel, 'update_status'):
                status_panel.update_status("Preprocessing completed", 'success')
        else:
            error_msg = result.get('message', 'Preprocessing failed')
            raise RuntimeError(f"Preprocessing failed: {error_msg}")
            
    finally:
        # Always re-enable buttons when done
        ui_utils.enable_buttons(ui_components)

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="execute_cleanup"
)
def _execute_cleanup_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute cleanup with proper error handling and button management
    
    Args:
        ui_components: Dictionary containing UI components including logger
        
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    try:
        logger_bridge = ui_components.get('logger_bridge')
        if not logger_bridge:
            raise ValueError("Logger bridge not initialized in UI components")
        
        ui_utils.disable_buttons(ui_components)
        ui_utils.setup_progress(ui_components, "🗑️ Starting cleanup with new API...")
        
        # Import cleanup API  
        from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files
        
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        data_dir = config.get('data', {}).get('dir', 'data')
        
        progress_callback = _create_progress_callback(ui_components)
        
        logger_bridge.info(f"🧹 Cleaning up {cleanup_target} files...")
        
        # Execute cleanup with new API
        result = cleanup_preprocessing_files(
            data_dir=data_dir,
            target=cleanup_target,
            splits=target_splits,
            confirm=True,
            progress_callback=progress_callback,
            ui_components=ui_components
        )
            
        # Handle results
        if result.get('success', False):
            _process_cleanup_result(ui_components, result)
            logger_bridge.success("✅ Cleanup completed successfully")
            return True
        else:
            error_msg = result.get('message', 'Cleanup failed')
            ui_utils.error_progress(ui_components, error_msg)
            logger_bridge.error(f"❌ {error_msg}")
            return False
            
    finally:
        # Always re-enable buttons when done
        ui_utils.enable_buttons(ui_components)

def _process_success_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Process dan display success results
    
    Args:
        ui_components: Dictionary berisi komponen UI
        result: Hasil dari operasi preprocessing
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        stats = result.get('stats', {})
        processing_time = result.get('processing_time', 0)
        
        # Extract processing statistics
        overview = stats.get('overview', {})
        processed_count = overview.get('total_files', 0)
        success_rate = overview.get('success_rate', '100%')
        
        success_msg = f"✅ Preprocessing berhasil: {processed_count:,} files diproses dalam {processing_time:.1f}s (Success rate: {success_rate})"
        
        ui_utils.complete_progress(ui_components, success_msg)
        logger_bridge.success(success_msg)
        
        # Log banknote analysis jika ada
        if 'main_banknotes' in stats:
            banknote_stats = stats['main_banknotes']
            total_objects = banknote_stats.get('total_objects', 0)
            logger_bridge.info(f"🏦 Main banknotes processed: {total_objects:,} objects")
            
    except Exception as e:
        error_msg = f"Gagal memproses hasil sukses: {str(e)}"
        logger_bridge.error(error_msg)
        ui_utils.show_error_ui(ui_components, error_msg)

def _process_cleanup_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Process dan display cleanup results
    
    Args:
        ui_components: Dictionary berisi komponen UI
        result: Hasil dari operasi cleanup
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        stats = result.get('stats', {})
        files_removed = stats.get('files_removed', 0)
        splits_cleaned = stats.get('splits_cleaned', [])
        
        success_msg = f"✅ Cleanup berhasil: {files_removed:,} files dihapus dari {len(splits_cleaned)} splits"
        
        ui_utils.complete_progress(ui_components, success_msg)
        logger_bridge.success(success_msg)
        
        # Log detail per split jika ada
        if 'split_stats' in stats:
            for split, split_stat in stats['split_stats'].items():
                removed = split_stat.get('files_removed', 0)
                logger_bridge.info(f"  📁 {split}: {removed:,} files removed")
                
    except Exception as e:
        error_msg = f"Gagal memproses hasil cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        ui_utils.show_error_ui(ui_components, error_msg)

# === PROGRESS CALLBACK ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="create_progress_callback"
)
def _create_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create a progress callback for preprocessing API with error handling.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Callback function to report progress
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.debug("Creating progress callback")
    
    def progress_callback(level: str, current: int, total: int, message: str = "") -> None:
        """Callback function to report operation progress.
        
        Args:
            level: Progress level (overall, current, step, batch)
            current: Current progress value
            total: Total progress value
            message: Optional progress message
        """
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                if logger_bridge:
                    logger_bridge.warning("Progress tracker not found in UI components")
                return
            
            # Validate inputs
            if total <= 0:
                if logger_bridge:
                    logger_bridge.warning(f"Invalid total progress value: {total}")
                return
                
            # Calculate percentage with bounds checking
            progress_percent = max(0, min(100, int((current / total) * 100)))
            
            # Log progress updates at appropriate levels
            if logger_bridge and logger_bridge.isEnabledFor("DEBUG"):
                logger_bridge.debug(f"Progress update - Level: {level}, Current: {current}, "
                            f"Total: {total}, Percent: {progress_percent}%, Message: {message}")
            
            # Map API level to tracker methods
            method_map = {
                'overall': 'update_overall',
                'current': 'update_current',
                'step': 'update_step',
                'batch': 'update_batch'
            }
            
            method_name = method_map.get(level)
            if method_name and hasattr(progress_tracker, method_name):
                method = getattr(progress_tracker, method_name)
                method(progress_percent, message)
            elif logger_bridge:
                logger_bridge.warning(f"No handler for progress level: {level}")
                
            # Log important progress updates at INFO level
            if logger_bridge and level in ['overall', 'current']:
                log_message = f"{message} ({progress_percent}%)" if message else f"Progress: {progress_percent}%"
                logger_bridge.info(log_message)
                
                # Update status panel for major progress updates
                status_panel = ui_components.get('status_panel')
                if status_panel and hasattr(status_panel, 'update_status'):
                    status_panel.update_status(log_message, 'info')
                
        except Exception as e:
            error_context = create_error_context(
                component="progress_callback",
                operation=level,
                current=current,
                total=total,
                message=message
            )
            error_handler.handle_error(
                e,
                context=error_context,
                ui_components=ui_components,
                log_level="warning"  # Use warning level to avoid spamming errors for progress updates
            )
    
    return progress_callback

# === CONFIRMATION HANDLERS ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="show_preprocessing_confirmation"
)
def _show_preprocessing_confirmation(ui_components: Dict[str, Any]) -> None:
    """Show preprocessing confirmation dialog with API info
    
    Args:
        ui_components: Dictionary containing UI components
    """
    # Show confirmation dialog using the imported function
    show_confirmation_dialog(
        ui_components=ui_components,
        title="Konfirmasi Preprocessing",
        message="Apakah Anda yakin ingin memproses dataset?",
        on_confirm=lambda: _set_preprocessing_confirmed(ui_components),
        on_cancel=lambda: _handle_preprocessing_cancel(ui_components),
        confirm_text="Ya, Proses",
        cancel_text="Batal"
    )
    
    # Log the confirmation dialog display
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info("⏳ Menunggu konfirmasi preprocessing...")

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="show_cleanup_confirmation"
)
def _show_cleanup_confirmation(ui_components: Dict[str, Any]) -> None:
    """Show cleanup confirmation dialog with preview info
    
    Args:
        ui_components: Dictionary containing UI components
    """
    # Get cleanup target from config
    config = _extract_config(ui_components)
    cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
    
    target_descriptions = {
        'preprocessed': 'file preprocessing (pre_*.npy + pre_*.txt)',
        'samples': 'sample images (sample_*.jpg)',
        'both': 'file preprocessing dan sample images'
    }
    
    target_desc = target_descriptions.get(cleanup_target, cleanup_target)
    
    # Show confirmation dialog using the imported function
    show_confirmation_dialog(
        ui_components=ui_components,
        title="🧹 Konfirmasi Cleanup",
        message=f"Hapus {target_desc}?\n\nTindakan ini akan menghapus file-file yang sudah diproses.",
        on_confirm=lambda: _set_cleanup_confirmed(ui_components),
        on_cancel=lambda: _handle_cleanup_cancel(ui_components),
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True
    )
    
    # Log the confirmation dialog display
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info(f"⏳ Menunggu konfirmasi cleanup untuk: {target_desc}...")

# === CONFIRMATION STATE MANAGEMENT ===

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="set_preprocessing_confirmed"
)
def _set_preprocessing_confirmed(ui_components: Dict[str, Any]) -> None:
    """Set preprocessing confirmation flag and trigger execution.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info("✅ Konfirmasi diterima, memulai preprocessing...")
    
    # Set confirmation flag
    ui_components['_preprocessing_confirmed'] = True
    
    try:
        # Execute preprocessing with error handling
        _execute_preprocessing_with_api(ui_components)
    except Exception as e:
        # Create rich error context
        error_context = create_error_context(
            component="preprocessing_handlers",
            operation="execute_preprocessing",
            details={
                'step': 'preprocessing_confirmed',
                'ui_components': list(ui_components.keys())
            }
        )
        raise UIError("Gagal memproses konfirmasi preprocessing", context=error_context) from e

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="set_cleanup_confirmed"
)
def _set_cleanup_confirmed(ui_components: Dict[str, Any]) -> None:
    """Set cleanup confirmation flag and trigger execution.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info("✅ Konfirmasi cleanup diterima, memulai pembersihan...")
    
    # Set confirmation flag
    ui_components['_cleanup_confirmed'] = True
    
    try:
        # Execute cleanup with error handling
        _execute_cleanup_with_api(ui_components)
    except Exception as e:
        # Create rich error context
        error_context = create_error_context(
            component="preprocessing_handlers",
            operation="execute_cleanup",
            details={
                'step': 'cleanup_confirmed',
                'ui_components': list(ui_components.keys())
            }
        )
        raise UIError("Gagal memproses konfirmasi cleanup", context=error_context) from e
    finally:
        # Ensure confirmation area is hidden after operation
        clear_dialog_area(ui_components)

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="handle_preprocessing_cancel"
)
def _handle_preprocessing_cancel(ui_components: Dict[str, Any]) -> None:
    """Handle preprocessing cancellation
    
    Args:
        ui_components: Dictionary containing UI components
    """
    # Update UI state
    ui_components['_preprocessing_confirmed'] = False
    
    # Log the cancellation
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.info("❌ Preprocessing dibatalkan")
    
    # Reset UI to initial state
    ui_utils.reset_ui_state(ui_components)

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="handle_cleanup_cancel"
)
def _handle_cleanup_cancel(ui_components: Dict[str, Any]) -> None:
    """Handle cleanup cancellation with proper UI state reset.
    
    Args:
        ui_components: Dictionary containing UI components
    """
    # Clear the confirmation flag
    ui_components['_cleanup_confirmed'] = False
    
    # Log the cancellation
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.warning("🚫 Cleanup dibatalkan")
    
    # Clear any existing dialog
    clear_dialog_area(ui_components)
    
    # Update status panel if available
    status_panel = ui_components.get('status_panel')
    if status_panel and hasattr(status_panel, 'update_status'):
        status_panel.update_status("Cleanup dibatalkan", 'warning')
    
    # Reset UI state
    ui_utils.enable_buttons(ui_components)

def _should_execute_preprocessing(ui_components: Dict[str, Any]) -> bool:
    """Check if preprocessing should execute (consume confirmation flag)
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        bool: True jika preprocessing sudah dikonfirmasi, False jika tidak
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        confirmed = ui_components.pop('_preprocessing_confirmed', False)
        if not confirmed:
            logger_bridge.debug("Preprocessing belum dikonfirmasi")
        return confirmed
    except Exception as e:
        error_msg = f"Gagal memeriksa status konfirmasi preprocessing: {str(e)}"
        logger_bridge.error(error_msg)
        return False

def _should_execute_cleanup(ui_components: Dict[str, Any]) -> bool:
    """Check if cleanup should execute (consume confirmation flag)
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        bool: True jika cleanup sudah dikonfirmasi, False jika tidak
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        confirmed = ui_components.pop('_cleanup_confirmed', False)
        if not confirmed:
            logger_bridge.debug("Cleanup belum dikonfirmasi")
        return confirmed
    except Exception as e:
        error_msg = f"Gagal memeriksa status konfirmasi cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        return False

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="is_confirmation_pending"
)
def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check if any confirmation dialog is pending.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        bool: True if any confirmation is pending, False otherwise
    """
    # Check for pending confirmation flags
    pending = any(key in ui_components for key in ['_preprocessing_confirmed', '_cleanup_confirmed'])
    
    # Log the status if we have a logger
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge and logger_bridge.isEnabledFor("DEBUG"):
        logger_bridge.debug(f"Status konfirmasi: {'pending' if pending else 'tidak ada'}")
    
    return pending

@with_error_handling(
    error_handler=error_handler,
    component="preprocessing_handlers",
    operation="extract_config"
)
def _extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract config from UI components with fallback.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Dict[str, Any]: Konfigurasi yang telah diekstrak
        
    """
    # Try to get config from config handler first
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'config'):
        return config_handler.config
    
    # Fallback to direct config from UI components
    config = ui_components.get('config')
    if config is not None:
        return config
    
    # Log warning if no config found
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        logger_bridge.warning("Config not found in UI components, using empty config")
    
    return {}
