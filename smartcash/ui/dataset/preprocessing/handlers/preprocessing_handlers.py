"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Handlers untuk preprocessing dengan proper error handling dan button state management
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.preprocessing.utils import (
    hide_confirmation_area as _hide_confirmation_area,
    show_confirmation_area as _show_confirmation_area,
    clear_outputs as _clear_outputs,
    disable_buttons as _disable_buttons,
    enable_buttons as _enable_buttons,
    setup_progress as _setup_progress,
    complete_progress as _complete_progress,
    error_progress as _error_progress
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

def _setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup save/reset handlers dengan UI logging integration
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
    
    try:
        # Get save and reset buttons
        save_btn = ui_components.get('save_config_btn')
        reset_btn = ui_components.get('reset_config_btn')
        
        if not save_btn or not reset_btn:
            logger_bridge.warning("Save or reset button not found in UI components")
            return
            
        # Clear existing handlers to prevent duplicates
        if hasattr(save_btn, '_click_handlers'):
            save_btn._click_handlers.callbacks.clear()
        if hasattr(reset_btn, '_click_handlers'):
            reset_btn._click_handlers.callbacks.clear()
            
        # Setup save handler
        @save_btn.on_click
        def on_save():
            """Handle save button click"""
            try:
                logger_bridge.info("Saving preprocessing configuration...")
                # Get config from UI
                config = _extract_config(ui_components)
                # Save config
                config_handler = PreprocessingConfigHandler()
                config_handler.ui_components = ui_components
                config_handler.logger_bridge = logger_bridge
                config_handler.save_config(config)
                logger_bridge.success("Preprocessing configuration saved successfully!")
            except Exception as e:
                error_msg = f"Error saving config: {str(e)}"
                logger_bridge.error(error_msg)
                _handle_error(e, ui_components, logger_bridge)
                
        # Setup reset handler
        @reset_btn.on_click
        def on_reset():
            """Handle reset button click"""
            try:
                logger_bridge.info("Resetting preprocessing configuration...")
                # Reset UI to default values
                config_handler = PreprocessingConfigHandler()
                config_handler.ui_components = ui_components
                config_handler.logger_bridge = logger_bridge
                config_handler.reset_ui()
                logger_bridge.success("Preprocessing configuration reset successfully!")
            except Exception as e:
                error_msg = f"Error resetting config: {str(e)}"
                logger_bridge.error(error_msg)
                _handle_error(e, ui_components, logger_bridge)
                
    except Exception as e:
        logger_bridge.error(f"Error setting up config handlers: {str(e)}")
        _handle_error(e, ui_components, logger_bridge)

# === OPERATION HANDLERS ===

def _setup_operation_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup operation handlers dengan API integration
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
    
    try:
        # Get operation buttons
        preprocess_btn = ui_components.get('preprocess_btn')
        check_btn = ui_components.get('check_btn')
        cleanup_btn = ui_components.get('cleanup_btn')
        
        if not all([preprocess_btn, check_btn, cleanup_btn]):
            logger_bridge.warning("Beberapa tombol operasi tidak ditemukan")
            return
            
        # Clear existing handlers
        for btn in [preprocess_btn, check_btn, cleanup_btn]:
            if hasattr(btn, '_click_handlers'):
                btn._click_handlers.callbacks.clear()
                
        # Setup preprocessing handler
        @preprocess_btn.on_click
        def on_preprocess():
            _handle_preprocessing_operation(ui_components)
            
        # Setup check handler
        @check_btn.on_click
        def on_check():
            _handle_check_operation(ui_components)
            
        # Setup cleanup handler
        @cleanup_btn.on_click
        def on_cleanup():
            _handle_cleanup_operation(ui_components)
            
    except Exception as e:
        error_msg = f"Error setting up operation handlers: {str(e)}"
        logger.error(error_msg)
        _handle_error(ui_components, error_msg, logger)

# === OPERATION IMPLEMENTATIONS ===

def _handle_preprocessing_operation(ui_components: Dict[str, Any]) -> None:
    """Handle preprocessing dengan confirmation dan proper error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
    
    try:
        # Check if confirmation is pending
        if _is_confirmation_pending(ui_components):
            logger_bridge.warning("Ada operasi konfirmasi yang sedang menunggu")
            return
            
        # Show confirmation dialog
        _show_preprocessing_confirmation(ui_components)
        
        # If confirmed, execute preprocessing
        if _should_execute_preprocessing(ui_components):
            _execute_preprocessing_with_api(ui_components)
            
    except Exception as e:
        error_msg = f"Error in preprocessing operation: {str(e)}"
        logger.error(error_msg)
        _handle_error(ui_components, error_msg, logger)
        return False

def _handle_check_operation(ui_components: Dict[str, Any]) -> None:
    """Handle dataset check dengan preprocessing API dan error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        # Clear previous outputs
        _clear_outputs(ui_components)
        
        # Show loading state
        _setup_progress(ui_components, "Memeriksa dataset...")
        
        # Get config from UI
        config = _extract_config(ui_components)
        
        # Call preprocessing API to check dataset
        from smartcash.api.preprocessing import check_dataset
        
        logger_bridge.info("🔍 Memeriksa dataset...")
        
        # Execute with progress callback
        progress_callback = _create_progress_callback(ui_components)
        result = check_dataset(
            config=config,
            progress_callback=progress_callback
        )
        
        # Process and display results
        _process_status_result(ui_components, result)
        logger_bridge.success("✅ Pemeriksaan dataset selesai")
        
    except Exception as e:
        error_msg = f"Error saat memeriksa dataset: {str(e)}"
        logger.error(error_msg)
        _handle_error(ui_components, error_msg, logger)
        return False

def _handle_cleanup_operation(ui_components: Dict[str, Any]) -> None:
    """Handle cleanup operation dengan proper error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        # Clear outputs
        _clear_outputs(ui_components)
        
        # Check if confirmation is pending
        if _is_confirmation_pending(ui_components):
            logger_bridge.warning("Ada operasi konfirmasi yang sedang menunggu")
            return
            
        # Show confirmation dialog
        _show_cleanup_confirmation(ui_components)
        
        # If confirmed, execute cleanup
        if _should_execute_cleanup(ui_components):
            _execute_cleanup_with_api(ui_components)
            
    except Exception as e:
        error_msg = f"Error in cleanup operation: {str(e)}"
        logger.error(error_msg)
        _handle_error(ui_components, error_msg, logger)
        return False

# === API EXECUTION FUNCTIONS ===

def _execute_preprocessing_with_api(ui_components: Dict[str, Any]) -> bool:
    """Execute preprocessing dengan proper error handling dan button management"""
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
            progress_callback=progress_callback,
            ui_components=ui_components
        )
        
        # Handle results
        if result.get('success', False):
            _process_success_result(ui_components, result)
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
    """Execute cleanup dengan proper error handling dan button management"""
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
            confirm=True,
            progress_callback=progress_callback,
            ui_components=ui_components
        )
        
        # Handle results
        if result.get('success', False):
            _process_cleanup_result(ui_components, result)
        else:
            error_msg = result.get('message', 'Cleanup failed')
            _error_progress(ui_components, error_msg)
            _log_to_ui(ui_components, f"❌ {error_msg}", "error")
            return False
            
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        error_msg = f"Gagal memproses hasil status: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)
        return
        
    # Log layer analysis jika ada
    if 'layer_analysis' in file_stats:
        layer_info = file_stats['layer_analysis']
        main_objects = layer_info.get('l1_main', {}).get('objects', 0)
        logger_bridge.info(f"🏦 Main banknotes detected: {main_objects:,} objects")

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
        
        _complete_progress(ui_components, success_msg)
        logger_bridge.success(success_msg)
        
        # Log banknote analysis jika ada
        if 'main_banknotes' in stats:
            banknote_stats = stats['main_banknotes']
            total_objects = banknote_stats.get('total_objects', 0)
            logger_bridge.info(f"🏦 Main banknotes processed: {total_objects:,} objects")
            
    except Exception as e:
        error_msg = f"Gagal memproses hasil sukses: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)

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
        
        _complete_progress(ui_components, success_msg)
        logger_bridge.success(success_msg)
        
        # Log detail per split jika ada
        if 'split_stats' in stats:
            for split, split_stat in stats['split_stats'].items():
                removed = split_stat.get('files_removed', 0)
                logger_bridge.info(f"  📁 {split}: {removed:,} files removed")
                
    except Exception as e:
        error_msg = f"Gagal memproses hasil cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)

# === PROGRESS CALLBACK ===

def _create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk preprocessing API
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Fungsi callback untuk melaporkan progress
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
    
    def progress_callback(level: str, current: int, total: int, message: str):
        """Callback untuk melaporkan progress operasi
        
        Args:
            level: Level progress (overall, current, step, batch)
            current: Nilai progress saat ini
            total: Total nilai progress
            message: Pesan progress opsional
        """
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
                progress_tracker.update_current(progress_percent, message)
            
            # Log progress updates
            if message and progress_percent < 100:
                logger_bridge.info(f"⏳ {message}")
                
        except Exception as e:
            error_msg = f"Gagal memperbarui progress: {str(e)}"
            logger_bridge.warning(error_msg)
            
    return progress_callback

# === CONFIRMATION HANDLERS ===

def _show_preprocessing_confirmation(ui_components: Dict[str, Any]) -> None:
    """Show preprocessing confirmation dengan API info
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        # Show confirmation area
        _show_confirmation_area(ui_components)
        
        # Get confirmation components
        confirm_title = ui_components.get('confirm_title')
        confirm_message = ui_components.get('confirm_message')
        confirm_btn = ui_components.get('confirm_btn')
        cancel_btn = ui_components.get('cancel_btn')
        
        if not all([confirm_title, confirm_message, confirm_btn, cancel_btn]):
            logger_bridge.warning("Komponen konfirmasi tidak ditemukan")
            return
            
        # Update confirmation UI
        confirm_title.value = "Konfirmasi Preprocessing"
        confirm_message.value = "Apakah Anda yakin ingin memproses dataset?"
        
        # Clear previous handlers
        confirm_btn.on_click(lambda _: _set_preprocessing_confirmed(ui_components))
        cancel_btn.on_click(lambda _: _handle_preprocessing_cancel(ui_components))
        
        logger_bridge.info("⏳ Menunggu konfirmasi preprocessing...")
        
    except Exception as e:
        error_msg = f"Gagal menampilkan konfirmasi preprocessing: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)

def _show_cleanup_confirmation(ui_components: Dict[str, Any]) -> None:
    """Show cleanup confirmation dengan preview info
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        # Show confirmation area
        _show_confirmation_area(ui_components)
        
        # Get confirmation components
        confirm_title = ui_components.get('confirm_title')
        confirm_message = ui_components.get('confirm_message')
        confirm_btn = ui_components.get('confirm_btn')
        cancel_btn = ui_components.get('cancel_btn')
        
        if not all([confirm_title, confirm_message, confirm_btn, cancel_btn]):
            logger_bridge.warning("Komponen konfirmasi tidak ditemukan")
            return
            
        # Get cleanup target from config
        config = _extract_config(ui_components)
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        target_descriptions = {
            'preprocessed': 'file preprocessing (pre_*.npy + pre_*.txt)',
            'samples': 'sample images (sample_*.jpg)',
            'both': 'file preprocessing dan sample images'
        }
        
        target_desc = target_descriptions.get(cleanup_target, cleanup_target)
            
        # Update confirmation UI
        confirm_title.value = "🧹 Konfirmasi Cleanup"
        confirm_message.value = f"Hapus {target_desc}?\n\nTindakan ini akan menghapus file-file yang sudah diproses."
        
        # Clear previous handlers
        confirm_btn.on_click(lambda _: _set_cleanup_confirmed(ui_components))
        cancel_btn.on_click(lambda _: _handle_cleanup_cancel(ui_components))
        
        logger_bridge.info(f"⏳ Menunggu konfirmasi cleanup untuk: {target_desc}...")
        
    except Exception as e:
        error_msg = f"Gagal menampilkan konfirmasi cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)

# === CONFIRMATION STATE MANAGEMENT ===

def _set_preprocessing_confirmed(ui_components: Dict[str, Any]) -> None:
    """Set preprocessing confirmation flag dan trigger execution
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        ui_components['_preprocessing_confirmed'] = True
        logger_bridge.info("✅ Konfirmasi diterima, memulai preprocessing...")
        _execute_preprocessing_with_api(ui_components)
    except Exception as e:
        error_msg = f"Gagal memproses konfirmasi preprocessing: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)

def _set_cleanup_confirmed(ui_components: Dict[str, Any]):
    """Set cleanup confirmation flag dan trigger execution"""
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        ui_components['_cleanup_confirmed'] = True
        logger_bridge.info("✅ Konfirmasi diterima, memulai cleanup...")
        _execute_cleanup_with_api(ui_components)
    except Exception as e:
        error_msg = f"Gagal memproses konfirmasi cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)
    _hide_confirmation_area(ui_components)
    _execute_cleanup_with_api(ui_components)

def _handle_preprocessing_cancel(ui_components: Dict[str, Any]) -> None:
    """Handle preprocessing cancellation dengan proper cleanup
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        _hide_confirmation_area(ui_components)
        logger_bridge.warning("❌ Preprocessing dibatalkan")
    except Exception as e:
        error_msg = f"Gagal menangani pembatalan preprocessing: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)
    _enable_buttons(ui_components)  # Enable buttons after cancel

def _handle_cleanup_cancel(ui_components: Dict[str, Any]):
    """Handle cleanup cancellation dengan proper cleanup"""
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        _hide_confirmation_area(ui_components)
        logger_bridge.warning("❌ Cleanup dibatalkan")
    except Exception as e:
        error_msg = f"Gagal menangani pembatalan cleanup: {str(e)}"
        logger_bridge.error(error_msg)
        _show_error_ui(ui_components, error_msg)
    _clear_outputs(ui_components)
    _log_to_ui(ui_components, "🚫 Cleanup dibatalkan oleh user", "info")
    _enable_buttons(ui_components)

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

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check if confirmation dialog is pending
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        bool: True jika ada konfirmasi yang sedang menunggu, False jika tidak
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        pending = any(key in ui_components for key in ['_preprocessing_confirmed', '_cleanup_confirmed'])
        if pending:
            logger_bridge.debug(f"Status konfirmasi: {'pending' if pending else 'tidak ada'}")
        return pending
    except Exception as e:
        error_msg = f"Gagal memeriksa status konfirmasi: {str(e)}"
        logger_bridge.error(error_msg)
        return False

def _extract_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI components dengan fallback
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dict[str, Any]: Konfigurasi yang telah diekstrak
        
    Raises:
        ValueError: Jika logger bridge belum diinisialisasi
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi. Pastikan PreprocessingInitializer digunakan")
        
    try:
        config_handler = ui_components.get('config_handler')
        if config_handler and hasattr(config_handler, 'get_config'):
            config = config_handler.get_config()
            logger_bridge.debug("Konfigurasi berhasil diekstrak dari config handler")
            return config
            
        config = ui_components.get('config', {})
        if not config:
            logger_bridge.warning("Menggunakan konfigurasi kosong karena tidak ada config handler atau config yang valid")
        return config
        
    except Exception as e:
        error_msg = f"Gagal mengekstrak konfigurasi: {str(e)}"
        logger_bridge.error(error_msg)
        return {}
