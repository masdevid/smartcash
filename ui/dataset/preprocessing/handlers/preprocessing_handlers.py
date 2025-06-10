"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Handler untuk preprocessing dataset (UI layer)
"""

from typing import Dict, Any, Callable, Optional, Tuple
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox, Button, HTML, Output

# Import from local utils with clear grouping
from smartcash.ui.dataset.preprocessing.utils import (
    # UI Utilities
    clear_outputs,
    handle_ui_error,
    show_ui_success,
    log_to_accordion,
    
    # Button Management
    with_button_management,
    ButtonStateManager,
    
    # Progress Utilities
    create_dual_progress_callback,
    setup_dual_progress_tracker,
    complete_progress_tracker,
    error_progress_tracker,
    
    # Backend Integration
    validate_dataset_ready,
    create_backend_preprocessor,
    create_backend_checker,
    create_backend_cleanup_service,
    check_preprocessed_exists,
    _convert_ui_to_backend_config
)

from smartcash.common.logger import get_logger
from .operation_handlers import execute_operation, get_operation_config

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handler preprocessing dengan one-liner style"""
    try:
        # Setup progress callback
        ui_components['progress_callback'] = create_dual_progress_callback(ui_components)
        
        # Setup config handler
        if (config_handler := ui_components.get('config_handler')) and hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(ui_components)
        
        # Setup operation dan config handlers
        _setup_operation_handlers(ui_components, config)
        _setup_config_handlers(ui_components, config)
        
        return ui_components
    except Exception as e:
        log_to_service(ui_components.get('logger'), f"❌ Error setup preprocessing handlers: {str(e)}", "error")
        return ui_components

def _setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """
    Setup operation handlers dengan one-liner style
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi preprocessing
    """
    logger = get_logger('preprocessing_handlers')
    
    # Inisialisasi button state manager
    button_manager = ButtonStateManager(ui_components)
    ui_components['button_manager'] = button_manager
    
    # Register UI utility functions
    ui_components.update({
        # UI Utilities
        'clear_outputs': clear_outputs,
        'handle_ui_error': handle_ui_error,
        'show_ui_success': show_ui_success,
        'log_to_accordion': log_to_accordion,
        
        # Progress Utilities
        'setup_dual_progress_tracker': setup_dual_progress_tracker,
        'complete_progress_tracker': complete_progress_tracker,
        'error_progress_tracker': error_progress_tracker,
        
        # Backend Integration
        'validate_dataset_ready': lambda cfg: validate_dataset_ready(cfg, logger=logger),
        'check_preprocessed_exists': check_preprocessed_exists,
        'create_backend_preprocessor': lambda ui: create_backend_preprocessor(ui, logger=logger),
        'create_backend_checker': lambda cfg: create_backend_checker(cfg, logger=logger),
        'create_backend_cleanup_service': lambda cfg: create_backend_cleanup_service(cfg, logger=logger),
        '_convert_ui_to_backend_config': _convert_ui_to_backend_config
    })
    
    def _on_operation_click(btn, operation_type: str):
        """Handle click event for operation buttons"""
        clear_outputs(ui_components)
        
        # Dapatkan konfigurasi operasi
        _, button_key, _ = get_operation_config(operation_type)
        
        # Jalankan operasi dengan button management
        @with_button_management(ui_components, button_key)
        def _execute():
            return execute_operation(ui_components, operation_type, config)
            
        # Eksekusi operasi
        _execute()
    
    # Setup button handlers
    for op_type in ['preprocess', 'check', 'cleanup']:
        button = ui_components.get(f"{op_type}_button")
        if button and hasattr(button, 'on_click'):
            button.on_click(lambda btn, op=op_type: _on_operation_click(btn, op))
    
def _setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup configuration change handlers"""
    logger = get_logger('config_handlers')
    
    def on_config_change(change):
        """Handle configuration changes"""
        try:
            key = change['owner'].description
            value = change['new']
            
            # Update config based on changed value
            if key == 'target_split':
                config['preprocessing']['target_split'] = value
            elif key == 'normalization_method':
                config['preprocessing']['normalization']['method'] = value
            elif key == 'target_width':
                config['preprocessing']['normalization']['target_size'][0] = value
            elif key == 'target_height':
                config['preprocessing']['normalization']['target_size'][1] = value
                
            logger.debug(f"Config updated - {key}: {value}")
            
        except Exception as e:
            logger.error(f"Gagal memperbarui konfigurasi: {str(e)}", exc_info=True)
    
    # Setup config change observers
    for key in ['target_split', 'normalization_method', 'target_width', 'target_height']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'observe'):
            widget.observe(on_config_change, 'value')
    
    # Initialize UI components with config values
    def _init_ui_component(component_key: str, config_path: str, default=None):
        """Initialize UI component with config value"""
        component = ui_components.get(component_key)
        if component:
            try:
                # Navigate config path (e.g., 'preprocessing.normalization.method')
                keys = config_path.split('.')
                value = config
                for k in keys:
                    value = value.get(k, {})
                
                if value is not None and hasattr(component, 'value'):
                    component.value = value
                elif default is not None:
                    component.value = default
                    
            except Exception as e:
                logger.warning(f"Gagal menginisialisasi {component_key}: {str(e)}")
    
    # Initialize components
    _init_ui_component('target_split', 'preprocessing.target_split', 'train')
    _init_ui_component('normalization_method', 'preprocessing.normalization.method', 'minmax')
    _init_ui_component('target_width', 'preprocessing.normalization.target_size.0', 640)
    _init_ui_component('target_height', 'preprocessing.normalization.target_size.1', 640)

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any]):
    """Tampilkan dialog konfirmasi sebelum membersihkan dataset"""
    from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
    
    ui_config = extract_preprocessing_config(ui_components)
    if (count := check_preprocessed_exists(ui_config)) <= 0:
        return handle_ui_error(ui_components, "❌ Tidak ada file preprocessed yang ditemukan")
    
    _show_confirmation_in_area(
        ui_components=ui_components,
        title="⚠️ Konfirmasi Cleanup",
        message=f"Akan menghapus {count:,} file preprocessed.\n\n⚠️ Tindakan ini tidak dapat dibatalkan!\n\nLanjutkan?",
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        on_confirm=lambda: _execute_backend_operation(ui_components, 'cleanup'),
        on_cancel=lambda: show_ui_success(ui_components, "✅ Operasi cleanup dibatalkan")
    )

def _bind_handlers_safe(ui_components: Dict[str, Any], handlers: Dict[str, Callable]):
    """
    Safe handler binding dengan one-liner style
    
    Args:
        ui_components: Dictionary berisi komponen UI
        handlers: Dictionary berisi mapping antara nama tombol dan handler-nya
    """
    logger = ui_components.get('logger', get_logger('bind_handlers'))
    
    for button_key, handler in handlers.items():
        if not (button := ui_components.get(button_key)) or not hasattr(button, 'on_click'):
            logger.warning(f"Tombol {button_key} tidak ditemukan atau tidak mendukung event on_click")
            continue
            
        try:
            button.on_click(handler)
            logger.debug(f"✅ {button_key.replace('_', ' ').title()} handler bound")
        except Exception as e:
            logger.error(f"⚠️ Error binding {button_key}: {str(e)}", exc_info=True)

def _execute_backend_operation(ui_components: Dict[str, Any], operation_type: str):
    """Execute backend operation dengan one-liner style"""
    @with_button_management
    def _backend_operation(ui_components):
        logger, progress_tracker = ui_components.get('logger'), ui_components.get('progress_tracker')
        operation_handlers = {
            'preprocessing_pipeline': _execute_preprocessing_pipeline,
            'dataset_check': _execute_dataset_check,
            'cleanup': _execute_cleanup_operation
        }
        
        try:
            if handler := operation_handlers.get(operation_type):
                return handler(ui_components, logger, progress_tracker)
            raise ValueError(f"Handler tidak ditemukan untuk operasi: {operation_type}")
        except Exception as e:
            error_msg = f"❌ {operation_type.replace('_', ' ').title()} error: {str(e)}"
            log_to_service(logger, error_msg, "error")
            handle_ui_error(ui_components, error_msg)
            if progress_tracker:
                progress_tracker.error(error_msg)
    
    _backend_operation(ui_components)

def _execute_preprocessing_pipeline(ui_components: Dict[str, Any], logger, progress_tracker) -> bool:
    """Execute preprocessing pipeline dengan one-liner style"""
    from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
    
    log_to_service(logger, "🚀 Memulai preprocessing pipeline", "info")
    
    # Validasi konfigurasi
    ui_config = extract_preprocessing_config(ui_components)
    if not (valid := validate_dataset_ready(ui_config, logger))[0]:
        return handle_ui_error(ui_components, f"❌ {valid[1]}")
    
    setup_dual_progress_tracker(ui_components, "Dataset Preprocessing")
    
    # Buat dan konfigurasi preprocessor
    if not (preprocessor := create_backend_preprocessor(ui_config, logger)):
        return handle_ui_error(ui_components, "❌ Gagal membuat preprocessing service")
    
    # Register progress callback jika tersedia
    if hasattr(preprocessor, 'register_progress_callback'):
        preprocessor.register_progress_callback(ui_components['progress_callback'])
    
    # Eksekusi preprocessing
    if not (result := preprocessor.preprocess_dataset()) or not result.get('success'):
        error_msg = result.get('message', 'Tidak ada respons dari service') if result else 'Gagal memproses dataset'
        error_progress_tracker(ui_components, f"Preprocessing gagal: {error_msg}")
        return handle_ui_error(ui_components, error_msg)
    
    # Tampilkan hasil sukses
    stats = result.get('stats', {})
    total_images = stats.get('total_processed', 0)
    success_msg = f"✅ Preprocessing berhasil: {total_images:,} gambar"
    
    complete_progress_tracker(ui_components, "Preprocessing selesai")
    show_ui_success(ui_components, success_msg)
    
    _show_success_info(
        ui_components,
        "Preprocessing Berhasil!",
        f"✅ Preprocessing selesai!\n\n📊 Hasil:\n• {total_images:,} gambar diproses\n• Output: {result.get('output_dir', 'data/preprocessed')}"
    )
    return True

def _execute_dataset_check(ui_components: Dict[str, Any], logger, progress_tracker):
    """Execute dataset check"""
    from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
    
    if logger:
        logger.info("🔍 Checking dataset")
    
    ui_config = extract_preprocessing_config(ui_components)
    setup_dual_progress_tracker(ui_components, "Dataset Check")
    
    valid, source_msg = validate_dataset_ready(ui_config, logger)
    
    if valid:
        if logger:
            logger.success(f"✅ {source_msg}")
        
        # Check preprocessed data
        preprocessed_exists, preprocessed_count = check_preprocessed_exists(ui_config)
        
        if preprocessed_exists:
            if logger:
                logger.success(f"💾 Preprocessed dataset: {preprocessed_count:,} gambar")
        else:
            if logger:
                logger.info("ℹ️ Belum ada preprocessed dataset")
        
        complete_progress_tracker(ui_components, "Dataset check selesai")
        
        check_msg = f"Check: {source_msg.split(': ')[1] if ': ' in source_msg else source_msg}"
        if preprocessed_exists:
            check_msg += f" + {preprocessed_count:,} preprocessed"
        
        show_ui_success(ui_components, check_msg)
        
        # Show check info in confirmation area
        _show_success_info(ui_components, "Dataset Check Berhasil!",
            f"✅ Dataset check selesai!\n\n📊 Status:\n• {source_msg}\n• Preprocessed: {preprocessed_count:,} files" if preprocessed_exists else f"✅ Dataset check selesai!\n\n📊 Status:\n• {source_msg}\n• Preprocessed: Belum ada")
    else:
        error_progress_tracker(ui_components, f"Check gagal: {source_msg}")
        handle_ui_error(ui_components, f"❌ {source_msg}")

def _execute_cleanup_operation(ui_components: Dict[str, Any], logger, progress_tracker):
    """Execute cleanup operation"""
    from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
    
    if logger:
        logger.info("🧹 Memulai cleanup preprocessed data")
    
    ui_config = extract_preprocessing_config(ui_components)
    setup_dual_progress_tracker(ui_components, "Dataset Cleanup")
    
    cleanup_service = create_backend_cleanup_service(ui_config, logger)
    if not cleanup_service:
        handle_ui_error(ui_components, "❌ Gagal membuat cleanup service")
        return
    
    result = cleanup_service.cleanup_preprocessed_data()
    
    if result and result.get('success', False):
        stats = result.get('stats', {})
        files_removed = stats.get('files_removed', 0)
        
        complete_progress_tracker(ui_components, f"Cleanup selesai: {files_removed:,} file")
        show_ui_success(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file")
        
        # Show cleanup success info
        _show_success_info(ui_components, "Cleanup Berhasil!",
            f"✅ Cleanup selesai!\n\n📊 Summary:\n• {files_removed:,} file dihapus\n• Storage space dikosongkan")
    else:
        error_msg = result.get('message', 'Unknown cleanup error') if result else 'No response from service'
        error_progress_tracker(ui_components, f"Cleanup gagal: {error_msg}")
        handle_ui_error(ui_components, error_msg)

def _show_confirmation_in_area(ui_components: Dict[str, Any], title: str, message: str, 
                            confirm_text: str, cancel_text: str, 
                            on_confirm: Callable, on_cancel: Callable):
    """Tampilkan dialog konfirmasi di area yang ditentukan
    
    Args:
        ui_components: Dictionary berisi komponen UI
        title: Judul dialog
        message: Pesan konfirmasi
        confirm_text: Teks tombol konfirmasi
        cancel_text: Teks tombol batal
        on_confirm: Fungsi yang dipanggil saat konfirmasi
        on_cancel: Fungsi yang dipanggil saat batal
    """
    from ipywidgets import VBox, HBox, Button, HTML, Output
    from IPython.display import display, clear_output
    
    output = Output()
    
    def _on_confirm(btn):
        with output:
            clear_output()
            on_confirm()
    
    def _on_cancel(btn):
        with output:
            clear_output()
            on_cancel()
    
    confirm_btn = Button(description=confirm_text, button_style='danger')
    cancel_btn = Button(description=cancel_text, button_style='')
    
    confirm_btn.on_click(_on_confirm)
    cancel_btn.on_click(_on_cancel)
    
    dialog = VBox([
        HTML(f"<h4>{title}</h4>"),
        HTML(f'<p>{"<br>".join(message.splitlines())}</p>'),
        HBox([confirm_btn, cancel_btn])
    ])
    
    with output:
        clear_output()
        display(dialog)
    
    # Tampilkan dialog di area konfirmasi
    if 'confirmation_area' in ui_components:
        with ui_components['confirmation_area']:
            display(output)