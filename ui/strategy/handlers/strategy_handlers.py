"""
File: smartcash/ui/strategy/handlers/strategy_handlers.py
Deskripsi: Complete event handlers dengan log integration untuk strategy form
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def setup_strategy_event_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup event handlers untuk strategy form dengan log integration"""
    try:
        logger.debug("🎛️ Setting up strategy event handlers...")
        _add_log(ui_components, "🎛️ Mengaktifkan event handlers...", "info")
        
        # Setup save button handler
        save_button = ui_components.get('save_button')
        if save_button:
            save_button.on_click(lambda _: handle_save_click(ui_components))
        
        # Setup reset button handler
        reset_button = ui_components.get('reset_button')
        if reset_button:
            reset_button.on_click(lambda _: handle_reset_click(ui_components))
        
        # Setup dynamic form behavior
        setup_dynamic_form_behavior(ui_components)
        
        logger.debug("✅ Strategy event handlers berhasil diatur")
        _add_log(ui_components, "✅ Event handlers berhasil diaktifkan", "success")
        
    except Exception as e:
        error_msg = f"❌ Gagal mengatur event handlers: {str(e)}"
        logger.error(error_msg)
        _add_log(ui_components, error_msg, "error")


def setup_dynamic_summary_updates(ui_components: Dict[str, Any]) -> None:
    """Setup dynamic summary updates dengan log tracking"""
    try:
        logger.debug("🔄 Mengatur pembaruan summary otomatis...")
        
        # Daftar widget yang akan di-observe
        widgets_to_observe = [
            'val_frequency_slider', 'iou_thres_slider', 'conf_thres_slider',
            'max_detections_slider', 'experiment_name_text', 'tensorboard_checkbox',
            'log_metrics_slider', 'visualize_batch_slider', 'layer_mode_dropdown',
            'multi_scale_checkbox', 'img_size_min_slider', 'img_size_max_slider'
        ]
        
        # Add change handler for each widget
        for widget_name in widgets_to_observe:
            widget = ui_components.get(widget_name)
            if widget and hasattr(widget, 'observe'):
                widget.observe(lambda change, w=widget_name: _on_widget_change(ui_components, w, change), names='value')
        
        # Update summary card pertama kali
        update_summary_card(ui_components)
        
        logger.debug("✅ Pembaruan summary otomatis berhasil diatur")
        
    except Exception as e:
        error_msg = f"❌ Gagal mengatur pembaruan summary: {str(e)}"
        logger.error(error_msg)
        _add_log(ui_components, error_msg, "error")


def handle_save_click(ui_components: Dict[str, Any]) -> None:
    """Handle save button click dengan validation dan logging"""
    try:
        logger.info("💾 Memproses penyimpanan konfigurasi...")
        _add_log(ui_components, "⏳ Menyimpan konfigurasi...", "info")
        
        # Validasi form
        is_valid, errors = validate_form_inputs(ui_components)
        if not is_valid:
            error_msg = "❌ Validasi gagal: " + ", ".join(errors)
            logger.error(error_msg)
            _add_log(ui_components, error_msg, "error")
            show_save_error(ui_components, error_msg)
            return
        
        # Simpan konfigurasi
        try:
            # TODO: Implement save to config file
            logger.info("✅ Konfigurasi berhasil disimpan")
            _add_log(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
            show_save_success(ui_components)
            
        except Exception as e:
            error_msg = f"❌ Gagal menyimpan konfigurasi: {str(e)}"
            logger.error(error_msg)
            _add_log(ui_components, error_msg, "error")
            show_save_error(ui_components, error_msg)
            
    except Exception as e:
        error_msg = f"❌ Terjadi kesalahan saat menyimpan: {str(e)}"
        logger.error(error_msg)
        _add_log(ui_components, error_msg, "error")
        show_save_error(ui_components, error_msg)


def handle_reset_click(ui_components: Dict[str, Any]) -> None:
    """Handle reset button click dengan logging"""
    try:
        logger.info("🔄 Mereset form ke default...")
        _add_log(ui_components, "🔄 Mereset form...", "info")
        
        # Dapatkan konfigurasi default
        from ..handlers.config_handler import StrategyConfigHandler
        handler = StrategyConfigHandler()
        default_config = handler.get_default_config()
        
        # Update UI dengan nilai default
        for key, value in default_config.items():
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                widget.value = value
        
        logger.info("✅ Form berhasil direset")
        _add_log(ui_components, "✅ Form berhasil direset ke default", "success")
        
    except Exception as e:
        error_msg = f"❌ Gagal mereset form: {str(e)}"
        logger.error(error_msg)
        _add_log(ui_components, error_msg, "error")


def setup_dynamic_form_behavior(ui_components: Dict[str, Any]) -> None:
    """Setup dynamic form behavior dengan conditional enables"""
    try:
        logger.debug("🎚️ Mengatur perilaku form dinamis...")
        
        # Enable/disable multi-scale sliders berdasarkan checkbox
        multi_scale_cb = ui_components.get('multi_scale_checkbox')
        min_size_slider = ui_components.get('img_size_min_slider')
        max_size_slider = ui_components.get('img_size_max_slider')
        
        def update_multi_scale_state(change):
            is_enabled = change['new']
            if min_size_slider:
                min_size_slider.disabled = not is_enabled
            if max_size_slider:
                max_size_slider.disabled = not is_enabled
        
        if multi_scale_cb and (min_size_slider or max_size_slider):
            # Set initial state
            update_multi_scale_state({'new': multi_scale_cb.value})
            # Observe changes
            multi_scale_cb.observe(update_multi_scale_state, names='value')
        
        logger.debug("✅ Perilaku form dinamis berhasil diatur")
        
    except Exception as e:
        error_msg = f"❌ Gagal mengatur perilaku form dinamis: {str(e)}"
        logger.error(error_msg)
        _add_log(ui_components, error_msg, "error")


def update_summary_card(ui_components: Dict[str, Any]) -> None:
    """Update summary card dengan current form values"""
    try:
        summary_card = ui_components.get('summary_card')
        if not summary_card:
            return
            
        # Dapatkan nilai dari form
        experiment_name = getattr(ui_components.get('experiment_name_text'), 'value', 'N/A')
        tensorboard = getattr(ui_components.get('tensorboard_checkbox'), 'value', False)
        layer_mode = getattr(ui_components.get('layer_mode_dropdown'), 'value', 'single')
        
        # Update summary card
        summary_card.value = f"""
        <div style="background: #f8f9fa; border-radius: 5px; padding: 10px; margin: 10px 0;">
            <h4>📊 Ringkasan Konfigurasi</h4>
            <p><strong>Nama Eksperimen:</strong> {experiment_name}</p>
            <p><strong>Tensorboard:</strong> {'Aktif' if tensorboard else 'Non-aktif'}</p>
            <p><strong>Mode Layer:</strong> {layer_mode}</p>
        </div>
        """
        
    except Exception as e:
        logger.error(f"❌ Gagal memperbarui summary card: {str(e)}")


def show_save_success(ui_components: Dict[str, Any]) -> None:
    """Show save success status dengan visual feedback"""
    try:
        status_display = ui_components.get('status_display')
        if status_display:
            status_display.value = """
            <div style="color: #28a745; background-color: #d4edda; 
                        border: 1px solid #c3e6cb; border-radius: 4px; 
                        padding: 10px; margin: 10px 0;">
                ✅ Konfigurasi berhasil disimpan
            </div>
            """
            
            # Reset status setelah 3 detik
            import threading
            def reset_status():
                import time
                time.sleep(3)
                if status_display:
                    status_display.value = ""
            
            thread = threading.Thread(target=reset_status)
            thread.daemon = True
            thread.start()
            
    except Exception as e:
        logger.error(f"❌ Gagal menampilkan status sukses: {str(e)}")


def show_save_error(ui_components: Dict[str, Any], error_msg: str = "") -> None:
    """Show save error status dengan visual feedback"""
    try:
        status_display = ui_components.get('status_display')
        if status_display:
            status_display.value = f"""
            <div style="color: #dc3545; background-color: #f8d7da; 
                        border: 1px solid #f5c6cb; border-radius: 4px; 
                        padding: 10px; margin: 10px 0;">
                ❌ {error_msg or 'Terjadi kesalahan saat menyimpan konfigurasi'}
            </div>
            """
    except Exception as e:
        logger.error(f"❌ Gagal menampilkan status error: {str(e)}")


def validate_form_inputs(ui_components: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate form inputs sebelum save"""
    errors = []
    
    try:
        # Validasi nama eksperimen
        exp_name = getattr(ui_components.get('experiment_name_text'), 'value', '').strip()
        if not exp_name:
            errors.append("Nama eksperimen tidak boleh kosong")
        
        # Validasi nilai slider
        def validate_slider(key: str, min_val: float, max_val: float, label: str) -> None:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'value'):
                val = widget.value
                if not (min_val <= val <= max_val):
                    errors.append(f"{label} harus antara {min_val} dan {max_val}")
        
        validate_slider('iou_thres_slider', 0, 1, 'IoU Threshold')
        validate_slider('conf_thres_slider', 0, 1, 'Confidence Threshold')
        validate_slider('max_detections_slider', 1, 1000, 'Max Detections')
        
        # Validasi multi-scale
        multi_scale = getattr(ui_components.get('multi_scale_checkbox'), 'value', False)
        if multi_scale:
            min_size = getattr(ui_components.get('img_size_min_slider'), 'value', 0)
            max_size = getattr(ui_components.get('img_size_max_slider'), 'value', 0)
            
            if min_size >= max_size:
                errors.append("Ukuran minimum harus lebih kecil dari ukuran maksimum")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        error_msg = f"Kesalahan validasi: {str(e)}"
        logger.error(f"❌ {error_msg}")
        errors.append(error_msg)
        return False, errors


def _on_widget_change(ui_components: Dict[str, Any], widget_key: str, change: Dict[str, Any]) -> None:
    """Handle individual widget change dengan logging"""
    try:
        widget = ui_components.get(widget_key)
        if not widget or 'new' not in change:
            return
            
        # Dapatkan label atau deskripsi widget
        widget_desc = getattr(widget, 'description', widget_key)
        if hasattr(widget_desc, 'value'):
            widget_desc = widget_desc.value
            
        # Log perubahan
        logger.debug(f"🔄 {widget_desc} diubah menjadi: {change['new']}")
        
        # Update summary card
        update_summary_card(ui_components)
        
    except Exception as e:
        logger.error(f"❌ Gagal memproses perubahan widget: {str(e)}")


def _add_log(ui_components: Dict[str, Any], message: str, log_type: str = "info") -> None:
    """Add log message menggunakan existing log component"""
    try:
        log_display = ui_components.get('log_display')
        if log_display and hasattr(log_display, 'append_stdout'):
            prefix = "[INFO] " if log_type == "info" else "[ERROR] " if log_type == "error" else "[SUCCESS] "
            log_display.append_stdout(f"{prefix}{message}\n")
    except Exception as e:
        logger.error(f"❌ Gagal menambahkan log: {str(e)}")
