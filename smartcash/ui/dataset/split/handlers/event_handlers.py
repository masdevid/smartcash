"""
File: smartcash/ui/dataset/split/handlers/event_handlers.py
Deskripsi: Event handlers untuk split UI dengan integrasi parent component
"""

from typing import Dict, Any, TYPE_CHECKING
import ipywidgets as widgets

if TYPE_CHECKING:
    from smartcash.ui.dataset.split.split_init import SplitConfigInitializer

def setup_event_handlers(initializer: 'SplitConfigInitializer', ui_components: Dict[str, Any]) -> None:
    """Setup event handlers untuk split UI dengan logger bridge integration.
    
    Args:
        initializer: Instance SplitConfigInitializer dengan akses ke parent component
        ui_components: Dictionary berisi komponen UI
    """
    # Get logger bridge dari initializer
    logger_bridge = initializer._logger_bridge
    
    # Get handler
    handler = initializer.handler
    
    # Setup save/reset button handlers
    _setup_save_reset_handlers(handler, ui_components, logger_bridge)
    
    # Setup slider handlers
    _setup_slider_handlers(ui_components, logger_bridge)
    
    # Setup checkbox handlers
    _setup_checkbox_handlers(ui_components, logger_bridge)
    
    logger_bridge.debug("✅ Event handlers berhasil di-setup")


def _setup_save_reset_handlers(handler, ui_components: Dict[str, Any], logger_bridge) -> None:
    """Setup handlers untuk save/reset buttons."""
    
    save_reset = ui_components.get('save_reset_buttons', {})
    save_btn = save_reset.get('save_button')
    reset_btn = save_reset.get('reset_button')
    
    if save_btn:
        def on_save_click(btn):
            """Handle save button click dengan status updates."""
            try:
                # Update button state
                save_btn.disabled = True
                save_btn.description = "💾 Menyimpan..."
                
                # Update status panel via parent
                _update_status_panel(ui_components, "💾 Menyimpan konfigurasi...", "info")
                
                # Save config
                success = handler.save_config(ui_components)
                
                if success:
                    logger_bridge.info("✅ Konfigurasi berhasil disimpan")
                    _update_status_panel(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
                else:
                    raise Exception("Gagal menyimpan konfigurasi")
                    
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                logger_bridge.error(error_msg)
                _update_status_panel(ui_components, error_msg, "error")
                
            finally:
                # Restore button state
                save_btn.disabled = False
                save_btn.description = "💾 Simpan"
        
        save_btn.on_click(on_save_click)
        logger_bridge.debug("✅ Save button handler attached")
    
    if reset_btn:
        def on_reset_click(btn):
            """Handle reset button click dengan konfirmasi."""
            try:
                # Log action
                logger_bridge.info("🔄 Reset konfigurasi ke default...")
                
                # Update status
                _update_status_panel(ui_components, "🔄 Mereset konfigurasi...", "info")
                
                # Reset UI
                handler.reset_ui(ui_components)
                
                # Update total label
                _update_total_label(ui_components)
                
                logger_bridge.info("✅ Konfigurasi berhasil di-reset")
                _update_status_panel(ui_components, "✅ Konfigurasi di-reset ke default", "success")
                
            except Exception as e:
                error_msg = f"❌ Error reset: {str(e)}"
                logger_bridge.error(error_msg)
                _update_status_panel(ui_components, error_msg, "error")
        
        reset_btn.on_click(on_reset_click)
        logger_bridge.debug("✅ Reset button handler attached")


def _setup_slider_handlers(ui_components: Dict[str, Any], logger_bridge) -> None:
    """Setup handlers untuk ratio sliders dengan real-time updates."""
    
    train_slider = ui_components.get('train_slider')
    valid_slider = ui_components.get('valid_slider')
    test_slider = ui_components.get('test_slider')
    
    if not all([train_slider, valid_slider, test_slider]):
        logger_bridge.warning("⚠️ Tidak semua slider ditemukan")
        return
    
    def on_ratio_change(change):
        """Handle perubahan ratio dengan normalisasi otomatis."""
        if change['name'] != 'value':
            return
            
        try:
            # Get current values
            total = train_slider.value + valid_slider.value + test_slider.value
            
            if abs(total - 1.0) > 0.001:  # Tolerance untuk floating point
                # Normalize ratios
                if total > 0:
                    train_slider.value = round(train_slider.value / total, 2)
                    valid_slider.value = round(valid_slider.value / total, 2)
                    test_slider.value = round(1.0 - train_slider.value - valid_slider.value, 2)
                    
                    logger_bridge.debug(f"📊 Ratios normalized - Train: {train_slider.value}, Valid: {valid_slider.value}, Test: {test_slider.value}")
            
            # Update total label
            _update_total_label(ui_components)
            
        except Exception as e:
            logger_bridge.error(f"❌ Error updating ratios: {str(e)}")
    
    # Attach handlers
    train_slider.observe(on_ratio_change, names='value')
    valid_slider.observe(on_ratio_change, names='value')
    test_slider.observe(on_ratio_change, names='value')
    
    logger_bridge.debug("✅ Slider handlers attached")


def _setup_checkbox_handlers(ui_components: Dict[str, Any], logger_bridge) -> None:
    """Setup handlers untuk checkboxes dengan dependent field management."""
    
    # Backup checkbox handler
    backup_checkbox = ui_components.get('backup_checkbox')
    backup_fields = ['backup_dir', 'backup_count', 'auto_backup']
    
    if backup_checkbox:
        def on_backup_change(change):
            enabled = change['new']
            for field_name in backup_fields:
                field = ui_components.get(field_name)
                if field and hasattr(field, 'disabled'):
                    field.disabled = not enabled
            
            status = "aktif" if enabled else "nonaktif"
            logger_bridge.debug(f"💾 Backup {status}")
            
        backup_checkbox.observe(on_backup_change, names='value')
        # Trigger initial state
        on_backup_change({'new': backup_checkbox.value})
    
    # Validation checkbox handler
    validation_checkbox = ui_components.get('validation_enabled')
    validation_fields = ['fix_issues', 'move_invalid', 'invalid_dir', 'visualize_issues']
    
    if validation_checkbox:
        def on_validation_change(change):
            enabled = change['new']
            for field_name in validation_fields:
                field = ui_components.get(field_name)
                if field and hasattr(field, 'disabled'):
                    field.disabled = not enabled
                    
            status = "aktif" if enabled else "nonaktif"
            logger_bridge.debug(f"✅ Validasi {status}")
            
        validation_checkbox.observe(on_validation_change, names='value')
        # Trigger initial state
        on_validation_change({'new': validation_checkbox.value})
        
    logger_bridge.debug("✅ Checkbox handlers attached")


def _update_total_label(ui_components: Dict[str, Any]) -> None:
    """Update total ratio label dengan visual feedback."""
    
    total_label = ui_components.get('total_label')
    if not total_label:
        return
        
    train = ui_components.get('train_slider', {}).value or 0
    valid = ui_components.get('valid_slider', {}).value or 0  
    test = ui_components.get('test_slider', {}).value or 0
    
    total = train + valid + test
    
    # Format dengan warna berdasarkan validitas
    if abs(total - 1.0) < 0.001:
        color = "green"
        icon = "✅"
    else:
        color = "red"
        icon = "❌"
        
    total_label.value = f'<span style="color: {color}; font-weight: bold;">{icon} Total: {total:.2f}</span>'


def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str) -> None:
    """Update status panel melalui parent component.
    
    Args:
        ui_components: UI components dictionary
        message: Status message
        status_type: Type of status (success, error, info, warning)
    """
    # Try to find status panel
    status_panel = ui_components.get('status_panel')
    
    # Try from parent if available
    if not status_panel and 'parent' in ui_components:
        parent = ui_components['parent']
        if hasattr(parent, 'status_panel'):
            status_panel = parent.status_panel
            
    if status_panel:
        try:
            from smartcash.ui.components import update_status_panel
            update_status_panel(status_panel, message, status_type)
        except Exception:
            # Don't fail main operation if status update fails
            pass