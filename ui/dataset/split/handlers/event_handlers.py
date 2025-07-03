"""
File: smartcash/ui/dataset/split/handlers/event_handlers.py
Deskripsi: Event handlers untuk split UI dengan centralized error handling
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
import logging

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors

# Import base handler
from smartcash.ui.dataset.split.handlers.base_split_handler import BaseSplitHandler

# Logger
logger = logging.getLogger(__name__)

@handle_ui_errors(log_error=True)
def setup_event_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup event handlers untuk split UI dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger.debug("Menyiapkan event handlers untuk split UI")
    
    # Get handler if available
    handler = ui_components.get('handler')
    
    # Setup save/reset button handlers
    _setup_save_reset_handlers(handler, ui_components)
    
    # Setup checkbox handlers
    _setup_checkbox_handlers(ui_components)
    
    logger.debug("Event handlers berhasil di-setup")


@handle_ui_errors(log_error=True)
def _setup_save_reset_handlers(handler: Optional[BaseSplitHandler], ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk save/reset buttons dengan centralized error handling
    
    Args:
        handler: Handler instance untuk operasi save/reset
        ui_components: Dictionary berisi komponen UI
    """
    # Get save and reset buttons
    save_btn = ui_components.get('save_button')
    reset_btn = ui_components.get('reset_button')
    
    if save_btn and handler:
        @handle_ui_errors(log_error=True)
        def on_save_click(btn):
            """Handle save button click dengan centralized error handling"""
            try:
                # Update button state
                save_btn.disabled = True
                save_btn.description = "💾 Menyimpan..."
                
                logger.info("Menyimpan konfigurasi split dataset...")
                
                # Save config
                result = handler.save_config(ui_components)
                
                # Check status key for consistency
                if result and result.get('status', False):
                    logger.info("Konfigurasi split dataset berhasil disimpan")
                else:
                    error_msg = result.get('message', 'Gagal menyimpan konfigurasi')
                    logger.error(f"Error: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Error saat menyimpan konfigurasi: {str(e)}")
                
            finally:
                # Restore button state
                save_btn.disabled = False
                save_btn.description = "💾 Simpan"
        
        save_btn.on_click(on_save_click)
        logger.debug("Save button handler berhasil dipasang")
    
    if reset_btn and handler:
        @handle_ui_errors(log_error=True)
        def on_reset_click(btn):
            """Handle reset button click dengan centralized error handling"""
            try:
                logger.info("Mereset konfigurasi split dataset ke default...")
                
                # Reset UI
                handler.reset_config(ui_components)
                
                # Update total label using slider_handlers utility
                from smartcash.ui.dataset.split.handlers.slider_handlers import _update_total_display
                
                # Get sliders
                sliders = {name: ui_components.get(f'{name}_slider') for name in ['train', 'valid', 'test']}
                _update_total_display(ui_components, sliders)
                
                logger.info("Konfigurasi split dataset berhasil di-reset")
                
            except Exception as e:
                logger.error(f"Error saat mereset konfigurasi: {str(e)}")
        
        reset_btn.on_click(on_reset_click)
        logger.debug("Reset button handler berhasil dipasang")


@handle_ui_errors(log_error=True)
def _setup_checkbox_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk checkboxes dengan centralized error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Backup checkbox handler
    backup_checkbox = ui_components.get('backup_checkbox')
    backup_fields = ['backup_dir', 'backup_count', 'auto_backup']
    
    if backup_checkbox:
        @handle_ui_errors(log_error=True)
        def on_backup_change(change):
            """Handle backup checkbox change dengan centralized error handling"""
            enabled = change['new']
            for field_name in backup_fields:
                field = ui_components.get(field_name)
                if field and hasattr(field, 'disabled'):
                    field.disabled = not enabled
            
            status = "aktif" if enabled else "nonaktif"
            logger.debug(f"Backup {status}")
            
        backup_checkbox.observe(on_backup_change, names='value')
        # Trigger initial state
        on_backup_change({'new': backup_checkbox.value})
        logger.debug("Backup checkbox handler berhasil dipasang")
    
    # Validation checkbox handler
    validation_checkbox = ui_components.get('validation_enabled')
    validation_fields = ['fix_issues', 'move_invalid', 'invalid_dir', 'visualize_issues']
    
    if validation_checkbox:
        @handle_ui_errors(log_error=True)
        def on_validation_change(change):
            """Handle validation checkbox change dengan centralized error handling"""
            enabled = change['new']
            for field_name in validation_fields:
                field = ui_components.get(field_name)
                if field and hasattr(field, 'disabled'):
                    field.disabled = not enabled
                    
            status = "aktif" if enabled else "nonaktif"
            logger.debug(f"Validasi {status}")
            
        validation_checkbox.observe(on_validation_change, names='value')
        # Trigger initial state
        on_validation_change({'new': validation_checkbox.value})
        logger.debug("Validation checkbox handler berhasil dipasang")
