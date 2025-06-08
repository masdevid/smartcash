"""
File: smartcash/ui/dataset/augmentation/utils/dialog_utils.py
Deskripsi: Dialog utilities untuk confirmation dan progress dengan integrasi shared dialog components
"""

from typing import Dict, Any, Callable, Optional
from IPython.display import display

def show_cleanup_confirmation(ui_components: Dict[str, Any], on_confirm: Callable, on_cancel: Callable = None):
    """Show cleanup confirmation dengan shared dialog components"""
    try:
        from smartcash.ui.components.dialogs import show_destructive_confirmation
        
        dialog = show_destructive_confirmation(
            title="Konfirmasi Cleanup Dataset",
            message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
            item_name="file augmented",
            on_confirm=on_confirm,
            on_cancel=on_cancel or (lambda btn: _log_ui(ui_components, "❌ Cleanup dibatalkan", 'info'))
        )
        
        return _display_dialog_safe(ui_components, dialog)
        
    except ImportError:
        # Fallback ke simple confirmation
        return _show_simple_cleanup_confirmation(ui_components, on_confirm, on_cancel)

def show_reset_confirmation(ui_components: Dict[str, Any], on_confirm: Callable, on_cancel: Callable = None):
    """Show reset confirmation dengan shared dialog components"""
    try:
        from smartcash.ui.components.dialogs import show_confirmation
        
        dialog = show_confirmation(
            title="Konfirmasi Reset Konfigurasi",
            message="Apakah Anda yakin ingin reset konfigurasi ke default?\n\nSemua pengaturan saat ini akan hilang.",
            on_confirm=on_confirm,
            on_cancel=on_cancel or (lambda btn: _log_ui(ui_components, "❌ Reset dibatalkan", 'info'))
        )
        
        return _display_dialog_safe(ui_components, dialog)
        
    except ImportError:
        # Fallback ke simple confirmation
        return _show_simple_reset_confirmation(ui_components, on_confirm, on_cancel)

def show_validation_errors_dialog(ui_components: Dict[str, Any], validation_result: Dict[str, Any]):
    """Show validation errors dalam dialog format"""
    if not validation_result.get('errors') and not validation_result.get('warnings'):
        return
    
    # Build error message
    error_messages = []
    for error in validation_result.get('errors', []):
        error_messages.append(error)
    
    for warning in validation_result.get('warnings', []):
        error_messages.append(warning)
    
    message = "Form validation menemukan masalah:\n\n" + "\n".join(error_messages)
    
    try:
        from smartcash.ui.components.dialogs import show_warning
        
        dialog = show_warning(
            title="Validation Error",
            message=message,
            on_close=lambda btn: None
        )
        
        return _display_dialog_safe(ui_components, dialog)
        
    except ImportError:
        # Fallback ke UI log
        _log_ui(ui_components, "⚠️ Form validation errors - check inputs", 'warning')

def show_operation_progress(ui_components: Dict[str, Any], operation_name: str, on_cancel: Callable = None):
    """Show progress dialog untuk long operations"""
    try:
        from smartcash.ui.components.dialogs import show_progress
        
        widget, dialog_id = show_progress(
            title=f"Progress {operation_name}",
            message=f"Memproses {operation_name}...\nHarap tunggu hingga selesai.",
            on_cancel=on_cancel
        )
        
        # Store dialog info untuk updates
        if 'active_dialogs' not in ui_components:
            ui_components['active_dialogs'] = {}
        
        ui_components['active_dialogs'][operation_name] = {
            'widget': widget,
            'dialog_id': dialog_id
        }
        
        return _display_dialog_safe(ui_components, widget)
        
    except ImportError:
        # Fallback: log operation start
        _log_ui(ui_components, f"🚀 {operation_name} dimulai...", 'info')
        return None

def update_operation_progress(ui_components: Dict[str, Any], operation_name: str, 
                            progress: int, status: str = None):
    """Update progress dialog dengan nilai baru"""
    try:
        from smartcash.ui.components.dialogs import get_dialog_factory
        
        dialog_info = ui_components.get('active_dialogs', {}).get(operation_name)
        if dialog_info and 'dialog_id' in dialog_info:
            factory = get_dialog_factory()
            factory.update_progress(dialog_info['dialog_id'], progress, status)
            return True
            
    except ImportError:
        pass
    
    # Fallback ke log
    status_text = f" - {status}" if status else ""
    _log_ui(ui_components, f"📊 {operation_name}: {progress}%{status_text}", 'info')
    return False

def complete_operation_progress(ui_components: Dict[str, Any], operation_name: str, 
                              success: bool = True, message: str = None):
    """Complete progress dialog dengan hasil akhir"""
    try:
        from smartcash.ui.components.dialogs import get_dialog_factory
        
        dialog_info = ui_components.get('active_dialogs', {}).get(operation_name)
        if dialog_info and 'dialog_id' in dialog_info:
            factory = get_dialog_factory()
            factory.complete_progress(dialog_info['dialog_id'], success, message)
            
            # Clean up
            if 'active_dialogs' in ui_components:
                ui_components['active_dialogs'].pop(operation_name, None)
            
            return True
            
    except ImportError:
        pass
    
    # Fallback ke log
    status_emoji = "✅" if success else "❌"
    final_message = message or f"{operation_name} {'berhasil' if success else 'gagal'}"
    _log_ui(ui_components, f"{status_emoji} {final_message}", 'success' if success else 'error')
    return False

def _display_dialog_safe(ui_components: Dict[str, Any], dialog_widget) -> bool:
    """Display dialog dalam confirmation area dengan safe error handling"""
    if not dialog_widget:
        return False
    
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        try:
            confirmation_area.clear_output()
            with confirmation_area:
                display(dialog_widget)
            return True
        except Exception:
            pass
    
    return False

def _show_simple_cleanup_confirmation(ui_components: Dict[str, Any], on_confirm: Callable, on_cancel: Callable):
    """Fallback simple cleanup confirmation tanpa shared components"""
    import ipywidgets as widgets
    
    # Create simple confirm/cancel buttons
    confirm_btn = widgets.Button(description="Ya, Hapus", button_style='danger', layout=widgets.Layout(width='100px'))
    cancel_btn = widgets.Button(description="Batal", button_style='', layout=widgets.Layout(width='100px'))
    
    def handle_confirm(btn):
        _clear_confirmation_area(ui_components)
        on_confirm(btn)
    
    def handle_cancel(btn):
        _clear_confirmation_area(ui_components)
        if on_cancel:
            on_cancel(btn)
    
    confirm_btn.on_click(handle_confirm)
    cancel_btn.on_click(handle_cancel)
    
    # Create dialog HTML
    dialog_html = widgets.HTML(f"""
    <div style="padding: 15px; background-color: #f8d7da; border: 1px solid #dc3545; 
                border-radius: 8px; color: #721c24; margin: 10px 0;">
        <h4>⚠️ Konfirmasi Cleanup Dataset</h4>
        <p>Apakah Anda yakin ingin menghapus semua file augmented?</p>
        <p><strong>Tindakan ini tidak dapat dibatalkan!</strong></p>
    </div>
    """)
    
    buttons_row = widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content='flex-start'))
    dialog_container = widgets.VBox([dialog_html, buttons_row])
    
    return _display_dialog_safe(ui_components, dialog_container)

def _show_simple_reset_confirmation(ui_components: Dict[str, Any], on_confirm: Callable, on_cancel: Callable):
    """Fallback simple reset confirmation tanpa shared components"""
    import ipywidgets as widgets
    
    # Create simple confirm/cancel buttons
    confirm_btn = widgets.Button(description="Ya, Reset", button_style='warning', layout=widgets.Layout(width='100px'))
    cancel_btn = widgets.Button(description="Batal", button_style='', layout=widgets.Layout(width='100px'))
    
    def handle_confirm(btn):
        _clear_confirmation_area(ui_components)
        on_confirm(btn)
    
    def handle_cancel(btn):
        _clear_confirmation_area(ui_components)
        if on_cancel:
            on_cancel(btn)
    
    confirm_btn.on_click(handle_confirm)
    cancel_btn.on_click(handle_cancel)
    
    # Create dialog HTML
    dialog_html = widgets.HTML(f"""
    <div style="padding: 15px; background-color: #fff3cd; border: 1px solid #ffc107; 
                border-radius: 8px; color: #856404; margin: 10px 0;">
        <h4>🔄 Konfirmasi Reset Konfigurasi</h4>
        <p>Apakah Anda yakin ingin reset konfigurasi ke default?</p>
        <p>Semua pengaturan saat ini akan hilang.</p>
    </div>
    """)
    
    buttons_row = widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content='flex-start'))
    dialog_container = widgets.VBox([dialog_html, buttons_row])
    
    return _display_dialog_safe(ui_components, dialog_container)

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area setelah dialog action"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        try:
            confirmation_area.clear_output(wait=True)
        except Exception:
            pass

def _log_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Internal logging dengan fallback ke print"""
    try:
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
    except Exception:
        pass
    
    # Fallback
    emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
    print(f"{emoji_map.get(level, 'ℹ️')} {message}")

# One-liner utilities untuk common dialog operations
show_cleanup_dialog = lambda ui_components, on_confirm, on_cancel=None: show_cleanup_confirmation(ui_components, on_confirm, on_cancel)
show_reset_dialog = lambda ui_components, on_confirm, on_cancel=None: show_reset_confirmation(ui_components, on_confirm, on_cancel)
show_validation_dialog = lambda ui_components, validation_result: show_validation_errors_dialog(ui_components, validation_result)
show_progress_dialog = lambda ui_components, operation, on_cancel=None: show_operation_progress(ui_components, operation, on_cancel)
update_progress_dialog = lambda ui_components, operation, progress, status=None: update_operation_progress(ui_components, operation, progress, status)
complete_progress_dialog = lambda ui_components, operation, success=True, message=None: complete_operation_progress(ui_components, operation, success, message)
clear_confirmation = lambda ui_components: _clear_confirmation_area(ui_components)