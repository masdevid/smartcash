"""
File: smartcash/ui/components/dialogs/__init__.py
Deskripsi: Main dialog components export dengan backward compatibility dan one-liner access
"""

from .dialog_manager import (
    DialogManager, DialogType, DialogState, DialogConfig, DialogInstance,
    get_dialog_manager
)
from .dialog_factory import (
    DialogFactory, get_dialog_factory,
    show_confirmation, show_destructive_confirmation, show_info, show_warning, 
    show_error, show_progress, show_save_confirmation, show_exit_confirmation,
    confirm, info, warn, error, progress,
    confirm_delete, confirm_reset, confirm_cleanup
)

# Backward compatibility dengan confirmation_dialog.py lama
def create_confirmation_dialog(title: str, message: str, on_confirm, on_cancel,
                              confirm_text: str = "Ya, Lanjutkan", cancel_text: str = "Batal",
                              dialog_width: str = "600px", danger_mode: bool = False):
    """Backward compatibility function untuk existing code"""
    factory = get_dialog_factory()
    
    if danger_mode:
        dialog_id = factory.create_destructive_confirmation(
            title, message, "item", on_confirm, on_cancel,
            width=dialog_width, confirm_text=confirm_text, cancel_text=cancel_text
        )
    else:
        dialog_id = factory.create_confirmation(
            title, message, on_confirm, on_cancel,
            width=dialog_width, confirm_text=confirm_text, cancel_text=cancel_text
        )
    
    return factory.get_dialog_widget(dialog_id)

def create_destructive_confirmation(title: str, message: str, on_confirm, on_cancel,
                                   item_name: str = "item", confirm_text: str = None, 
                                   cancel_text: str = "Batal"):
    """Backward compatibility function untuk existing code"""
    return show_destructive_confirmation(title, message, item_name, on_confirm, on_cancel)

def create_info_dialog(title: str, message: str, on_close=None,
                      close_text: str = "OK", dialog_width: str = "500px"):
    """Backward compatibility function untuk existing code"""
    return show_info(title, message, on_close)

def create_progress_dialog(title: str, message: str, progress_callback=None,
                          cancel_callback=None, dialog_width: str = "600px"):
    """Backward compatibility function untuk existing code"""
    widget, dialog_id = show_progress(title, message, cancel_callback)
    
    # Return format yang kompatibel dengan kode lama
    return {
        'dialog': widget,
        'update_progress': lambda value, status=None: get_dialog_factory().update_progress(dialog_id, value, status),
        'complete_progress': lambda success=True, message=None: get_dialog_factory().complete_progress(dialog_id, success, message),
        'dialog_id': dialog_id
    }

def cleanup_all_dialogs():
    """Backward compatibility function untuk existing code"""
    get_dialog_manager().cleanup_all_dialogs()

def get_active_dialog_count():
    """Backward compatibility function untuk existing code"""
    return len(get_dialog_manager().active_dialogs)

# Modern one-liner convenience functions
class Dialogs:
    """Modern static class untuk one-liner dialog access"""
    
    @staticmethod
    def confirm(title: str, message: str, on_yes=None, on_no=None, **kwargs):
        """Static method untuk confirmation dialog"""
        return show_confirmation(title, message, on_yes, on_no)
    
    @staticmethod
    def info(title: str, message: str, on_close=None, **kwargs):
        """Static method untuk info dialog"""
        return show_info(title, message, on_close)
    
    @staticmethod
    def warn(title: str, message: str, on_close=None, **kwargs):
        """Static method untuk warning dialog"""
        return show_warning(title, message, on_close)
    
    @staticmethod
    def error(title: str, message: str, on_close=None, **kwargs):
        """Static method untuk error dialog"""
        return show_error(title, message, on_close)
    
    @staticmethod
    def progress(title: str, message: str, on_cancel=None, **kwargs):
        """Static method untuk progress dialog"""
        return show_progress(title, message, on_cancel)
    
    @staticmethod
    def delete(item_name: str, on_confirm=None, on_cancel=None, **kwargs):
        """Static method untuk delete confirmation"""
        return confirm_delete(item_name, on_confirm, on_cancel)
    
    @staticmethod
    def save(title: str = "Simpan Perubahan?", message: str = None, on_save=None, on_cancel=None, **kwargs):
        """Static method untuk save confirmation"""
        message = message or "Apakah Anda yakin ingin menyimpan perubahan?"
        return show_save_confirmation(title, message, on_save, on_cancel)
    
    @staticmethod
    def exit(title: str = "Keluar dari Aplikasi?", message: str = None, on_exit=None, on_cancel=None, **kwargs):
        """Static method untuk exit confirmation"""
        message = message or "Perubahan yang belum disimpan akan hilang."
        return show_exit_confirmation(title, message, on_exit, on_cancel)
    
    @staticmethod
    def reset(on_confirm=None, on_cancel=None, **kwargs):
        """Static method untuk reset confirmation"""
        return confirm_reset(on_confirm, on_cancel)
    
    @staticmethod
    def cleanup(on_confirm=None, on_cancel=None, **kwargs):
        """Static method untuk cleanup confirmation"""
        return confirm_cleanup(on_confirm, on_cancel)
    
    @staticmethod
    def custom(title: str, message: str, buttons=None, callbacks=None, **kwargs):
        """Static method untuk custom dialog"""
        factory = get_dialog_factory()
        dialog_id = factory.create_custom_dialog(title, message, buttons, callbacks, **kwargs)
        return factory.get_dialog_widget(dialog_id)

# Dialog helper functions untuk common use cases
class DialogHelpers:
    """Helper functions untuk common dialog patterns"""
    
    @staticmethod
    def confirm_with_validation(title: str, message: str, validation_func, 
                               on_confirm=None, on_cancel=None):
        """Confirmation dialog dengan validation function"""
        
        def validated_confirm(button):
            try:
                if validation_func and callable(validation_func):
                    if not validation_func():
                        return
                if on_confirm:
                    on_confirm(button)
            except Exception:
                pass
        
        return show_confirmation(title, message, validated_confirm, on_cancel)
    
    @staticmethod
    def confirm_with_timeout(title: str, message: str, timeout_seconds: int = 10,
                            on_confirm=None, on_cancel=None, on_timeout=None):
        """Confirmation dialog dengan auto-timeout"""
        import threading
        import time
        
        dialog_closed = False
        
        def timeout_handler():
            nonlocal dialog_closed
            time.sleep(timeout_seconds)
            if not dialog_closed:
                dialog_closed = True
                if on_timeout:
                    on_timeout()
        
        def safe_confirm(button):
            nonlocal dialog_closed
            dialog_closed = True
            if on_confirm:
                on_confirm(button)
        
        def safe_cancel(button):
            nonlocal dialog_closed
            dialog_closed = True
            if on_cancel:
                on_cancel(button)
        
        # Start timeout thread
        threading.Thread(target=timeout_handler, daemon=True).start()
        
        timeout_message = f"{message}\n\n⏰ Dialog akan ditutup otomatis dalam {timeout_seconds} detik."
        return show_confirmation(title, timeout_message, safe_confirm, safe_cancel)
    
    @staticmethod
    def progressive_confirmation(steps: list, on_complete=None, on_cancel=None):
        """Multi-step confirmation dialog"""
        current_step = 0
        
        def next_step(button):
            nonlocal current_step
            current_step += 1
            
            if current_step < len(steps):
                step = steps[current_step]
                return show_confirmation(step['title'], step['message'], next_step, on_cancel)
            else:
                if on_complete:
                    on_complete(button)
        
        if steps:
            first_step = steps[0]
            return show_confirmation(first_step['title'], first_step['message'], next_step, on_cancel)
    
    @staticmethod
    def conditional_confirmation(condition_func, title: str, message: str,
                                on_confirm=None, on_cancel=None):
        """Conditional confirmation berdasarkan function"""
        try:
            if condition_func and callable(condition_func):
                if condition_func():
                    return show_confirmation(title, message, on_confirm, on_cancel)
                else:
                    # Skip confirmation jika kondisi tidak terpenuhi
                    if on_confirm:
                        on_confirm(None)
                    return None
        except Exception:
            # Jika error, tetap tampilkan confirmation
            return show_confirmation(title, message, on_confirm, on_cancel)

# Progress dialog dengan advanced features
class ProgressDialog:
    """Advanced progress dialog dengan features tambahan"""
    
    def __init__(self, title: str, message: str, on_cancel=None):
        widget, self.dialog_id = show_progress(title, message, on_cancel)
        self.widget = widget
        self.factory = get_dialog_factory()
        self.is_completed = False
    
    def update(self, progress: int, status: str = None):
        """Update progress dengan safe execution"""
        if not self.is_completed:
            return self.factory.update_progress(self.dialog_id, progress, status)
        return False
    
    def complete(self, success: bool = True, message: str = None):
        """Complete progress dengan status"""
        if not self.is_completed:
            self.is_completed = True
            return self.factory.complete_progress(self.dialog_id, success, message)
        return False
    
    def close(self):
        """Close progress dialog"""
        return self.factory.close_dialog(self.dialog_id)
    
    def set_indeterminate(self, status: str = "Processing..."):
        """Set progress ke indeterminate mode"""
        return self.update(-1, status)  # -1 indicates indeterminate

# Export semua untuk backward compatibility dan ease of use
__all__ = [
    # Manager dan Factory
    'DialogManager', 'DialogFactory', 'DialogType', 'DialogState', 'DialogConfig',
    'get_dialog_manager', 'get_dialog_factory',
    
    # Modern one-liner functions
    'show_confirmation', 'show_destructive_confirmation', 'show_info', 'show_warning', 
    'show_error', 'show_progress', 'show_save_confirmation', 'show_exit_confirmation',
    'confirm', 'info', 'warn', 'error', 'progress',
    'confirm_delete', 'confirm_reset', 'confirm_cleanup',
    
    # Static classes
    'Dialogs', 'DialogHelpers', 'ProgressDialog',
    
    # Backward compatibility
    'create_confirmation_dialog', 'create_destructive_confirmation', 'create_info_dialog',
    'create_progress_dialog', 'cleanup_all_dialogs', 'get_active_dialog_count'
]