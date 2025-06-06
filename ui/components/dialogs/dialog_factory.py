"""
File: smartcash/ui/components/dialogs/dialog_factory.py
Deskripsi: Factory untuk membuat berbagai tipe dialog dengan one-liner methods dan preset configurations
"""

from typing import Callable, Optional, Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.dialogs.dialog_manager import (
    DialogManager, DialogType, DialogConfig, get_dialog_manager
)

class DialogFactory:
    """Factory untuk membuat berbagai tipe dialog dengan preset configurations"""
    
    def __init__(self):
        self.manager = get_dialog_manager()
        self._preset_configs = self._initialize_presets()
    
    def _initialize_presets(self) -> Dict[str, DialogConfig]:
        """Initialize preset configurations untuk berbagai use case"""
        return {
            'default_confirmation': DialogConfig(
                width="500px", confirm_text="Ya", cancel_text="Tidak",
                title_icon="❓", border_color="#007bff"
            ),
            'destructive_action': DialogConfig(
                width="450px", confirm_text="Ya, Hapus", cancel_text="Batal",
                title_icon="⚠️", confirm_style="danger", border_color="#dc3545"
            ),
            'save_confirmation': DialogConfig(
                width="400px", confirm_text="Simpan", cancel_text="Batal",
                title_icon="💾", border_color="#28a745"
            ),
            'exit_confirmation': DialogConfig(
                width="450px", confirm_text="Ya, Keluar", cancel_text="Batal",
                title_icon="🚪", border_color="#ffc107"
            ),
            'info_message': DialogConfig(
                width="400px", show_cancel=False, close_text="OK",
                title_icon="ℹ️", border_color="#17a2b8"
            ),
            'warning_message': DialogConfig(
                width="450px", show_cancel=False, close_text="Mengerti",
                title_icon="⚠️", border_color="#ffc107"
            ),
            'error_message': DialogConfig(
                width="500px", show_cancel=False, close_text="OK",
                title_icon="❌", border_color="#dc3545"
            ),
            'progress_task': DialogConfig(
                width="600px", show_confirm=False, cancel_text="Cancel",
                title_icon="⏳", border_color="#28a745"
            )
        }
    
    def create_confirmation(self, title: str, message: str, 
                          on_confirm: Callable = None, on_cancel: Callable = None,
                          preset: str = 'default_confirmation', **config_overrides) -> str:
        """
        Create confirmation dialog dengan preset configuration
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides(preset, {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_confirm: callbacks['on_confirm'] = on_confirm
        if on_cancel: callbacks['on_cancel'] = on_cancel
        
        return self.manager.create_dialog(DialogType.CONFIRMATION, config, callbacks)
    
    def create_destructive_confirmation(self, title: str, message: str,
                                      item_name: str = "item",
                                      on_confirm: Callable = None, on_cancel: Callable = None,
                                      **config_overrides) -> str:
        """
        Create destructive confirmation dialog untuk operasi berbahaya
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('destructive_action', {
            'title': title, 'message': message,
            'confirm_text': f"Ya, Hapus {item_name}",
            **config_overrides
        })
        
        callbacks = {}
        if on_confirm: callbacks['on_confirm'] = on_confirm
        if on_cancel: callbacks['on_cancel'] = on_cancel
        
        return self.manager.create_dialog(DialogType.DESTRUCTIVE, config, callbacks)
    
    def create_save_confirmation(self, title: str = "Simpan Perubahan?",
                               message: str = "Apakah Anda yakin ingin menyimpan perubahan?",
                               on_save: Callable = None, on_cancel: Callable = None,
                               **config_overrides) -> str:
        """
        Create save confirmation dialog
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('save_confirmation', {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_save: callbacks['on_confirm'] = on_save
        if on_cancel: callbacks['on_cancel'] = on_cancel
        
        return self.manager.create_dialog(DialogType.CONFIRMATION, config, callbacks)
    
    def create_exit_confirmation(self, title: str = "Keluar dari Aplikasi?",
                               message: str = "Perubahan yang belum disimpan akan hilang.",
                               on_exit: Callable = None, on_cancel: Callable = None,
                               **config_overrides) -> str:
        """
        Create exit confirmation dialog
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('exit_confirmation', {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_exit: callbacks['on_confirm'] = on_exit
        if on_cancel: callbacks['on_cancel'] = on_cancel
        
        return self.manager.create_dialog(DialogType.CONFIRMATION, config, callbacks)
    
    def create_info_dialog(self, title: str, message: str,
                          on_close: Callable = None,
                          preset: str = 'info_message', **config_overrides) -> str:
        """
        Create info dialog dengan single close button
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides(preset, {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_close: callbacks['on_confirm'] = on_close
        
        return self.manager.create_dialog(DialogType.INFO, config, callbacks)
    
    def create_warning_dialog(self, title: str, message: str,
                            on_close: Callable = None, **config_overrides) -> str:
        """
        Create warning dialog
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('warning_message', {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_close: callbacks['on_confirm'] = on_close
        
        return self.manager.create_dialog(DialogType.WARNING, config, callbacks)
    
    def create_error_dialog(self, title: str, message: str,
                          on_close: Callable = None, **config_overrides) -> str:
        """
        Create error dialog
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('error_message', {
            'title': title, 'message': message, **config_overrides
        })
        
        callbacks = {}
        if on_close: callbacks['on_confirm'] = on_close
        
        return self.manager.create_dialog(DialogType.INFO, config, callbacks)
    
    def create_progress_dialog(self, title: str, message: str,
                             on_cancel: Callable = None,
                             auto_close_delay: int = 3, **config_overrides) -> str:
        """
        Create progress dialog dengan cancel support
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = self._get_config_with_overrides('progress_task', {
            'title': title, 'message': message,
            'auto_close_delay': auto_close_delay,
            **config_overrides
        })
        
        callbacks = {}
        if on_cancel: callbacks['on_cancel'] = on_cancel
        
        return self.manager.create_dialog(DialogType.PROGRESS, config, callbacks)
    
    def create_custom_dialog(self, title: str, message: str,
                           buttons: Dict[str, Dict[str, str]] = None,
                           callbacks: Dict[str, Callable] = None,
                           **config_overrides) -> str:
        """
        Create custom dialog dengan button dan callback yang dapat disesuaikan
        
        Args:
            buttons: Dict dengan format {"action": {"text": "Button Text", "style": "primary"}}
            callbacks: Dict dengan format {"on_action": callback_function}
        
        Returns:
            dialog_id untuk manage dialog
        """
        config = DialogConfig(
            title=title, message=message,
            width="500px", **config_overrides
        )
        
        # Override button configuration based on custom buttons
        if buttons:
            first_button = list(buttons.keys())[0]
            if first_button in buttons:
                config.confirm_text = buttons[first_button].get('text', 'OK')
                config.confirm_style = buttons[first_button].get('style', 'primary')
            
            if len(buttons) > 1:
                second_button = list(buttons.keys())[1]
                config.cancel_text = buttons[second_button].get('text', 'Cancel')
                config.cancel_style = buttons[second_button].get('style', '')
            else:
                config.show_cancel = False
        
        return self.manager.create_dialog(DialogType.CUSTOM, config, callbacks or {})
    
    def _get_config_with_overrides(self, preset: str, overrides: Dict[str, Any]) -> DialogConfig:
        """Get preset config dengan overrides yang diterapkan"""
        base_config = self._preset_configs.get(preset, DialogConfig())
        
        # Create new config dengan overrides
        config_dict = base_config.__dict__.copy()
        config_dict.update(overrides)
        
        return DialogConfig(**config_dict)
    
    def get_dialog_widget(self, dialog_id: str) -> Optional[widgets.Widget]:
        """Get widget container dari dialog untuk display"""
        instance = self.manager.get_dialog(dialog_id)
        return instance.container if instance else None
    
    def update_progress(self, dialog_id: str, progress: int, status: str = None) -> bool:
        """Update progress dialog dengan value dan status baru"""
        return self.manager.update_dialog(dialog_id, progress=progress, status=status)
    
    def complete_progress(self, dialog_id: str, success: bool = True, 
                         final_message: str = None) -> bool:
        """Complete progress dialog dengan status akhir"""
        message = final_message or ("Selesai dengan sukses!" if success else "Operasi gagal!")
        self.manager.complete_dialog(dialog_id, success, message)
        return True
    
    def close_dialog(self, dialog_id: str) -> bool:
        """Close dialog berdasarkan ID"""
        return self.manager.close_dialog(dialog_id)
    
    def add_preset(self, name: str, config: DialogConfig):
        """Tambahkan preset configuration baru"""
        self._preset_configs[name] = config
    
    def get_preset_names(self) -> list:
        """Get daftar nama preset yang tersedia"""
        return list(self._preset_configs.keys())

# Singleton factory instance
_dialog_factory = None

def get_dialog_factory() -> DialogFactory:
    """Get singleton dialog factory instance"""
    global _dialog_factory
    if _dialog_factory is None:
        _dialog_factory = DialogFactory()
    return _dialog_factory

# One-liner convenience functions untuk kemudahan penggunaan
def show_confirmation(title: str, message: str, on_confirm: Callable = None, on_cancel: Callable = None) -> widgets.Widget:
    """One-liner untuk show confirmation dialog dan return widget"""
    factory = get_dialog_factory()
    dialog_id = factory.create_confirmation(title, message, on_confirm, on_cancel)
    return factory.get_dialog_widget(dialog_id)

def show_destructive_confirmation(title: str, message: str, item_name: str = "item",
                                 on_confirm: Callable = None, on_cancel: Callable = None) -> widgets.Widget:
    """One-liner untuk show destructive confirmation dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_destructive_confirmation(title, message, item_name, on_confirm, on_cancel)
    return factory.get_dialog_widget(dialog_id)

def show_info(title: str, message: str, on_close: Callable = None) -> widgets.Widget:
    """One-liner untuk show info dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_info_dialog(title, message, on_close)
    return factory.get_dialog_widget(dialog_id)

def show_warning(title: str, message: str, on_close: Callable = None) -> widgets.Widget:
    """One-liner untuk show warning dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_warning_dialog(title, message, on_close)
    return factory.get_dialog_widget(dialog_id)

def show_error(title: str, message: str, on_close: Callable = None) -> widgets.Widget:
    """One-liner untuk show error dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_error_dialog(title, message, on_close)
    return factory.get_dialog_widget(dialog_id)

def show_progress(title: str, message: str, on_cancel: Callable = None) -> tuple:
    """One-liner untuk show progress dialog, return (widget, dialog_id)"""
    factory = get_dialog_factory()
    dialog_id = factory.create_progress_dialog(title, message, on_cancel)
    return factory.get_dialog_widget(dialog_id), dialog_id

def show_save_confirmation(title: str = "Simpan Perubahan?", 
                          message: str = "Apakah Anda yakin ingin menyimpan perubahan?",
                          on_save: Callable = None, on_cancel: Callable = None) -> widgets.Widget:
    """One-liner untuk show save confirmation dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_save_confirmation(title, message, on_save, on_cancel)
    return factory.get_dialog_widget(dialog_id)

def show_exit_confirmation(title: str = "Keluar dari Aplikasi?",
                          message: str = "Perubahan yang belum disimpan akan hilang.",
                          on_exit: Callable = None, on_cancel: Callable = None) -> widgets.Widget:
    """One-liner untuk show exit confirmation dialog"""
    factory = get_dialog_factory()
    dialog_id = factory.create_exit_confirmation(title, message, on_exit, on_cancel)
    return factory.get_dialog_widget(dialog_id)

# Quick access functions dengan minimal parameters
confirm = lambda title, message, on_yes=None, on_no=None: show_confirmation(title, message, on_yes, on_no)
info = lambda title, message, on_close=None: show_info(title, message, on_close)
warn = lambda title, message, on_close=None: show_warning(title, message, on_close)
error = lambda title, message, on_close=None: show_error(title, message, on_close)
progress = lambda title, message, on_cancel=None: show_progress(title, message, on_cancel)

# Preset-specific convenience functions
confirm_delete = lambda item_name, on_confirm=None, on_cancel=None: show_destructive_confirmation(
    f"Hapus {item_name}?", f"Apakah Anda yakin ingin menghapus {item_name}?\n\nTindakan ini tidak dapat dibatalkan.", 
    item_name, on_confirm, on_cancel
)

confirm_reset = lambda on_confirm=None, on_cancel=None: show_confirmation(
    "Reset ke Default?", "Apakah Anda yakin ingin mereset semua pengaturan ke nilai default?", 
    on_confirm, on_cancel
)

confirm_cleanup = lambda on_confirm=None, on_cancel=None: show_destructive_confirmation(
    "Bersihkan Data?", "Apakah Anda yakin ingin membersihkan semua data?\n\nSemua file dan folder akan dihapus.", 
    "data", on_confirm, on_cancel
)