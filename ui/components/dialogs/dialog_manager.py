"""
File: smartcash/ui/components/dialogs/dialog_manager.py
Deskripsi: Core dialog manager yang mengelola semua tipe dialog dengan state management dan auto-cleanup
"""

import uuid
import atexit
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import ipywidgets as widgets

class DialogType(Enum):
    """Enum untuk tipe dialog yang tersedia"""
    CONFIRMATION = "confirmation"
    DESTRUCTIVE = "destructive" 
    INFO = "info"
    WARNING = "warning"
    PROGRESS = "progress"
    CUSTOM = "custom"

class DialogState(Enum):
    """Enum untuk state dialog"""
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

@dataclass
class DialogConfig:
    """Konfigurasi dialog yang dapat disesuaikan"""
    title: str = ""
    message: str = ""
    width: str = "600px"
    max_width: str = "100%"
    padding: str = "25px"
    border_radius: str = "8px"
    auto_close_delay: int = 0  # 0 = tidak auto close
    show_backdrop: bool = True
    backdrop_click_close: bool = False
    escape_key_close: bool = True
    
    # Button configurations
    confirm_text: str = "OK"
    cancel_text: str = "Batal"
    close_text: str = "Tutup"
    show_confirm: bool = True
    show_cancel: bool = True
    confirm_style: str = "primary"
    cancel_style: str = ""
    
    # Icons
    title_icon: str = ""
    confirm_icon: str = "check"
    cancel_icon: str = "times"
    
    # Colors and styling
    border_color: str = "#ddd"
    background_color: str = "white"
    title_color: str = "#2c3e50"
    message_color: str = "#34495e"
    box_shadow: str = "0 4px 12px rgba(0,0,0,0.15)"

@dataclass
class DialogInstance:
    """Instance dialog yang sedang aktif"""
    dialog_id: str
    dialog_type: DialogType
    config: DialogConfig
    container: widgets.Widget
    state: DialogState = DialogState.ACTIVE
    callbacks: Dict[str, Callable] = field(default_factory=dict)
    components: Dict[str, widgets.Widget] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())

class DialogManager:
    """Singleton dialog manager untuk mengelola semua dialog dalam aplikasi"""
    
    _instance = None
    _cleanup_registered = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inisialisasi dialog manager"""
        self.active_dialogs: Dict[str, DialogInstance] = {}
        self.dialog_history: List[str] = []
        self.max_history: int = 50
        self._register_cleanup()
    
    def _register_cleanup(self):
        """Register cleanup handler untuk session end"""
        if not DialogManager._cleanup_registered:
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython and hasattr(ipython, 'events'):
                    ipython.events.register('pre_run_cell', self._cleanup_completed_dialogs)
                    DialogManager._cleanup_registered = True
            except Exception:
                pass
            
            # Fallback cleanup saat module exit
            atexit.register(self.cleanup_all_dialogs)
    
    def create_dialog(self, dialog_type: DialogType, config: DialogConfig, 
                     callbacks: Optional[Dict[str, Callable]] = None) -> str:
        """
        Create dialog dengan tipe dan konfigurasi tertentu
        
        Returns:
            dialog_id: ID unik untuk dialog yang dibuat
        """
        dialog_id = str(uuid.uuid4())
        callbacks = callbacks or {}
        
        # Buat container dialog berdasarkan tipe
        container = self._create_dialog_container(dialog_type, config, dialog_id)
        
        # Buat instance dialog
        instance = DialogInstance(
            dialog_id=dialog_id,
            dialog_type=dialog_type,
            config=config,
            container=container,
            callbacks=callbacks,
            metadata={'created_by': dialog_type.value}
        )
        
        # Setup handlers berdasarkan tipe
        self._setup_dialog_handlers(instance)
        
        # Register dialog
        self.active_dialogs[dialog_id] = instance
        self.dialog_history.append(dialog_id)
        
        # Maintain history limit
        if len(self.dialog_history) > self.max_history:
            self.dialog_history = self.dialog_history[-self.max_history:]
        
        return dialog_id
    
    def _create_dialog_container(self, dialog_type: DialogType, config: DialogConfig, dialog_id: str) -> widgets.VBox:
        """Buat container dialog berdasarkan tipe"""
        # Style mapping berdasarkan tipe
        type_styles = {
            DialogType.CONFIRMATION: {"border_color": "#007bff", "title_icon": "❓"},
            DialogType.DESTRUCTIVE: {"border_color": "#dc3545", "title_icon": "⚠️", "confirm_style": "danger"},
            DialogType.INFO: {"border_color": "#17a2b8", "title_icon": "ℹ️"},
            DialogType.WARNING: {"border_color": "#ffc107", "title_icon": "⚠️"},
            DialogType.PROGRESS: {"border_color": "#28a745", "title_icon": "⏳"},
            DialogType.CUSTOM: {}
        }
        
        # Apply type-specific styles
        style_overrides = type_styles.get(dialog_type, {})
        for key, value in style_overrides.items():
            if not getattr(config, key, None):  # Only override if not explicitly set
                setattr(config, key, value)
        
        # Create components
        components = self._create_dialog_components(config, dialog_type)
        
        # Create main container
        container = widgets.VBox(
            list(components.values()),
            layout=widgets.Layout(
                width=config.width,
                max_width=config.max_width,
                padding=config.padding,
                background_color=config.background_color,
                border=f'1px solid {config.border_color}',
                border_radius=config.border_radius,
                box_shadow=config.box_shadow,
                margin='10px auto',
                overflow='hidden'
            )
        )
        
        # Add dialog metadata
        setattr(container, '_dialog_id', dialog_id)
        setattr(container, '_dialog_type', dialog_type.value)
        
        return container
    
    def _create_dialog_components(self, config: DialogConfig, dialog_type: DialogType) -> Dict[str, widgets.Widget]:
        """Buat komponen dialog berdasarkan konfigurasi"""
        components = {}
        
        # Title component
        if config.title:
            title_text = f"{config.title_icon} {config.title}" if config.title_icon else config.title
            components['title'] = widgets.HTML(
                f'<div style="font-size: 18px; font-weight: bold; color: {config.title_color}; margin-bottom: 15px;">{title_text}</div>',
                layout=widgets.Layout(margin='0 0 10px 0')
            )
        
        # Message component
        if config.message:
            components['message'] = widgets.HTML(
                f'<div style="font-size: 14px; line-height: 1.6; color: {config.message_color}; white-space: pre-wrap;">{config.message}</div>',
                layout=widgets.Layout(margin='0 0 20px 0')
            )
        
        # Progress component (khusus untuk progress dialog)
        if dialog_type == DialogType.PROGRESS:
            components['progress'] = widgets.IntProgress(
                value=0, min=0, max=100,
                bar_style='info',
                layout=widgets.Layout(width='100%', margin='10px 0')
            )
            components['status'] = widgets.HTML(
                '<div style="text-align: center; color: #666; font-size: 12px;">Memulai...</div>',
                layout=widgets.Layout(margin='5px 0 15px 0')
            )
        
        # Button components
        buttons = self._create_dialog_buttons(config, dialog_type)
        if buttons:
            components['buttons'] = widgets.HBox(
                buttons,
                layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0')
            )
        
        return components
    
    def _create_dialog_buttons(self, config: DialogConfig, dialog_type: DialogType) -> List[widgets.Button]:
        """Buat tombol dialog berdasarkan konfigurasi"""
        buttons = []
        
        # Determine which buttons to show based on dialog type
        if dialog_type == DialogType.INFO:
            config.show_cancel = False
            config.confirm_text = config.close_text
        elif dialog_type == DialogType.PROGRESS:
            config.show_confirm = False
            config.cancel_text = "Cancel"
        
        # Create confirm button
        if config.show_confirm:
            confirm_button = widgets.Button(
                description=config.confirm_text,
                button_style=config.confirm_style,
                icon=config.confirm_icon,
                layout=widgets.Layout(width='auto', margin='0 10px 0 0' if config.show_cancel else '0')
            )
            setattr(confirm_button, '_action', 'confirm')
            buttons.append(confirm_button)
        
        # Create cancel button
        if config.show_cancel:
            cancel_button = widgets.Button(
                description=config.cancel_text,
                button_style=config.cancel_style,
                icon=config.cancel_icon,
                layout=widgets.Layout(width='auto')
            )
            setattr(cancel_button, '_action', 'cancel')
            buttons.append(cancel_button)
        
        return buttons
    
    def _setup_dialog_handlers(self, instance: DialogInstance):
        """Setup event handlers untuk dialog instance"""
        container = instance.container
        
        # Find and setup button handlers
        for widget in self._get_all_widgets(container):
            if isinstance(widget, widgets.Button) and hasattr(widget, '_action'):
                action = getattr(widget, '_action')
                widget.on_click(lambda btn, act=action, inst=instance: self._handle_button_click(inst, act, btn))
        
        # Store component references
        for widget in self._get_all_widgets(container):
            if hasattr(widget, 'description') and widget.description:
                widget_type = type(widget).__name__
                instance.components[f"{widget_type}_{len(instance.components)}"] = widget
    
    def _get_all_widgets(self, widget) -> List[widgets.Widget]:
        """Recursively get all widgets from a container"""
        widgets_list = [widget]
        if hasattr(widget, 'children'):
            for child in widget.children:
                widgets_list.extend(self._get_all_widgets(child))
        return widgets_list
    
    def _handle_button_click(self, instance: DialogInstance, action: str, button: widgets.Button):
        """Handle button click dengan safe execution"""
        try:
            # Update state
            if action == 'confirm':
                instance.state = DialogState.PROCESSING
            elif action == 'cancel':
                instance.state = DialogState.CANCELLED
            
            # Disable buttons
            self._disable_dialog_buttons(instance)
            
            # Execute callback if available
            callback = instance.callbacks.get(f'on_{action}')
            if callback and callable(callback):
                callback(button)
            
            # Auto close atau update state
            if action in ['confirm', 'cancel'] or instance.dialog_type == DialogType.INFO:
                self._close_dialog(instance.dialog_id)
            
        except Exception as e:
            instance.state = DialogState.ERROR
            instance.metadata['error'] = str(e)
    
    def _disable_dialog_buttons(self, instance: DialogInstance):
        """Disable semua button dalam dialog"""
        for widget in self._get_all_widgets(instance.container):
            if isinstance(widget, widgets.Button):
                widget.disabled = True
    
    def get_dialog(self, dialog_id: str) -> Optional[DialogInstance]:
        """Get dialog instance berdasarkan ID"""
        return self.active_dialogs.get(dialog_id)
    
    def update_dialog(self, dialog_id: str, **kwargs) -> bool:
        """Update dialog properties"""
        instance = self.active_dialogs.get(dialog_id)
        if not instance:
            return False
        
        try:
            # Update progress (khusus untuk progress dialog)
            if 'progress' in kwargs and instance.dialog_type == DialogType.PROGRESS:
                progress_widget = None
                status_widget = None
                
                for widget in self._get_all_widgets(instance.container):
                    if isinstance(widget, widgets.IntProgress):
                        progress_widget = widget
                    elif isinstance(widget, widgets.HTML) and 'text-align: center' in widget.value:
                        status_widget = widget
                
                if progress_widget:
                    progress_widget.value = max(0, min(100, kwargs['progress']))
                
                if status_widget and 'status' in kwargs:
                    status_widget.value = f'<div style="text-align: center; color: #666; font-size: 12px;">{kwargs["status"]}</div>'
            
            # Update other properties
            instance.metadata.update(kwargs)
            return True
            
        except Exception:
            return False
    
    def complete_dialog(self, dialog_id: str, success: bool = True, message: str = None):
        """Complete dialog dengan status tertentu"""
        instance = self.active_dialogs.get(dialog_id)
        if not instance:
            return
        
        instance.state = DialogState.COMPLETED
        instance.metadata.update({
            'completed_at': __import__('time').time(),
            'success': success,
            'final_message': message
        })
        
        # Auto close jika ada delay
        if instance.config.auto_close_delay > 0:
            import threading
            def delayed_close():
                import time
                time.sleep(instance.config.auto_close_delay)
                self._close_dialog(dialog_id)
            
            threading.Thread(target=delayed_close, daemon=True).start()
    
    def _close_dialog(self, dialog_id: str):
        """Close dan cleanup dialog"""
        instance = self.active_dialogs.get(dialog_id)
        if not instance:
            return
        
        # Hide dialog
        if hasattr(instance.container, 'layout'):
            instance.container.layout.display = 'none'
            instance.container.layout.visibility = 'hidden'
        
        # Remove from active dialogs
        del self.active_dialogs[dialog_id]
    
    def close_dialog(self, dialog_id: str) -> bool:
        """Public method untuk close dialog"""
        if dialog_id in self.active_dialogs:
            self._close_dialog(dialog_id)
            return True
        return False
    
    def _cleanup_completed_dialogs(self):
        """Cleanup dialog yang sudah completed"""
        completed_dialogs = [
            dialog_id for dialog_id, instance in self.active_dialogs.items()
            if instance.state in [DialogState.COMPLETED, DialogState.CANCELLED, DialogState.ERROR]
        ]
        
        for dialog_id in completed_dialogs:
            self._close_dialog(dialog_id)
    
    def cleanup_all_dialogs(self):
        """Force cleanup semua active dialogs"""
        dialog_ids = list(self.active_dialogs.keys())
        for dialog_id in dialog_ids:
            self._close_dialog(dialog_id)
        
        if dialog_ids:
            print(f"🧹 Auto-cleaned {len(dialog_ids)} active dialogs")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get status dari dialog manager untuk debugging"""
        return {
            'active_dialogs': len(self.active_dialogs),
            'dialog_types': [instance.dialog_type.value for instance in self.active_dialogs.values()],
            'dialog_states': [instance.state.value for instance in self.active_dialogs.values()],
            'cleanup_registered': DialogManager._cleanup_registered,
            'history_count': len(self.dialog_history)
        }

# Singleton instance
def get_dialog_manager() -> DialogManager:
    """Get singleton dialog manager instance"""
    return DialogManager()

# One-liner utility functions
create_dialog_config = lambda **kwargs: DialogConfig(**kwargs)
get_active_dialog_count = lambda: len(get_dialog_manager().active_dialogs)
cleanup_all = lambda: get_dialog_manager().cleanup_all_dialogs()
get_dialog_by_id = lambda dialog_id: get_dialog_manager().get_dialog(dialog_id)