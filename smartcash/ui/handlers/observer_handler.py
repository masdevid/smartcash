"""
File: smartcash/ui/handlers/observer_handler.py
Deskripsi: Handler untuk sistem observer dengan integrasi UI minimalis dan efisien
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, List, Optional, Union, Callable

def setup_observer_handlers(
    ui_components: Dict[str, Any], 
    observer_group: str = "cell_observers"
) -> Dict[str, Any]:
    """
    Setup handler untuk observer di UI components tanpa duplikasi implementasi.
    
    Args:
        ui_components: Dictionary berisi widget UI
        observer_group: Nama group untuk observer
        
    Returns:
        Dictionary UI components yang telah ditambahkan observer handler
    """
    try:
        from smartcash.components.observer.manager_observer import ObserverManager
        
        # Gunakan instance ObserverManager yang sudah ada atau buat baru jika belum ada
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            observer_manager = ObserverManager()
            ui_components['observer_manager'] = observer_manager
        
        # Catat group observer pada UI components
        ui_components['observer_group'] = observer_group
        
        # Tambahkan fungsi cleanup jika belum ada
        if 'cleanup' not in ui_components:
            def cleanup():
                """Cleanup resources"""
                if observer_manager:
                    observer_manager.unregister_group(observer_group)
            
            ui_components['cleanup'] = cleanup
            
            # Daftarkan ke IPython events jika dalam notebook environment
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython:
                    ipython.events.register('pre_run_cell', cleanup)
            except (ImportError, AttributeError):
                pass
    except ImportError:
        # Fallback jika ObserverManager tidak tersedia tanpa log yang berisik
        pass
    
    return ui_components

def register_ui_observer(
    ui_components: Dict[str, Any],
    event_type: Union[str, List[str]],
    callback: Optional[Callable] = None,
    output_widget_key: str = 'status',
    observer_group: Optional[str] = None
) -> bool:
    """
    Register observer untuk update UI berdasarkan event dengan delegasi ke ObserverManager.
    
    Args:
        ui_components: Dictionary berisi widget UI
        event_type: Tipe event atau list tipe event
        callback: Callback function untuk observer (opsional)
        output_widget_key: Key untuk output widget
        observer_group: Nama group untuk observer (jika None, gunakan dari ui_components)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Dapatkan observer manager dari ui_components
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager:
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            observer_manager = ObserverManager()
            ui_components['observer_manager'] = observer_manager
        except ImportError:
            return False
    
    # Dapatkan output widget
    output_widget = ui_components.get(output_widget_key)
    
    # Dapatkan group
    group = observer_group or ui_components.get('observer_group', 'cell_observers')
    
    # Gunakan callback yang disediakan atau buat callback default untuk update UI
    if not callback and output_widget:
        def default_callback(event_type, sender, message=None, **kwargs):
            if message and hasattr(output_widget, 'clear_output'):
                status = kwargs.get('status', 'info')
                with output_widget:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    display(create_status_indicator(status, message))
        callback = default_callback
    
    if not callback:
        return False
    
    # Delegasi pembuatan observer ke ObserverManager
    try:
        if isinstance(event_type, list):
            for et in event_type:
                observer_manager.create_simple_observer(
                    event_type=et,
                    callback=callback,
                    name=f"UI_{et}_Observer",
                    group=group
                )
        else:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=callback,
                name=f"UI_{event_type}_Observer",
                group=group
            )
        return True
    except Exception:
        return False

def create_progress_observer(
    ui_components: Dict[str, Any],
    event_types: Union[str, List[str]],
    progress_widget_key: str = 'progress_bar',
    progress_label_key: str = 'progress_message',
    observer_group: Optional[str] = None
) -> bool:
    """
    Buat observer untuk progress tracking dengan delegasi ke ObserverManager.
    
    Args:
        ui_components: Dictionary berisi widget UI
        event_types: Tipe event atau list tipe event
        progress_widget_key: Key untuk progress bar widget
        progress_label_key: Key untuk progress label widget
        observer_group: Nama group untuk observer (jika None, gunakan dari ui_components)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Dapatkan observer manager dari ui_components
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager:
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            observer_manager = ObserverManager()
            ui_components['observer_manager'] = observer_manager
        except ImportError:
            return False
    
    # Dapatkan progress widget
    progress_widget = ui_components.get(progress_widget_key)
    progress_label = ui_components.get(progress_label_key)
    
    if not progress_widget:
        return False
    
    # Dapatkan group
    group = observer_group or ui_components.get('observer_group', 'cell_observers')
    
    # Buat callback function
    def update_progress_callback(event_type, sender, progress=None, message=None, **kwargs):
        # Update progress bar
        if progress is not None and hasattr(progress_widget, 'value'):
            total = kwargs.get('total', progress_widget.max)
            progress_widget.max = total
            progress_widget.value = progress
            
            # Update persentase jika adalah widget IntProgress
            if hasattr(progress_widget, 'description'):
                percentage = int((progress / total) * 100) if total > 0 else 0
                progress_widget.description = f"Proses: {percentage}%"
        
        # Update progress label
        if message and progress_label and hasattr(progress_label, 'value'):
            progress_label.value = message
    
    # Register observer dengan delegasi ke ObserverManager
    try:
        event_types_list = event_types if isinstance(event_types, list) else [event_types]
        
        for et in event_types_list:
            observer_manager.create_simple_observer(
                event_type=et,
                callback=update_progress_callback,
                name=f"Progress_{et}_Observer",
                group=group
            )
        return True
    except Exception:
        return False