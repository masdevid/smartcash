"""
File: smartcash/ui/handlers/observer_handler.py
Deskripsi: Handler untuk sistem observer dengan integrasi UI tanpa ThreadPool
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, List, Optional, Union, Callable

def setup_observer_handlers(
    ui_components: Dict[str, Any], 
    observer_group: str = "cell_observers"
) -> Dict[str, Any]:
    """
    Setup handler untuk observer di UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
        observer_group: Nama group untuk observer
        
    Returns:
        Dictionary UI components yang telah ditambahkan observer handler
    """
    try:
        from smartcash.components.observer.manager_observer import ObserverManager
        
        # Dapatkan observer manager
        observer_manager = ObserverManager()
        if observer_manager:
            # Unregister group yang lama jika ada
            observer_manager.unregister_group(observer_group)
            
            # Tambahkan observer manager ke UI components
            ui_components['observer_manager'] = observer_manager
            
            # Catat group observer pada UI components
            ui_components['observer_group'] = observer_group
            
            # Tambahkan fungsi cleanup
            if 'cleanup' not in ui_components:
                def cleanup():
                    """Cleanup resources"""
                    if observer_manager:
                        observer_manager.unregister_group(observer_group)
                
                ui_components['cleanup'] = cleanup
    except ImportError:
        pass
    
    return ui_components

def register_ui_observer(
    ui_components: Dict[str, Any],
    event_type: Union[str, List[str]],
    output_widget_key: str = 'status',
    observer_group: Optional[str] = None
) -> bool:
    """
    Register observer untuk update UI berdasarkan event.
    
    Args:
        ui_components: Dictionary berisi widget UI
        event_type: Tipe event atau list tipe event
        output_widget_key: Key untuk output widget
        observer_group: Nama group untuk observer (jika None, gunakan dari ui_components)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    try:
        # Dapatkan observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                # Ubah dari .get_instance() ke cara inisialisasi yang benar
                observer_manager = ObserverManager()
            except ImportError:
                return False
                
        if not observer_manager:
            return False
            
        # Dapatkan output widget
        output_widget = ui_components.get(output_widget_key)
        if not output_widget:
            return False
            
        # Dapatkan group
        group = observer_group or ui_components.get('observer_group', 'cell_observers')
        
        # Buat callback function
        def update_ui_callback(event_type, sender, message=None, **kwargs):
            if message:
                status = kwargs.get('status', 'info')
                with output_widget:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    display(create_status_indicator(status, message))
        
        # Register observer
        if isinstance(event_type, list):
            for et in event_type:
                observer_manager.create_simple_observer(
                    event_type=et,
                    callback=update_ui_callback,
                    name=f"UI_{et}_Observer",
                    group=group
                )
        else:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=update_ui_callback,
                name=f"UI_{event_type}_Observer",
                group=group
            )
            
        return True
    except Exception:
        return False

def create_progress_observer(
    ui_components: Dict[str, Any],
    event_type: Union[str, List[str]],
    total: int,
    progress_widget_key: str = 'progress_bar',
    observer_group: Optional[str] = None,
    update_output: bool = True,
    output_widget_key: str = 'status'
) -> bool:
    """
    Buat dan register observer untuk progress bar.
    
    Args:
        ui_components: Dictionary berisi widget UI
        event_type: Tipe event atau list tipe event
        total: Total progress
        progress_widget_key: Key untuk progress bar widget
        observer_group: Nama group untuk observer (jika None, gunakan dari ui_components)
        update_output: Juga update output widget
        output_widget_key: Key untuk output widget
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    try:
        # Dapatkan observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                observer_manager = ObserverManager()
            except ImportError:
                return False
                
        if not observer_manager:
            return False
            
        # Dapatkan progress widget
        progress_widget = ui_components.get(progress_widget_key)
        if not progress_widget:
            return False
            
        # Setup progress widget
        progress_widget.max = total
        progress_widget.value = 0
            
        # Dapatkan output widget
        output_widget = ui_components.get(output_widget_key) if update_output else None
            
        # Dapatkan group
        group = observer_group or ui_components.get('observer_group', 'cell_observers')
        
        # Buat callback function
        def update_progress_callback(event_type, sender, progress=None, message=None, **kwargs):
            if progress is not None:
                progress_widget.value = progress
                
            if message and output_widget:
                status = kwargs.get('status', 'info')
                with output_widget:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    display(create_status_indicator(status, message))
        
        # Register observer
        if isinstance(event_type, list):
            for et in event_type:
                observer_manager.create_simple_observer(
                    event_type=et,
                    callback=update_progress_callback,
                    name=f"Progress_{et}_Observer",
                    group=group
                )
        else:
            observer_manager.create_simple_observer(
                event_type=event_type,
                callback=update_progress_callback,
                name=f"Progress_{event_type}_Observer",
                group=group
            )
            
        return True
    except Exception:
        return False