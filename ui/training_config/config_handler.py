"""
File: smartcash/ui/training_config/config_handler.py
Deskripsi: Handler bersama untuk operasi konfigurasi (save/load/reset) dengan ui_helpers
"""

import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from IPython.display import display, clear_output

from smartcash.ui.utils.alerts import create_status_indicator
from smartcash.ui.helpers.ui_helpers import update_output_area

def get_config_manager():
    """
    Dapatkan config manager dengan exception handling.
    
    Returns:
        Config manager atau None jika tidak tersedia
    """
    try:
        from smartcash.common.config import get_config_manager
        return get_config_manager()
    except ImportError as e:
        print(f"⚠️ ConfigManager tidak tersedia: {e}")
        return None

def get_observer_manager():
    """
    Dapatkan observer manager dengan exception handling.
    
    Returns:
        Observer manager atau None jika tidak tersedia
    """
    try:
        from smartcash.components.observer.manager_observer import ObserverManager
        # Buat instance baru jika tidak ada method get_instance
        return ObserverManager()
    except ImportError as e:
        print(f"⚠️ ObserverManager tidak tersedia: {e}")
        return None

def save_config(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    config_path: str,
    update_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    config_name: str = "Konfigurasi"
) -> None:
    """
    Fungsi generik untuk menyimpan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Dictionary konfigurasi
        config_path: Path file konfigurasi
        update_func: Fungsi untuk update konfigurasi dari UI
        config_name: Nama konfigurasi untuk pesan
    """
    status_widget = ui_components.get('status')
    if not status_widget:
        return
    
    with status_widget:
        clear_output(wait=True)
        display(create_status_indicator("info", f"🔄 Menyimpan {config_name.lower()}..."))
        
        try:
            # Update config dengan nilai UI terbaru
            updated_config = update_func(config)
            
            # Fallback yang langsung menyimpan ke file YAML
            try:
                import yaml
                import shutil
                # Pastikan direktori ada
                config_dir = os.path.dirname(config_path)
                os.makedirs(config_dir, exist_ok=True)
                
                # Buat backup manual jika diperlukan
                if os.path.exists(config_path):
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_path = f"{config_path}.{timestamp}.backup"
                    try:
                        shutil.copy2(config_path, backup_path)
                    except Exception:
                        pass  # Abaikan error backup
                
                # Simpan file baru
                with open(config_path, 'w') as f:
                    yaml.dump(updated_config, f, default_flow_style=False)
                
                display(create_status_indicator("success", f"✅ {config_name} berhasil disimpan ke {config_path}"))
                
                # Notifikasi observer jika tersedia
                observer_manager = get_observer_manager()
                if observer_manager:
                    try:
                        from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
                        from smartcash.components.observer.event_topics_observer import EventTopics
                        
                        # Pastikan EventTopics.CONFIG_UPDATED tersedia
                        if hasattr(EventTopics, 'CONFIG_UPDATED'):
                            EventDispatcher.notify(
                                event_type=EventTopics.CONFIG_UPDATED,
                                sender=ui_components.get('module_name', 'config_handler'),
                                message=f"{config_name} diperbarui",
                                config_path=config_path
                            )
                        else:
                            # Fallback jika CONFIG_UPDATED tidak tersedia
                            EventDispatcher.notify(
                                event_type="config.updated",  # Hardcoded string sebagai fallback
                                sender=ui_components.get('module_name', 'config_handler'),
                                message=f"{config_name} diperbarui",
                                config_path=config_path
                            )
                    except (ImportError, AttributeError) as e:
                        if logger:
                            logger.warning(f"⚠️ Error notifikasi observer: {str(e)}")
                        pass
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error menyimpan file: {str(e)}"))
                
        except Exception as e:
            display(create_status_indicator("error", f"❌ Error: {str(e)}"))

def reset_config(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    default_config: Dict[str, Any],
    update_ui_func: Callable[[Dict[str, Any]], None],
    config_name: str = "Konfigurasi"
) -> None:
    """
    Fungsi generik untuk reset konfigurasi ke default.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Dictionary konfigurasi yang akan diupdate
        default_config: Dictionary konfigurasi default
        update_ui_func: Fungsi untuk update UI dari konfigurasi
        config_name: Nama konfigurasi untuk pesan
    """
    status_widget = ui_components.get('status')
    if not status_widget:
        return
    
    with status_widget:
        clear_output(wait=True)
        display(create_status_indicator("info", f"🔄 Reset {config_name.lower()} ke default..."))
        
        try:
            # Update konfigurasi dengan nilai default
            config.clear()
            config.update(default_config)
            
            # Update komponen UI
            update_ui_func()
            
            display(create_status_indicator("success", f"✅ {config_name} berhasil direset ke default"))
            
        except Exception as e:
            display(create_status_indicator("error", f"❌ Error saat reset: {str(e)}"))