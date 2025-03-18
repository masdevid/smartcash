"""
File: smartcash/ui/utils/cell_utils.py
Deskripsi: Utilitas untuk cell notebook dengan fungsi setup environment dan komponen UI yang lebih konsisten
"""

import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_notebook_environment(
    cell_name: str,
    config_path: str = "configs/colab_config.yaml"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Setup environment notebook dengan konfigurasi cell.
    
    Args:
        cell_name: Nama cell untuk identifikasi dan logging
        config_path: Path relatif ke file konfigurasi
        
    Returns:
        Tuple berisi (environment_manager, config_dict)
    """
    # Pastikan smartcash dalam path
    if '.' not in sys.path:
        sys.path.append('.')
        
    # Import dependencies dengan fallback
    try:
        import logging
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Setup komponen
        logger = setup_ipython_logging(cell_name, log_level=logging.INFO)
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        
        # Load konfigurasi
        try:
            if Path(config_path).exists():
                config = config_manager.load_config(config_path)
                logger.info(f"🔄 Loaded config from {config_path}")
            else:
                config = config_manager.config
                logger.warning(f"⚠️ Config file {config_path} not found, using default")
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            config = {}
            
        return env_manager, config
        
    except ImportError as e:
        print(f"⚠️ Limited functionality mode - {str(e)}")
        
        # Fallback implementation
        env = type('DummyEnv', (), {
            'is_colab': 'google.colab' in sys.modules,
            'base_dir': os.getcwd(),
            'get_path': lambda p: os.path.join(os.getcwd(), p)
        })
        
        # Fallback config loading
        config = {}
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"🔄 Loaded config from {config_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            
        return env, config

def create_default_ui_components(cell_name: str) -> Dict[str, Any]:
    """
    Buat komponen UI default untuk cell.
    
    Args:
        cell_name: Nama cell
        
    Returns:
        Dictionary berisi widget UI default
    """
    # Format judul cell dari nama
    title = " ".join(word.capitalize() for word in cell_name.split("_"))
    
    # Buat komponen UI default dengan styling yang lebih baik
    header = widgets.HTML(
        f"""<div style="background-color: {COLORS['header_bg']}; padding: 15px; 
                      color: {COLORS['dark']}; border-radius: 5px; 
                      margin-bottom: 15px; border-left: 5px solid {COLORS['header_border']};">
            <h2 style="color: {COLORS['secondary']}; margin-top: 0;">{title}</h2>
            <p style="color: {COLORS['secondary']}; margin-bottom: 0;">Cell untuk {title}</p>
        </div>"""
    )
    
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='8px',
            overflow='auto'
        )
    )
    
    # Return dictionary
    return {
        'ui': widgets.VBox([header, status], 
            layout=widgets.Layout(width='100%', padding='10px')),
        'header': header,
        'status': status,
        'module_name': cell_name
    }

def setup_ui_component(
    env: Any, 
    config: Dict[str, Any], 
    component_name: str
) -> Dict[str, Any]:
    """
    Setup UI component untuk notebook cell.
    
    Args:
        env: Environment manager
        config: Konfigurasi
        component_name: Nama komponen UI
        
    Returns:
        Dictionary berisi widget UI
    """
    # Buat default UI components
    ui_components = create_default_ui_components(component_name)
    
    # Import komponen UI dengan fallback, termasuk pencarian di subdirektori
    try:
        # Coba import dari lokasi baru (setelah refactor)
        import_locations = [
            f"smartcash.ui.setup.{component_name}_component",
            f"smartcash.ui.dataset.{component_name}_component",
            f"smartcash.ui.training_config.{component_name}_component",
            f"smartcash.ui.training_execution.{component_name}_component",
            f"smartcash.ui.model_evaluation.{component_name}_component",
            f"smartcash.ui.detection.{component_name}_component"
        ]
        
        # Coba impor dari semua lokasi yang mungkin
        for module_path in import_locations:
            try:
                ui_module = importlib.import_module(module_path)
                create_func = getattr(ui_module, f"create_{component_name}_ui")
                ui_components = create_func(env, config)
                break
            except (ImportError, AttributeError):
                continue
        else:
            # Jika tidak ditemukan
            with ui_components['status']:
                display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                          color:{COLORS['alert_warning_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_warning_text']};">
                    <p style="margin:5px 0">{ICONS['warning']} UI component '{component_name}' tidak ditemukan. 
                       Menggunakan implementasi minimal.</p>
                </div>
                """))
                
    except Exception as e:
        # Catch any other errors during import
        with ui_components['status']:
            display(HTML(f"""
            <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                      color:{COLORS['alert_danger_text']}; 
                      border-radius:4px; margin:5px 0;
                      border-left:4px solid {COLORS['alert_danger_text']};">
                <p style="margin:5px 0">{ICONS['error']} Error saat import komponen '{component_name}': {str(e)}</p>
            </div>
            """))
    
    # Setup handler jika tersedia
    try:
        handler_locations = [
            f"smartcash.ui.handlers.{component_name}",
            f"smartcash.ui_handlers.{component_name}"
        ]
        
        for handler_path in handler_locations:
            try:
                handler_module = importlib.import_module(handler_path)
                setup_handlers_func = getattr(handler_module, f"setup_{component_name}_handlers")
                ui_components = setup_handlers_func(ui_components, env, config)
                break
            except (ImportError, AttributeError):
                continue
    except Exception:
        pass  # Handler tidak tersedia, lanjutkan
    
    return ui_components

def display_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display UI components dalam notebook.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    if 'ui' in ui_components:
        display(ui_components['ui'])
    else:
        # Fallback jika 'ui' tidak ada
        for key, component in ui_components.items():
            if isinstance(component, (widgets.Widget, widgets.widgets.Widget)):
                display(component)
                break
        else:
            display(HTML(f"""
            <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                      color:{COLORS['alert_warning_text']}; 
                      border-radius:4px; margin:5px 0;">
                <p style="margin:5px 0">{ICONS['warning']} Tidak ada komponen UI yang dapat ditampilkan</p>
            </div>
            """))

def cleanup_resources(ui_components: Dict[str, Any]) -> None:
    """
    Cleanup resources yang digunakan oleh UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    if 'cleanup' in ui_components and callable(ui_components['cleanup']):
        ui_components['cleanup']()
    
    # Cleanup observer jika ada
    try:
        from smartcash.components.observer import ObserverManager
        observer_manager = ObserverManager.get_instance()
        if observer_manager:
            group_name = ui_components.get('observer_group', 'cell_observers')
            observer_manager.unregister_group(group_name)
    except ImportError:
        pass