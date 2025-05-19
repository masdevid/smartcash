"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment
"""

from typing import Dict, Any
from IPython.display import display
import ipywidgets as widgets

from smartcash.common.environment import EnvironmentManager
from smartcash.common.config.manager import ConfigManager

def create_env_config_ui(env_manager: EnvironmentManager, config_manager: ConfigManager) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment
    
    Args:
        env_manager: Environment manager instance
        config_manager: Config manager instance
        
    Returns:
        Dictionary UI components
    """
    # Buat progress bar
    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        style={'bar_color': 'maroon'},
        orientation='horizontal'
    )
    progress_bar.layout.visibility = 'hidden'
    
    # Buat progress message
    progress_message = widgets.HTML(
        value="",
        placeholder='',
        description='',
    )
    progress_message.layout.visibility = 'hidden'
    
    # Buat tombol sinkronisasi
    sync_button = widgets.Button(
        description='Sinkronkan dengan Drive',
        disabled=False,
        button_style='info',
        tooltip="Sinkronkan konfigurasi dengan Google Drive",
        icon='sync'
    )
    
    # Buat tombol cek environment
    check_button = widgets.Button(
        description='Cek Environment',
        disabled=False,
        button_style='info',
        tooltip="Cek status environment",
        icon='check'
    )
    
    # Buat tombol simpan
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        disabled=False,
        button_style='success',
        tooltip="Simpan konfigurasi ke file",
        icon='save'
    )
    
    # Buat output area
    output = widgets.Output()
    
    # Buat layout
    button_box = widgets.HBox([sync_button, check_button, save_button])
    progress_box = widgets.VBox([progress_bar, progress_message])
    ui = widgets.VBox([button_box, progress_box, output])
    
    return {
        'ui': ui,
        'sync_button': sync_button,
        'check_button': check_button,
        'save_button': save_button,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'output': output
    }
