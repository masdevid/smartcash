"""
File: smartcash/ui/dataset/downloader/components/ui_form.py
Deskripsi: Form fields untuk downloader UI dengan styling konsisten
"""

import ipywidgets as widgets
import os
from typing import Dict, Any, Optional

from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.constants import COLORS

def create_form_fields(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Membuat form fields untuk downloader UI dengan styling konsisten."""
    config = config or {}
    
    # Deteksi API key secara silent
    api_key = _detect_api_key_silent()
    
    # Form fields dengan styling konsisten
    workspace_field = widgets.Text(
        value=config.get('workspace', 'smartcash-wo2us'),
        placeholder='Masukkan nama workspace Roboflow',
        description='Workspace:',
        layout=widgets.Layout(width='100%')
    )
    
    project_field = widgets.Text(
        value=config.get('project', 'rupiah-emisi-2022'),
        placeholder='Masukkan nama project Roboflow',
        description='Project:',
        layout=widgets.Layout(width='100%')
    )
    
    version_field = widgets.Text(
        value=config.get('version', '3'),
        placeholder='Masukkan versi dataset',
        description='Version:',
        layout=widgets.Layout(width='100%')
    )
    
    api_key_field = widgets.Password(
        value=api_key,
        placeholder='Masukkan API Key Roboflow atau biarkan kosong jika sudah di environment',
        description='API Key:',
        layout=widgets.Layout(width='100%')
    )
    
    # Directory fields
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    output_dir_field = widgets.Text(
        value=config.get('output_dir', paths['download']),
        placeholder='Lokasi untuk menyimpan dataset',
        description='Output Dir:',
        layout=widgets.Layout(width='100%')
    )
    
    backup_dir_field = widgets.Text(
        value=config.get('backup_dir', paths['backup']),
        placeholder='Lokasi untuk backup dataset lama',
        description='Backup Dir:',
        layout=widgets.Layout(width='100%')
    )
    
    # Options checkboxes
    backup_checkbox = widgets.Checkbox(
        value=config.get('backup_enabled', True),
        description='Backup dataset lama',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    organize_dataset = widgets.Checkbox(
        value=config.get('organize_dataset', True),
        description='Organisir dataset setelah download',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Structure info widget
    structure_info = widgets.HTML(
        value="""
        <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #6c757d;">
            <p style="margin: 0; color: #495057;"><strong>📁 Struktur Dataset:</strong></p>
            <pre style="margin: 5px 0 0; font-size: 12px; color: #495057;">
/data/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
            </pre>
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    return {
        'workspace_field': workspace_field,
        'project_field': project_field,
        'version_field': version_field,
        'api_key_field': api_key_field,
        'output_dir_field': output_dir_field,
        'backup_dir_field': backup_dir_field,
        'backup_checkbox': backup_checkbox,
        'organize_dataset': organize_dataset,
        'structure_info': structure_info
    }

def _detect_api_key_silent() -> str:
    """Deteksi API key tanpa logging."""
    # 1. Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # 2. Google Colab userdata
    try:
        from google.colab import userdata
        api_key = userdata.get('ROBOFLOW_API_KEY')
        if api_key:
            return api_key
        
        # Check alternative names
        alternative_keys = ['roboflow_api_key', 'ROBOFLOW_KEY', 'roboflow_key', 'API_KEY']
        for key_name in alternative_keys:
            try:
                api_key = userdata.get(key_name)
                if api_key:
                    return api_key
            except Exception:
                continue
                
    except ImportError:
        pass
    except Exception:
        pass
        
    return ''
