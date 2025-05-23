"""
File: smartcash/ui/dataset/download/components/form_fields.py
Deskripsi: Updated form fields dengan Drive path defaults
"""

import ipywidgets as widgets
import os
from smartcash.common.environment import get_environment_manager

def output_dir_field(config):
    """Output directory field dengan Drive default."""
    env_manager = get_environment_manager()
    
    # Default ke Drive jika tersedia
    if env_manager.is_colab and env_manager.is_drive_mounted:
        default_value = str(env_manager.drive_path / 'downloads')
    else:
        default_value = config.get('dir', 'data')
    
    return widgets.Text(
        value=default_value,
        placeholder='Path penyimpanan dataset',
        description='Output Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )

def backup_dir_field(default_path=None):
    """Backup directory field dengan Drive default."""
    env_manager = get_environment_manager()
    
    if default_path:
        value = default_path
    elif env_manager.is_colab and env_manager.is_drive_mounted:
        value = str(env_manager.drive_path / 'backups')
    else:
        value = 'data/backup'
    
    return widgets.Text(
        value=value,
        placeholder='Path backup dataset',
        description='Backup Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )

def workspace_field(config):
    value = config.get('workspace', 'smartcash-wo2us')
    return widgets.Text(
        value=value,
        placeholder='Workspace ID',
        description='Workspace:',
        layout=widgets.Layout(width='100%')
    )

def project_field(config):
    value = config.get('project', 'rupiah-emisi-2022')
    return widgets.Text(
        value=value,
        placeholder='Project ID',
        description='Project:',
        layout=widgets.Layout(width='100%')
    )

def version_field(config):
    value = config.get('version', '3')
    return widgets.Text(
        value=value,
        placeholder='Version',
        description='Version:',
        layout=widgets.Layout(width='100%')
    )

def api_key_field():
    value = os.environ.get('ROBOFLOW_API_KEY', '')
    return widgets.Password(
        value=value,
        placeholder='API Key',
        description='API Key:',
        layout=widgets.Layout(width='100%')
    )

def validate_dataset_field():
    return widgets.Checkbox(
        value=True,
        description='Validasi dataset setelah download',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )

def backup_checkbox_field():
    return widgets.Checkbox(
        value=True,
        description='Backup dataset sebelum menghapus',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )
