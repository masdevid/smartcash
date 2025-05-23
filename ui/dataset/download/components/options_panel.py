"""
File: smartcash/ui/dataset/download/components/options_panel.py
Deskripsi: Updated options panel dengan Drive path info
"""

import ipywidgets as widgets
from .form_fields import (
    workspace_field, project_field, version_field, api_key_field,
    output_dir_field, validate_dataset_field, backup_checkbox_field, backup_dir_field
)
from smartcash.ui.utils.constants import COLORS

def create_options_panel(config, env_manager=None):
    """Create options panel dengan Drive integration."""
    
    # Form fields
    workspace = workspace_field(config)
    project = project_field(config)
    version = version_field(config)
    api_key = api_key_field()
    
    # Output directory dengan Drive default
    if env_manager and env_manager.is_drive_mounted:
        default_output = str(env_manager.drive_path / 'downloads')
        output_dir = output_dir_field({**config, 'dir': default_output})
        backup_dir = backup_dir_field(str(env_manager.drive_path / 'backups'))
    else:
        output_dir = output_dir_field(config)
        backup_dir = backup_dir_field()
    
    validate_dataset = validate_dataset_field()
    backup_checkbox = backup_checkbox_field()
    
    # Create panel
    panel = widgets.VBox([
        workspace, project, version, api_key, 
        output_dir, validate_dataset, 
        backup_checkbox, backup_dir
    ], layout=widgets.Layout(
        width='100%', margin='10px 0', padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px'
    ))
    
    return {
        'panel': panel,
        'workspace': workspace,
        'project': project, 
        'version': version,
        'api_key': api_key,
        'output_dir': output_dir,
        'validate_dataset': validate_dataset,
        'backup_checkbox': backup_checkbox,
        'backup_dir': backup_dir
    }