"""
File: smartcash/ui/dataset/download/components/options_panel.py
Deskripsi: Updated options panel dengan struktur info dan field yang diperbaiki
"""

import ipywidgets as widgets
from .form_fields import (
    workspace_field, project_field, version_field, api_key_field,
    output_dir_field, validate_dataset_field, backup_checkbox_field, 
    backup_dir_field, organize_dataset_field, show_structure_info
)
from smartcash.ui.utils.constants import COLORS
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

def create_options_panel(config, env_manager=None):
    """Create options panel dengan struktur info dan field yang tepat."""
    
    # Use provided env_manager atau create new
    if env_manager is None:
        env_manager = get_environment_manager()
    
    # Form fields
    workspace = workspace_field(config)
    project = project_field(config)
    version = version_field(config)
    api_key = api_key_field()
    
    # Directory fields dengan default yang benar
    output_dir = output_dir_field(config)
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    backup_dir = backup_dir_field(paths['backup'])
    
    # Options checkboxes
    validate_dataset = validate_dataset_field()
    backup_checkbox = backup_checkbox_field()
    organize_dataset = organize_dataset_field()  # Always enabled
    
    # Structure info widget
    structure_info = show_structure_info()
    
    # Create panel dengan urutan yang logis
    panel = widgets.VBox([
        # Dataset info section
        widgets.HTML('<h4 style="margin: 0 0 10px 0; color: #495057;">📊 Dataset Information</h4>'),
        workspace, 
        project, 
        version, 
        api_key,
        
        # Directory section  
        widgets.HTML('<h4 style="margin: 15px 0 10px 0; color: #495057;">📁 Directory Settings</h4>'),
        output_dir,
        backup_dir,
        
        # Options section
        widgets.HTML('<h4 style="margin: 15px 0 10px 0; color: #495057;">⚙️ Process Options</h4>'),
        organize_dataset,  # Always enabled untuk konsistensi
        validate_dataset,
        backup_checkbox,
        
        # Info section
        structure_info
        
    ], layout=widgets.Layout(
        width='100%', 
        margin='10px 0', 
        padding='15px',
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
        'backup_dir': backup_dir,
        'validate_dataset': validate_dataset,
        'backup_checkbox': backup_checkbox,
        'organize_dataset': organize_dataset,
        'structure_info': structure_info
    }