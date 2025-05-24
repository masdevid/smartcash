"""
File: smartcash/ui/dataset/download/components/options_panel.py
Deskripsi: Fixed layout tanpa horizontal scrollbar dan responsive
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
    """Create options panel dengan layout responsive tanpa horizontal scroll."""
    
    if env_manager is None:
        env_manager = get_environment_manager()
    
    # Form fields
    workspace = workspace_field(config)
    project = project_field(config)
    version = version_field(config)
    api_key = api_key_field()
    
    # Directory fields
    output_dir = output_dir_field(config)
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    backup_dir = backup_dir_field(paths['backup'])
    
    # Options checkboxes
    validate_dataset = validate_dataset_field()
    backup_checkbox = backup_checkbox_field()
    organize_dataset = organize_dataset_field()
    
    # Structure info widget
    structure_info = show_structure_info()
    
    # 📊 Left Column: Dataset Information
    dataset_info_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 8px;">📊 Dataset Information</h4>'),
        workspace, 
        project, 
        version, 
        api_key
    ], layout=widgets.Layout(
        width='calc(50% - 8px)',
        margin='0 4px 0 0',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # ⚙️ Right Column: Process Options  
    process_options_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #28a745; padding-bottom: 8px;">⚙️ Process Options</h4>'),
        organize_dataset,
        validate_dataset,
        backup_checkbox,
        widgets.HTML('<div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 4px; border-left: 4px solid #28a745;"><small style="color: #155724;"><strong>💡 Tips:</strong> Organisasi otomatis akan memindahkan dataset ke struktur final /data/train, /data/valid, /data/test</small></div>')
    ], layout=widgets.Layout(
        width='calc(50% - 8px)',
        margin='0 0 0 4px',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # 📁 Row 1: Two Columns Layout
    row1_container = widgets.HBox([
        dataset_info_section,
        process_options_section
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start',
        margin='0 0 15px 0',
        overflow='hidden'
    ))
    
    # 📁 Row 2: Directory Settings (Full Width)
    directory_settings_section = widgets.VBox([
        widgets.HTML('<h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #ffc107; padding-bottom: 8px;">📁 Directory Settings</h4>'),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML('<label style="font-weight: bold; color: #495057; margin-bottom: 5px; display: block;">📥 Download Directory</label>'),
                output_dir,
                widgets.HTML('<small style="color: #6c757d; margin-top: 5px; display: block;">Lokasi sementara untuk download dataset sebelum diorganisir</small>')
            ], layout=widgets.Layout(width='calc(50% - 24px)', margin='0 4px 0 0')),
            widgets.VBox([
                widgets.HTML('<label style="font-weight: bold; color: #495057; margin-bottom: 5px; display: block;">💾 Backup Directory</label>'),
                backup_dir,
                widgets.HTML('<small style="color: #6c757d; margin-top: 5px; display: block;">Lokasi penyimpanan backup dataset lama (jika diperlukan)</small>')
            ], layout=widgets.Layout(width='calc(50% - 24px)', margin='0 0 0 4px'))
        ], layout=widgets.Layout(width='100%', overflow='hidden')),
        structure_info
    ], layout=widgets.Layout(
        width='100%',
        margin='0',
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # 📦 Main Panel Container
    panel = widgets.VBox([
        row1_container,
        directory_settings_section
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        overflow='hidden'
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
        'structure_info': structure_info,
        'dataset_info_section': dataset_info_section,
        'process_options_section': process_options_section,
        'directory_settings_section': directory_settings_section
    }