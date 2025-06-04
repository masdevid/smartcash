"""
File: smartcash/ui/dataset/download/components/download_forms.py
Deskripsi: Consolidated form components untuk download dengan responsive layout dan smart defaults
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.common.constants.paths import get_paths_for_environment
import os

def create_download_forms(config: Optional[Dict[str, Any]] = None, env_manager=None) -> Dict[str, Any]:
    """Create all form collections untuk download dengan responsive design."""
    config = config or {}
    
    # Primary forms (dataset info)
    primary_forms = create_dataset_info_forms(config)
    
    # Configuration forms (paths & options)  
    config_forms = create_path_and_options_forms(config, env_manager)
    
    # Action forms (save/reset)
    action_forms = create_action_forms()
    
    # Container assembly dengan responsive grid
    forms_containers = _assemble_form_containers(primary_forms, config_forms, action_forms)
    
    return {
        **primary_forms, **config_forms, **action_forms, **forms_containers,
        'form_sections': ['dataset_info', 'path_options', 'action_buttons']
    }

def create_dataset_info_forms(config: Dict[str, Any]) -> Dict[str, Any]:
    """Primary forms untuk dataset information."""
    # Dataset identification widgets
    dataset_widgets = {
        'workspace': widgets.Text(
            description="Workspace:", value=config.get('workspace', 'smartcash-wo2us'),
            layout=widgets.Layout(width='100%'), style={'description_width': '120px'}
        ),
        'project': widgets.Text(
            description="Project:", value=config.get('project', 'rupiah-emisi-2022'),
            layout=widgets.Layout(width='100%'), style={'description_width': '120px'}
        ),
        'version': widgets.Text(
            description="Version:", value=config.get('version', '3'),
            layout=widgets.Layout(width='100%'), style={'description_width': '120px'}
        ),
        'api_key': widgets.Password(
            value=_detect_api_key(), placeholder='Masukkan API Key Roboflow atau biarkan kosong jika sudah di environment',
            description='API Key:', layout=widgets.Layout(width='100%'), style={'description_width': '120px'}
        )
    }
    
    # Dataset info container
    dataset_container = create_responsive_container(
        list(dataset_widgets.values()),
        title="📊 Dataset Information", container_type="vbox"
    )
    
    return {**dataset_widgets, 'dataset_info_container': dataset_container}

def create_path_and_options_forms(config: Dict[str, Any], env_manager=None) -> Dict[str, Any]:
    """Configuration forms untuk paths dan processing options."""
    # Get environment-appropriate paths
    if env_manager:
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab, is_drive_mounted=env_manager.is_drive_mounted
        )
    else:
        paths = {'downloads': 'data/downloads', 'backup': 'data/backup'}
    
    # Path widgets
    path_widgets = {
        'output_dir': widgets.Text(
            value=config.get('output_dir', paths['downloads']),
            placeholder='Path download sementara dataset',
            layout=widgets.Layout(width='90%')
        ),
        'backup_dir': widgets.Text(
            value=config.get('backup_dir', paths['backup']),
            placeholder='Path backup dataset',
            layout=widgets.Layout(width='90%')
        )
    }
    
    # Options widgets
    option_widgets = {
        'organize_dataset': widgets.Checkbox(
            value=config.get('organize_dataset', True),
            description='Organisir dataset ke struktur final (train/valid/test)',
            disabled=True, layout=widgets.Layout(width='100%')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=config.get('backup_before_download', False),
            description='Backup dataset lama sebelum replace (jika ada)',
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Path section dengan labels
    path_section = widgets.VBox([
        widgets.HTML('<h4 style="margin:0 0 15px 0;color:#495057;border-bottom:2px solid #ffc107;padding-bottom:8px;">📁 Directory Settings</h4>'),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML('<label style="font-weight:bold;color:#495057;margin-bottom:5px;">📥 Download Directory</label>'),
                path_widgets['output_dir'],
                widgets.HTML('<small style="color:#6c757d;margin-top:5px;">Lokasi sementara untuk download dataset</small>')
            ], layout=widgets.Layout(width='calc(50% - 8px)', margin='0 4px 0 0')),
            widgets.VBox([
                widgets.HTML('<label style="font-weight:bold;color:#495057;margin-bottom:5px;">💾 Backup Directory</label>'),
                path_widgets['backup_dir'],
                widgets.HTML('<small style="color:#6c757d;margin-top:5px;">Lokasi backup dataset lama</small>')
            ], layout=widgets.Layout(width='calc(50% - 8px)', margin='0 0 0 4px'))
        ], layout=widgets.Layout(width='100%')),
        
        # Options section
        widgets.HTML('<h4 style="margin:15px 0 10px 0;color:#495057;border-bottom:2px solid #28a745;padding-bottom:8px;">⚙️ Process Options</h4>'),
        option_widgets['organize_dataset'],
        option_widgets['backup_checkbox'],
        
        # Info sections
        widgets.HTML('<div style="margin-top:15px;padding:10px;background:#e3f2fd;border-radius:4px;border-left:4px solid #2196f3;"><small style="color:#0d47a1;"><strong>🔍 Verifikasi:</strong> Setelah download selesai, gunakan tombol <strong>"Check Dataset"</strong> untuk memverifikasi hasil.</small></div>'),
        widgets.HTML('<div style="margin-top:8px;padding:10px;background:#e8f5e8;border-radius:4px;border-left:4px solid #28a745;"><small style="color:#155724;"><strong>💡 Organisasi:</strong> Dataset akan otomatis dipindah ke struktur final /data/train, /data/valid, /data/test</small></div>')
    ], layout=widgets.Layout(width='100%', padding='15px', border='1px solid #ddd', border_radius='5px', background_color='#f8f9fa'))
    
    return {**path_widgets, **option_widgets, 'path_options_container': path_section}

def create_action_forms() -> Dict[str, Any]:
    """Action forms dengan save/reset buttons."""
    save_reset = create_save_reset_buttons(
        "Simpan", "Reset", save_tooltip="Simpan konfigurasi download saat ini",
        reset_tooltip="Reset ke nilai default", with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan saat disimpan atau direset.",
        button_width="120px"
    )
    
    return {
        'save_reset_buttons': save_reset, 'save_button': save_reset['save_button'],
        'reset_button': save_reset['reset_button'], 'action_buttons_container': save_reset['container']
    }

def _assemble_form_containers(primary: Dict, config: Dict, action: Dict) -> Dict[str, Any]:
    """Assemble form containers dengan responsive two-column layout."""
    # Two column layout untuk dataset info dan path options
    forms_grid = create_responsive_two_column(
        primary.get('dataset_info_container'),
        config.get('path_options_container'),
        left_width="48%", right_width="48%"
    )
    
    # Main forms container
    forms_main_container = create_responsive_container([
        forms_grid, action.get('action_buttons_container')
    ], container_type="vbox")
    
    return {'forms_main_container': forms_main_container, 'forms_grid': forms_grid}

def _detect_api_key() -> str:
    """Detect API key dari environment sources."""
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key: return api_key
    
    # Google Colab userdata
    try:
        from google.colab import userdata
        for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']:
            try:
                api_key = userdata.get(key_name, '')
                if api_key: return api_key
            except: continue
    except: pass
    
    return ''

def extract_forms_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration dari semua forms dengan proper field mapping."""
    field_mapping = {
        'workspace': 'workspace', 'project': 'project', 'version': 'version',
        'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
        'backup_checkbox': 'backup_before_download', 'organize_dataset': 'organize_dataset'
    }
    
    config = {}
    for ui_key, config_key in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = ui_components[ui_key].value
            if value is not None and (isinstance(value, bool) or (isinstance(value, str) and value.strip())):
                config[config_key] = value.strip() if isinstance(value, str) else value
    
    return config

def update_forms_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update forms dari configuration dengan proper mapping."""
    field_mapping = {
        'workspace': 'workspace', 'project': 'project', 'version': 'version',
        'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
        'backup_before_download': 'backup_checkbox', 'organize_dataset': 'organize_dataset'
    }
    
    [setattr(ui_components[ui_key], 'value', config.get(config_key, ''))
     for config_key, ui_key in field_mapping.items()
     if ui_key in ui_components and hasattr(ui_components[ui_key], 'value')]