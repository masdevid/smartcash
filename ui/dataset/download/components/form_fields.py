"""
File: smartcash/ui/dataset/download/components/form_fields.py
Deskripsi: Silent form fields tanpa verbose logging
"""

import ipywidgets as widgets
import os
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment

def api_key_field():
    """API key field dengan silent detection."""
    api_key = _detect_api_key_silent()
    
    field = widgets.Password(
        value=api_key,
        placeholder='Masukkan API Key Roboflow atau biarkan kosong jika sudah di environment',
        description='API Key:',
        layout=widgets.Layout(width='100%')
    )
    
    return field

def _detect_api_key_silent() -> str:
    """Deteksi API key tanpa logging."""
    
    # 1. Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # 2. Google Colab userdata
    try:
        from google.colab import userdata
        
        # Check primary key
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

def output_dir_field(config):
    """Output directory field dengan default ke downloads folder."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    default_value = paths['downloads']
    
    return widgets.Text(
        value=default_value,
        placeholder='Path download sementara dataset (akan dipindah ke struktur final)',
        disabled=False,
        layout=widgets.Layout(width='100%', display='inline-block')
    )

def backup_dir_field(default_path=None):
    """Backup directory field dengan path yang konsisten."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    if default_path:
        value = default_path
    else:
        value = paths['backup']
    
    return widgets.Text(
        value=value,
        placeholder='Path backup dataset',
        disabled=False,
        layout=widgets.Layout(width='100%', display='inline-block')
    )

def workspace_field(config):
    """Workspace field dengan default SmartCash."""
    value = config.get('workspace', 'smartcash-wo2us')
    return widgets.Text(
        value=value,
        placeholder='Workspace ID Roboflow',
        description='Workspace:',
        layout=widgets.Layout(width='100%')
    )

def project_field(config):
    """Project field dengan default SmartCash."""
    value = config.get('project', 'rupiah-emisi-2022')
    return widgets.Text(
        value=value,
        placeholder='Project ID Roboflow',
        description='Project:',
        layout=widgets.Layout(width='100%')
    )

def version_field(config):
    """Version field dengan default version 3."""
    value = config.get('version', '3')
    return widgets.Text(
        value=value,
        placeholder='Dataset Version',
        description='Version:',
        layout=widgets.Layout(width='100%')
    )

def validate_dataset_field():
    """Checkbox untuk validasi dataset."""
    return widgets.Checkbox(
        value=True,
        description='Validasi dataset setelah download dan organisasi',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )

def backup_checkbox_field():
    """Checkbox untuk backup dataset."""
    return widgets.Checkbox(
        value=False,
        description='Backup dataset lama sebelum replace (jika ada)',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )

def organize_dataset_field():
    """Checkbox untuk organisasi dataset (selalu aktif)."""
    return widgets.Checkbox(
        value=True,
        description='Organisir dataset ke struktur final (train/valid/test)',
        disabled=True,
        indent=False,
        layout=widgets.Layout(width='100%')
    )

def show_structure_info():
    """Widget info untuk menjelaskan struktur dataset."""
    info_html = """
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 12px; margin: 10px 0;">
        <h4 style="margin-top: 0; color: #495057;">📁 Struktur Dataset</h4>
        <p style="margin: 8px 0; color: #6c757d; font-size: 0.9em;">
            Dataset akan didownload ke folder sementara, kemudian diorganisir ke struktur final:
        </p>
        <ul style="margin: 8px 0; color: #6c757d; font-size: 0.9em;">
            <li><code>data/train/</code> - Dataset training</li>
            <li><code>data/valid/</code> - Dataset validasi</li>
            <li><code>data/test/</code> - Dataset testing</li>
        </ul>
        <p style="margin: 8px 0 0 0; color: #6c757d; font-size: 0.85em;">
            💡 Setiap folder akan berisi subdirektori <code>images/</code> dan <code>labels/</code>
        </p>
    </div>
    """
    
    return widgets.HTML(info_html)