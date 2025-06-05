"""
File: smartcash/ui/dataset/downloader/components/ui_form.py
Deskripsi: Enhanced form fields dengan auto-detect API key dan responsive design
"""

import ipywidgets as widgets
import os
from typing import Dict, Any, Optional
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.constants import COLORS

def create_form_fields(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create form fields dengan auto-detect API key dan improved styling."""
    config = config or {}
    
    # Auto-detect API key dari Colab secrets
    api_key = _detect_api_key_from_colab_secrets()
    roboflow = config.get('roboflow', {})
    local = config.get('local', {})
    
    # Override config API key jika terdeteksi dari secrets
    if api_key and not roboflow.get('api_key'): roboflow['api_key'] = api_key
    
    # Get environment-specific paths
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(is_colab=env_manager.is_colab, is_drive_mounted=env_manager.is_drive_mounted)
    
    # Form fields dengan consistent styling
    workspace_field = widgets.Text(value=roboflow.get('workspace', 'smartcash-wo2us'), placeholder='Nama workspace di Roboflow (e.g., smartcash-wo2us)', description='Workspace:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'})
    project_field = widgets.Text(value=roboflow.get('project', 'rupiah-emisi-2022'), placeholder='Nama project di Roboflow (e.g., rupiah-emisi-2022)', description='Project:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'})
    version_field = widgets.Text(value=str(roboflow.get('version', '3')), placeholder='Versi dataset (e.g., 3)', description='Version:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'})
    
    # API key field dengan auto-detect dari secrets
    api_key_field = widgets.Password(value=api_key or roboflow.get('api_key', ''), placeholder='🔑 Terdeteksi otomatis dari Colab secrets' if api_key else 'Masukkan API Key Roboflow', description='API Key:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'}, disabled=bool(api_key))
    
    # Directory fields dengan smart defaults
    output_dir_field = widgets.Text(value=local.get('output_dir', paths.get('data_root', '/content/data')), placeholder='Direktori untuk menyimpan dataset (e.g., /content/data)', description='Output Dir:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'})
    backup_dir_field = widgets.Text(value=local.get('backup_dir', paths.get('backup', '/content/data/backup')), placeholder='Direktori untuk backup dataset lama', description='Backup Dir:', layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': '90px'})
    
    # Backup checkbox dengan improved description
    backup_checkbox = widgets.Checkbox(value=local.get('backup_enabled', False), description='Enable backup dataset existing sebelum download', indent=False, layout=widgets.Layout(width='100%', max_width='100%'), style={'description_width': 'initial'})
    
    # API key status indicator
    api_status_html = _create_api_status_indicator(api_key)
    
    # Environment info widget
    env_info_html = _create_environment_info(env_manager)
    
    # Dataset structure preview
    structure_info = _create_structure_preview()
    
    return {
        'workspace_field': workspace_field,
        'project_field': project_field,
        'version_field': version_field,
        'api_key_field': api_key_field,
        'output_dir_field': output_dir_field,
        'backup_dir_field': backup_dir_field,
        'backup_checkbox': backup_checkbox,
        'api_status_html': api_status_html,
        'env_info_html': env_info_html,
        'structure_info': structure_info
    }

def _detect_api_key_from_colab_secrets() -> str:
    """Detect API key dari Colab secrets dengan priority order."""
    # 1. Colab userdata (secrets) - highest priority
    try:
        from google.colab import userdata
        for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'API_KEY']:
            try:
                api_key = userdata.get(key_name, '').strip()
                if api_key and len(api_key) > 10:
                    return api_key
            except Exception:
                continue
    except ImportError:
        pass
    
    # 2. Environment variables - fallback
    for env_key in ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']:
        api_key = os.environ.get(env_key, '').strip()
        if api_key and len(api_key) > 10:
            return api_key
    
    return ''

def _create_api_status_indicator(api_key: str) -> widgets.HTML:
    """Create API key status indicator."""
    if api_key:
        status_html = """
        <div style="margin-top: 8px; padding: 8px 12px; background: #e8f5e8; 
                   border-radius: 6px; border-left: 4px solid #4caf50;">
            <small style="color: #2e7d32; font-weight: 500;">
                ✅ API Key terdeteksi dari Colab secrets
            </small>
        </div>
        """
    else:
        status_html = """
        <div style="margin-top: 8px; padding: 8px 12px; background: #fff3cd; 
                   border-radius: 6px; border-left: 4px solid #ffc107;">
            <small style="color: #856404; font-weight: 500;">
                ⚠️ API Key belum terdeteksi - masukkan manual atau set di Colab secrets
            </small>
        </div>
        """
    
    return widgets.HTML(status_html)

def _create_environment_info(env_manager) -> widgets.HTML:
    """Create environment information widget."""
    try:
        if env_manager.is_colab:
            if env_manager.is_drive_mounted:
                env_html = """
                <div style="margin: 10px 0; padding: 10px 12px; background: #e8f5e8; 
                           border-radius: 6px; border-left: 4px solid #4caf50;">
                    <small style="color: #2e7d32; line-height: 1.4;">
                        <strong>🌐 Environment:</strong> Google Colab + Drive terhubung<br>
                        <strong>💾 Storage:</strong> Dataset akan tersimpan permanen di Drive
                    </small>
                </div>
                """
            else:
                env_html = """
                <div style="margin: 10px 0; padding: 10px 12px; background: #fff3cd; 
                           border-radius: 6px; border-left: 4px solid #ffc107;">
                    <small style="color: #856404; line-height: 1.4;">
                        <strong>🌐 Environment:</strong> Google Colab (Drive tidak terhubung)<br>
                        <strong>⚠️ Warning:</strong> Dataset akan hilang saat runtime restart
                    </small>
                </div>
                """
        else:
            env_html = """
            <div style="margin: 10px 0; padding: 10px 12px; background: #e3f2fd; 
                       border-radius: 6px; border-left: 4px solid #2196f3;">
                <small style="color: #1976d2; line-height: 1.4;">
                    <strong>🖥️ Environment:</strong> Local development<br>
                    <strong>💾 Storage:</strong> Dataset akan tersimpan lokal
                </small>
            </div>
            """
    except Exception:
        env_html = """
        <div style="margin: 10px 0; padding: 10px 12px; background: #f5f5f5; 
                   border-radius: 6px; border-left: 4px solid #9e9e9e;">
            <small style="color: #616161;">
                <strong>🖥️ Environment:</strong> Unknown
            </small>
        </div>
        """
    
    return widgets.HTML(env_html)

def _create_structure_preview() -> widgets.HTML:
    """Create dataset structure preview widget."""
    structure_html = """
    <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; 
               border-radius: 6px; border: 1px solid #e9ecef;">
        <div style="font-weight: 600; color: #495057; margin-bottom: 8px;">
            📁 Struktur Dataset Hasil:
        </div>
        <pre style="margin: 0; font-size: 12px; color: #6c757d; line-height: 1.4; 
                   font-family: 'Courier New', monospace; background: none; 
                   border: none; padding: 0;">
/data/
├── train/
│   ├── images/     # Gambar training
│   └── labels/     # Label YOLO format
├── valid/
│   ├── images/     # Gambar validation  
│   └── labels/     # Label YOLO format
├── test/
│   ├── images/     # Gambar testing
│   └── labels/     # Label YOLO format
└── data.yaml       # Config file YOLOv5
        </pre>
    </div>
    """
    
    return widgets.HTML(structure_html)