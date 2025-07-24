"""
File: smartcash/ui/dataset/downloader/components/input_options.py
Deskripsi: Form input components untuk downloader dengan responsive layout
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.core.mixins.colab_secrets_mixin import ColabSecretsMixin


def create_downloader_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen input options untuk downloader dengan responsive layout.
    
    Args:
        config: Konfigurasi downloader
        
    Returns:
        widgets.VBox: Container input options dengan responsive layout
    """
    if not config:
        config = {}
    
    roboflow_config = config.get('data', {}).get('roboflow', {})
    download_config = config.get('download', {})
    
    # Cek API key dari secrets menggunakan ColabSecretsMixin
    secrets_mixin = ColabSecretsMixin()
    api_key = secrets_mixin.get_secret()
    
    # === KOLOM KIRI: Dataset Info (Fixed Width) ===
    
    # Workspace input
    workspace_input = widgets.Text(
        value=roboflow_config.get('workspace', 'smartcash-wo2us'),
        description='Workspace:',
        placeholder='Nama workspace Roboflow',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Project input
    project_input = widgets.Text(
        value=roboflow_config.get('project', 'rupiah-emisi-2022'),
        description='Project:',
        placeholder='Nama project Roboflow',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Version input
    version_input = widgets.Text(
        value=str(roboflow_config.get('version', '3')),
        description='Versi:',
        placeholder='Versi dataset',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # API Key input dengan auto-detection
    api_key_value = api_key or roboflow_config.get('api_key', '')
    api_key_input = widgets.Password(
        value=api_key_value,
        description='Kunci API:',
        placeholder='🔑 Otomatis dari Colab secrets' if api_key else 'Masukkan Kunci API Roboflow',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Left column dengan fixed padding
    left_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 6px; color: #666; font-weight: bold; font-size: 13px;'>🎯 Target Dataset</div>"),
        widgets.HTML("<div style='margin-bottom: 3px; color: #888; font-size: 11px;'>Pengenal workspace Roboflow</div>"),
        workspace_input,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Nama project dalam workspace</div>"),
        project_input,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Versi dataset yang akan diunduh</div>"),
        version_input,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Kunci API untuk mengakses Roboflow</div>"),
        api_key_input
    ], layout=widgets.Layout(width='47%', padding='8px'))
    
    # === KOLOM KANAN: Opsi Unduhan (Fixed Width) ===
    
    # Validation checkbox
    validate_checkbox = widgets.Checkbox(
        value=download_config.get('validate_download', True),
        description='Validasi hasil unduhan',
        style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Backup checkbox
    backup_checkbox = widgets.Checkbox(
        value=download_config.get('backup_existing', False),
        description='Cadangkan data yang ada',
        style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # UUID renaming info (readonly)
    uuid_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin: 6px 0;">
        <small style="color: #1976d2;"><strong>🔤 UUID Renaming:</strong> Aktif (otomatis)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # Format info (readonly)
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #f3e5f5; border-radius: 4px; margin: 6px 0;">
        <small style="color: #7b1fa2;"><strong>📦 Format:</strong> YOLOv5 PyTorch</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # API key status info
    api_key_status = widgets.HTML(
        value=secrets_mixin.create_api_key_info_html({
            'data': {
                'roboflow': {
                    'api_key': api_key or ''
                }
            }
        }),
        layout=widgets.Layout(width='100%', margin='6px 0')
    )
    
    # Right column dengan fixed padding
    right_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 6px; color: #666; font-weight: bold; font-size: 13px;'>⚙️ Download Options</div>"),
        widgets.HTML("<div style='margin-bottom: 3px; color: #888; font-size: 11px;'>Verifikasi integritas dataset</div>"),
        validate_checkbox,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Backup data sebelum replace</div>"),
        backup_checkbox,
        uuid_info,
        format_info,
        api_key_status
    ], layout=widgets.Layout(width='47%', padding='8px'))
    
    # Container 2 kolom dengan responsive spacing
    columns_container = widgets.HBox([
        left_column, 
        right_column
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start',
        margin='0px',
        padding='0px'
    ))
    
    # Safe icon dan color access
    def get_safe_icon(key: str, fallback: str = "📥") -> str:
        try:
            return ICONS.get(key, fallback)
        except (NameError, AttributeError):
            return fallback
    
    def get_safe_color(key: str, fallback: str = "#333") -> str:
        try:
            return COLORS.get(key, fallback)
        except (NameError, AttributeError):
            return fallback
    
    # Container utama dengan responsive design
    options_container = widgets.VBox([
        widgets.HTML(f"<h5 style='margin: 10px 0 8px 0; color: {get_safe_color('dark', '#333')};'>{get_safe_icon('download', '📥')} Dataset Configuration</h5>"),
        columns_container
    ], layout=widgets.Layout(
        padding='12px',
        width='100%',
        max_width='100%',
        border='1px solid #dee2e6',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'
    ))
    
    # Tambahkan referensi untuk akses mudah
    options_container.workspace_input = workspace_input
    options_container.project_input = project_input
    options_container.version_input = version_input
    options_container.api_key_input = api_key_input
    options_container.validate_checkbox = validate_checkbox
    options_container.backup_checkbox = backup_checkbox
    
    return options_container