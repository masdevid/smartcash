"""
File: smartcash/ui/dataset/download/components/form_fields.py
Deskripsi: Updated form fields dengan default path yang benar dan konsisten dengan struktur baru
"""

import ipywidgets as widgets
import os
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment

def api_key_field():
    """API key field dengan comprehensive detection."""
    # Deteksi API key dari semua sumber
    api_key = _detect_api_key_comprehensive()
    
    field = widgets.Password(
        value=api_key,
        placeholder='Masukkan API Key Roboflow atau biarkan kosong jika sudah di environment',
        description='API Key:',
        layout=widgets.Layout(width='100%')
    )
    
    # Log status API key untuk debugging
    if api_key:
        print(f"🔑 API key ditemukan: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    else:
        print("⚠️ API key tidak ditemukan. Silakan isi manual atau set di Google Secrets.")
    
    return field

def _detect_api_key_comprehensive() -> str:
    """Deteksi API key dari semua sumber yang tersedia dengan logging."""
    print("🔍 Mencari API key...")
    
    # 1. Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        print("✅ API key ditemukan dari environment variable")
        return api_key
    else:
        print("❌ API key tidak ditemukan di environment variable")
    
    # 2. Google Colab userdata (Google Secrets)
    try:
        from google.colab import userdata
        print("🔍 Checking Google Colab secrets...")
        
        # Check primary key
        api_key = userdata.get('ROBOFLOW_API_KEY')
        if api_key:
            print("✅ API key ditemukan dari Google Colab secrets (ROBOFLOW_API_KEY)")
            return api_key
        
        # Check alternative names
        alternative_keys = ['roboflow_api_key', 'ROBOFLOW_KEY', 'roboflow_key', 'API_KEY']
        for key_name in alternative_keys:
            try:
                api_key = userdata.get(key_name)
                if api_key:
                    print(f"✅ API key ditemukan dari Google Colab secrets ({key_name})")
                    return api_key
            except Exception:
                continue
                
        print("❌ API key tidak ditemukan di Google Colab secrets")
        
    except ImportError:
        print("ℹ️ Tidak di lingkungan Google Colab")
    except Exception as e:
        print(f"⚠️ Error accessing Google Colab secrets: {str(e)}")
    
    print("❌ API key tidak ditemukan dari semua sumber")
    return ''

def output_dir_field(config):
    """Output directory field dengan default ke downloads folder yang benar."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    print(f"🌍 Environment detection:")
    print(f"   • is_colab: {env_manager.is_colab}")
    print(f"   • is_drive_mounted: {env_manager.is_drive_mounted}")
    print(f"   • downloads_path: {paths['downloads']}")
    
    # Gunakan downloads path sebagai default
    default_value = paths['downloads']
    print(f"📁 Output directory default: {default_value}")
    
    return widgets.Text(
        value=default_value,
        placeholder='Path download sementara dataset (akan dipindah ke struktur final)',
        description='Download Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
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
        print(f"💾 Backup akan disimpan di: {value}")
    
    return widgets.Text(
        value=value,
        placeholder='Path backup dataset',
        description='Backup Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
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
        value=False,  # Default false untuk proses yang lebih cepat
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
        disabled=True,  # Always enabled untuk proses yang konsisten
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