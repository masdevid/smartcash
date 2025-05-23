"""
File: smartcash/ui/dataset/download/components/form_fields.py
Deskripsi: Fixed form fields dengan API key detection yang akurat dan Drive path defaults
"""

import ipywidgets as widgets
import os
from smartcash.common.environment import get_environment_manager

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
    """Output directory field dengan Drive detection yang akurat."""
    env_manager = get_environment_manager()
    
    print(f"🌍 Environment detection:")
    print(f"   • is_colab: {env_manager.is_colab}")
    print(f"   • is_drive_mounted: {env_manager.is_drive_mounted}")
    print(f"   • drive_path: {env_manager.drive_path}")
    
    # Tentukan default path berdasarkan environment
    if env_manager.is_colab and env_manager.is_drive_mounted:
        default_value = str(env_manager.drive_path / 'downloads')
        print(f"✅ Menggunakan Drive path: {default_value}")
    else:
        default_value = config.get('dir', 'data')
        print(f"📁 Menggunakan local path: {default_value}")
    
    return widgets.Text(
        value=default_value,
        placeholder='Path penyimpanan dataset',
        description='Output Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )

def backup_dir_field(default_path=None):
    """Backup directory field dengan Drive detection yang akurat."""
    env_manager = get_environment_manager()
    
    if default_path:
        value = default_path
    elif env_manager.is_colab and env_manager.is_drive_mounted:
        value = str(env_manager.drive_path / 'backups')
        print(f"✅ Backup akan disimpan di Drive: {value}")
    else:
        value = 'data/backup'
        print(f"📁 Backup akan disimpan lokal: {value}")
    
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
        description='Validasi dataset setelah download',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )

def backup_checkbox_field():
    """Checkbox untuk backup dataset."""
    return widgets.Checkbox(
        value=True,
        description='Backup dataset sebelum menghapus',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )