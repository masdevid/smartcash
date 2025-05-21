"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment
"""

from pathlib import Path
from IPython.display import display, HTML
import sys
import os

from smartcash.common.config import get_config_manager
from smartcash.common.utils import is_colab
from smartcash.ui.setup.env_config.components.manager_setup import setup_managers

def initialize_env_config_ui():
    """
    Inisialisasi konfigurasi environment
    
    Returns:
        ConfigManager instance
    """
    # Setup config managers dan direktori
    config_manager, base_dir, config_dir = setup_managers()
    
    # Tampilkan header dengan styling
    display(HTML("""
    <div style="background-color: #f8f9fa; padding: 10px; border-left: 5px solid #28a745; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #28a745;">SmartCash Environment Configuration</h3>
        <p style="margin: 5px 0 0 0; color: #666;">Environment setup completed successfully</p>
    </div>
    """))
    
    # Tampilkan informasi konfigurasi
    print(f"✅ Environment berhasil dikonfigurasi:")
    print(f"   📁 Base directory: {base_dir}")
    print(f"   📁 Config directory: {config_dir}")
    
    # Cek apakah config_dir adalah symlink
    if config_dir.is_symlink():
        target = Path(config_dir).resolve()
        print(f"   🔗 Config directory adalah symlink ke: {target}")
    
    # Tampilkan informasi environment sistem
    print(f"\n📊 Informasi Environment:")
    print(f"   🐍 Python version: {sys.version.split()[0]}")
    
    # Cek apakah sedang berjalan di Colab
    colab_status = "Ya" if is_colab() else "Tidak"
    print(f"   💻 Running di Google Colab: {colab_status}")
    
    # Jika di Colab, tampilkan informasi konfigurasi Colab
    if is_colab():
        try:
            # Ambil konfigurasi Colab
            colab_config = config_manager.get_config('colab')
            
            if colab_config:
                # Tampilkan informasi drive
                if 'drive' in colab_config:
                    drive_config = colab_config['drive']
                    print(f"\n🗄️ Pengaturan Google Drive:")
                    print(f"   - Sinkronisasi aktif: {drive_config.get('use_drive', False)}")
                    print(f"   - Strategi sinkronisasi: {drive_config.get('sync_strategy', 'none')}")
                    print(f"   - Gunakan symlinks: {drive_config.get('symlinks', False)}")
                    
                    # Tampilkan paths jika ada
                    if 'paths' in drive_config:
                        paths = drive_config['paths']
                        print(f"   - SmartCash dir: {paths.get('smartcash_dir', 'SmartCash')}")
                        print(f"   - Configs dir: {paths.get('configs_dir', 'configs')}")
                
                # Tampilkan informasi model jika menggunakan GPU/TPU
                if 'model' in colab_config:
                    model_config = colab_config['model']
                    print(f"\n⚡ Pengaturan Hardware:")
                    print(f"   - Gunakan GPU: {model_config.get('use_gpu', False)}")
                    print(f"   - Gunakan TPU: {model_config.get('use_tpu', False)}")
                    print(f"   - Precision: {model_config.get('precision', 'float32')}")
                
                # Informasi performa
                if 'performance' in colab_config:
                    perf_config = colab_config['performance']
                    print(f"\n🚀 Pengaturan Performa:")
                    print(f"   - Auto garbage collect: {perf_config.get('auto_garbage_collect', False)}")
                    print(f"   - Simpan checkpoint ke Drive: {perf_config.get('checkpoint_to_drive', False)}")
        except Exception as e:
            print(f"⚠️ Gagal memuat konfigurasi Colab: {str(e)}")
    
    # Tampilkan konfigurasi file yang tersedia
    try:
        available_configs = config_manager.get_available_configs()
        print(f"\n📝 File Konfigurasi Tersedia:")
        for config in available_configs:
            print(f"   - {config}")
    except Exception as e:
        print(f"⚠️ Gagal mendapatkan daftar konfigurasi: {str(e)}")
    
    return config_manager
