"""
File: smartcash/ui/dataset/download/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi dan hasil download dataset
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

def handle_save_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol save pada UI download.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Simpan konfigurasi dan hasil
        save_config_and_results(ui_components)
        
        # Log save berhasil
        from smartcash.ui.utils.ui_logger import log_to_ui
        log_to_ui(ui_components, "Konfigurasi dan hasil download berhasil disimpan", "success", "💾")
        if logger: logger.info("💾 Konfigurasi dan hasil download berhasil disimpan")
        
        # Update status panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel'](ui_components, "success", "Konfigurasi dan hasil download berhasil disimpan")
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.ui_logger import log_to_ui
        error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
        log_to_ui(ui_components, error_msg, "error", "❌")
        if logger: logger.error(f"❌ {error_msg}")

def save_config_and_results(ui_components: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Simpan konfigurasi dan hasil download ke file JSON dan sinkronisasi dengan Google Drive jika tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        output_path: Path untuk menyimpan file (opsional)
        
    Returns:
        Path file yang disimpan
    """
    logger = ui_components.get('logger')
    
    # Dapatkan nilai dari komponen UI
    config = {
        'workspace': ui_components.get('rf_workspace', {}).value if 'rf_workspace' in ui_components else '',
        'project': ui_components.get('rf_project', {}).value if 'rf_project' in ui_components else '',
        'version': ui_components.get('rf_version', {}).value if 'rf_version' in ui_components else '',
        'output_dir': ui_components.get('output_dir', {}).value if 'output_dir' in ui_components else 'data',
        'validate_dataset': ui_components.get('validate_dataset', {}).value if 'validate_dataset' in ui_components else True,
        'timestamp': ui_components.get('download_timestamp', '') or datetime.now().isoformat()
    }
    
    # Tambahkan statistik dataset jika tersedia
    if 'dataset_stats' in ui_components:
        config['dataset_stats'] = ui_components['dataset_stats']
    
    # Tentukan path output
    if not output_path:
        # Gunakan direktori output dari UI atau default ke 'data'
        base_dir = config['output_dir'] or 'data'
        # Buat direktori jika belum ada
        os.makedirs(base_dir, exist_ok=True)
        # Path file konfigurasi
        output_path = os.path.join(base_dir, 'download_config.json')
    
    # Simpan ke file JSON
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if logger: logger.info(f"💾 Konfigurasi disimpan ke {output_path}")
    
    # Sinkronisasi dengan Google Drive jika dalam environment Colab
    try:
        # Cek apakah kita berada di Google Colab
        from google.colab import drive
        
        # Coba mount Google Drive jika belum di-mount
        try:
            drive_mounted = os.path.exists('/content/drive/MyDrive')
            if not drive_mounted:
                if logger: logger.info("🔄 Mounting Google Drive...")
                drive.mount('/content/drive')
                if logger: logger.info("✅ Google Drive berhasil di-mount")
        except Exception as e:
            if logger: logger.warning(f"⚠️ Gagal mount Google Drive: {str(e)}")
            return output_path
        
        # Tentukan path di Google Drive
        drive_dir = f"/content/drive/MyDrive/SmartCash/{config['project']}_v{config['version']}"
        os.makedirs(drive_dir, exist_ok=True)
        drive_config_path = os.path.join(drive_dir, 'download_config.json')
        
        # Salin konfigurasi ke Google Drive
        import shutil
        shutil.copy2(output_path, drive_config_path)
        
        # Salin dataset ke Google Drive jika ukurannya tidak terlalu besar
        dataset_dir = config['output_dir']
        if os.path.exists(dataset_dir):
            # Hitung ukuran dataset
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dataset_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            # Konversi ke MB
            size_mb = total_size / (1024 * 1024)
            
            # Jika ukuran < 500MB, salin ke Drive
            if size_mb < 500:
                if logger: logger.info(f"🔄 Menyalin dataset ({size_mb:.2f} MB) ke Google Drive...")
                
                # Buat direktori di Drive
                drive_dataset_dir = os.path.join(drive_dir, os.path.basename(dataset_dir))
                
                # Salin dataset
                if os.path.exists(drive_dataset_dir):
                    shutil.rmtree(drive_dataset_dir)
                shutil.copytree(dataset_dir, drive_dataset_dir)
                
                if logger: logger.info(f"✅ Dataset berhasil disalin ke {drive_dataset_dir}")
            else:
                if logger: logger.warning(f"⚠️ Dataset terlalu besar ({size_mb:.2f} MB) untuk disalin ke Google Drive")
        
        if logger: logger.info(f"✅ Konfigurasi berhasil disinkronkan ke Google Drive: {drive_config_path}")
        return drive_config_path
    except ImportError:
        # Bukan di Google Colab
        if logger: logger.debug("ℹ️ Bukan di Google Colab, tidak perlu sinkronisasi dengan Google Drive")
    except Exception as e:
        # Error lainnya
        if logger: logger.warning(f"⚠️ Gagal sinkronisasi dengan Google Drive: {str(e)}")
    
    return output_path

def load_saved_config(config_path: str) -> Dict[str, Any]:
    """
    Muat konfigurasi yang tersimpan dari file JSON.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def apply_saved_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Terapkan konfigurasi yang dimuat ke UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang dimuat
    """
    # Terapkan nilai ke komponen UI
    if 'rf_workspace' in ui_components and 'workspace' in config:
        ui_components['rf_workspace'].value = config['workspace']
    
    if 'rf_project' in ui_components and 'project' in config:
        ui_components['rf_project'].value = config['project']
    
    if 'rf_version' in ui_components and 'version' in config:
        ui_components['rf_version'].value = config['version']
    
    if 'output_dir' in ui_components and 'output_dir' in config:
        ui_components['output_dir'].value = config['output_dir']
    
    if 'validate_dataset' in ui_components and 'validate_dataset' in config:
        ui_components['validate_dataset'].value = config['validate_dataset']
    
    # Simpan timestamp download jika ada
    if 'timestamp' in config:
        ui_components['download_timestamp'] = config['timestamp']
    
    # Simpan statistik dataset jika ada
    if 'dataset_stats' in config:
        ui_components['dataset_stats'] = config['dataset_stats']
