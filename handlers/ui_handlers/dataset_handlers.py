"""
File: handlers/ui_handlers/dataset_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen dataset, menangani event listener dan logika bisnis.
"""

import os
import shutil
from pathlib import Path
from IPython.display import display, clear_output
import yaml

def setup_dataset_dirs(base_path):
    """
    Siapkan direktori dataset dengan struktur yang diperlukan.
    
    Args:
        base_path: Path dasar untuk direktori dataset
        
    Returns:
        List direktori yang dibuat
    """
    dirs = [
        f"{base_path}/data/train/images",
        f"{base_path}/data/train/labels",
        f"{base_path}/data/valid/images", 
        f"{base_path}/data/valid/labels",
        f"{base_path}/data/test/images",
        f"{base_path}/data/test/labels"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def check_dataset_status(storage_option, drive_path=None):
    """
    Cek status dataset di lokasi penyimpanan yang dipilih.
    
    Args:
        storage_option: Opsi penyimpanan ('local' atau 'drive')
        drive_path: Path ke Google Drive jika di-mount
        
    Returns:
        Dict berisi informasi status dataset
    """
    base_path = drive_path if storage_option == 'drive' else '.'
    
    train_dir = Path(f"{base_path}/data/train/images")
    valid_dir = Path(f"{base_path}/data/valid/images")
    test_dir = Path(f"{base_path}/data/test/images")
    
    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        return {
            'status': 'missing',
            'message': f"⚠️ Beberapa direktori dataset tidak ditemukan di {base_path}/data"
        }
    
    train_count = len(list(train_dir.glob('*.*')))
    valid_count = len(list(valid_dir.glob('*.*')))
    test_count = len(list(test_dir.glob('*.*')))
    total_count = train_count + valid_count + test_count
    
    if total_count == 0:
        return {
            'status': 'empty',
            'message': f"⚠️ Dataset kosong di {base_path}/data"
        }
    
    return {
        'status': 'ok',
        'message': f"✅ Dataset ditemukan di {base_path}/data dengan {total_count} gambar",
        'stats': {
            'train': train_count,
            'valid': valid_count,
            'test': test_count,
            'total': total_count
        }
    }

def on_storage_option_change(change, drive_path=None):
    """
    Handler untuk perubahan opsi penyimpanan.
    
    Args:
        change: Change event dari widget
        drive_path: Path ke Google Drive jika di-mount
    """
    if change['new'] == 'drive' and drive_path is None:
        print("⚠️ Google Drive tidak tersedia. Menggunakan penyimpanan lokal.")
        # Kembalikan nilai widget ke 'local', harus diimplementasikan dari luar
        return 'local'
    return change['new']

def on_download_button_clicked(ui_components, config, downloader_class, drive_path=None, logger=None):
    """
    Handler untuk tombol download dataset.
    
    Args:
        ui_components: Dictionary komponen UI dari create_dataset_ui()
        config: Dictionary konfigurasi aplikasi
        downloader_class: Class untuk download dataset
        drive_path: Path ke Google Drive jika di-mount
        logger: Logger untuk mencatat aktivitas
    """
    # Disable tombol saat proses berlangsung
    ui_components['download_button'].disabled = True
    ui_components['download_button'].description = "Sedang Mengunduh..."
    
    with ui_components['output']:
        clear_output()
        
        try:
            # Load konfigurasi
            config_path = 'configs/experiment_config.yaml'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open('configs/base_config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
            
            # Verifikasi konfigurasi minimum
            if not config.get('data', {}).get('roboflow', {}).get('api_key'):
                print("❌ API key tidak ditemukan dalam konfigurasi")
                print("💡 Tip: Gunakan tombol 'Simpan Konfigurasi' untuk menyimpan API key")
                ui_components['download_button'].disabled = False
                ui_components['download_button'].description = "Download Dataset"
                return
            
            # Update konfigurasi untuk menyimpan di Drive jika dipilih
            storage_option = ui_components['storage_options'].value
            if storage_option == 'drive' and drive_path:
                # Setup direktori dataset di Drive
                print(f"📂 Menyiapkan direktori dataset di Google Drive...")
                setup_dataset_dirs(drive_path)
                
                # Update path dalam konfigurasi
                if 'data' not in config:
                    config['data'] = {}
                if 'local' not in config['data']:
                    config['data']['local'] = {}
                
                config['data']['local']['train'] = f"{drive_path}/data/train"
                config['data']['local']['valid'] = f"{drive_path}/data/valid"
                config['data']['local']['test'] = f"{drive_path}/data/test"
                
                # Simpan konfigurasi yang diupdate
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"✅ Konfigurasi diperbarui untuk menggunakan Google Drive")
                
                # Buat instance RoboflowDownloader dengan konfigurasi yang diperbarui
                downloader = downloader_class(
                    config=config, 
                    output_dir=f"{drive_path}/data",
                    config_path=config_path
                )
                
                # Jalankan proses download
                results = downloader.run_download_pipeline(
                    force_download=ui_components['force_download_checkbox'].value
                )
            else:
                # Gunakan lokasi default dan downloader function
                from smartcash.utils.roboflow_downloader import download_dataset_from_roboflow
                results = download_dataset_from_roboflow(
                    config, 
                    force=ui_components['force_download_checkbox'].value
                )
            
            if results.get('success', False):
                # Tampilkan statistik akhir jika sukses
                stats = results.get('stats', {}).get('final', results.get('stats', {}).get('existing', {}))
                
                train_count = stats.get('train', 0)
                valid_count = stats.get('valid', 0) 
                test_count = stats.get('test', 0)
                total_count = train_count + valid_count + test_count
                
                print(f"\n✅ Dataset berhasil diunduh dan disiapkan")
                print(f"📊 Statistik dataset:")
                print(f"   • Total gambar: {total_count}")
                print(f"   • Training: {train_count} gambar")
                print(f"   • Validasi: {valid_count} gambar")
                print(f"   • Testing: {test_count} gambar")
                print(f"📁 Lokasi: {drive_path + '/data' if storage_option == 'drive' else './data'}")
                
                # Update tombol untuk menunjukkan selesai
                ui_components['download_button'].button_style = "success"
                ui_components['download_button'].description = "Download Selesai"
                ui_components['download_button'].icon = "check"
            else:
                print("\n⚠️ Proses download tidak berhasil diselesaikan")
                # Reset tombol
                ui_components['download_button'].disabled = False
                ui_components['download_button'].description = "Download Dataset"
                ui_components['download_button'].button_style = "danger"
                
        except Exception as e:
            print(f"❌ Error saat proses download: {str(e)}")
            # Reset tombol
            ui_components['download_button'].disabled = False
            ui_components['download_button'].description = "Download Dataset"
            ui_components['download_button'].button_style = "danger"
        finally:
            # Re-enable tombol setelah selesai
            ui_components['download_button'].disabled = False

def on_cleanup_button_clicked(ui_components):
    """
    Handler untuk tombol membersihkan file sementara.
    
    Args:
        ui_components: Dictionary komponen UI dari create_dataset_ui()
    """
    ui_components['cleanup_button'].disabled = True
    ui_components['cleanup_button'].description = "Sedang Membersihkan..."
    
    with ui_components['output']:
        clear_output()
        
        try:
            # Bersihkan file zip
            print("🧹 Mencari dan membersihkan file zip...")
            zip_found = False
            for zip_file in Path('.').glob('*.zip'):
                zip_found = True
                print(f"  → Menghapus file zip: {zip_file}")
                try:
                    zip_file.unlink()
                except Exception as e:
                    print(f"⚠️ Gagal menghapus file zip {zip_file}: {str(e)}")
            
            if not zip_found:
                print("✅ Tidak ditemukan file zip.")
            
            # Bersihkan direktori sementara hasil ekstraksi
            print("🧹 Mencari direktori hasil ekstraksi...")
            temp_found = False
            temp_dirs = [d for d in Path('.').glob('*') 
                        if d.is_dir() and (
                            d.name.startswith('rupiah') or 
                            d.name.endswith('-yolov5') or
                            d.name.endswith('_sample')
                        )]
            
            for temp_dir in temp_dirs:
                temp_found = True
                print(f"  → Menghapus direktori sementara: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"⚠️ Gagal menghapus direktori {temp_dir}: {str(e)}")
            
            if not temp_found:
                print("✅ Tidak ditemukan direktori sementara.")
            
            print("✨ Pembersihan selesai!")
            
        except Exception as e:
            print(f"❌ Error saat membersihkan: {str(e)}")
        finally:
            # Reset tombol
            ui_components['cleanup_button'].disabled = False
            ui_components['cleanup_button'].description = "Bersihkan File Sementara"

def on_check_status_button_clicked(ui_components, drive_path=None):
    """
    Handler untuk tombol mengecek status dataset.
    
    Args:
        ui_components: Dictionary komponen UI dari create_dataset_ui()
        drive_path: Path ke Google Drive jika di-mount
    """
    ui_components['check_status_button'].disabled = True
    ui_components['check_status_button'].description = "Sedang Memeriksa..."
    
    with ui_components['output']:
        clear_output()
        
        try:
            storage_option = ui_components['storage_options'].value
            result = check_dataset_status(storage_option, drive_path)
            print(result['message'])
            if result['status'] == 'ok':
                stats = result['stats']
                print(f"📊 Statistik dataset:")
                print(f"   • Total gambar: {stats['total']}")
                print(f"   • Training: {stats['train']} gambar")
                print(f"   • Validasi: {stats['valid']} gambar")
                print(f"   • Testing: {stats['test']} gambar")
        except Exception as e:
            print(f"❌ Error saat memeriksa status: {str(e)}")
        finally:
            ui_components['check_status_button'].disabled = False
            ui_components['check_status_button'].description = "Cek Status Dataset"

def setup_dataset_handlers(ui_components, config, downloader_class, drive_path=None, logger=None):
    """
    Setup semua event handler untuk UI dataset.
    
    Args:
        ui_components: Dictionary komponen UI dari create_dataset_ui()
        config: Dictionary konfigurasi aplikasi
        downloader_class: Class untuk download dataset
        drive_path: Path ke Google Drive jika di-mount
        logger: Logger untuk mencatat aktivitas
    """
    # Handler untuk storage options
    def handle_storage_change(change):
        new_value = on_storage_option_change(change, drive_path)
        if new_value != change['new']:
            ui_components['storage_options'].value = new_value
    
    ui_components['storage_options'].observe(handle_storage_change, names='value')
    
    # Handler untuk tombol download
    ui_components['download_button'].on_click(
        lambda b: on_download_button_clicked(ui_components, config, downloader_class, drive_path, logger)
    )
    
    # Handler untuk tombol cleanup
    ui_components['cleanup_button'].on_click(
        lambda b: on_cleanup_button_clicked(ui_components)
    )
    
    # Handler untuk tombol check status
    ui_components['check_status_button'].on_click(
        lambda b: on_check_status_button_clicked(ui_components, drive_path)
    )
    
    return ui_components