import os
import shutil
from pathlib import Path
from IPython.display import display, HTML, clear_output

def setup_dataset_handlers(ui_components, roboflow_downloader=None, drive_path=None, logger=None):
    """Setup handlers for dataset UI components.
    
    Args:
        ui_components: Dictionary of UI components from create_dataset_ui()
        roboflow_downloader: Instance of RoboflowDownloader class
        drive_path: Path to Google Drive if mounted
        logger: Instance of logger for logging messages
    """
    # Extract UI components
    storage_options = ui_components['storage_options']
    force_download_checkbox = ui_components['force_download_checkbox']
    download_button = ui_components['download_button']
    cleanup_button = ui_components['cleanup_button']
    check_status_button = ui_components['check_status_button']
    output_area = ui_components['output_area']
    
    # Define handlers
    def on_storage_option_change(change):
        """Handler for storage option change."""
        if change['new'] == 'drive' and drive_path is None:
            with output_area:
                clear_output()
                print("⚠️ Google Drive tidak tersedia. Menggunakan penyimpanan lokal.")
            storage_options.value = 'local'
    
    def on_download_button_clicked(b):
        """Handler for download button click."""
        # Disable button during processing
        download_button.disabled = True
        download_button.description = "Sedang Mengunduh..."
        
        with output_area:
            clear_output()
            
            try:
                # Load configuration
                config_path = 'configs/experiment_config.yaml'
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    with open('configs/base_config.yaml', 'r') as f:
                        config = yaml.safe_load(f)
                
                # Check for API key
                if not config.get('data', {}).get('roboflow', {}).get('api_key'):
                    print("❌ API key tidak ditemukan dalam konfigurasi")
                    print("💡 Tip: Gunakan tombol 'Simpan Konfigurasi' untuk menyimpan API key")
                    download_button.disabled = False
                    download_button.description = "Download Dataset"
                    return
                
                # Update config based on selected storage option
                storage_option = storage_options.value
                if storage_option == 'drive' and drive_path:
                    # Setup dataset directories in Drive
                    setup_dirs = setup_dataset_dirs(drive_path)
                    
                    # Update paths in configuration
                    if 'data' not in config:
                        config['data'] = {}
                    if 'local' not in config['data']:
                        config['data']['local'] = {}
                    
                    config['data']['local']['train'] = f"{drive_path}/data/train"
                    config['data']['local']['valid'] = f"{drive_path}/data/valid"
                    config['data']['local']['test'] = f"{drive_path}/data/test"
                    
                    # Save updated configuration
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    print(f"✅ Konfigurasi diperbarui untuk menggunakan Google Drive")
                    
                    # Create RoboflowDownloader instance with updated config
                    downloader = roboflow_downloader(
                        config=config, 
                        output_dir=f"{drive_path}/data",
                        config_path=config_path
                    )
                    
                    # Run download pipeline
                    results = downloader.run_download_pipeline(force_download=force_download_checkbox.value)
                else:
                    # Use default location
                    results = roboflow_downloader(config, force=force_download_checkbox.value)
                
                # Display results
                if results.get('success', False):
                    # Show final statistics
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
                    
                    # Update button to show success
                    download_button.button_style = "success"
                    download_button.description = "Download Selesai"
                    download_button.icon = "check"
                else:
                    print("\n⚠️ Proses download tidak berhasil diselesaikan")
                    # Reset button
                    download_button.disabled = False
                    download_button.description = "Download Dataset"
                    download_button.button_style = "danger"
                    
            except Exception as e:
                print(f"❌ Error saat proses download: {str(e)}")
                # Reset button
                download_button.disabled = False
                download_button.description = "Download Dataset"
                download_button.button_style = "danger"
            finally:
                # Re-enable button after completion
                download_button.disabled = False
    
    def on_cleanup_button_clicked(b):
        """Handler for cleanup button click."""
        cleanup_button.disabled = True
        cleanup_button.description = "Sedang Membersihkan..."
        
        with output_area:
            clear_output()
            
            try:
                # Clean up zip files
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
                
                # Clean up temporary extraction directories
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
                # Reset button
                cleanup_button.disabled = False
                cleanup_button.description = "Bersihkan File Sementara"
    
    def check_dataset_status():
        """Check the status of the dataset in the selected storage location."""
        storage_option = storage_options.value
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
    
    def on_check_status_button_clicked(b):
        """Handler for check status button click."""
        check_status_button.disabled = True
        check_status_button.description = "Sedang Memeriksa..."
        
        with output_area:
            clear_output()
            
            try:
                result = check_dataset_status()
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
                check_status_button.disabled = False
                check_status_button.description = "Cek Status Dataset"
    
    def setup_dataset_dirs(base_path):
        """Setup necessary dataset directories."""
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
    
    # Attach handlers to UI components
    storage_options.observe(on_storage_option_change, names='value')
    download_button.on_click(on_download_button_clicked)
    cleanup_button.on_click(on_cleanup_button_clicked)
    check_status_button.on_click(on_check_status_button_clicked)