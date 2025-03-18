"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk setup download dataset dengan integrasi logging
"""

from typing import Dict, Any
import logging

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk download dataset dengan integrasi logging.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Setup logger yang terintegrasi dengan UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, UILogger
        
        # Buat atau dapatkan logger
        logger = setup_ipython_logging(
            ui_components, 
            logger_name="dataset_download", 
            log_level=logging.INFO
        )
        
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Komponen download dataset siap digunakan")
    except ImportError as e:
        print(f"⚠️ Tidak dapat mengintegrasikan logger: {str(e)}")
    
    try:
        # Setup handler untuk download option
        from smartcash.ui.dataset.download_ui_handler import setup_ui_handlers
        ui_components = setup_ui_handlers(ui_components, env, config)
        
        # Setup handler untuk download initialization
        from smartcash.ui.dataset.download_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika tersedia - harus dilakukan sebelum setup click handlers
        try:
            from smartcash.dataset.manager import DatasetManager
            
            if config:
                dataset_manager = DatasetManager(config=config, logger=ui_components.get('logger'))
                ui_components['dataset_manager'] = dataset_manager
                
                if 'logger' in ui_components:
                    ui_components['logger'].info("✅ Dataset Manager berhasil diinisialisasi")
        except ImportError as e:
            if 'logger' in ui_components:
                ui_components['logger'].warning(f"⚠️ Tidak dapat menggunakan DatasetManager: {str(e)}")
                ui_components['logger'].info("ℹ️ Beberapa fitur mungkin tidak tersedia")
        
        # Setup handler untuk click button download
        from smartcash.ui.dataset.download_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Validasi dataset jika sudah ada
        try:
            # Import handler konfirmasi untuk mendapatkan fungsi pengecekan dataset
            from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
            
            # Dapatkan direktori data
            data_dir = config.get('data', {}).get('dir', 'data')
            if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
                data_dir = str(env.drive_path / 'data')
                
            # Cek dataset yang sudah ada
            if check_existing_dataset(data_dir):
                stats = get_dataset_stats(data_dir)
                
                if 'logger' in ui_components:
                    ui_components['logger'].info(f"📊 Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})")
                    
                # Jalankan validasi struktur jika memungkinkan
                if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
                    ui_components['validate_dataset_structure'](data_dir)
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].warning(f"⚠️ Gagal memeriksa dataset yang ada: {str(e)}")
    
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error saat setup handlers: {str(e)}")
        else:
            print(f"❌ Error saat setup handlers: {str(e)}")
    
    return ui_components