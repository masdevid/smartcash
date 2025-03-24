"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk download dataset SmartCash dengan integrasi logger dan observer yang lebih konsisten
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk download dataset dengan integrasi logging yang terintegrasi."""
    # Setup logging terintegrasi UI dengan utils standar
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "dataset_download")
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Download dataset handler diinisialisasi")
    except ImportError:
        pass
    
    # Setup observer handlers dengan komponen standar
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        ui_components = setup_observer_handlers(ui_components, "dataset_download_observers")
        if logger: logger.info(f"{ICONS['success']} Observer handlers berhasil diinisialisasi")
    except ImportError as e:
        if logger: logger.warning(f"{ICONS['warning']} Observer handlers tidak tersedia: {str(e)}")
    
    # Initialize progress handler dengan komponen standar dari handlers
    try:
        from smartcash.ui.handlers.download_progress_handler import DownloadProgressHandler
        ui_components['progress_handler'] = DownloadProgressHandler(ui_components)
        if logger: logger.info(f"{ICONS['success']} Progress handler berhasil diinisialisasi")
    except ImportError as e:
        if logger: logger.warning(f"{ICONS['warning']} Progress handler tidak tersedia: {str(e)}")
    
    # Initialization handler dengan fungsi standar
    try:
        from smartcash.ui.dataset.download_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika tidak ada dengan validasi fallback menggunakan utils
        if 'dataset_manager' not in ui_components:
            try:
                from smartcash.ui.utils.fallback_utils import get_dataset_manager
                ui_components['dataset_manager'] = get_dataset_manager(config, logger)
                if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
            except ImportError as e:
                if logger: logger.warning(f"{ICONS['warning']} DatasetManager tidak tersedia: {str(e)}")
        
        # Setup handlers UI dengan error handling standar
        try:
            from smartcash.ui.handlers.error_handler import try_except_decorator
            from smartcash.ui.dataset.download_ui_handler import setup_ui_handlers
            
            # Gunakan decorator untuk penanganan error yang terintegrasi
            @try_except_decorator(ui_components.get('status'))
            def setup_ui_handlers_safe():
                return setup_ui_handlers(ui_components, env, config)
                
            ui_components = setup_ui_handlers_safe() or ui_components
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error setup UI handlers: {str(e)}")
            
        # Setup click handlers dengan error handling standar
        try:
            from smartcash.ui.dataset.download_click_handler import setup_click_handlers
            ui_components = setup_click_handlers(ui_components, env, config)
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error setup click handlers: {str(e)}")
        
        # Cek dataset yang sudah ada dengan utilitas validasi menggunakan utils
        try:
            from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
            
            # Setup direktori data dengan logic yang terstandarisasi
            from smartcash.ui.utils.drive_utils import detect_drive_mount
            drive_mounted, drive_path = detect_drive_mount()
            
            data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
            if drive_mounted and drive_path:
                data_dir = f"{drive_path}/data"
                
            # Periksa dataset yang sudah ada dan update UI    
            if check_existing_dataset(data_dir):
                stats = get_dataset_stats(data_dir)
                if logger: logger.info(f"{ICONS['folder']} Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})")
                
                # Validasi struktur jika fungsi tersedia
                if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
                    ui_components['validate_dataset_structure'](data_dir)
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error cek dataset: {str(e)}")
                
    except Exception as e:
        # Gunakan fungsi standar dari error handler
        from smartcash.ui.handlers.error_handler import handle_ui_error
        handle_ui_error(e, ui_components.get('status', None), True, f"{ICONS['error']} Error saat setup handlers")
    
    # Tambahkan fungsi cleanup dengan standar
    def cleanup_resources():
        """Cleanup resources yang digunakan oleh handler."""
        # Cleanup observers jika ada
        if 'observer_group' in ui_components:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                observer_manager = ObserverManager()
                observer_manager.unregister_group(ui_components['observer_group'])
                if logger: logger.info(f"{ICONS['cleanup']} Observer handlers dibersihkan")
            except ImportError:
                pass
    
    ui_components['cleanup'] = cleanup_resources
    
    return ui_components