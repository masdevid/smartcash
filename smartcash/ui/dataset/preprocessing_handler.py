"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator utama untuk handler preprocessing dataset tanpa ThreadPool
"""

from typing import Dict, Any
import logging
from smartcash.ui.utils.constants import ICONS

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk preprocessing dataset tanpa ThreadPool.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "preprocessing", log_level=logging.INFO)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Komponen preprocessing dataset siap digunakan")
    except ImportError:
        pass
    
    # Setup handlers komponen secara berurutan
    try:
        # Import dan setup semua handler yang diperlukan
        from smartcash.ui.dataset.preprocessing_initialization import setup_initialization
        from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
        from smartcash.ui.dataset.preprocessing_cleanup_handler import setup_cleanup_handler
        from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
        
        # Setup inisialisasi
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika belum ada
        if 'dataset_manager' not in ui_components and config:
            from smartcash.dataset.manager import DatasetManager
            ui_components['dataset_manager'] = DatasetManager(config=config, logger=logger)
            if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
        
        # Setup progress handler
        ui_components = setup_progress_handler(ui_components, env, config)
        
        # Setup click handler
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup cleanup handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Fungsi ringkas untuk update summary hasil preprocessing
        def update_summary(result):
            if not result or 'summary_container' not in ui_components: return
            
            from IPython.display import display, clear_output, HTML
            with ui_components['summary_container']:
                clear_output(wait=True)
                ui_components['summary_container'].layout.display = 'block'
                
                # Buat ringkasan dalam format HTML
                html = f"""
                <div style="padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
                    <h3 style="margin-top:0">{ICONS.get('stats', '📊')} Hasil Preprocessing</h3>
                    <ul>
                        <li><b>Jumlah Gambar:</b> {result.get('total_images', 0)}</li>
                        <li><b>Waktu Proses:</b> {result.get('processing_time', 0):.2f} detik</li>
                        <li><b>Resolusi:</b> {result.get('image_size', [0, 0])[0]}x{result.get('image_size', [0, 0])[1]}</li>
                        <li><b>Path Output:</b> {result.get('output_dir', '')}</li>
                    </ul>
                </div>
                """
                display(HTML(html))
        
        # Tambahkan fungsi update summary ke komponen
        ui_components['update_summary'] = update_summary
    
    except ImportError as e:
        if logger: logger.warning(f"{ICONS['warning']} Modul tidak tersedia: {str(e)}")
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
    
    return ui_components