"""
File: smartcash/utils/cell_header_utils.py
Author: Refactored
Deskripsi: Utilitas untuk setup standard cell header, menyederhanakan kode boilerplate di notebook cells.
"""

import sys
import os
import atexit
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, Tuple

def setup_notebook_environment(
    cell_name: str,
    config_path: str = "configs/base_config.yaml",
    create_dirs: Optional[List[str]] = None,
    register_cleanup: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Setup environment standar untuk cell notebook:
    - Menambahkan smartcash ke path
    - Membuat direktori yang diperlukan
    - Setting up logger
    - Loading dan sinkronisasi konfigurasi
    - Setup dan cleanup observer manager
    
    Args:
        cell_name: Nama unik untuk cell (untuk logging dan observer group)
        config_path: Path ke file konfigurasi
        create_dirs: List direktori yang perlu dibuat
        register_cleanup: Register cleanup handler otomatis
    
    Returns:
        Tuple (environment components, config)
    """
    # Pastikan smartcash ada di path
    if not any('smartcash' in p for p in sys.path):
        sys.path.append('.')
    
    # Buat direktori standar
    os.makedirs("configs", exist_ok=True)
    os.makedirs("smartcash/ui_components", exist_ok=True)
    os.makedirs("smartcash/ui_handlers", exist_ok=True)
    
    # Buat direktori tambahan jika diperlukan
    if create_dirs:
        for directory in create_dirs:
            os.makedirs(directory, exist_ok=True)
    
    # Setup komponen
    env = {}
    config = {}
    
    # Setup logging, config dan observer
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        # Setup logger
        logger = get_logger(cell_name)
        env['logger'] = logger
        
        # Setup config manager
        config_manager = get_config_manager(logger)
        env['config_manager'] = config_manager
        
        # Coba muat config dari Google Drive jika di Colab
        if config_manager.is_colab and config_manager.drive_mounted:
            config = config_manager.load_from_drive()
        
        # Jika belum ada konfigurasi, muat dari file lokal
        if not config:
            config = config_manager.load_config(config_path, logger=logger)
        
        # Setup observer manager
        observer_manager = ObserverManager(auto_register=True)
        env['observer_manager'] = observer_manager
        
        # Buat observer group berdasarkan nama cell
        observer_group = f"{cell_name}_observers"
        
        # Bersihkan observer dari sesi sebelumnya
        observer_manager.unregister_group(observer_group)
        
        # Setup cleanup
        if register_cleanup:
            def cleanup():
                try:
                    observer_manager.unregister_group(observer_group)
                    logger.info(f"✅ Observer {observer_group} berhasil dibersihkan")
                except Exception as e:
                    logger.error(f"❌ Error saat cleanup: {str(e)}")
            
            atexit.register(cleanup)
            env['cleanup'] = cleanup
        
    except ImportError as e:
        print(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}")
    
    return env, config

def setup_ui_component(
    env: Dict[str, Any],
    config: Dict[str, Any],
    component_name: str,
    register_cleanup: bool = True
) -> Dict[str, Any]:
    """
    Setup komponen UI dengan handler-nya.
    
    Args:
        env: Dict environment components
        config: Dict konfigurasi
        component_name: Nama komponen (sesuai nama file)
        register_cleanup: Register cleanup handler dari UI
    
    Returns:
        Dict komponen UI
    """
    try:
        # Dynamic import komponen dan handler
        ui_module = __import__(f"smartcash.ui_components.{component_name}", fromlist=['create_*'])
        handler_module = __import__(f"smartcash.ui_handlers.{component_name}", fromlist=['setup_*'])
        
        # Cari fungsi create_*_ui
        create_func = None
        for item in dir(ui_module):
            if item.startswith('create_') and item.endswith('_ui'):
                create_func = getattr(ui_module, item)
                break
        
        # Jika tidak ditemukan, coba dengan nama variasi lain
        if not create_func:
            create_func = getattr(ui_module, f"create_{component_name}_ui", None)
        
        # Cari fungsi setup_*_handlers
        setup_func = None
        for item in dir(handler_module):
            if item.startswith('setup_') and item.endswith('_handlers'):
                setup_func = getattr(handler_module, item)
                break
        
        # Jika tidak ditemukan, coba dengan nama variasi lain
        if not setup_func:
            setup_func = getattr(handler_module, f"setup_{component_name}_handlers", None)
        
        # Validasi fungsi yang ditemukan
        if not create_func or not setup_func:
            raise ImportError(f"Fungsi create/setup tidak ditemukan untuk {component_name}")
        
        # Buat dan setup UI
        ui_components = create_func()
        ui_components = setup_func(ui_components, config)
        
        # Register cleanup jika ada dan diminta
        if register_cleanup and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            atexit.register(ui_components['cleanup'])
        
        return ui_components
        
    except Exception as e:
        print(f"❌ Error saat setup komponen {component_name}: {str(e)}")
        
        # Fallback UI kosong
        import ipywidgets as widgets
        return {
            'ui': widgets.HTML(f"<h3>⚠️ Komponen {component_name} tidak tersedia</h3><p>Error: {str(e)}</p>")
        }

def display_ui(ui_components):
    """
    Tampilkan UI components dengan penanganan error.
    
    Args:
        ui_components: Dict hasil dari setup_ui_component
    """
    try:
        from IPython.display import display
        
        if 'ui' in ui_components:
            display(ui_components['ui'])
        else:
            import ipywidgets as widgets
            display(widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3>"))
            
    except Exception as e:
        print(f"❌ Error saat menampilkan UI: {str(e)}")