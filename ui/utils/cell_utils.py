"""
File: smartcash/ui/utils/cell_utils.py
Deskripsi: Utilitas terpadu untuk setup dan konfigurasi sel notebook
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def setup_notebook_environment(cell_name: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Setup environment dan konfigurasi untuk sel notebook.
    
    Args:
        cell_name: Nama sel/modul
        
    Returns:
        Tuple (environment_manager, config)
    """
    # Inisialisasi default
    env, config = None, {}
    
    try:
        # Import environment manager
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        
        # Import config manager dan load configuration
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        
        # Coba load konfigurasi dari file
        try:
            # Load konfigurasi base terlebih dahulu
            if Path('configs/base_config.yaml').exists():
                config = config_manager.load_config('configs/base_config.yaml')
            
            # Coba load konfigurasi spesifik cell jika ada
            cell_config_file = f"configs/{cell_name.lower()}_config.yaml"
            if Path(cell_config_file).exists():
                config = config_manager.merge_config(cell_config_file)
        except Exception as e:
            print(f"⚠️ Error saat load konfigurasi: {str(e)}")
            
            # Coba fallback ke default config
            try:
                from smartcash.common.default_config import ensure_base_config_exists
                if ensure_base_config_exists():
                    config = config_manager.load_config('configs/base_config.yaml')
            except Exception:
                pass
    except ImportError as e:
        print(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
        # Fallback untuk deteksi environment dasar
        try:
            from smartcash.common.utils import is_colab
            class SimpleEnv:
                def __init__(self):
                    self.is_colab = is_colab()
                    self.is_drive_mounted = Path('/content/drive/MyDrive').exists() if self.is_colab else False
            env = SimpleEnv()
        except ImportError:
            # Ultra fallback
            class UltraSimpleEnv:
                def __init__(self):
                    self.is_colab = 'google.colab' in sys.modules
                    self.is_drive_mounted = Path('/content/drive/MyDrive').exists() if self.is_colab else False
            env = UltraSimpleEnv()
    
    # Inisialisasi drive sync jika perlu
    if env and hasattr(env, 'is_colab') and env.is_colab:
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_configs
            initialize_configs()
        except ImportError:
            pass
    
    return env, config

def register_resource(ui_components: Dict[str, Any], resource: Any, 
                     cleanup_func: Optional[callable] = None) -> None:
    """
    Register resource untuk auto-cleanup.
    
    Args:
        ui_components: Dictionary UI components
        resource: Resource yang perlu di-cleanup
        cleanup_func: Fungsi cleanup (opsional, default: resource.close())
    """
    if 'resources' not in ui_components:
        ui_components['resources'] = []
    
    ui_components['resources'].append((resource, cleanup_func))

def setup_drive_sync(ui_components: Dict[str, Any]) -> None:
    """
    Setup sinkronisasi drive jika diperlukan.
    
    Args:
        ui_components: Dictionary UI components
    """
    logger = ui_components.get('logger')
    
    try:
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        
        if env.is_colab and env.is_drive_mounted:
            # Coba sync drive
            try:
                from smartcash.ui.utils.drive_utils import sync_drive_to_local
                from smartcash.common.config import get_config_manager
                
                config = get_config_manager().config
                result = sync_drive_to_local(config, env, logger)
                
                if logger:
                    if result.get("status") == "success":
                        logger.info(f"✅ Sinkronisasi Drive berhasil: {result.get('drive_to_local', 0)} file disalin")
                    else:
                        logger.warning(f"⚠️ Sinkronisasi Drive: {result.get('reason', 'unknown')}")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error saat sinkronisasi drive: {str(e)}")
    except ImportError:
        pass