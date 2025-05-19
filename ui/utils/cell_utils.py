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
        from smartcash.common.utils import is_colab
        from pathlib import Path
        
        # Determine base directory
        if is_colab():
            base_dir = Path("/content")
        else:
            base_dir = Path(os.getcwd())
            
        # Ensure config directory exists
        config_dir = base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment manager with base directory
        env = get_environment_manager(base_dir=str(base_dir))
        
        # Import config manager dan load configuration
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager(base_dir=str(base_dir), config_file=str(config_dir / "base_config.yaml"))
        
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
                
        return env, config
        
    except Exception as e:
        print(f"❌ Error saat setup environment: {str(e)}")
        return None, {}

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