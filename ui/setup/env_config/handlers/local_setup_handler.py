"""
File: smartcash/ui/setup/env_config/handlers/local_setup_handler.py
Deskripsi: Handler untuk setup environment di lingkungan lokal (non-Colab)
"""

from pathlib import Path
from typing import Tuple, Any, Optional, Dict, Callable
import logging

class LocalSetupHandler:
    """
    Handler untuk operasi setup di lingkungan lokal
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None):
        """
        Inisialisasi handler untuk setup lokal
        
        Args:
            ui_callback: Dictionary callback untuk update UI
        """
        self.ui_callback = ui_callback or {}
        self.logger = logging.getLogger(__name__)
    
    def _log_message(self, message: str):
        """Log message to UI if callback exists"""
        self.logger.info(message)
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message)
    
    def setup_local_environment(self) -> Tuple[Path, Path]:
        """
        Setup environment untuk local (non-Colab)
        
        Returns:
            Tuple of (base_dir, config_dir)
        """
        # Gunakan project root sebagai base_dir
        base_dir = Path(__file__).resolve().parents[5]
        config_dir = base_dir / "configs"
        
        # Pastikan direktori config ada
        config_dir.mkdir(parents=True, exist_ok=True)
        
        self._log_message(f"✅ Setup local environment: base_dir={base_dir}, config_dir={config_dir}")
        
        return base_dir, config_dir 