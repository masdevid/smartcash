"""
File: smartcash/utils/augmentation/augmentation_base.py
Author: Alfrida Sabar
Deskripsi: Kelas dasar untuk augmentasi dengan fungsionalitas umum untuk pemrosesan gambar dan label
"""

import time
import threading
from pathlib import Path
from typing import Dict, Optional

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class AugmentationBase:
    """Kelas dasar untuk augmentasi dengan fungsionalitas umum."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi kelas dasar augmentasi."""
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = self._validate_active_layers(config.get('layers', ['banknote']))
        
        # Thread lock untuk statistik
        self._stats_lock = threading.RLock()
    
    def _validate_active_layers(self, layers: list) -> list:
        """Validasi dan filter layer yang aktif."""
        valid_layers = [
            layer for layer in layers 
            if layer in self.layer_config_manager.get_layer_names()
        ]
        
        if not valid_layers:
            self.logger.warning("⚠️ Tidak ada layer valid, menggunakan 'banknote'")
            return ['banknote']
        
        return valid_layers
    
    def reset_stats(self) -> Dict:
        """Reset statistik augmentasi."""
        with self._stats_lock:
            return {
                'processed': 0,
                'augmented': 0,
                'failed': 0,
                'skipped_invalid': 0,
                'layer_stats': {layer: 0 for layer in self.active_layers},
                'per_type': {
                    'position': 0,
                    'lighting': 0,
                    'combined': 0,
                    'extreme_rotation': 0
                },
                'duration': 0.0,
                'start_time': time.time()
            }