"""
File: smartcash/utils/augmentation/augmentation_base.py
Author: Alfrida Sabar
Deskripsi: Kelas dasar untuk augmentasi dengan fungsionalitas umum untuk pemrosesan gambar dan label
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import random
from tqdm.auto import tqdm
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class AugmentationBase:
    """
    Kelas dasar untuk augmentasi dengan fungsionalitas umum.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi kelas dasar augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Validasi layer yang aktif
        self._validate_layers()
        
        # Thread lock untuk statistik
        self._stats_lock = threading.RLock()
    
    def _validate_layers(self) -> None:
        """Validasi layer yang aktif."""
        for layer in self.active_layers[:]:
            if layer not in self.layer_config_manager.get_layer_names():
                self.logger.warning(f"⚠️ Layer tidak dikenali: {layer}")
                self.active_layers.remove(layer)
        
        if not self.active_layers:
            self.logger.warning("⚠️ Tidak ada layer aktif yang valid, fallback ke 'banknote'")
            self.active_layers = ['banknote']
    
    def reset_stats(self) -> Dict:
        """
        Reset statistik augmentasi.
        
        Returns:
            Dict statistik kosong
        """
        with self._stats_lock:
            stats = {
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
            return stats