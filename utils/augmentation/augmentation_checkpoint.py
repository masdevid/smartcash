"""
File: smartcash/utils/augmentation/augmentation_checkpoint.py
Author: Alfrida Sabar
Deskripsi: Pengelola checkpoint augmentasi untuk melanjutkan proses yang terganggu
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase

class AugmentationCheckpoint(AugmentationBase):
    """Pengelola checkpoint augmentasi untuk melanjutkan proses yang terganggu."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        checkpoint_interval: int = 50
    ):
        """Inisialisasi pengelola checkpoint augmentasi."""
        super().__init__(config, output_dir, logger)
        self.checkpoint_interval = checkpoint_interval
        self._checkpoint_file = None
        self._last_checkpoint = None
    
    def load_checkpoint(self, split: str) -> List[str]:
        """Load checkpoint augmentasi jika ada."""
        checkpoint_file = self.output_dir / f".aug_checkpoint_{split}.json"
        self._checkpoint_file = checkpoint_file
        
        if not checkpoint_file.exists():
            return []
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            processed_files = checkpoint_data.get('processed_files', [])
            
            # Update stats dari checkpoint
            with self._stats_lock:
                stats = checkpoint_data.get('stats', {})
                for key, value in stats.items():
                    if isinstance(value, dict):
                        self.stats.setdefault(key, {}).update(value)
                    else:
                        self.stats[key] = value
            
            if processed_files:
                self.logger.info(f"🔄 Melanjutkan dari checkpoint: {len(processed_files)} file telah diproses")
            
            return processed_files
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memuat checkpoint: {str(e)}")
            return []
    
    def save_checkpoint(self, processed_files: List[str], stats: Dict) -> bool:
        """Simpan checkpoint untuk melanjutkan proses yang terganggu."""
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'processed_files': processed_files,
                'stats': stats
            }
            
            with open(self._checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            
            self._last_checkpoint = time.time()
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            return False
    
    def should_checkpoint(self, processed_count: int, last_time: float) -> bool:
        """Periksa apakah perlu melakukan checkpoint."""
        interval_check = processed_count % self.checkpoint_interval == 0
        time_check = (time.time() - last_time) > 300
        
        return interval_check or time_check