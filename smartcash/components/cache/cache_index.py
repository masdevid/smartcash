"""
File: smartcash/utils/cache/cache_index.py
Author: Alfrida Sabar
Deskripsi: Modul untuk mengelola index cache dengan persistensi dan validasi efisien.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import threading

from smartcash.utils.logger import SmartCashLogger

class CacheIndex:
    def __init__(self, cache_dir: Path, logger: Optional[SmartCashLogger] = None):
        self.cache_dir = cache_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.index_path = cache_dir / "cache_index.json"
        self._lock = threading.RLock()
        self.index = self._create_default_index()

    def _create_default_index(self) -> Dict:
        return {
            'files': {},
            'total_size': 0,
            'last_cleanup': None,
            'creation_time': datetime.now().isoformat()
        }

    def load_index(self) -> bool:
        with self._lock:
            if self.index_path.exists():
                try:
                    with open(self.index_path, 'r') as f:
                        loaded_index = json.load(f)
                    
                    # Validasi struktur index minimal
                    if all(key in loaded_index for key in ['files', 'total_size']):
                        # Bersihkan entri file yang tidak valid
                        loaded_index['files'] = {
                            k: v for k, v in loaded_index['files'].items()
                            if all(key in v for key in ['size', 'timestamp'])
                        }
                        self.index = loaded_index
                        self.logger.info(f"📂 Cache index dimuat: {len(self.index['files'])} entri")
                        return True
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membaca cache index: {str(e)}")
            
            self.index = self._create_default_index()
            return False

    def save_index(self) -> bool:
        with self._lock:
            self.index['last_updated'] = datetime.now().isoformat()
            temp_path = self.index_path.with_suffix('.tmp')
            
            try:
                with open(temp_path, 'w') as f:
                    json.dump(self.index, f, indent=2)
                shutil.move(str(temp_path), str(self.index_path))
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan cache index: {str(e)}")
                return False

    def get_files(self) -> Dict:
        with self._lock:
            return self.index['files']

    def get_file_info(self, key: str) -> Dict:
        with self._lock:
            return self.index['files'].get(key, {})

    def add_file(self, key: str, size: int) -> None:
        with self._lock:
            now = datetime.now().isoformat()
            self.index['files'][key] = {
                'size': size,
                'timestamp': now,
                'last_accessed': now
            }
            self.index['total_size'] += size
            self.save_index()

    def remove_file(self, key: str) -> int:
        with self._lock:
            if key in self.index['files']:
                size = self.index['files'][key].get('size', 0)
                del self.index['files'][key]
                self.index['total_size'] -= size
                self.save_index()
                return size
            return 0

    def update_access_time(self, key: str) -> None:
        with self._lock:
            if key in self.index['files']:
                self.index['files'][key]['last_accessed'] = datetime.now().isoformat()
                self.save_index()

    def update_cleanup_time(self) -> None:
        with self._lock:
            self.index['last_cleanup'] = datetime.now().isoformat()
            self.save_index()

    def get_total_size(self) -> int:
        with self._lock:
            return self.index.get('total_size', 0)

    def set_total_size(self, size: int) -> None:
        with self._lock:
            self.index['total_size'] = size
            self.save_index()