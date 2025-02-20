# File: utils/preprocessing_cache.py
# Author: Alfrida Sabar
# Deskripsi: Cache handler untuk menyimpan dan mengelola hasil preprocessing

import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger

class PreprocessingCache:
    """Sistem caching untuk hasil preprocessing"""
    
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessing",
        max_size_gb: float = 1.0,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        # Load cache index
        self.index_path = self.cache_dir / "cache_index.json"
        self.load_index()
        
        # Track statistik
        self.hits = 0
        self.misses = 0
        
    def load_index(self) -> None:
        """Load cache index dari disk"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
            self.logger.info(
                f"📂 Cache index dimuat: {len(self.index['files'])} entries"
            )
        else:
            self.index = {
                'files': {},
                'total_size': 0,
                'last_cleanup': None
            }
            
    def save_index(self) -> None:
        """Simpan cache index ke disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
            
    def get_cache_key(
        self,
        file_path: str,
        params: Dict
    ) -> str:
        """Generate cache key dari file path dan parameter"""
        # Hash file content
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        # Hash parameters
        param_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        
        return f"{file_hash}_{param_hash}"
        
    def get(
        self,
        key: str
    ) -> Optional[Dict]:
        """
        Ambil hasil preprocessing dari cache
        Args:
            key: Cache key
        Returns:
            Dict hasil preprocessing atau None jika tidak ada
        """
        if key in self.index['files']:
            cache_path = self.cache_dir / f"{key}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.hits += 1
                    return pickle.load(f)
                    
        self.misses += 1
        return None
        
    def put(
        self,
        key: str,
        data: Dict,
        file_size: int
    ) -> None:
        """
        Simpan hasil preprocessing ke cache
        Args:
            key: Cache key
            data: Data yang akan disimpan
            file_size: Ukuran file dalam bytes
        """
        # Check ukuran cache
        if self.index['total_size'] + file_size > self.max_size_bytes:
            self.cleanup()
            
        # Simpan data
        cache_path = self.cache_dir / f"{key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Update index
        self.index['files'][key] = {
            'size': file_size,
            'timestamp': datetime.now().isoformat()
        }
        self.index['total_size'] += file_size
        self.save_index()
        
    def cleanup(self) -> None:
        """Bersihkan cache lama ketika ukuran melebihi batas"""
        self.logger.info("🧹 Memulai cache cleanup...")
        
        # Sort berdasarkan timestamp
        sorted_files = sorted(
            self.index['files'].items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Hapus file lama sampai ukuran di bawah batas
        freed_space = 0
        current_size = self.index['total_size']
        for key, info in sorted_files:
            if current_size <= self.max_size_bytes:
                break
                
            cache_path = self.cache_dir / f"{key}.pkl"
            if cache_path.exists():
                cache_path.unlink()
                freed_space += info['size']
                
            current_size -= info['size']
            del self.index['files'][key]
            
        self.index['total_size'] = current_size
        self.index['last_cleanup'] = datetime.now().isoformat()
        self.save_index()
        
        self.logger.success(
            f"✨ Cache cleanup selesai\n"
            f"🗑️ Space freed: {freed_space / 1024 / 1024:.1f} MB"
        )
        
    def get_stats(self) -> Dict:
        """
        Dapatkan statistik cache
        Returns:
            Dict berisi statistik cache
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': self.index['total_size'] / 1024 / 1024,  # MB
            'num_files': len(self.index['files']),
            'last_cleanup': self.index['last_cleanup']
        }