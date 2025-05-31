"""
File: smartcash/ui/pretrained_model/utils/model_utils.py
Deskripsi: Utilitas untuk mengelola model pretrained dengan metadata dan validasi
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

class ModelManager:
    """Kelas untuk mengelola model pretrained dengan metadata dan validasi."""
    
    def __init__(self, models_dir: str = '/content/models'):
        """
        Inisialisasi manager model pretrained.
        
        Args:
            models_dir: Direktori untuk menyimpan model
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_file = self.models_dir / 'model_metadata.json'
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata dari file JSON."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_metadata(self) -> bool:
        """Simpan metadata ke file JSON."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            return True
        except Exception:
            return False
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Hitung hash SHA-256 dari file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def validate_model(self, model_path: Path, model_id: str) -> bool:
        """
        Validasi model berdasarkan hash yang tersimpan di metadata.
        
        Args:
            model_path: Path ke file model
            model_id: ID model dalam metadata
            
        Returns:
            True jika model valid, False jika tidak
        """
        if not model_path.exists() or model_id not in self.metadata:
            return False
            
        stored_hash = self.metadata.get(model_id, {}).get('hash')
        if not stored_hash:
            return False
            
        current_hash = self._calculate_hash(model_path)
        return current_hash == stored_hash
    
    def update_model_metadata(self, model_path: Path, model_id: str, 
                              version: str, source: str) -> Dict[str, Any]:
        """
        Update metadata untuk model yang baru diunduh.
        
        Args:
            model_path: Path ke file model
            model_id: ID unik untuk model
            version: Versi model
            source: Sumber model
            
        Returns:
            Dict berisi metadata model yang diupdate
        """
        if not model_path.exists():
            return {}
            
        model_hash = self._calculate_hash(model_path)
        self.metadata[model_id] = {
            'path': str(model_path),
            'version': version,
            'source': source,
            'hash': model_hash,
            'date_downloaded': str(model_path.stat().st_mtime),
            'size_mb': round(model_path.stat().st_size / (1024 * 1024), 2)
        }
        self._save_metadata()
        
        return self.metadata[model_id]
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Dapatkan informasi tentang model tertentu.
        
        Args:
            model_id: ID model dalam metadata
            
        Returns:
            Dict berisi informasi model
        """
        return self.metadata.get(model_id, {})
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi semua model yang tersedia.
        
        Returns:
            Dict berisi informasi semua model
        """
        info = {
            'models_dir': str(self.models_dir),
            'models': {}
        }
        
        for model_id, metadata in self.metadata.items():
            model_path = Path(metadata.get('path', ''))
            if model_path.exists():
                model_name = model_path.name
                info['models'][model_name] = {
                    'path': str(model_path),
                    'size_mb': metadata.get('size_mb', 0),
                    'version': metadata.get('version', ''),
                    'source': metadata.get('source', ''),
                    'date_downloaded': metadata.get('date_downloaded', ''),
                    'is_valid': self.validate_model(model_path, model_id)
                }
        
        return info
