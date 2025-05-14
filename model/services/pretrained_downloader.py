"""
File: smartcash/model/services/pretrained_downloader.py
Deskripsi: Layanan untuk mengunduh dan mengelola model pre-trained
"""

import os
import torch
import timm
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

class PretrainedModelDownloader:
    """Layanan untuk mengunduh dan mengelola model pre-trained."""
    
    def __init__(self, models_dir: str = '/content/models'):
        """
        Inisialisasi downloader model pre-trained.
        
        Args:
            models_dir: Direktori untuk menyimpan model
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
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
    
    def download_yolov5(self, force: bool = False) -> Dict[str, Any]:
        """
        Download model YOLOv5s pre-trained.
        
        Args:
            force: Paksa download ulang meskipun sudah ada
            
        Returns:
            Dict berisi informasi model
        """
        yolo_path = self.models_dir / 'yolov5s.pt'
        yolo_version = 'v6.2'
        yolo_source = 'ultralytics/yolov5'
        yolo_model_id = f"yolov5s_{yolo_version}"
        
        # Cek apakah model sudah ada dan valid
        if not force and yolo_path.exists() and yolo_model_id in self.metadata:
            return {
                'path': str(yolo_path),
                'version': yolo_version,
                'source': yolo_source,
                'metadata': self.metadata.get(yolo_model_id, {})
            }
        
        # Download model
        model = torch.hub.load(yolo_source, 'yolov5s', pretrained=True, force_reload=True)
        torch.save(model.state_dict(), yolo_path)
        
        # Update metadata
        model_hash = self._calculate_hash(yolo_path)
        self.metadata[yolo_model_id] = {
            'path': str(yolo_path),
            'version': yolo_version,
            'source': yolo_source,
            'hash': model_hash,
            'date_downloaded': str(Path(yolo_path).stat().st_mtime)
        }
        self._save_metadata()
        
        return {
            'path': str(yolo_path),
            'version': yolo_version,
            'source': yolo_source,
            'metadata': self.metadata.get(yolo_model_id, {})
        }
    
    def download_efficientnet(self, force: bool = False) -> Dict[str, Any]:
        """
        Download model EfficientNet-B4 pre-trained.
        
        Args:
            force: Paksa download ulang meskipun sudah ada
            
        Returns:
            Dict berisi informasi model
        """
        efficientnet_path = self.models_dir / 'efficientnet_b4.pt'
        efficientnet_version = 'timm-1.0'
        efficientnet_source = 'timm'
        efficientnet_model_id = f"efficientnet_b4_{efficientnet_version}"
        
        # Cek apakah model sudah ada dan valid
        if not force and efficientnet_path.exists() and efficientnet_model_id in self.metadata:
            return {
                'path': str(efficientnet_path),
                'version': efficientnet_version,
                'source': efficientnet_source,
                'metadata': self.metadata.get(efficientnet_model_id, {})
            }
        
        # Download model
        model = timm.create_model('efficientnet_b4', pretrained=True)
        torch.save(model.state_dict(), efficientnet_path)
        
        # Update metadata
        model_hash = self._calculate_hash(efficientnet_path)
        self.metadata[efficientnet_model_id] = {
            'path': str(efficientnet_path),
            'version': efficientnet_version,
            'source': efficientnet_source,
            'hash': model_hash,
            'date_downloaded': str(Path(efficientnet_path).stat().st_mtime)
        }
        self._save_metadata()
        
        return {
            'path': str(efficientnet_path),
            'version': efficientnet_version,
            'source': efficientnet_source,
            'metadata': self.metadata.get(efficientnet_model_id, {})
        }
    
    def download_all_models(self, force: bool = False) -> Dict[str, Any]:
        """
        Download semua model pre-trained.
        
        Args:
            force: Paksa download ulang meskipun sudah ada
            
        Returns:
            Dict berisi informasi semua model
        """
        yolo_info = self.download_yolov5(force)
        efficientnet_info = self.download_efficientnet(force)
        
        return {
            'yolov5': yolo_info,
            'efficientnet_b4': efficientnet_info
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi semua model yang tersedia.
        
        Returns:
            Dict berisi informasi model
        """
        yolo_path = self.models_dir / 'yolov5s.pt'
        efficientnet_path = self.models_dir / 'efficientnet_b4.pt'
        
        info = {
            'models_dir': str(self.models_dir),
            'models': {}
        }
        
        if yolo_path.exists():
            yolo_size = yolo_path.stat().st_size / (1024 * 1024)  # Convert to MB
            info['models']['yolov5s'] = {
                'path': str(yolo_path),
                'size_mb': round(yolo_size, 2),
                'metadata': self.metadata.get('yolov5s_v6.2', {})
            }
        
        if efficientnet_path.exists():
            efficientnet_size = efficientnet_path.stat().st_size / (1024 * 1024)  # Convert to MB
            info['models']['efficientnet_b4'] = {
                'path': str(efficientnet_path),
                'size_mb': round(efficientnet_size, 2),
                'metadata': self.metadata.get('efficientnet_b4_timm-1.0', {})
            }
        
        return info
