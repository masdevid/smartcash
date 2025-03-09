# File: smartcash/handlers/evaluation/integration/model_manager_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk loading dan inisialisasi model

import os
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.models.yolov5_model import YOLOv5Model

class ModelManagerAdapter:
    """
    Adapter untuk ModelManager.
    Menyediakan antarmuka untuk loading dan persiapan model untuk evaluasi.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi ModelManagerAdapter.
        
        Args:
            config: Konfigurasi untuk evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("model_adapter")
        
        # Setup cache untuk model
        self.model_cache = {}
        
        # Konfigurasi model
        self.model_config = self.config.get('model', {})
        self.default_backbone = self.model_config.get('backbone', 'cspdarknet')
        self.num_classes = self.model_config.get('num_classes', 17)  # Default untuk SmartCash
        
        self.logger.debug(f"🔧 ModelManagerAdapter diinisialisasi (backbone={self.default_backbone}, num_classes={self.num_classes})")
    
    def load_model(
        self,
        model_path: str,
        backbone: Optional[str] = None,
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Load model dari checkpoint.
        
        Args:
            model_path: Path ke checkpoint model
            backbone: Jenis backbone ('efficientnet', 'cspdarknet')
            device: Device untuk model ('cuda', 'cpu')
            force_reload: Paksa reload meskipun ada di cache
            
        Returns:
            Model yang sudah dimuat
            
        Raises:
            FileNotFoundError: Jika file tidak ditemukan
            RuntimeError: Jika gagal memuat model
        """
        # Gunakan cache jika ada dan tidak force_reload
        cache_key = f"{model_path}_{backbone}_{device}"
        if cache_key in self.model_cache and not force_reload:
            self.logger.info(f"🔄 Menggunakan model dari cache: {os.path.basename(model_path)}")
            return self.model_cache[cache_key]
        
        try:
            # Cek keberadaan file
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ File model tidak ditemukan: {model_path}")
            
            # Tentukan device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.logger.info(f"🔄 Memuat model dari {os.path.basename(model_path)} ke {device}...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Tentukan backbone
            if backbone is None:
                # Coba ambil dari checkpoint config
                if 'config' in checkpoint and 'backbone' in checkpoint['config']:
                    backbone = checkpoint['config']['backbone']
                # Atau dari nama file
                elif 'efficientnet' in model_path.lower():
                    backbone = 'efficientnet'
                else:
                    backbone = self.default_backbone

            # Inisialisasi model
            if backbone == 'efficientnet':
                self.logger.info(f"🏗️ Menggunakan arsitektur EfficientNet-B4 backbone")
                model = YOLOv5Model(
                    backbone_type='efficientnet',
                    num_classes=self.num_classes
                )
            else:
                self.logger.info(f"🏗️ Menggunakan arsitektur CSPDarknet backbone")
                model = YOLOv5Model(
                    backbone_type='cspdarknet',
                    num_classes=self.num_classes
                )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                # Coba load langsung
                model.load_state_dict(checkpoint)
            
            # Pindahkan ke device yang tepat
            model = model.to(device)
            
            # Set ke mode evaluasi
            model.eval()
            
            # Simpan ke cache
            self.model_cache[cache_key] = model
            
            self.logger.success(f"✅ Model berhasil dimuat: {os.path.basename(model_path)} ({backbone})")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise RuntimeError(f"Gagal memuat model: {str(e)}")
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Dapatkan informasi tentang model.
        
        Args:
            model_path: Path ke checkpoint model
            
        Returns:
            Dictionary berisi informasi model
        """
        try:
            # Cek keberadaan file
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ File model tidak ditemukan: {model_path}")
            
            # Load checkpoint (header saja jika memungkinkan)
            try:
                # Coba load dengan map_location untuk menghindari OOM
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat checkpoint penuh: {str(e)}")
                return {
                    'filename': os.path.basename(model_path),
                    'path': model_path,
                    'size': os.path.getsize(model_path),
                    'error': str(e)
                }
            
            # Ekstrak info
            info = {
                'filename': os.path.basename(model_path),
                'path': model_path,
                'size': os.path.getsize(model_path),
                'last_modified': os.path.getmtime(model_path),
                'config': checkpoint.get('config', {}),
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
            }
            
            # Tentukan backbone
            if 'config' in checkpoint and 'backbone' in checkpoint['config']:
                info['backbone'] = checkpoint['config']['backbone']
            elif 'efficientnet' in model_path.lower():
                info['backbone'] = 'efficientnet'
            else:
                info['backbone'] = 'cspdarknet'
            
            # Tentukan jumlah kelas
            if 'config' in checkpoint and 'num_classes' in checkpoint['config']:
                info['num_classes'] = checkpoint['config']['num_classes']
            else:
                info['num_classes'] = self.num_classes
            
            return info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mendapatkan info model: {str(e)}")
            # Return info minimal
            return {
                'filename': os.path.basename(model_path),
                'path': model_path,
                'size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
                'error': str(e)
            }
    
    def prepare_model_for_evaluation(
        self, 
        model: torch.nn.Module,
        half_precision: Optional[bool] = None
    ) -> torch.nn.Module:
        """
        Siapkan model untuk evaluasi.
        
        Args:
            model: Model PyTorch
            half_precision: Gunakan half precision (FP16)
            
        Returns:
            Model yang siap untuk evaluasi
        """
        # Pastikan model dalam mode evaluasi
        model.eval()
        
        # Gunakan half precision jika diminta
        if half_precision is None:
            half_precision = self.model_config.get('half_precision', False)
        
        if half_precision and torch.cuda.is_available():
            self.logger.info("🔄 Menggunakan half precision (FP16)")
            model = model.half()
        
        return model
    
    def clear_cache(self):
        """Bersihkan cache model."""
        n_models = len(self.model_cache)
        self.model_cache.clear()
        self.logger.info(f"🧹 Cache model dibersihkan ({n_models} model)")