# File: smartcash/handlers/detection/integration/model_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan ModelManager

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import ModelError

class ModelAdapter:
    """
    Adapter untuk integrasi dengan ModelManager.
    Mengelola loading model dan persiapan untuk deteksi.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger = None,
        colab_mode: bool = False
    ):
        """
        Inisialisasi model adapter.
        
        Args:
            config: Konfigurasi
            logger: Logger kustom (opsional)
            colab_mode: Flag untuk mode Google Colab
        """
        self.config = config
        self.logger = logger or get_logger("model_adapter")
        self.colab_mode = colab_mode
        
        # Parameter dari konfigurasi
        model_config = config.get('model', {})
        self.weights_path = model_config.get('weights')
        self.backbone = model_config.get('backbone', 'efficientnet_b4')
        self.half_precision = model_config.get('half_precision', True)
        
        # Caching untuk model
        self._model = None
        self._model_path = None
    
    def get_model(
        self,
        model_path: Optional[str] = None,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Dapatkan model deteksi yang sudah dimuat.
        
        Args:
            model_path: Path ke model (opsional, gunakan konfigurasi jika None)
            force_reload: Flag untuk paksa reload model
            
        Returns:
            Model yang sudah dimuat
        """
        # Gunakan path dari konfigurasi jika tidak diberikan
        if model_path is None:
            model_path = self.weights_path
            
        if model_path is None:
            raise ModelError("Path model tidak ditemukan di konfigurasi dan tidak diberikan")
            
        # Konversi ke Path
        model_path = Path(model_path)
        
        # Cek apakah perlu reload
        if (self._model is None or 
            self._model_path != str(model_path) or 
            force_reload):
            
            # Load model
            try:
                # Deteksi device yang tersedia
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                
                self.logger.info(f"🧠 Loading model dari {model_path} (device: {device})")
                
                # Check eksistensi file model
                if not model_path.exists():
                    raise ModelError(f"File model tidak ditemukan: {model_path}")
                
                # Import ModelManager hanya saat diperlukan untuk menghindari
                # circular import dan lazy-load dependency
                try:
                    from smartcash.handlers.model.model_manager import ModelManager
                    model_manager = ModelManager(self.config, colab_mode=self.colab_mode)
                    
                    # Load model menggunakan ModelManager
                    model = model_manager.load_model(
                        model_path=str(model_path),
                        backbone=self.backbone,
                        device=str(device)
                    )
                    
                    # Prepare model untuk inferensi
                    model = model_manager.prepare_model_for_inference(
                        model, 
                        half_precision=self.half_precision
                    )
                    
                except ImportError:
                    # Fallback ke loading basic dengan torch jika ModelManager tidak tersedia
                    self.logger.warning("⚠️ ModelManager tidak tersedia, menggunakan fallback loading")
                    model = self._fallback_load_model(model_path, device)
                
                self._model = model
                self._model_path = str(model_path)
                
                self.logger.info(f"✅ Model loaded dan siap untuk inferensi")
                
                return model
                
            except Exception as e:
                self.logger.error(f"❌ Gagal load model: {str(e)}")
                raise ModelError(f"Gagal load model dari {model_path}: {str(e)}")
        
        return self._model
    
    def _fallback_load_model(
        self, 
        model_path: Path, 
        device: torch.device
    ) -> torch.nn.Module:
        """
        Fallback untuk load model tanpa ModelManager.
        
        Args:
            model_path: Path ke file model
            device: Device target
            
        Returns:
            Model yang sudah dimuat
        """
        try:
            # Load model dengan torch
            model = torch.load(model_path, map_location=device)
            
            # Handle jika yang diload adalah state_dict atau checkpoint
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
                
            # Pastikan model dalam mode eval
            model.eval()
            
            # Half precision jika diminta dan di GPU
            if self.half_precision and device.type == 'cuda':
                model = model.half()
                
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Fallback load gagal: {str(e)}")
            raise ModelError(f"Gagal load model (fallback): {str(e)}")