# File: smartcash/models/backbones/cspdarknet.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi CSPDarknet backbone yang lebih robust dengan fallback ke local weights

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import os
import urllib.request
import warnings
from tqdm import tqdm

from .base import BaseBackbone
from smartcash.utils.logger import SmartCashLogger

class DownloadProgressBar(tqdm):
    """Progress bar kustom untuk download file."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class CSPDarknet(BaseBackbone):
    """
    CSPDarknet backbone implementation.
    
    Mendukung:
    1. Download otomatis dari torch.hub dengan validasi
    2. Fallback ke weights lokal
    3. Validasi output feature maps
    4. Deteksi struktur model yang tidak sesuai
    """
    
    # Konfigurasi model YOLOv5
    YOLOV5_CONFIG = {
        'yolov5s': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
            'feature_indices': [4, 6, 9],  # P3, P4, P5 layers
            'expected_channels': [128, 256, 512],
            'expected_shapes': [(80, 80), (40, 40), (20, 20)],  # untuk input 640x640
        },
        'yolov5m': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt',
            'feature_indices': [4, 6, 9],
            'expected_channels': [192, 384, 768],
            'expected_shapes': [(80, 80), (40, 40), (20, 20)],
        },
        'yolov5l': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt',
            'feature_indices': [4, 6, 9],
            'expected_channels': [256, 512, 1024],
            'expected_shapes': [(80, 80), (40, 40), (20, 20)],
        }
    }
    
    def __init__(
        self,
        pretrained: bool = True,
        model_size: str = 'yolov5s',
        weights_path: Optional[str] = None,
        fallback_to_local: bool = True,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi CSPDarknet backbone.
        
        Args:
            pretrained: Gunakan pretrained weights atau tidak
            model_size: Ukuran model ('yolov5s', 'yolov5m', 'yolov5l')
            weights_path: Path ke weights file kustom (opsional)
            fallback_to_local: Fallback ke weights lokal jika download gagal
            logger: Logger untuk mencatat proses
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
        # Validasi model size
        if model_size not in self.YOLOV5_CONFIG:
            supported = list(self.YOLOV5_CONFIG.keys())
            raise ValueError(f"❌ Model size {model_size} tidak didukung. Pilihan yang tersedia: {supported}")
            
        self.model_size = model_size
        self.config = self.YOLOV5_CONFIG[model_size]
        self.feature_indices = self.config['feature_indices']
        self.expected_channels = self.config['expected_channels']
        
        try:
            # Setup model
            if pretrained:
                self._setup_pretrained_model(weights_path, fallback_to_local)
            else:
                self._setup_model_from_scratch()
                
            # Verifikasi struktur model
            self._verify_model_structure()
                
            self.logger.success(
                f"✨ CSPDarknet backbone berhasil diinisialisasi:\n"
                f"   • Model size: {model_size}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Feature layers: {self.feature_indices}\n"
                f"   • Channels: {self.get_output_channels()}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Gagal inisialisasi CSPDarknet: {str(e)}")
            raise
    
    def _setup_pretrained_model(self, weights_path: Optional[str], fallback_to_local: bool):
        """Setup pretrained model dengan fallback ke local weights."""
        # Tentukan weights path
        if weights_path is None:
            # Setup default pretrained directory
            pretrained_dir = Path('./pretrained')
            pretrained_dir.mkdir(exist_ok=True)
            weights_file = pretrained_dir / f"{self.model_size}.pt"
            
            # Check jika weights sudah ada secara lokal
            if weights_file.exists():
                self.logger.info(f"💾 Menggunakan weights lokal: {weights_file}")
                weights_path = str(weights_file)
            else:
                # Download weights jika tidak ada
                try:
                    self.logger.info(f"⬇️ Mengunduh weights untuk {self.model_size}...")
                    self._download_weights(self.config['url'], weights_file)
                    weights_path = str(weights_file)
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal mengunduh weights: {str(e)}")
                    if fallback_to_local:
                        # Coba gunakan torch.hub sebagai fallback
                        self.logger.info("🔄 Mencoba fallback ke torch.hub...")
                        weights_path = None
                    else:
                        raise
        
        # Load model
        try:
            if weights_path is not None:
                # Load from local path
                self.logger.info(f"📂 Memuat model dari: {weights_path}")
                model = torch.load(weights_path, map_location='cpu')
                
                # Check if it's a YOLOv5 model format
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
            else:
                # Fallback to torch.hub
                self.logger.info("🔌 Memuat model dari torch.hub...")
                
                # Set torch.hub dir
                hub_dir = Path('./hub')
                hub_dir.mkdir(exist_ok=True)
                torch.hub.set_dir(str(hub_dir))
                
                # Load YOLOv5 model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    yolo = torch.hub.load(
                        'ultralytics/yolov5',
                        self.model_size,
                        pretrained=True,
                        trust_repo=True
                    )
                model = yolo.model
            
            # Extract backbone dari model
            self._extract_backbone(model)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise
    
    def _setup_model_from_scratch(self):
        """Setup model dari awal tanpa pretrained weights."""
        try:
            # Import YOLOv5 repo locally if available
            try:
                from yolov5.models.yolo import Model
                from yolov5.models.common import Focus, Conv, C3
                
                self.logger.info("📦 Menggunakan YOLOv5 local repository")
                
                # Setup backbone manually
                # Simplified implementation - in a real scenario this would be more complete
                self.backbone = nn.Sequential(
                    # First layer - Focus
                    Focus(3, 64, 3),
                    # Conv layer
                    Conv(64, 128, 3, 2),
                    # C3 layer
                    C3(128, 128),
                    # Layer for P3
                    Conv(128, 256, 3, 2),
                    # C3 layer
                    C3(256, 256),
                    # Layer for P4
                    Conv(256, 512, 3, 2),
                    # C3 layer
                    C3(512, 512),
                    # Layer for P5
                    Conv(512, 1024, 3, 2),
                    # Final C3 layer
                    C3(1024, 1024)
                )
                
            except ImportError:
                # Fallback: build simplified version if YOLOv5 not available
                self.logger.warning("⚠️ YOLOv5 repo tidak tersedia, menggunakan implementasi sederhana")
                
                # Build a minimal CSPDarknet variant
                self.backbone = nn.Sequential(
                    # Initial convolution
                    nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=2),
                    nn.BatchNorm2d(32),
                    nn.SiLU(inplace=True),
                    
                    # Downsampling blocks with residual connections for P3, P4, P5
                    self._make_downsample_block(32, 64),
                    self._make_downsample_block(64, 128),  # P3
                    self._make_downsample_block(128, 256), # P4
                    self._make_downsample_block(256, 512), # P5
                )
                
                # Override expected channels based on simplified implementation
                self.expected_channels = [128, 256, 512]
                self.feature_indices = [2, 3, 4]  # Adjusted for simplified implementation
                
        except Exception as e:
            self.logger.error(f"❌ Gagal setup model dari awal: {str(e)}")
            raise
    
    def _download_weights(self, url: str, output_path: Union[str, Path]):
        """Download weights dengan progress bar."""
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=f"⬇️ Downloading {self.model_size}") as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    def _extract_backbone(self, model):
        """Extract backbone dari YOLOv5 model dengan validasi struktur."""
        if hasattr(model, 'model'):
            modules = list(model.model.children())
        else:
            modules = list(model.children())
            
        # Validasi modul
        if len(modules) < 10:
            raise ValueError(
                f"❌ Struktur model yang tidak valid! Diharapkan minimal 10 layer, "
                f"tetapi hanya menemukan {len(modules)}"
            )
        
        # Ambil backbone layers (0 sampai maksimum P5 index + 1)
        max_index = max(self.feature_indices) + 1
        backbone_modules = modules[:max_index]
        
        # Buat backbone Sequential
        self.backbone = nn.Sequential(*backbone_modules)
        
        # Validasi feature indices
        if max(self.feature_indices) >= len(backbone_modules):
            actual_layers = len(backbone_modules)
            problematic = [i for i in self.feature_indices if i >= actual_layers]
            raise ValueError(
                f"❌ Feature indices yang tidak valid! Indices {problematic} "
                f"di luar batas (total layers: {actual_layers})"
            )
    
    def _verify_model_structure(self):
        """Verifikasi struktur model dengan dummy forward pass."""
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 640, 640)
                features = self.forward(dummy_input)
                
                # Validasi jumlah features
                if len(features) != len(self.feature_indices):
                    raise ValueError(
                        f"❌ Jumlah feature maps ({len(features)}) tidak sesuai dengan "
                        f"jumlah indeks feature ({len(self.feature_indices)})!"
                    )
                
                # Validasi channel dimensions
                actual_channels = [feat.shape[1] for feat in features]
                if actual_channels != self.expected_channels:
                    self.logger.warning(
                        f"⚠️ Channel dimensions ({actual_channels}) tidak sesuai dengan "
                        f"yang diharapkan ({self.expected_channels})!"
                    )
                    
                    # Update expected channels untuk integrasi yang benar
                    self.expected_channels = actual_channels
                
                # Verifikasi spatial dimensions
                for i, feat in enumerate(features):
                    _, _, h, w = feat.shape
                    expected_h, expected_w = self.config['expected_shapes'][i]
                    
                    if h != expected_h or w != expected_w:
                        self.logger.warning(
                            f"⚠️ Spatial dimensions dari feature {i} ({h}x{w}) tidak sesuai dengan "
                            f"yang diharapkan ({expected_h}x{expected_w})!"
                        )
                
                self.logger.info(
                    f"✅ Verifikasi struktur backbone berhasil:\n"
                    f"   • Channels: {actual_channels}\n"
                    f"   • Shapes: {[f.shape[2:] for f in features]}"
                )
                    
        except Exception as e:
            self.logger.error(f"❌ Verifikasi model gagal: {str(e)}")
            raise
    
    def _make_downsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Helper untuk membuat blok downsampling sederhana untuk implementasi CSPDarknet."""
        return nn.Sequential(
            # Downsample
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            
            # Residual block
            nn.Conv2d(out_channels, out_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, return P3, P4, P5 feature maps."""
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for P3, P4, P5."""
        return self.expected_channels
    
    def get_output_shapes(self) -> List[Tuple[int, int]]:
        """Get output shapes for feature maps with 640x640 input."""
        return self.config['expected_shapes']
        
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load state dictionary dengan validasi dan logging."""
        try:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            
            if missing_keys:
                self.logger.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
                
            self.logger.info("✅ Berhasil memuat state dictionary")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat state dictionary: {str(e)}")
            raise