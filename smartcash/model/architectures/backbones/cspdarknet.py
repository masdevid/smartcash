"""
File: smartcash/model/architectures/backbones/cspdarknet.py
Deskripsi: CSPDarknet backbone implementation for YOLOv5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import sys
from pathlib import Path
import os
import urllib.request
import warnings

from smartcash.common.logger import SmartCashLogger
from smartcash.common.exceptions import BackboneError
from smartcash.common.utils.progress_utils import download_with_progress
from smartcash.model.architectures.backbones.base import BaseBackbone
from smartcash.model.config.model_constants import YOLOV5_CONFIG, YOLO_CHANNELS

class CSPDarknet(BaseBackbone):
    """Enhanced CSPDarknet backbone dengan multi-layer detection support dan model building capabilities."""
    
    def __init__(self, pretrained: bool = True, model_size: str = 'yolov5s', weights_path: Optional[str] = None, 
                 fallback_to_local: bool = True, pretrained_dir: str = './pretrained', testing_mode: bool = False, 
                 build_mode: str = 'detection', multi_layer_heads: bool = False, 
                 logger: Optional[SmartCashLogger] = None):
        """Inisialisasi CSPDarknet backbone dengan konfigurasi model dan pretrained weights."""
        super().__init__(logger=logger)
        
        self.build_mode = build_mode
        self.multi_layer_heads = multi_layer_heads
        
        # Validasi model size
        if model_size not in YOLOV5_CONFIG: raise BackboneError(f"❌ Model size {model_size} tidak didukung. Pilihan yang tersedia: {list(YOLOV5_CONFIG.keys())}")
            
        self.model_size, self.config, self.pretrained_dir = model_size, YOLOV5_CONFIG[model_size], Path(pretrained_dir)
        self.feature_indices, self.expected_channels = self.config['feature_indices'], self.config['expected_channels']
        self.out_channels = self.expected_channels  # Ensure out_channels is set for BaseBackbone compatibility
        self.testing_mode = testing_mode
        
        try:
            # Setup model
            if testing_mode:
                self._setup_dummy_model_for_testing()
            elif pretrained:
                self._setup_pretrained_model(weights_path, fallback_to_local)
            else:
                self._setup_model_from_scratch()
                
            # Verifikasi struktur model
            self._verify_model_structure()
                
            self.logger.info(
                f"✨ CSPDarknet backbone berhasil diinisialisasi:\n"
                f"   • Model size: {model_size}\n"
                f"   • Pretrained: {pretrained}\n"
                f"   • Feature layers: {self.feature_indices}\n"
                f"   • Channels: {self.get_output_channels()}"
            )
            
        except BackboneError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"❌ Gagal inisialisasi CSPDarknet: {str(e)}")
            raise BackboneError(f"Gagal inisialisasi CSPDarknet: {str(e)}")
    
    def _setup_pretrained_model(self, weights_path: Optional[str], fallback_to_local: bool):
        """Setup pretrained model dengan download otomatis dan fallback ke local weights."""
        # Tentukan weights path
        if weights_path is None:
            self.pretrained_dir.mkdir(exist_ok=True)
            weights_file = self.pretrained_dir / f"{self.model_size}.pt"
            
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
                        raise BackboneError(f"Gagal mengunduh weights dan fallback dinonaktifkan: {str(e)}")
        
        # Load model
        try:
            if weights_path is not None:
                # Load from local path
                self.logger.info(f"📂 Memuat model dari: {weights_path}")
                model = torch.load(weights_path, map_location='cpu')
                
                # Check if it's a YOLOv5 model format
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                elif isinstance(model, dict) and 'models' in model:
                    model = model['models']
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
            raise BackboneError(f"Gagal memuat model: {str(e)}")
    
    def _setup_dummy_model_for_testing(self):
        """Buat model dummy untuk testing tanpa perlu mengunduh model pretrained."""
        self.logger.info("🧪 Membuat model dummy untuk testing...")
        
        # Gunakan YOLO_CHANNELS langsung untuk output channels
        self.expected_channels = YOLO_CHANNELS
        
        # Buat layer dummy untuk setiap output channel yang diharapkan
        self.dummy_layers = nn.ModuleList()
        in_channels = 3
        
        # Buat layer dummy untuk setiap feature map yang diharapkan
        for i, out_ch in enumerate(self.expected_channels):
            # Buat layer konvolusi sederhana yang langsung menghasilkan channel yang diharapkan
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.dummy_layers.append(layer)
            in_channels = out_ch
        
        # Override metode forward untuk menggunakan model dummy
        self.forward = self._forward_dummy
        
        self.logger.info(f"✅ Model dummy berhasil dibuat dengan {len(self.dummy_layers)} feature maps")
        self.logger.info(f"   • Output channels: {self.expected_channels} (sesuai dengan YOLO_CHANNELS)")
    
    def _setup_model_from_scratch(self):
        """Setup model dari awal tanpa pretrained weights."""
        try:
            # Import YOLOv5 repo locally if available
            try:
                from yolov5.models.yolo import Model
                from yolov5.models.common import Focus, Conv, C3
                
                self.logger.info("📦 Menggunakan YOLOv5 local repository")
                
                # Setup backbone manually
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
            raise BackboneError(f"Gagal setup model dari awal: {str(e)}")
    
    def _download_weights(self, url: str, output_path: Union[str, Path]):
        """Download weights dengan progress callback dari URL ke output_path."""
        download_with_progress(url, str(output_path), logger=self.logger)
    
    def _extract_backbone(self, model):
        """Extract backbone dari YOLOv5 model dengan validasi struktur."""
        if hasattr(model, 'model'):
            modules = list(model.model.children())
        else:
            modules = list(model.children())
            
        # Validasi modul
        if len(modules) < 10:
            raise BackboneError(
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
            raise BackboneError(
                f"❌ Feature indices yang tidak valid! Indices {problematic} "
                f"di luar batas (total layers: {actual_layers})"
            )
    
    def _verify_model_structure(self):
        """Verifikasi struktur model dan output channels."""
        # Skip verifikasi jika dalam mode testing
        if self.testing_mode:
            self.logger.info("🧪 Skip verifikasi model dalam mode testing")
            return
            
        # Validasi feature indices
        if max(self.feature_indices) >= len(self.backbone):
            raise BackboneError(f"❌ Feature indices {self.feature_indices} melebihi jumlah layer backbone ({len(self.backbone)})")
        
        # Validasi output channels dengan dummy input
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 640, 640)
                features = self.forward(dummy_input)
                actual_channels = [f.shape[1] for f in features]
                
                # Validasi jumlah feature maps
                if len(actual_channels) != len(self.expected_channels):
                    raise BackboneError(f"❌ Jumlah feature maps tidak sesuai: {len(actual_channels)} vs {len(self.expected_channels)}")
                
                # Validasi channels per feature map
                for i, (actual, expected) in enumerate(zip(actual_channels, self.expected_channels)):
                    if actual != expected:
                        self.logger.warning(f"⚠️ Channel output pada index {i} tidak sesuai: {actual} vs {expected}")
                
                self.logger.debug(f"🔍 Feature shapes: {[f.shape for f in features]}")
                    
        except Exception as e:
            self.logger.error(f"❌ Verifikasi model gagal: {str(e)}")
            raise BackboneError(f"Verifikasi model gagal: {str(e)}")
    
    def _make_downsample_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Membuat blok downsampling untuk CSPDarknet."""
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True), 
                            nn.Conv2d(out_channels, out_channels//2, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels//2), nn.SiLU(inplace=True),
                            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
    
    def _forward_dummy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass dengan model dummy untuk testing."""
        features = []
        
        # Buat dummy feature maps dengan channel yang sesuai dengan YOLO_CHANNELS
        for i, out_ch in enumerate(self.expected_channels):
            # Downsample input sesuai dengan level feature map
            # P3 = 1/8, P4 = 1/16, P5 = 1/32 dari input asli
            scale_factor = 1 / (2 ** (i + 3))
            h, w = x.shape[2], x.shape[3]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Buat dummy tensor dengan channel yang sesuai
            batch_size = x.shape[0]
            dummy_feature = torch.zeros(batch_size, out_ch, new_h, new_w, device=x.device)
            
            # Isi dengan nilai random untuk simulasi feature map
            dummy_feature.normal_(0, 0.02)
            
            features.append(dummy_feature)
            
            # Log untuk debugging
            self.logger.debug(f"🔍 Feature map {i}: shape={dummy_feature.shape}, channels={dummy_feature.shape[1]}")
        
        # Pastikan jumlah feature maps sesuai dengan yang diharapkan
        if len(features) != len(self.expected_channels):
            self.logger.warning(f"⚠️ Jumlah feature maps ({len(features)}) tidak sesuai dengan yang diharapkan ({len(self.expected_channels)})")
        
        return features
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass, mengembalikan feature maps P3, P4, P5."""
        try:
            features = []
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i in self.feature_indices: features.append(x)
            self.validate_output(features, self.expected_channels)
            return features
        except Exception as e:
            self.logger.error(f"❌ Forward pass gagal: {str(e)}")
            raise BackboneError(f"Forward pass gagal: {str(e)}")
    
    def get_info(self):
        """Dapatkan informasi backbone dalam bentuk dictionary.
        
        Returns:
            Dict: Informasi tentang backbone
        """
        return {
            'type': 'CSPDarknet',
            'variant': self.model_size,
            'out_channels': self.expected_channels,
            'feature_stages': self.feature_indices,
            'pretrained': not self.testing_mode,
            'build_mode': self.build_mode,
            'multi_layer_heads': self.multi_layer_heads,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'feature_count': len(self.expected_channels),
            'fpn_compatible': len(self.expected_channels) == 3
        }
    
    def build_for_yolo(self, head_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build CSPDarknet backbone specifically for YOLO architecture with multi-layer support"""
        try:
            # Setup multi-layer detection heads configuration
            if self.multi_layer_heads:
                layer_specs = {
                    'layer_1': {'classes': ['001', '002', '005', '010', '020', '050', '100'], 'description': 'Full banknote detection'},
                    'layer_2': {'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'], 'description': 'Nominal-defining features'},
                    'layer_3': {'classes': ['l3_sign', 'l3_text', 'l3_thread'], 'description': 'Common features'}
                }
            else:
                layer_specs = {
                    'layer_1': {'classes': ['001', '002', '005', '010', '020', '050', '100'], 'description': 'Single layer detection'}
                }
            
            build_result = {
                'backbone': self,
                'backbone_info': self.get_info(),
                'output_channels': self.expected_channels,
                'feature_shapes': self.get_output_shapes(),
                'layer_specifications': layer_specs,
                'recommended_neck': 'FPN-PAN',
                'compatible_heads': ['YOLOv5Head', 'MultiLayerHead'],
                'phase_training': {
                    'phase_1': 'Freeze backbone, train detection heads only',
                    'phase_2': 'Unfreeze entire model for fine-tuning'
                },
                'optimizer_config': {
                    'backbone_lr': 1e-5,
                    'head_lr': 1e-3,
                    'differential_lr': True
                },
                'success': True
            }
            
            self.logger.info(f"✅ CSPDarknet backbone built for YOLO with {len(layer_specs)} detection layers")
            return build_result
            
        except Exception as e:
            error_msg = f"Failed to build CSPDarknet for YOLO: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def prepare_for_training(self, freeze_backbone: bool = True) -> Dict[str, Any]:
        """Prepare backbone for training with specified freeze configuration"""
        try:
            if freeze_backbone:
                self.freeze()
                self.logger.info("❄️ Backbone frozen for phase 1 training")
            else:
                self.unfreeze()
                self.logger.info("🔥 Backbone unfrozen for phase 2 fine-tuning")
            
            return {
                'frozen': freeze_backbone,
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'total_params': sum(p.numel() for p in self.parameters()),
                'phase': 'phase_1' if freeze_backbone else 'phase_2',
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Failed to prepare backbone for training: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_output_channels(self) -> List[int]: return self.expected_channels
    
    def get_output_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> List[Tuple[int, int]]: return self.config['expected_shapes']
        
    def load_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """Load state dictionary dengan validasi dan logging."""
        try:
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
            if missing_keys and self.logger: self.logger.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys and self.logger: self.logger.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
            self.logger.info("✅ Berhasil memuat state dictionary")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat state dictionary: {str(e)}")
            raise BackboneError(f"Gagal memuat state dictionary: {str(e)}")