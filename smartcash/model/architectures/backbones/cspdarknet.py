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
from smartcash.model.config.model_constants import YOLOV5_CONFIG

class CSPDarknet(BaseBackbone):
    """Enhanced CSPDarknet backbone dengan multi-layer detection support dan model building capabilities."""
    
    def __init__(self, pretrained: bool = True, model_size: str = 'yolov5s', weights_path: Optional[str] = None, 
                 fallback_to_local: bool = True, pretrained_dir: str = './pretrained', 
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
        
        try:
            # Setup model
            if pretrained:
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
                
                # Use safe globals for PyTorch 2.6+ compatibility
                import torch.serialization
                try:
                    from models.yolo import Model as YOLOModel
                    from models.common import Conv, C3, SPPF, Bottleneck
                    safe_globals = [YOLOModel, Conv, C3, SPPF, Bottleneck]
                except ImportError:
                    safe_globals = []
                
                with torch.serialization.safe_globals(safe_globals):
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
        # Handle different YOLOv5 model structures
        modules = None
        
        # Try different ways to access the model layers
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            # YOLOv5 with AutoShape wrapper: model.model.model
            modules = list(model.model.model.children())
            self.logger.debug("📋 Using model.model.model structure")
        elif hasattr(model, 'model'):
            # Direct YOLOv5 model: model.model
            modules = list(model.model.children())
            self.logger.debug("📋 Using model.model structure")
        else:
            # Fallback: direct model
            modules = list(model.children())
            self.logger.debug("📋 Using direct model structure")
        
        self.logger.debug(f"🔍 Found {len(modules)} modules in model")
        
        # For YOLOv5, we need to look for the backbone part
        # YOLOv5 structure: [model layers] where backbone is usually first N layers
        if len(modules) == 1 and hasattr(modules[0], 'model'):
            # If we get a single module that contains the actual model
            modules = list(modules[0].model.children())
            self.logger.debug(f"🔍 Unwrapped single module, now have {len(modules)} modules")
        
        # Validate we have enough modules
        if len(modules) < max(self.feature_indices) + 1:
            self.logger.warning(f"⚠️ Only {len(modules)} modules found, need at least {max(self.feature_indices) + 1}")
            
            # If we still don't have enough modules, try to build a custom backbone
            if len(modules) < 3:
                self.logger.info("🔧 Building custom backbone due to insufficient modules")
                self._build_custom_backbone()
                return
        
        # Extract backbone layers (0 to maximum P5 index + 1)
        max_index = min(max(self.feature_indices) + 1, len(modules))
        backbone_modules = modules[:max_index]
        
        self.logger.debug(f"📋 Extracting {len(backbone_modules)} backbone modules (indices 0-{max_index-1})")
        
        # Build backbone Sequential
        self.backbone = nn.Sequential(*backbone_modules)
        
        # Adjust feature indices if necessary
        if max(self.feature_indices) >= len(backbone_modules):
            self.logger.warning(f"⚠️ Adjusting feature indices to fit available layers")
            # Adjust feature indices to available layers
            available_indices = list(range(len(backbone_modules)))
            if len(available_indices) >= 3:
                # Take the last 3 indices for P3, P4, P5
                self.feature_indices = available_indices[-3:]
                self.logger.info(f"📋 Adjusted feature indices to: {self.feature_indices}")
            else:
                raise BackboneError(
                    f"❌ Not enough backbone layers ({len(backbone_modules)}) for multi-scale features"
                )
    
    def _build_custom_backbone(self):
        """Build a custom CSPDarknet-like backbone when YOLOv5 extraction fails."""
        self.logger.info("🔧 Building custom CSPDarknet backbone")
        
        # Build a simplified CSPDarknet-like structure
        self.backbone = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 32, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Downsampling blocks for P3, P4, P5
            self._make_downsample_block(32, 64),
            self._make_downsample_block(64, 128),  # P3 - index 4
            self._make_downsample_block(128, 256), # P4 - index 5  
            self._make_downsample_block(256, 512), # P5 - index 6
        )
        
        # Update configuration for custom backbone
        self.feature_indices = [4, 5, 6]  # Indices where we extract features
        self.expected_channels = [128, 256, 512]  # Match YOLO_CHANNELS
        
        self.logger.info(f"✅ Custom backbone built with indices {self.feature_indices}")
    
    def _verify_model_structure(self):
        """Verifikasi struktur model dan output channels."""
            
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
            'pretrained': hasattr(self, 'pretrained_weights_loaded'),
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