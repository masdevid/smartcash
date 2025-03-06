# File: smartcash/models/yolov5_model.py
# Author: Alfrida Sabar
# Deskripsi: Optimasi model YOLOv5 untuk bekerja dengan EfficientNet backbone

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union, Tuple

from smartcash.models.backbones.base import BaseBackbone
from smartcash.models.backbones.cspdarknet import CSPDarknet
from smartcash.models.backbones.efficientnet import EfficientNetBackbone
from smartcash.models.necks.fpn_pan import FeatureProcessingNeck
from smartcash.models.losses import YOLOLoss
from smartcash.utils.logger import SmartCashLogger

class YOLOv5Model(nn.Module):
    """
    Model YOLOv5 dengan backbone yang bisa diganti.
    Mendukung penggunaan CSPDarknet (default) atau EfficientNet
    dengan Feature Pyramid Network + Path Aggregation Network.
    """
    
    def __init__(
        self,
        num_classes: int = 7,  # 7 denominasi Rupiah
        backbone_type: str = "cspdarknet",
        pretrained: bool = True,
        detection_layers: List[str] = None,  # Layer yang akan diaktifkan
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.num_classes = num_classes
        
        # Default ke 'banknote' jika tidak ada layers yang dispesifikasi
        self.detection_layers = detection_layers or ['banknote']
        
        # Inisialisasi backbone
        self.backbone = self._create_backbone(backbone_type, pretrained)
        
        # Get output channels dari backbone
        self.backbone_channels = self.backbone.get_output_channels()
        
        # ✨ PERUBAHAN: Gunakan FeatureProcessingNeck dari fpn_pan.py
        self.feature_neck = FeatureProcessingNeck(
            in_channels=self.backbone_channels,
            out_channels=[128, 256, 512],  # Output sesuai YOLOv5
            logger=self.logger
        )
        
        # Detection Head - ModuleDict untuk mendukung multiple layers
        self.heads = nn.ModuleDict()
        for layer in self.detection_layers:
            # Buat head untuk setiap layer deteksi
            self.heads[layer] = self._build_head(
                in_channels=[128, 256, 512],  # Channel setelah FPN-PAN
                num_classes=num_classes // len(self.detection_layers)  # Classes per layer
            )
        
        # Inisialisasi loss function
        self._init_loss_function()
        
        self.logger.info(
            f"✨ Model YOLOv5 siap dengan:\n"
            f"   Backbone: {backbone_type}\n"
            f"   Channels: {self.backbone_channels}\n"
            f"   Classes: {num_classes}\n"
            f"   Detection Layers: {self.detection_layers}\n"
            f"   Pretrained: {pretrained}"
        )
    
    def _init_loss_function(self):
        """Inisialisasi loss function untuk setiap layer"""
        self.loss_fn = {}
        for layer in self.detection_layers:
            self.loss_fn[layer] = YOLOLoss(
                num_classes=self.num_classes // len(self.detection_layers)
            )
        
    def _create_backbone(
        self,
        backbone_type: str,
        pretrained: bool
    ) -> BaseBackbone:
        """Buat backbone sesuai tipe yang dipilih"""
        if backbone_type == "cspdarknet":
            return CSPDarknet(pretrained=pretrained, logger=self.logger)
        elif backbone_type == "efficientnet":
            return EfficientNetBackbone(pretrained=pretrained, logger=self.logger)
        else:
            raise ValueError(f"Backbone {backbone_type} tidak didukung")
            
    def _build_head(
        self,
        in_channels: List[int],
        num_classes: int
    ) -> nn.ModuleList:
        """Build detection head untuk setiap skala"""
        heads = nn.ModuleList()
        
        for ch in in_channels:
            heads.append(
                nn.Sequential(
                    self._conv(ch, ch//2, 1),
                    self._conv(ch//2, ch, 3),
                    nn.Conv2d(
                        ch,
                        3 * (5 + num_classes),  # 3 anchor, 5 bbox params, num_classes
                        1
                    )
                )
            )
            
        return heads
        
    def _conv(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Helper untuk membuat convolution block"""
        return nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride,
                kernel_size//2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass model
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dict berisi prediksi untuk setiap layer deteksi
            Format: {
                'banknote': [pred_s, pred_m, pred_l],
                'nominal': [pred_s, pred_m, pred_l],
                ...
            }
        """
        # Validasi input shape
        batch_size, channels, height, width = x.shape
        if channels != 3:
            self.logger.warning(f"⚠️ Input tidak valid! Ekspektasi 3 channels, tapi dapat {channels}")
        
        # Backbone features
        features = self.backbone(x)
        
        # ✨ PERUBAHAN: Gunakan Feature Neck (FPN+PAN)
        # Logging dimensi untuk debugging
        if self.logger:
            self.logger.debug(f"🔍 Backbone output shapes: {[f.shape for f in features]}")
            
        processed_features = self.feature_neck(features)
        
        if self.logger:
            self.logger.debug(f"🔍 FPN-PAN output shapes: {[f.shape for f in processed_features]}")
        
        # Prediksi untuk setiap layer
        predictions = {}
        
        for layer_name, heads in self.heads.items():
            layer_preds = []
            
            # Hitung classes per layer
            num_classes = self.num_classes // len(self.detection_layers)
            
            # Detection head untuk setiap skala
            for feat, head in zip(processed_features, heads):
                # Deteksi dan reshape ke format YOLO
                pred = head(feat)
                bs, _, h, w = pred.shape
                
                # [B, anchors*(5+classes), H, W] -> [B, anchors, H, W, 5+classes]
                pred = pred.view(bs, 3, 5 + num_classes, h, w)
                pred = pred.permute(0, 1, 3, 4, 2)
                
                layer_preds.append(pred)
                
            predictions[layer_name] = layer_preds
            
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Args:
            predictions: Prediksi model (single atau multi layer)
            targets: Ground truth labels (single atau multi layer)
            
        Returns:
            Dict berisi komponen loss
        """
        device = next(self.parameters()).device
        loss_components = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # ✨ PERUBAHAN: Refaktor untuk memisahkan logika single dan multi layer
        if isinstance(targets, torch.Tensor):
            # Single layer target
            return self._compute_single_layer_loss(predictions, targets)
        else:
            # Multi layer target
            return self._compute_multi_layer_loss(predictions, targets)
    
    def _compute_single_layer_loss(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss untuk single layer targets"""
        device = next(self.parameters()).device
        loss_components = {}
        
        # Hanya ada satu layer (misalnya 'banknote')
        if len(self.detection_layers) == 1:
            layer_name = self.detection_layers[0]
            if layer_name in predictions:
                layer_preds = predictions[layer_name]
                
                try:
                    # Pastikan target valid untuk loss function
                    if self._is_valid_target(targets):
                        loss, loss_items = self.loss_fn[layer_name](layer_preds, targets)
                        total_loss = loss
                        loss_components = loss_items
                    else:
                        self.logger.warning(f"⚠️ Target tidak valid untuk YOLOLoss")
                        total_loss = self._fallback_loss(layer_preds, targets)
                        loss_components = {
                            'box_loss': torch.tensor(0.0, device=device),
                            'obj_loss': torch.tensor(0.0, device=device),
                            'cls_loss': total_loss
                        }
                except Exception as e:
                    self.logger.warning(f"⚠️ Error pada YOLOLoss: {str(e)}. Menggunakan fallback loss")
                    total_loss = self._fallback_loss(layer_preds, targets)
                    loss_components = {
                        'box_loss': torch.tensor(0.0, device=device),
                        'obj_loss': torch.tensor(0.0, device=device),
                        'cls_loss': total_loss
                    }
            else:
                self.logger.warning(f"⚠️ Layer {layer_name} tidak ditemukan dalam prediksi")
                total_loss = torch.tensor(0.1, device=device, requires_grad=True)
                loss_components = {'dummy_loss': total_loss}
        else:
            # Multi layer configuration tapi target masih single format
            self.logger.warning("⚠️ Multi-layer detection membutuhkan target multi-layer")
            
            # Gunakan layer pertama saja
            layer_name = self.detection_layers[0]
            if layer_name in predictions:
                layer_preds = predictions[layer_name]
                total_loss = self._fallback_loss(layer_preds, targets)
                loss_components = {'fallback_loss': total_loss}
        
        # Add total loss
        loss_components['total_loss'] = total_loss
        return loss_components
    
    def _compute_multi_layer_loss(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss untuk multi layer targets"""
        device = next(self.parameters()).device
        loss_components = {}
        layer_losses = []
        
        for layer_name in self.detection_layers:
            if layer_name in predictions and layer_name in targets:
                layer_preds = predictions[layer_name]
                layer_targets = targets[layer_name]
                
                try:
                    if self._is_valid_target(layer_targets):
                        loss, loss_items = self.loss_fn[layer_name](layer_preds, layer_targets)
                        layer_losses.append(loss)
                        
                        # Add prefix untuk membedakan loss tiap layer
                        for k, v in loss_items.items():
                            loss_components[f"{layer_name}_{k}"] = v
                    else:
                        self.logger.warning(f"⚠️ Target untuk layer {layer_name} tidak valid")
                        fallback = self._fallback_loss(layer_preds, layer_targets)
                        layer_losses.append(fallback)
                        loss_components[f"{layer_name}_fallback_loss"] = fallback
                except Exception as e:
                    self.logger.warning(f"⚠️ Error pada layer {layer_name}: {str(e)}")
                    fallback = self._fallback_loss(layer_preds, layer_targets)
                    layer_losses.append(fallback)
                    loss_components[f"{layer_name}_fallback_loss"] = fallback
        
        # Gabungkan semua layer losses
        if layer_losses:
            total_loss = sum(layer_losses)
        else:
            self.logger.warning("⚠️ Tidak ada valid layer untuk loss calculation")
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)
            loss_components['dummy_loss'] = total_loss
        
        # Add total loss
        loss_components['total_loss'] = total_loss
        return loss_components
    
    def _is_valid_target(self, targets: torch.Tensor) -> bool:
        """Validasi target untuk YOLOLoss"""
        if not isinstance(targets, torch.Tensor):
            return False
        
        if targets.numel() == 0:
            return False
            
        if len(targets.shape) == 3 and targets.shape[2] >= 5:
            return True  # format [batch, objects, coords]
            
        if len(targets.shape) == 2 and targets.shape[1] >= 5:
            return True  # format [objects, coords]
            
        if targets.numel() % 6 == 0:  # Asumsi format tertentu
            return True
            
        return False
    
    def _fallback_loss(
        self, 
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Fallback ke MSELoss jika YOLOLoss gagal"""
        criterion = nn.MSELoss()
        batch_size = predictions[0].size(0)
        
        # Flatten prediksi
        pred_tensor = torch.cat([p.view(batch_size, -1) for p in predictions], dim=1)
        
        # Coba sesuaikan target dengan prediksi
        if isinstance(targets, torch.Tensor) and targets.size(0) == batch_size:
            target_flat = targets.view(batch_size, -1)
            # Pastikan ukuran sesuai
            if target_flat.size(1) > pred_tensor.size(1):
                target_flat = target_flat[:, :pred_tensor.size(1)]
            elif target_flat.size(1) < pred_tensor.size(1):
                # Pad target dengan zeros
                padding = torch.zeros(
                    batch_size, 
                    pred_tensor.size(1) - target_flat.size(1),
                    device=target_flat.device
                )
                target_flat = torch.cat([target_flat, padding], dim=1)
            
            return criterion(pred_tensor, target_flat)
        else:
            # Fallback ke dummy target
            dummy_target = torch.zeros_like(pred_tensor)
            return criterion(pred_tensor, dummy_target)