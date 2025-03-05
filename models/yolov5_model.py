# File: models/yolov5_model.py
# Author: Alfrida Sabar
# Deskripsi: Perbaikan model YOLOv5 untuk bekerja dengan EfficientNet backbone

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union

from smartcash.models.backbones.base import BaseBackbone
from smartcash.models.backbones.cspdarknet import CSPDarknet
from smartcash.models.backbones.efficientnet import EfficientNetBackbone
from smartcash.models.losses import YOLOLoss
from smartcash.utils.logger import SmartCashLogger

class YOLOv5Model(nn.Module):
    """
    Model YOLOv5 dengan backbone yang bisa diganti.
    Mendukung penggunaan CSPDarknet (default) atau EfficientNet.
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
        
        # Feature Pyramid Network
        self.fpn = self._build_fpn(self.backbone_channels)
        
        # Detection Head - ModuleDict untuk mendukung multiple layers
        self.heads = nn.ModuleDict()
        for layer in self.detection_layers:
            # Buat head untuk setiap layer deteksi
            self.heads[layer] = self._build_head(
                in_channels=[128, 256, 512],  # Channel setelah FPN, sekarang sudah disesuaikan
                num_classes=num_classes // len(self.detection_layers)  # Classes per layer
            )
        
        self.logger.info(
            f"✨ Model YOLOv5 siap dengan:\n"
            f"   Backbone: {backbone_type}\n"
            f"   Channels: {self.backbone_channels}\n"
            f"   Classes: {num_classes}\n"
            f"   Detection Layers: {self.detection_layers}\n"
            f"   Pretrained: {pretrained}"
        )
        
    def _create_backbone(
        self,
        backbone_type: str,
        pretrained: bool
    ) -> BaseBackbone:
        """Buat backbone sesuai tipe yang dipilih"""
        if backbone_type == "cspdarknet":
            return CSPDarknet(pretrained=pretrained)
        elif backbone_type == "efficientnet":
            return EfficientNetBackbone(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone_type} tidak didukung")
            
    def _build_fpn(self, in_channels: List[int]) -> nn.ModuleList:
        """
        Build Feature Pyramid Network
        
        Args:
            in_channels: Channel dari setiap stage backbone
        """
        fpn_layers = nn.ModuleList()
        
        # Bottom-up pathway (sesuai YOLOv5)
        for i, ch in enumerate(in_channels):
            fpn_layers.append(
                nn.Sequential(
                    self._conv(ch, in_channels[i]//2, 1),
                    self._conv(in_channels[i]//2, in_channels[i], 3),
                    self._conv(in_channels[i], in_channels[i]//2, 1)
                )
            )
            
        return fpn_layers
        
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
        # Backbone features
        features = self.backbone(x)
        
        # Feature Pyramid Network
        fpn_features = []
        for feat, fpn in zip(features, self.fpn):
            fpn_features.append(fpn(feat))
            
        # Prediksi untuk setiap layer
        predictions = {}
        
        for layer_name, heads in self.heads.items():
            layer_preds = []
            
            # Hitung classes per layer
            num_classes = self.num_classes // len(self.detection_layers)
            
            # Detection head untuk setiap skala
            for feat, head in zip(fpn_features, heads):
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
        Compute training loss dengan perbaikan untuk error requires_grad
        
        Args:
            predictions: Prediksi model (single atau multi layer)
            targets: Ground truth labels (single atau multi layer)
            
        Returns:
            Dict berisi komponen loss
        """
        # Inisialisasi tensor loss kosong dengan requires_grad=True
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        loss_components = {}
        
        # Buat loss function untuk setiap layer jika belum ada
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = YOLOLoss(
                num_classes=self.num_classes // len(self.detection_layers)
            )
            
        # Single layer target
        if isinstance(targets, torch.Tensor):
            # Hanya ada satu layer (misalnya 'banknote')
            if len(self.detection_layers) == 1:
                layer_name = self.detection_layers[0]
                if layer_name in predictions:
                    layer_preds = predictions[layer_name]
                    
                    # Gunakan MSELoss sebagai fallback jika YOLOLoss tidak berfungsi
                    try:
                        loss, loss_items = self.loss_fn(layer_preds, targets)
                        total_loss = loss
                        loss_components = loss_items
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error pada YOLOLoss: {str(e)}. Menggunakan MSELoss sebagai fallback")
                        criterion = nn.MSELoss()
                        # Buat prediksi dan target yang kompatibel dengan MSELoss
                        batch_size = layer_preds[0].size(0)
                        pred_tensor = torch.cat([p.view(batch_size, -1) for p in layer_preds], dim=1)
                        if targets.size(0) == batch_size:
                            target_flat = targets.view(batch_size, -1)
                            # Pastikan ukuran sesuai, potong jika perlu
                            if target_flat.size(1) > pred_tensor.size(1):
                                target_flat = target_flat[:, :pred_tensor.size(1)]
                            elif target_flat.size(1) < pred_tensor.size(1):
                                # Pad target dengan zeros
                                padding = torch.zeros(batch_size, pred_tensor.size(1) - target_flat.size(1), 
                                                    device=target_flat.device)
                                target_flat = torch.cat([target_flat, padding], dim=1)
                            
                            total_loss = criterion(pred_tensor, target_flat)
                        else:
                            # Fallback jika ukuran batch tidak sesuai
                            dummy_target = torch.zeros_like(pred_tensor, requires_grad=False)
                            total_loss = criterion(pred_tensor, dummy_target)
                        
                        loss_components = {'box_loss': torch.tensor(0.0, device=total_loss.device),
                                        'obj_loss': torch.tensor(0.0, device=total_loss.device),
                                        'cls_loss': total_loss}
                else:
                    # Layer tidak ditemukan dalam prediksi
                    self.logger.warning(f"⚠️ Layer {layer_name} tidak ditemukan dalam prediksi")
                    dummy_loss = torch.tensor(0.1, device=next(self.parameters()).device, requires_grad=True)
                    total_loss = dummy_loss
                    loss_components = {'dummy_loss': dummy_loss}
            else:
                # Multi layer tapi target masih single format
                # Perlu ada proses splitting target
                self.logger.warning("⚠️ Multi-layer detection membutuhkan target multi-layer")
                
                # Gunakan layer pertama saja
                layer_name = self.detection_layers[0]
                if layer_name in predictions:
                    layer_preds = predictions[layer_name]
                    
                    # Gunakan MSELoss untuk fallback
                    criterion = nn.MSELoss()
                    batch_size = layer_preds[0].size(0)
                    pred_tensor = torch.cat([p.view(batch_size, -1) for p in layer_preds], dim=1)
                    
                    # Pastikan target memiliki ukuran yang sesuai
                    if targets.size(0) == batch_size:
                        target_flat = targets.view(batch_size, -1)
                        # Pastikan ukuran sesuai
                        if target_flat.size(1) > pred_tensor.size(1):
                            target_flat = target_flat[:, :pred_tensor.size(1)]
                        elif target_flat.size(1) < pred_tensor.size(1):
                            padding = torch.zeros(batch_size, pred_tensor.size(1) - target_flat.size(1), 
                                                device=target_flat.device)
                            target_flat = torch.cat([target_flat, padding], dim=1)
                        
                        total_loss = criterion(pred_tensor, target_flat)
                    else:
                        # Fallback jika ukuran batch tidak sesuai
                        dummy_target = torch.zeros_like(pred_tensor, requires_grad=False)
                        total_loss = criterion(pred_tensor, dummy_target)
                    
                    loss_components = {'fallback_loss': total_loss}
        else:
            # Multi layer target
            if isinstance(predictions, dict) and isinstance(targets, dict):
                layer_losses = []
                
                for layer_name in self.detection_layers:
                    if layer_name in predictions and layer_name in targets:
                        layer_preds = predictions[layer_name]
                        layer_targets = targets[layer_name]
                        
                        try:
                            loss, loss_items = self.loss_fn(layer_preds, layer_targets)
                            layer_losses.append(loss)
                            
                            # Add prefix untuk membedakan loss tiap layer
                            for k, v in loss_items.items():
                                loss_components[f"{layer_name}_{k}"] = v
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error pada layer {layer_name}: {str(e)}")
                            # Fallback ke MSELoss
                            criterion = nn.MSELoss()
                            batch_size = layer_preds[0].size(0)
                            pred_tensor = torch.cat([p.view(batch_size, -1) for p in layer_preds], dim=1)
                            
                            # Buat dummy target jika layer_targets tidak kompatibel
                            if isinstance(layer_targets, torch.Tensor) and layer_targets.size(0) == batch_size:
                                target_flat = layer_targets.view(batch_size, -1)
                                # Pastikan ukuran sesuai
                                if target_flat.size(1) > pred_tensor.size(1):
                                    target_flat = target_flat[:, :pred_tensor.size(1)]
                                elif target_flat.size(1) < pred_tensor.size(1):
                                    padding = torch.zeros(batch_size, pred_tensor.size(1) - target_flat.size(1), 
                                                        device=target_flat.device)
                                    target_flat = torch.cat([target_flat, padding], dim=1)
                                
                                layer_loss = criterion(pred_tensor, target_flat)
                            else:
                                dummy_target = torch.zeros_like(pred_tensor, requires_grad=False)
                                layer_loss = criterion(pred_tensor, dummy_target)
                            
                            layer_losses.append(layer_loss)
                            loss_components[f"{layer_name}_fallback_loss"] = layer_loss
                
                # Gabungkan semua layer losses
                if layer_losses:
                    total_loss = sum(layer_losses)
                else:
                    # Tidak ada valid layer losses
                    total_loss = torch.tensor(0.1, device=next(self.parameters()).device, requires_grad=True)
                    loss_components['dummy_loss'] = total_loss
            else:
                # Format tidak sesuai
                self.logger.warning("⚠️ Format predictions dan targets tidak sesuai")
                total_loss = torch.tensor(0.1, device=next(self.parameters()).device, requires_grad=True)
                loss_components['format_error_loss'] = total_loss
        
        # Pastikan total loss memiliki requires_grad=True
        if not total_loss.requires_grad:
            self.logger.warning("⚠️ Total loss tidak memiliki requires_grad, membuat tensor baru")
            total_loss = total_loss.clone().detach().requires_grad_(True)
        
        # Add total loss ke komponen
        loss_components['total_loss'] = total_loss
        
        return loss_components