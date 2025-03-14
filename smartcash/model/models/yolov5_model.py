"""
File: smartcash/model/models/yolov5_model.py
Deskripsi: Implementasi model YOLOv5 terintegrasi yang menggabungkan backbone, neck, dan head
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

class YOLOv5Model(nn.Module):
    """
    Model YOLOv5 terintegrasi yang menggabungkan backbone, neck, dan head.
    Mendukung berbagai backbone termasuk EfficientNet dan CSPDarknet.
    """
    
    def __init__(
        self,
        backbone,
        neck,
        head,
        config: Dict
    ):
        """
        Inisialisasi YOLOv5Model.
        
        Args:
            backbone: Backbone network
            neck: Feature processing neck
            head: Detection head
            config: Konfigurasi model
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.config = config
        self.img_size = config.get('img_size', (640, 640))
        
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass model.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dict berisi prediksi untuk setiap layer deteksi
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Process features through neck
        processed_features = self.neck(features)
        
        # Get predictions from head
        predictions = self.head(processed_features)
        
        return predictions
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 300
    ) -> Dict[str, List[Dict]]:
        """
        Lakukan prediksi dengan post-processing.
        
        Args:
            x: Input tensor
            conf_threshold: Threshold confidence
            nms_threshold: Threshold IoU untuk NMS
            max_detections: Jumlah maksimum deteksi per gambar
            
        Returns:
            Dict berisi hasil deteksi untuk setiap layer
        """
        # Get raw predictions
        predictions = self(x)
        
        # Process predictions for each layer
        results = {}
        
        for layer_name, layer_preds in predictions.items():
            batch_results = []
            
            for pred in layer_preds:
                # Reshape prediction untuk kompatibilitas dengan NMS
                bs, na, h, w, no = pred.shape
                pred = pred.view(bs, -1, no)  # (batch, anchors*grid_h*grid_w, no)
                
                layer_detections = []
                
                for i in range(bs):  # Loop untuk setiap gambar dalam batch
                    # Extract prediksi untuk satu gambar
                    img_pred = pred[i]
                    
                    # Filter berdasarkan confidence threshold
                    conf_mask = img_pred[:, 4] > conf_threshold
                    detections = img_pred[conf_mask]
                    
                    # Skip jika tidak ada deteksi
                    if detections.shape[0] == 0:
                        layer_detections.append([])
                        continue
                    
                    # Extract class scores dan class IDs
                    class_scores, class_ids = detections[:, 5:].max(1, keepdim=True)
                    
                    # Gabungkan deteksi dalam format [x, y, w, h, conf, cls]
                    detections = torch.cat([
                        detections[:, :5],  # box coords dan confidence
                        class_ids.float(),  # class ID
                        class_scores       # class score
                    ], dim=1)
                    
                    # Terapkan Non-Maximum Suppression menggunakan fungsi yang ada
                    from smartcash.model.utils.metrics.metrics_nms import non_max_suppression
                    
                    nms_out = non_max_suppression(
                        detections.unsqueeze(0),  # Add batch dim for NMS function
                        conf_threshold,
                        nms_threshold,
                        max_det=max_detections
                    )
                    
                    # Store hasil NMS
                    layer_detections.append(nms_out[0] if nms_out else [])
                
                batch_results.append(layer_detections)
            
            # Konversi deteksi menjadi format Dictionary yang lebih informatif
            formatted_results = []
            
            for batch_idx, batch_det in enumerate(batch_results):
                batch_formatted = []
                
                for img_idx, img_det in enumerate(batch_det):
                    # Skip jika tidak ada deteksi
                    if isinstance(img_det, torch.Tensor) and img_det.shape[0] > 0:
                        img_results = []
                        
                        for *xyxy, conf, cls_id in img_det:
                            # Format hasil menjadi dictionary
                            det_dict = {
                                'bbox': [coord.item() for coord in xyxy],  # x1, y1, x2, y2
                                'confidence': conf.item(),
                                'class_id': int(cls_id.item()),
                                'layer': layer_name
                            }
                            img_results.append(det_dict)
                            
                        batch_formatted.append(img_results)
                    else:
                        # Tidak ada deteksi untuk gambar ini
                        batch_formatted.append([])
                
                formatted_results.append(batch_formatted)
            
            results[layer_name] = formatted_results
            
        return results
    
    def get_optimizer(self, learning_rate=0.001, weight_decay=0.0005):
        """
        Buat optimizer untuk model.
        
        Args:
            learning_rate: Learning rate untuk optimizer
            weight_decay: Weight decay untuk optimizer
            
        Returns:
            Optimizer untuk model
        """
        # Split parameter untuk grup berbeda
        backbone_params = []
        head_params = []
        
        # Iterasi parameter model dan kelompokkan berdasarkan nama
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Buat parameter groups dengan different learning rate
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Backbone dengan LR lebih rendah
            {'params': head_params, 'lr': learning_rate}             # Head dengan LR normal
        ]
        
        # Buat optimizer
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    
    def compute_loss(self, predictions, targets, loss_fn=None):
        """
        Hitung loss untuk prediksi dan target.
        
        Args:
            predictions: Prediksi dari model
            targets: Target untuk model
            loss_fn: Loss function (opsional)
            
        Returns:
            Loss value dan dictionary komponen loss
        """
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_components = {}
        
        # Gunakan loss function yang diberikan atau loss_fn class
        if loss_fn is None:
            if not hasattr(self, 'loss_fn'):
                return total_loss, loss_components
            loss_fn = self.loss_fn
        
        # Jika loss_fn adalah dict (per layer)
        if isinstance(loss_fn, dict):
            for layer_name, layer_loss_fn in loss_fn.items():
                if layer_name in predictions and layer_name in targets:
                    layer_pred = predictions[layer_name]
                    layer_target = targets[layer_name]
                    
                    # Hitung loss untuk layer ini
                    try:
                        layer_loss, components = layer_loss_fn(layer_pred, layer_target)
                        total_loss = total_loss + layer_loss
                        loss_components[layer_name] = components
                    except Exception as e:
                        print(f"⚠️ Error saat menghitung loss untuk layer {layer_name}: {str(e)}")
                        continue
        else:
            # Jika loss_fn adalah single function
            try:
                total_loss, loss_components = loss_fn(predictions, targets)
            except Exception as e:
                print(f"⚠️ Error saat menghitung loss: {str(e)}")
        
        return total_loss, loss_components