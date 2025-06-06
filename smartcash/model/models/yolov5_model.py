"""
File: smartcash/model/models/yolov5_model.py
Deskripsi: Simplified YOLOv5 model dengan training integration dan one-liner style
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

from smartcash.common.logger import get_logger
from smartcash.model.utils.metrics.metrics_nms import non_max_suppression

class YOLOv5Model(nn.Module):
    """Simplified YOLOv5 model dengan training integration dan efficient predictions"""
    
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module, 
                 detection_layers: List[str] = ['banknote'], layer_mode: str = 'single',
                 loss_fn: Optional[nn.Module] = None, config: Optional[Dict] = None,
                 logger = None, testing_mode: bool = False):
        """Inisialisasi dengan component integration dan validation"""
        super().__init__()
        self.logger = logger or get_logger(__name__)
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss_fn = loss_fn
        self.config = config or {}
        self.testing_mode = testing_mode
        
        # Validate dan set layer parameters
        self.layer_mode = layer_mode if layer_mode in ['single', 'multilayer'] else (
            self.logger.warning(f"⚠️ Layer mode '{layer_mode}' tidak valid, menggunakan 'single'") or 'single'
        )
        self.detection_layers = detection_layers or ['banknote']
        
        # Model properties dengan validation
        self.num_classes = self.config.get('num_classes', 7)
        self.img_size = self.config.get('img_size', (640, 640))
        
        # Validate multilayer configuration
        if self.layer_mode == 'multilayer' and len(self.detection_layers) < 2:
            self.logger.warning(
            f"⚠️ Mode multilayer dengan {len(self.detection_layers)} layers - tidak optimal"
        )
        
        self.logger.info(f"✅ YOLOv5Model initialized: {self.layer_mode} | {self.detection_layers} | {self.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """Efficient forward pass dengan feature processing"""
        # Feature extraction pipeline
        features = self.backbone(x)
        processed_features = self.neck(features)
        head_output = self.head(processed_features)
        
        # Filter output untuk single layer mode
        if self.layer_mode == 'single' and isinstance(head_output, dict) and len(head_output) > 1:
            # Return hanya layer pertama untuk mode single
            return {layer: head_output[layer] for layer in self.detection_layers[:1] if layer in head_output}
        
        return head_output
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25, nms_threshold: float = 0.45,
               max_detections: int = 300, layer_filter: Optional[List[str]] = None) -> List[torch.Tensor]:
        """Multi-layer prediction dengan hierarchical detection logic untuk 7 kelas utama"""
        # Testing mode dengan dummy results
        if self.testing_mode:
            batch_size = x.shape[0]
            return [torch.zeros((0, 6), device=x.device) for _ in range(batch_size)]
        
        # Auto-determine mode berdasarkan detection_layers length
        is_multi = len(self.detection_layers) >= 2
        predictions, batch_size = self(x), x.shape[0]
        
        # Layer-specific detections storage
        layer_detections = {layer: [[] for _ in range(batch_size)] for layer in self.detection_layers}
        
        # Process predictions untuk setiap layer
        for layer_name, layer_preds in predictions.items():
            for pred in layer_preds:
                bs, na, h, w, no = pred.shape
                pred = pred.contiguous().reshape(bs, -1, no)
                
                for i in range(bs):
                    img_pred = pred[i]
                    conf_mask = img_pred[:, 4] > conf_threshold
                    detections = img_pred[conf_mask]
                    
                    if detections.shape[0] == 0: continue
                    
                    # Format detections [x, y, w, h, conf, cls] - 7 kelas utama
                    class_scores, class_ids = detections[:, 5:].max(1, keepdim=True)
                    formatted_detections = torch.cat([detections[:, :4], detections[:, 4:5], class_ids.float()], dim=1)
                    
                    try:
                        nms_out = non_max_suppression(formatted_detections.unsqueeze(0), conf_threshold, nms_threshold, max_det=max_detections)
                        if len(nms_out) > 0 and len(nms_out[0]) > 0:
                            layer_detections[layer_name][i].append(nms_out[0])
                    except Exception as e:
                        self.logger.warning(f"⚠️ NMS error: {str(e)}")
        
        # Multi-layer fusion untuk 7 kelas utama
        if is_multi:
            return self._fuse_multilayer_predictions(layer_detections, batch_size, x.device)
        
        # Single layer mode - return primary layer saja
        primary_layer = self.detection_layers[0]
        return [torch.cat(layer_detections[primary_layer][i], dim=0) if layer_detections[primary_layer][i] 
                else torch.zeros((0, 6), device=x.device) for i in range(batch_size)]
    
    def _fuse_multilayer_predictions(self, layer_detections, batch_size, device) -> List[torch.Tensor]:
        """Fuse multilayer predictions dengan hierarchical logic untuk 7 kelas"""
        results = []
        
        for i in range(batch_size):
            # Layer 1 (banknote): Primary detections
            primary_dets = torch.cat(layer_detections.get('banknote', [[]])[i], dim=0) if layer_detections.get('banknote', [[]])[i] else torch.zeros((0, 6), device=device)
            
            # Layer 2 (nominal): Alternative predictions untuk low confidence atau missed detections
            if 'nominal' in layer_detections and layer_detections['nominal'][i]:
                nominal_dets = torch.cat(layer_detections['nominal'][i], dim=0)
                primary_dets = self._merge_alternative_predictions(primary_dets, nominal_dets, device)
            
            # Layer 3 (security): Validation boost untuk confidence scores
            if 'security' in layer_detections and layer_detections['security'][i] and len(primary_dets) > 0:
                security_dets = torch.cat(layer_detections['security'][i], dim=0)
                primary_dets = self._boost_confidence_with_security(primary_dets, security_dets)
            
            # Final NMS pada 7 kelas utama
            try:
                final_nms = non_max_suppression(primary_dets.unsqueeze(0), 0.2, 0.4, max_det=300) if len(primary_dets) > 0 else []
                if final_nms and len(final_nms[0]) > 0:
                    results.append(final_nms[0])
                else:
                    results.append(torch.zeros((0, 6), device=device))
            except:
                results.append(torch.zeros((0, 6), device=device))
        
        return results
    
    def _merge_alternative_predictions(self, primary_dets, nominal_dets, device) -> torch.Tensor:
        """Merge nominal sebagai alternatif untuk deteksi yang kurang confident"""
        if len(primary_dets) == 0: return nominal_dets
        if len(nominal_dets) == 0: return primary_dets
        
        # Compute IoU antara primary dan nominal
        from smartcash.model.utils.metrics.metrics_nms import box_iou
        ious = box_iou(nominal_dets[:, :4], primary_dets[:, :4])
        
        # Ambil nominal yang tidak overlap atau confidence lebih tinggi
        enhanced_dets = [primary_dets]
        for j, nom_det in enumerate(nominal_dets):
            max_iou, max_idx = ious[j].max(), ious[j].argmax()
            
            # Tidak overlap: tambahkan sebagai deteksi baru
            if max_iou < 0.3:
                enhanced_dets.append(nom_det.unsqueeze(0))
            # Overlap dengan confidence lebih tinggi: ganti
            elif nom_det[4] > primary_dets[max_idx][4] + 0.15:
                primary_dets[max_idx] = nom_det
        
        return torch.cat(enhanced_dets, dim=0) if len(enhanced_dets) > 1 else primary_dets
    
    def _boost_confidence_with_security(self, primary_dets, security_dets) -> torch.Tensor:
        """Boost confidence menggunakan security features sebagai validasi"""
        if len(primary_dets) == 0 or len(security_dets) == 0: return primary_dets
        
        from smartcash.model.utils.metrics.metrics_nms import box_iou
        ious = box_iou(security_dets[:, :4], primary_dets[:, :4])
        
        # Boost confidence untuk deteksi yang memiliki security features
        for i in range(len(primary_dets)):
            if ious[:, i].max() > 0.25:  # Ada security feature yang overlap
                boost = min(0.2, 0.1 * ious[:, i].max().item())
                primary_dets[i, 4] = min(0.95, primary_dets[i, 4] + boost)
        
        return primary_dets
    
    def compute_loss(self, predictions, targets, loss_fn=None):
        """Compute loss dengan fallback handling"""
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_components = {}
        
        # Use provided loss function atau class loss_fn
        loss_function = loss_fn or self.loss_fn
        if not loss_function: return total_loss, loss_components
        
        try:
            # Handle different loss function types
            if isinstance(loss_function, dict):
                # Per-layer loss functions
                for layer_name, layer_loss_fn in loss_function.items():
                    if layer_name in predictions and layer_name in targets:
                        try:
                            layer_loss, components = layer_loss_fn(predictions[layer_name], targets[layer_name])
                            total_loss = total_loss + layer_loss
                            loss_components[layer_name] = components
                        except Exception as e:
                            self.logger.warning(f"⚠️ Layer {layer_name} loss error: {str(e)}")
                            continue
            else:
                # Single loss function
                result = loss_function(predictions, targets)
                if isinstance(result, tuple):
                    total_loss, loss_components = result
                else:
                    total_loss = result
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Loss computation error: {str(e)}")
            # Return minimal loss untuk prevent training crash
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)
            
        return total_loss, loss_components
    
    def get_layer_predictions(self, x: torch.Tensor, conf_threshold: float = 0.25) -> Dict[str, List[torch.Tensor]]:
        """Get separated layer predictions untuk debugging dan visualization"""
        if self.testing_mode:
            batch_size = x.shape[0]
            return {layer: [torch.zeros((0, 6), device=x.device) for _ in range(batch_size)] for layer in self.detection_layers}
        
        predictions, batch_size = self(x), x.shape[0]
        layer_results = {layer: [[] for _ in range(batch_size)] for layer in self.detection_layers}
        
        # Process each layer separately
        for layer_name, layer_preds in predictions.items():
            for pred in layer_preds:
                bs, na, h, w, no = pred.shape
                pred = pred.contiguous().reshape(bs, -1, no)
                
                for i in range(bs):
                    img_pred = pred[i]
                    conf_mask = img_pred[:, 4] > conf_threshold
                    detections = img_pred[conf_mask]
                    
                    if detections.shape[0] == 0: continue
                    
                    class_scores, class_ids = detections[:, 5:].max(1, keepdim=True)
                    formatted = torch.cat([detections[:, :4], detections[:, 4:5], class_ids.float()], dim=1)
                    
                    try:
                        nms_out = non_max_suppression(formatted.unsqueeze(0), conf_threshold, 0.45, max_det=300)
                        if len(nms_out) > 0 and len(nms_out[0]) > 0:
                            layer_results[layer_name][i].append(nms_out[0])
                    except: continue
        
        # Combine per layer dan batch
        return {layer: [torch.cat(layer_results[layer][i], dim=0) if layer_results[layer][i] else torch.zeros((0, 6), device=x.device) 
                       for i in range(batch_size)] for layer in layer_results}