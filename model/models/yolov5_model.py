"""
File: smartcash/model/models/yolov5_model.py
Deskripsi: Simplified YOLOv5 model dengan training integration dan one-liner style
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any

from smartcash.common.logger import get_logger
from smartcash.model.utils.metrics.metrics_nms import non_max_suppression
from smartcash.model.config.model_constants import LAYER_CONFIG, DETECTION_THRESHOLDS

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
        self.layer_mode == 'multilayer' and len(self.detection_layers) < 2 and self.logger.warning(
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
        """Efficient prediction dengan post-processing dan multi-layer integration"""
        # Testing mode dengan dummy results
        if self.testing_mode:
            batch_size = x.shape[0]
            return [torch.zeros((0, 6), device=x.device) for _ in range(batch_size)]
        
        # Normal prediction pipeline
        predictions = self(x)
        batch_size = x.shape[0]
        
        # Struktur untuk menyimpan deteksi dari setiap layer
        layer_detections = {layer: [[] for _ in range(batch_size)] for layer in self.detection_layers}
        
        # Process predictions untuk setiap layer
        for layer_name, layer_preds in predictions.items():
            # Skip jika layer tidak dalam filter (jika filter diberikan)
            if layer_filter and layer_name not in layer_filter:
                continue
                
            # Gunakan threshold spesifik untuk layer jika tersedia
            layer_conf_threshold = DETECTION_THRESHOLDS.get(layer_name, conf_threshold)
            
            for pred in layer_preds:
                bs, na, h, w, no = pred.shape
                pred = pred.contiguous().reshape(bs, -1, no)
                
                # Process per batch
                for i in range(bs):
                    img_pred = pred[i]
                    conf_mask = img_pred[:, 4] > layer_conf_threshold
                    detections = img_pred[conf_mask]
                    
                    if detections.shape[0] == 0: continue
                    
                    # Extract class scores dan IDs
                    class_scores, class_ids = detections[:, 5:].max(1, keepdim=True)
                    
                    # Format detections [x, y, w, h, conf, cls, layer_idx]
                    # Tambahkan layer_idx untuk tracking sumber layer (0=banknote, 1=nominal, 2=security)
                    layer_idx = torch.full((class_ids.shape[0], 1), self.detection_layers.index(layer_name), 
                                          device=class_ids.device, dtype=torch.float)
                    
                    formatted_detections = torch.cat([
                        detections[:, :4],  # bbox
                        detections[:, 4:5], # confidence
                        class_ids.float(),  # class_id
                        layer_idx           # layer_idx (untuk tracking)
                    ], dim=1)
                    
                    # Apply NMS dengan error handling
                    try:
                        # NMS hanya menggunakan 6 kolom pertama, kolom ke-7 (layer_idx) dipertahankan
                        nms_out = non_max_suppression(
                            formatted_detections[:, :6].unsqueeze(0), 
                            layer_conf_threshold, nms_threshold, 
                            max_det=max_detections
                        )
                        
                        if len(nms_out) > 0 and len(nms_out[0]) > 0:
                            # Cari indeks yang dipertahankan oleh NMS
                            kept_indices = []
                            for box in nms_out[0]:
                                # Cari box yang sama di formatted_detections
                                matches = ((formatted_detections[:, :6] == box).all(dim=1).nonzero(as_tuple=True)[0])
                                if len(matches) > 0:
                                    kept_indices.append(matches[0].item())
                            
                            # Ambil deteksi lengkap termasuk layer_idx
                            if kept_indices:
                                kept_detections = formatted_detections[kept_indices]
                                layer_detections[layer_name][i].append(kept_detections)
                    except Exception as e:
                        self.logger.warning(f"⚠️ NMS error pada layer {layer_name}: {str(e)}")
                        continue
        
        # Implementasi multi-layer detection logic
        results = []
        for i in range(batch_size):
            # Kumpulkan deteksi dari semua layer untuk gambar ini
            image_detections = []
            
            # Proses untuk mode multilayer
            if self.layer_mode == 'multilayer' and len(self.detection_layers) > 1:
                # 1. Ambil deteksi dari layer utama (banknote)
                primary_dets = []
                if 'banknote' in layer_detections and layer_detections['banknote'][i]:
                    for dets in layer_detections['banknote'][i]:
                        primary_dets.append(dets)
                
                primary_boxes = []
                if primary_dets:
                    primary_dets = torch.cat(primary_dets, dim=0)
                    # Simpan bbox untuk matching dengan layer lain
                    primary_boxes = primary_dets[:, :4].clone()
                    image_detections.append(primary_dets)
                
                # 2. Proses layer 'nominal' sebagai alternatif saat layer 'banknote' tidak yakin
                if 'nominal' in layer_detections and layer_detections['nominal'][i]:
                    nominal_dets = []
                    for dets in layer_detections['nominal'][i]:
                        nominal_dets.append(dets)
                    
                    if nominal_dets:
                        nominal_dets = torch.cat(nominal_dets, dim=0)
                        
                        # Jika ada primary detections, filter nominal yang tumpang tindih
                        if len(primary_boxes) > 0:
                            # Hitung IoU antara nominal dan primary boxes
                            from smartcash.model.utils.metrics.metrics_nms import box_iou
                            ious = box_iou(nominal_dets[:, :4], primary_boxes)
                            
                            # Ambil nominal detections yang tidak tumpang tindih dengan primary
                            # atau yang memiliki confidence lebih tinggi
                            for j, nominal_det in enumerate(nominal_dets):
                                if ious[j].max() < 0.5:  # Tidak tumpang tindih
                                    # Konversi class id dari nominal ke banknote (keduanya 7 kelas)
                                    # Format: [x, y, w, h, conf, cls, layer_idx]
                                    nominal_det_copy = nominal_det.clone()
                                    # Ubah layer_idx menjadi 0 (banknote) untuk konsistensi output
                                    nominal_det_copy[6] = 0
                                    image_detections.append(nominal_det_copy.unsqueeze(0))
                                else:
                                    # Tumpang tindih, bandingkan confidence
                                    max_iou_idx = ious[j].argmax()
                                    if nominal_det[4] > primary_dets[max_iou_idx][4] + 0.1:  # Confidence lebih tinggi dengan margin
                                        # Ganti dengan nominal detection yang lebih confident
                                        nominal_det_copy = nominal_det.clone()
                                        nominal_det_copy[6] = 0  # Ubah layer_idx menjadi 0 (banknote)
                                        image_detections.append(nominal_det_copy.unsqueeze(0))
                        else:
                            # Tidak ada primary detections, gunakan semua nominal
                            for nominal_det in nominal_dets:
                                nominal_det_copy = nominal_det.clone()
                                nominal_det_copy[6] = 0  # Ubah layer_idx menjadi 0 (banknote)
                                image_detections.append(nominal_det_copy.unsqueeze(0))
                
                # 3. Gunakan layer 'security' untuk validasi tambahan
                if 'security' in layer_detections and layer_detections['security'][i]:
                    security_dets = []
                    for dets in layer_detections['security'][i]:
                        security_dets.append(dets)
                    
                    if security_dets:
                        security_dets = torch.cat(security_dets, dim=0)
                        
                        # Gunakan security detections untuk meningkatkan confidence deteksi yang ada
                        all_current_dets = []
                        for dets in image_detections:
                            if dets.dim() == 1:
                                all_current_dets.append(dets.unsqueeze(0))
                            else:
                                all_current_dets.append(dets)
                        
                        if all_current_dets:
                            all_current_dets = torch.cat(all_current_dets, dim=0)
                            
                            # Hitung IoU antara security dan current boxes
                            from smartcash.model.utils.metrics.metrics_nms import box_iou
                            ious = box_iou(security_dets[:, :4], all_current_dets[:, :4])
                            
                            # Tingkatkan confidence untuk deteksi yang tumpang tindih dengan security features
                            for j, current_det in enumerate(all_current_dets):
                                if ious[:, j].max() > 0.3:  # Ada fitur keamanan yang tumpang tindih
                                    # Tingkatkan confidence (max +0.15)
                                    confidence_boost = min(0.15, 0.05 * ious[:, j].max().item() * 3)
                                    current_det[4] = min(0.99, current_det[4] + confidence_boost)
                            
                            # Update image_detections
                            image_detections = [all_current_dets]
            else:
                # Mode single layer, gunakan deteksi dari layer pertama saja
                primary_layer = self.detection_layers[0]
                if primary_layer in layer_detections and layer_detections[primary_layer][i]:
                    for dets in layer_detections[primary_layer][i]:
                        image_detections.append(dets)
            
            # Gabungkan semua deteksi dan terapkan NMS final
            if image_detections:
                try:
                    combined = torch.cat(image_detections, dim=0)
                    
                    # Terapkan NMS final untuk menghilangkan duplikasi
                    final_dets = non_max_suppression(
                        combined[:, :6].unsqueeze(0),  # Gunakan 6 kolom pertama untuk NMS
                        conf_threshold, nms_threshold,
                        max_det=max_detections
                    )
                    
                    if len(final_dets) > 0 and len(final_dets[0]) > 0:
                        results.append(final_dets[0])
                    else:
                        results.append(torch.zeros((0, 6), device=x.device))
                except Exception as e:
                    self.logger.warning(f"⚠️ Final NMS error: {str(e)}")
                    results.append(torch.zeros((0, 6), device=x.device))
            else:
                results.append(torch.zeros((0, 6), device=x.device))
        
        return results
    
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
        """
        Mendapatkan prediksi terpisah untuk setiap layer tanpa menggabungkan.
        Berguna untuk debugging dan visualisasi multi-layer detection.
        
        Args:
            x: Input tensor dengan shape (batch_size, channels, height, width)
            conf_threshold: Confidence threshold untuk memfilter deteksi
            
        Returns:
            Dict berisi prediksi untuk setiap layer
        """
        # Testing mode dengan dummy results
        if self.testing_mode:
            batch_size = x.shape[0]
            return {layer: [torch.zeros((0, 6), device=x.device) for _ in range(batch_size)] 
                    for layer in self.detection_layers}
        
        # Forward pass
        predictions = self(x)
        batch_size = x.shape[0]
        
        # Hasil untuk setiap layer
        results = {layer: [[] for _ in range(batch_size)] for layer in self.detection_layers}
        
        # Process predictions untuk setiap layer
        for layer_name, layer_preds in predictions.items():
            # Gunakan threshold spesifik untuk layer jika tersedia
            layer_conf_threshold = DETECTION_THRESHOLDS.get(layer_name, conf_threshold)
            
            for pred in layer_preds:
                bs, na, h, w, no = pred.shape
                pred = pred.contiguous().reshape(bs, -1, no)
                
                # Process per batch
                for i in range(bs):
                    img_pred = pred[i]
                    conf_mask = img_pred[:, 4] > layer_conf_threshold
                    detections = img_pred[conf_mask]
                    
                    if detections.shape[0] == 0: 
                        continue
                    
                    # Extract class scores dan IDs
                    class_scores, class_ids = detections[:, 5:].max(1, keepdim=True)
                    
                    # Format detections [x, y, w, h, conf, cls]
                    formatted_detections = torch.cat([
                        detections[:, :4], detections[:, 4:5], class_ids.float()
                    ], dim=1)
                    
                    # Apply NMS
                    try:
                        nms_out = non_max_suppression(
                            formatted_detections.unsqueeze(0), 
                            layer_conf_threshold, 0.45, 
                            max_det=300
                        )
                        
                        if len(nms_out) > 0 and len(nms_out[0]) > 0:
                            results[layer_name][i].append(nms_out[0])
                    except Exception as e:
                        self.logger.warning(f"⚠️ NMS error pada layer {layer_name}: {str(e)}")
                        continue
        
        # Gabungkan deteksi untuk setiap layer dan batch
        final_results = {}
        for layer_name in results:
            layer_results = []
            for i in range(batch_size):
                if results[layer_name][i]:
                    combined = torch.cat(results[layer_name][i], dim=0)
                    layer_results.append(combined)
                else:
                    layer_results.append(torch.zeros((0, 6), device=x.device))
            final_results[layer_name] = layer_results
            
        return final_results