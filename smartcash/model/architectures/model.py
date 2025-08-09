import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from ultralytics import YOLO
from smartcash.model.architectures.backbones.yolov5_backbone import YOLOv5Backbone
from smartcash.model.architectures.backbones.efficientnet_backbone import EfficientNetBackbone
from smartcash.common.logger import get_logger

class SmartCashYOLOv5Model(nn.Module):
    """SmartCash YOLOv5 model with 17-class training and 7-class inference mapping"""
    
    def __init__(self, backbone: str = "yolov5s", num_classes: int = 17, img_size: int = 640, pretrained: bool = True, device: str = "auto"):
        super().__init__()
        self.logger = get_logger(__name__)
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone_type = backbone
        self.current_phase = 1
        
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.class_mapping = self._create_class_mapping()
        self.model = self._create_model(pretrained)
        
        # Get parameter count based on model type
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
            param_count = sum(p.numel() for p in self.model.model.parameters())
        else:
            param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"✅ SmartCashYOLOv5Model: {backbone} ({param_count:,} params, phase {self.current_phase})")
    
    def _create_model(self, pretrained: bool) -> Union[nn.Module, YOLO]:
        """Create model based on backbone type - supports research comparison between backbones"""
        if self.backbone_type == "efficientnet_b4":
            # Use custom EfficientNet-B4 backbone for research
            model = EfficientNetBackbone(self.num_classes, pretrained, self.device)
            self._set_phase_1_on_backbone(model)
            return model
        else:
            # Fallback to custom YOLOv5 backbone
            model = YOLOv5Backbone(self.backbone_type, self.num_classes, pretrained, self.device)
            self._set_phase_1_on_backbone(model)
            return model
    
    def _reinitialize_detection_head(self, detect_layer):
        """Reinitialize detection head weights for new class count"""
        if hasattr(detect_layer, 'm') and isinstance(detect_layer.m, nn.ModuleList):
            for conv in detect_layer.m:
                if hasattr(conv, 'weight'):
                    nn.init.normal_(conv.weight, 0, 0.01)
                if hasattr(conv, 'bias') and conv.bias is not None:
                    nn.init.constant_(conv.bias, 0)
    
    def _set_phase_1_on_model(self, model):
        """Phase 1: Freeze backbone, train detection head only - for YOLO models"""
        for name, param in model.model.named_parameters():
            if 'model.24' in name or 'head' in name.lower():  # Detection head
                param.requires_grad = True
            else:  # Backbone and neck
                param.requires_grad = False
        self.logger.info("🔒 Phase 1: Backbone frozen, head trainable")
    
    def _set_phase_1_on_backbone(self, model):
        """Phase 1: Freeze backbone, train detection head only - for custom backbones"""
        # Custom backbones already handle phase setup in their constructors
        pass
    
    def _set_phase_1(self):
        """Phase 1: Freeze backbone, train detection head only"""
        if hasattr(self.model, 'model'):  # YOLO model
            self._set_phase_1_on_model(self.model)
        else:  # Custom backbone
            self._set_phase_1_on_backbone(self.model)
    
    def _set_phase_2(self):
        """Phase 2: Unfreeze entire model for fine-tuning"""
        if hasattr(self.model, 'model'):  # YOLO model
            for param in self.model.model.parameters():
                param.requires_grad = True
        else:  # Custom backbone
            if hasattr(self.model, 'setup_phase_2'):
                self.model.setup_phase_2(self.model)
            else:
                for param in self.model.parameters():
                    param.requires_grad = True
        self.logger.info("🔓 Phase 2: Full model trainable")
    
    def _create_class_mapping(self) -> Dict[int, int]:
        mapping = {i: i for i in range(7)}
        for i in range(7):
            mapping[7 + i] = i
        mapping.update({14: -1, 15: -1, 16: -1})
        return mapping
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Forward pass with training/inference mode handling"""
        if self.training:
            # Training mode: return raw outputs for loss computation
            if hasattr(self.model, 'model'):  # YOLO model
                return self.model.model(x)
            else:  # Custom backbone
                return self.model(x)
        else:
            # Inference mode: return processed predictions with class mapping
            if hasattr(self.model, 'model'):  # YOLO model
                results = self.model(x, verbose=False)
                return self._process_inference_results(results)
            else:  # Custom backbone
                output = self.model(x)
                return self._process_inference_output_from_tensor(output, x.shape)
    
    def _process_inference_results(self, results) -> List[Dict[str, torch.Tensor]]:
        """Process YOLO inference results with 17→7 class mapping"""
        processed_results = []
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                processed_results.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'denomination_scores': torch.zeros(7)
                })
                continue
                
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            classes = result.boxes.cls.long()
            
            mapped_result = self._map_classes_and_adjust_confidence(boxes, scores, classes)
            processed_results.append(mapped_result)
            
        return processed_results
    
    def _process_inference_output_from_tensor(self, output: torch.Tensor, input_shape: Tuple[int, ...]) -> List[Dict[str, torch.Tensor]]:
        """Process tensor output from custom backbones"""
        from ultralytics.utils.ops import non_max_suppression
        
        predictions = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)
        processed_results = []
        
        for pred in predictions:
            if pred is None or len(pred) == 0:
                processed_results.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'denomination_scores': torch.zeros(7)
                })
                continue
                
            boxes, scores, classes = pred[:, :4], pred[:, 4], pred[:, 5].long()
            processed_results.append(self._map_classes_and_adjust_confidence(boxes, scores, classes))
            
        return processed_results
    
    def _map_classes_and_adjust_confidence(self, boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor) -> Dict[str, torch.Tensor]:
        denomination_detections = {i: [] for i in range(7)}
        authenticity_scores = []
        for i in range(len(classes)):
            cls = classes[i].item()
            mapped_cls = self.class_mapping.get(cls, -1)
            if mapped_cls >= 0:
                denomination_detections[mapped_cls].append({'box': boxes[i], 'score': scores[i].item(), 'source_class': cls})
            elif cls >= 14:
                authenticity_scores.append(scores[i].item())
        final_boxes, final_scores, final_labels = [], [], []
        denomination_scores = torch.zeros(7)
        for denom_id in range(7):
            detections = denomination_detections[denom_id]
            if not detections:
                continue
            best_det = max(detections, key=lambda x: x['score'])
            adjusted_score = self._adjust_confidence_with_supporting_evidence(best_det, detections, authenticity_scores)
            final_boxes.append(best_det['box'])
            final_scores.append(adjusted_score)
            final_labels.append(denom_id)
            denomination_scores[denom_id] = adjusted_score
        return {
            'boxes': torch.stack(final_boxes) if final_boxes else torch.empty(0, 4),
            'scores': torch.tensor(final_scores) if final_scores else torch.empty(0),
            'labels': torch.tensor(final_labels, dtype=torch.long) if final_labels else torch.empty(0, dtype=torch.long),
            'denomination_scores': denomination_scores
        }
    
    def _adjust_confidence_with_supporting_evidence(self, primary_detection: Dict, all_detections: List[Dict], authenticity_scores: List[float]) -> float:
        base_confidence = primary_detection['score']
        layer2_boost = max((det['score'] * 0.1 for det in all_detections if 7 <= det['source_class'] <= 13), default=0.0)
        auth_boost = (sum(authenticity_scores) / len(authenticity_scores)) * 0.15 if authenticity_scores else 0.0
        adjusted = base_confidence + layer2_boost + auth_boost
        if not authenticity_scores and base_confidence > 0.5:
            adjusted *= 0.8
        return min(adjusted, 1.0)
    
    def get_phase_info(self) -> Dict:
        """Get current phase information and parameter counts"""
        if hasattr(self.model, 'model'):  # YOLO model
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.model.parameters())
        else:  # Custom backbone
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
        return {
            'phase': self.current_phase,
            'backbone_frozen': self.current_phase == 1,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params,
            'model_size': self.backbone_type
        }
    
    def get_model_config(self) -> Dict:
        """Get model configuration"""
        if hasattr(self.model, 'model'):  # YOLO model
            total_params = sum(p.numel() for p in self.model.model.parameters())
        else:  # Custom backbone
            total_params = sum(p.numel() for p in self.model.parameters())
            
        return {
            'backbone': self.backbone_type,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'phase': self.current_phase,
            'class_mapping': self.class_mapping,
            'architecture': 'SmartCashYOLOv5Model',
            'total_params': total_params,
        }
    
    def setup_phase_2(self):
        """Switch to Phase 2: full model fine-tuning"""
        self.current_phase = 2
        self._set_phase_2()
    
    @staticmethod
    def get_supported_backbones() -> List[str]:
        """Get list of supported backbones for research comparison"""
        return ["cspdarknet", "efficientnet_b4", "yolov5s", "yolov5n", "yolov5m"]  # Optimized + research options
    
    @staticmethod
    def get_backbone_info() -> Dict[str, Dict]:
        """Get information about each backbone for research comparison"""
        return {
            "yolov5s": {"params": "~9.2M", "speed": "Fast", "use_case": "YOLOv5 baseline"},
            "yolov5n": {"params": "~2.7M", "speed": "Fastest", "use_case": "Lightweight option"},
            "cspdarknet": {"params": "~7.9M", "speed": "Fast", "use_case": "Optimized CSP backbone"}, 
            "efficientnet_b4": {"params": "~17.4M", "speed": "Medium", "use_case": "Optimized EfficientNet (reduced bloat)"},
            "yolov5m": {"params": "~21M", "speed": "Medium", "use_case": "High accuracy when needed"}
        }
    
    @staticmethod
    def get_class_names() -> Dict[str, List[str]]:
        """Get class name mappings"""
        return {
            'training_classes': [
                '001', '002', '005', '010', '020', '050', '100',
                'l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100',
                'l3_sign', 'l3_text', 'l3_thread'
            ],
            'inference_classes': ['001', '002', '005', '010', '020', '050', '100']
        }