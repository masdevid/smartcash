import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union
from ultralytics.utils.ops import non_max_suppression
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
        # Handle device selection properly
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
        self.current_phase = 1
        self.model = self._create_model(pretrained)
        self.logger.info(f"✅ Created SmartCashYOLOv5Model: {backbone}, {num_classes} classes, phase {self.current_phase}")
    
    def _create_model(self, pretrained: bool) -> nn.Module:
        if self.backbone_type == "efficientnet_b4":
            return EfficientNetBackbone(self.num_classes, pretrained, self.device)
        return YOLOv5Backbone(self.backbone_type, self.num_classes, pretrained, self.device)
    
    def _create_class_mapping(self) -> Dict[int, int]:
        mapping = {i: i for i in range(7)}
        for i in range(7):
            mapping[7 + i] = i
        mapping.update({14: -1, 15: -1, 16: -1})
        return mapping
    
    def forward(self, x: torch.Tensor, training: bool = None) -> Union[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if training is None:
            training = self.training
        output = self.model(x)
        return output if training else self._process_inference_output(output, x.shape)
    
    def _process_inference_output(self, output: torch.Tensor, input_shape: Tuple[int, ...]) -> List[Dict[str, torch.Tensor]]:
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
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'phase': self.current_phase,
            'backbone_frozen': not all(p.requires_grad for p in self.model.parameters()),
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params
        }
    
    def get_model_config(self) -> Dict:
        return {
            'backbone': self.backbone_type,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'phase': self.current_phase,
            'class_mapping': self.class_mapping,
            'architecture': 'SmartCashYOLOv5Model'
        }
    
    def setup_phase_2(self):
        self.current_phase = 2
        self.model.setup_phase_2(self.model)
    
    @staticmethod
    def get_supported_backbones() -> List[str]:
        """Get list of supported YOLOv5 backbones"""
        return ["yolov5s", "yolov5m", "yolov5l", "yolov5x", "efficientnet_b4"]
    
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