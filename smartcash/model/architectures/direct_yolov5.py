"""
Direct Ultralytics YOLOv5 Integration for SmartCash
Simplified architecture without wrapper layers or compatibility systems.

Training: 17 classes (fine-grained detection)
- Layer 1: 0-6 (whole bill denominations) 
- Layer 2: 7-13 (denomination-specific features)
- Layer 3: 14-16 (authenticity/security features)

Inference: Map to 7 main denominations post-prediction
- Classes 0-6 directly map to denominations
- Classes 7-13 & 14-16 used for confirmation scoring
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Import Ultralytics YOLOv5
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import initialize_weights
    from ultralytics.utils.ops import non_max_suppression, scale_boxes
    YOLOV5_AVAILABLE = True
except ImportError:
    try:
        # Fallback to torch.hub for ultralytics/yolov5
        import torch
        YOLO = None
        DetectionModel = None
        YOLOV5_AVAILABLE = True
    except ImportError:
        print("Warning: Neither ultralytics nor torch.hub YOLOv5 available")
        YOLOV5_AVAILABLE = False
        YOLO = None
        DetectionModel = None

# Import timm for EfficientNet backbone
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm not available - EfficientNet backbones disabled")
    TIMM_AVAILABLE = False

from smartcash.common.logger import SmartCashLogger


class SmartCashYOLOv5Model(nn.Module):
    """
    SmartCash YOLOv5 model with 17-class training and 7-class inference mapping
    """
    
    def __init__(
        self, 
        backbone: str = "yolov5s",
        num_classes: int = 17,
        img_size: int = 640,
        pretrained: bool = True,
        device: str = "auto"
    ):
        """
        Initialize direct YOLOv5 model
        
        Args:
            backbone: YOLOv5 variant (yolov5s, yolov5m, yolov5l, yolov5x)
            num_classes: Number of training classes (17 for SmartCash)
            img_size: Input image size
            pretrained: Use pretrained weights
            device: Device to use
        """
        super().__init__()
        
        self.logger = SmartCashLogger(__name__)
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone = backbone
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Class mapping for inference (17 → 7)
        self.class_mapping = self._create_class_mapping()
        
        # Phase configuration
        self.current_phase = 1
        self.frozen_backbone = True
        
        # Initialize YOLOv5 model
        self.model = self._create_yolo_model(pretrained)
        
        self.logger.info(f"✅ Created SmartCashYOLOv5Model: {backbone}, {num_classes} classes, phase {self.current_phase}")
        
    def _create_yolo_model(self, pretrained: bool):
        """Create the underlying YOLOv5 model using Ultralytics"""
        if not YOLOV5_AVAILABLE:
            raise RuntimeError("YOLOv5 is not available")
            
        try:
            if YOLO is not None and self.backbone != "efficientnet_b4":
                # Create YOLOv5s using our working EfficientNet approach as template
                self.logger.info(f"🔄 Creating custom YOLOv5s architecture with {self.num_classes} classes")
                
                # Import required modules
                from ultralytics.nn.modules import C2f, SPPF, Concat, Detect, Conv
                
                # Create a ResNet backbone similar to YOLOv5s structure
                if self.backbone == 'yolov5s':
                    # YOLOv5s channel progression: [32, 64, 128, 256, 512]
                    channels = [32, 64, 128, 256, 512]
                else:
                    # Default for other variants
                    channels = [64, 128, 256, 512, 1024]
                
                # Build custom YOLOv5-style backbone
                backbone_layers = nn.ModuleList([
                    # Input conv
                    Conv(3, channels[0], 6, 2, 2),  # 640x640 -> 320x320
                    Conv(channels[0], channels[1], 3, 2),  # 320x320 -> 160x160
                    C2f(channels[1], channels[1], 1, True),  # P2
                    Conv(channels[1], channels[2], 3, 2),  # 160x160 -> 80x80  
                    C2f(channels[2], channels[2], 2, True),  # P3
                    Conv(channels[2], channels[3], 3, 2),  # 80x80 -> 40x40
                    C2f(channels[3], channels[3], 3, True),  # P4
                    Conv(channels[3], channels[4], 3, 2),  # 40x40 -> 20x20
                    C2f(channels[4], channels[4], 1, True),  # P5
                    SPPF(channels[4], channels[4], 5)  # SPPF
                ])
                
                # Build YOLOv5 neck (PANet) - fix channel calculations
                neck_layers = nn.ModuleList([
                    # P5 -> P4 path
                    Conv(channels[4], channels[3], 1, 1),  # 512 -> 256
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    Concat(1),
                    C2f(channels[3] + channels[3], channels[3], 1, False),  # 256+256=512 -> 256
                    
                    # P4 -> P3 path  
                    Conv(channels[3], channels[2], 1, 1),  # 256 -> 128
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    Concat(1),
                    C2f(channels[2] + channels[2], channels[2], 1, False),  # 128+128=256 -> 128
                    
                    # P3 -> P4 path
                    Conv(channels[2], channels[2], 3, 2),  # 128 -> 128 (downsample)
                    Concat(1),
                    C2f(channels[2] + channels[3], channels[3], 1, False),  # 128+256=384 -> 256
                    
                    # P4 -> P5 path  
                    Conv(channels[3], channels[3], 3, 2),  # 256 -> 256 (downsample)
                    Concat(1),
                    C2f(channels[3] + channels[3], channels[4], 1, False),  # 256+256=512 -> 512
                ])
                
                # Create detection head for 17 classes
                detection_head = Detect(
                    nc=self.num_classes,
                    ch=[channels[2], channels[3], channels[4]]  # P3, P4, P5 channels
                )
                
                # Create custom YOLOv5 model
                class CustomYOLOv5(nn.Module):
                    def __init__(self, backbone, neck, head, num_classes):
                        super().__init__()
                        self.backbone = backbone
                        self.neck = neck  
                        self.head = head
                        self.nc = num_classes
                        
                    def forward(self, x):
                        # Forward through backbone
                        features = []
                        for i, layer in enumerate(self.backbone):
                            x = layer(x)
                            if i in [4, 6, 9]:  # P3, P4, P5 positions
                                features.append(x)
                        
                        p3, p4, p5 = features
                        
                        # P5 -> P4 upsampling path
                        p5_conv = self.neck[0](p5)
                        p5_up = self.neck[1](p5_conv)
                        p4_cat = self.neck[2]([p5_up, p4])
                        p4_out = self.neck[3](p4_cat)
                        
                        # P4 -> P3 upsampling path
                        p4_conv = self.neck[4](p4_out) 
                        p4_up = self.neck[5](p4_conv)
                        p3_cat = self.neck[6]([p4_up, p3])
                        p3_out = self.neck[7](p3_cat)
                        
                        # P3 -> P4 downsampling path
                        p3_down = self.neck[8](p3_out)
                        p4_cat2 = self.neck[9]([p3_down, p4_out])
                        p4_final = self.neck[10](p4_cat2)
                        
                        # P4 -> P5 downsampling path
                        p4_down = self.neck[11](p4_final)
                        p5_cat = self.neck[12]([p4_down, p5_conv])  # Connect to P5 conv output (256 channels)
                        p5_final = self.neck[13](p5_cat)
                        
                        # Detection head
                        return self.head([p3_out, p4_final, p5_final])
                
                model = CustomYOLOv5(backbone_layers, neck_layers, detection_head, self.num_classes)
                self.logger.info(f"✅ Created custom {self.backbone} with {self.num_classes} classes")
                
            elif self.backbone == "efficientnet_b4":
                # Handle real EfficientNet-B4 backbone using timm
                if not TIMM_AVAILABLE:
                    raise RuntimeError("timm is required for EfficientNet-B4 backbone")
                
                self.logger.info("🔄 Creating real EfficientNet-B4 backbone using timm")
                
                # Create EfficientNet-B4 backbone using timm
                efficientnet_b4 = timm.create_model(
                    'efficientnet_b4', 
                    pretrained=pretrained,
                    features_only=True,  # Get feature maps, not classification head
                    out_indices=[2, 3, 4]  # Get P3, P4, P5 equivalent feature maps
                )
                
                # Get the feature dimensions from EfficientNet-B4
                # EfficientNet-B4 feature dimensions: [56, 160, 448] for stages 2,3,4
                feature_dims = [56, 160, 448]
                
                # Create YOLOv5 neck (PANet) and detection head
                from ultralytics.nn.modules import C2f, SPPF, Concat, Detect
                
                # Build neck layers
                neck_layers = nn.ModuleList([
                    # P5 processing with SPPF
                    SPPF(feature_dims[2], feature_dims[2], 5),  # 0: SPPF
                    
                    # P5 to P4 upsampling path
                    nn.Upsample(scale_factor=2, mode='nearest'),  # 1: Upsample
                    Concat(1),  # 2: Concatenate P5_up with P4
                    C2f(feature_dims[2] + feature_dims[1], feature_dims[1], 1, False),  # 3: C2f
                    
                    # P4 to P3 upsampling path  
                    nn.Upsample(scale_factor=2, mode='nearest'),  # 4: Upsample
                    Concat(1),  # 5: Concatenate P4_up with P3
                    C2f(feature_dims[1] + feature_dims[0], feature_dims[0], 1, False),  # 6: C2f
                    
                    # P3 to P4 downsampling path
                    nn.Conv2d(feature_dims[0], feature_dims[0], 3, 2, 1, bias=False),  # 7: Conv
                    nn.BatchNorm2d(feature_dims[0]),  # 8: BN
                    nn.SiLU(),  # 9: Activation
                    Concat(1),  # 10: Concatenate P3_down with P4
                    C2f(feature_dims[0] + feature_dims[1], feature_dims[1], 1, False),  # 11: C2f
                    
                    # P4 to P5 downsampling path
                    nn.Conv2d(feature_dims[1], feature_dims[1], 3, 2, 1, bias=False),  # 12: Conv  
                    nn.BatchNorm2d(feature_dims[1]),  # 13: BN
                    nn.SiLU(),  # 14: Activation
                    Concat(1),  # 15: Concatenate P4_down with P5
                    C2f(feature_dims[1] + feature_dims[2], feature_dims[2], 1, False),  # 16: C2f
                ])
                
                # Create detection head
                detection_head = Detect(
                    nc=self.num_classes,
                    ch=[feature_dims[0], feature_dims[1], feature_dims[2]]  # P3, P4, P5 channels
                )
                
                # Create the complete model
                class EfficientNetYOLOv5(nn.Module):
                    def __init__(self, backbone, neck, head, num_classes):
                        super().__init__()
                        self.backbone = backbone
                        self.neck = neck
                        self.head = head
                        self.nc = num_classes
                        
                    def forward(self, x):
                        # Extract features from EfficientNet-B4 backbone
                        features = self.backbone(x)  # [P3, P4, P5]
                        
                        # Process through neck
                        p5 = features[2]
                        p4 = features[1] 
                        p3 = features[0]
                        
                        # SPPF on P5 (index 0)
                        p5 = self.neck[0](p5)
                        
                        # P5 -> P4 path
                        p5_up = self.neck[1](p5)  # Upsample (index 1)
                        p4_concat = self.neck[2]([p5_up, p4])  # Concat (index 2)
                        p4_out = self.neck[3](p4_concat)  # C2f (index 3)
                        
                        # P4 -> P3 path  
                        p4_up = self.neck[4](p4_out)  # Upsample (index 4)
                        p3_concat = self.neck[5]([p4_up, p3])  # Concat (index 5)
                        p3_out = self.neck[6](p3_concat)  # C2f (index 6)
                        
                        # P3 -> P4 path (indices 7, 8, 9, 10, 11)
                        p3_down = self.neck[9](self.neck[8](self.neck[7](p3_out)))  # Conv+BN+SiLU
                        p4_concat2 = self.neck[10]([p3_down, p4_out])  # Concat
                        p4_final = self.neck[11](p4_concat2)  # C2f
                        
                        # P4 -> P5 path (indices 12, 13, 14, 15, 16)
                        p4_down = self.neck[14](self.neck[13](self.neck[12](p4_final)))  # Conv+BN+SiLU
                        p5_concat = self.neck[15]([p4_down, p5])  # Concat  
                        p5_final = self.neck[16](p5_concat)  # C2f
                        
                        # Detection head
                        return self.head([p3_out, p4_final, p5_final])
                
                model = EfficientNetYOLOv5(efficientnet_b4, neck_layers, detection_head, self.num_classes)
                self.logger.info("✅ Real EfficientNet-B4 + YOLOv5 neck/head created")
                    
            else:
                # Fallback to torch.hub
                model = torch.hub.load('ultralytics/yolov5', self.backbone, pretrained=pretrained)
                
                # Modify for custom classes
                if hasattr(model.model, 'model') and len(model.model.model) > 0:
                    detection_layer = model.model.model[-1]
                    num_anchors = getattr(detection_layer, 'na', 3)
                    detection_layer.nc = self.num_classes
                    detection_layer.no = (self.num_classes + 5) * num_anchors
                    
                    # Reinitialize detection head with correct output channels
                    for i, m in enumerate(detection_layer.m):
                        if hasattr(m, 'weight') and hasattr(m, 'bias'):
                            out_channels = (self.num_classes + 5) * num_anchors
                            in_channels = m.in_channels
                            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
                            nn.init.zeros_(new_conv.weight)
                            nn.init.zeros_(new_conv.bias)
                            detection_layer.m[i] = new_conv
                        
            # Move to device
            if hasattr(model, 'to'):
                model = model.to(self.device)
            else:
                for param in model.parameters():
                    param.data = param.data.to(self.device)
                    
            # Set up phase 1: freeze backbone
            self._setup_phase_1(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create YOLOv5 model: {e}")
            raise
    
    def _load_pretrained_weights(self, model):
        """Load pretrained YOLOv5 weights and adapt for custom classes"""
        try:
            # Load pretrained weights
            pretrained_path = f"{self.backbone}.pt"
            
            # This will automatically download if not present
            ckpt = torch.hub.load('ultralytics/yolov5', self.backbone, pretrained=True, device='cpu')
            
            # Extract state dict
            state_dict = ckpt.state_dict()
            
            # Adapt detection head for custom number of classes
            model_state = model.state_dict()
            
            # Filter compatible weights (exclude detection head if classes differ)
            compatible_weights = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_weights[k] = v
                elif 'model.24' in k:  # Detection head
                    self.logger.info(f"Skipping detection head weight: {k} (shape mismatch)")
                else:
                    compatible_weights[k] = v
            
            # Load compatible weights
            model.load_state_dict(compatible_weights, strict=False)
            
            self.logger.info(f"✅ Loaded pretrained {self.backbone} weights with {self.num_classes} classes")
            
        except Exception as e:
            self.logger.warning(f"Failed to load pretrained weights: {e}, using random initialization")
            initialize_weights(model)
    
    def _create_class_mapping(self) -> Dict[int, int]:
        """
        Create mapping from 17 training classes to 7 inference denominations
        
        Returns:
            Dictionary mapping training class ID to denomination class ID
        """
        # Direct mapping for main denominations (0-6 → 0-6)
        mapping = {i: i for i in range(7)}
        
        # Layer 2: denomination-specific features (7-13 → 0-6)
        for i in range(7):
            mapping[7 + i] = i  # Classes 7-13 map to denominations 0-6
            
        # Layer 3: authenticity features (14-16 → special handling)
        # These don't map directly but affect confidence
        mapping[14] = -1  # Security sign (boosts confidence)
        mapping[15] = -1  # Serial text (boosts confidence)  
        mapping[16] = -1  # Security thread (boosts confidence)
        
        return mapping
    
    def _setup_phase_1(self, model):
        """Setup Phase 1: Freeze backbone, train head only"""
        self.current_phase = 1
        self.frozen_backbone = True
        
        # Freeze all parameters first
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                param.requires_grad = False
        elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            for param in model.model.parameters():
                param.requires_grad = False
                
        # Unfreeze only the detection head (last layer)
        detection_head = None
        
        if hasattr(model, 'model') and hasattr(model.model, 'model') and len(model.model.model) > 0:
            detection_head = model.model.model[-1]  # Ultralytics structure
        elif hasattr(model, 'model') and isinstance(model.model, nn.ModuleList) and len(model.model) > 0:
            detection_head = model.model[-1]  # Direct model list
        elif hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
            try:
                detection_head = model.model[-1]  # Try direct indexing
            except (IndexError, TypeError):
                pass
        elif hasattr(model, 'head'):  # Custom EfficientNet+YOLOv5 structure
            detection_head = model.head
                
        if detection_head is not None:
            for param in detection_head.parameters():
                param.requires_grad = True
            self.logger.info("🔒 Phase 1: Backbone frozen, head trainable")
        else:
            self.logger.warning("⚠️ Could not identify detection head for freezing")
    
    def setup_phase_2(self):
        """Setup Phase 2: Unfreeze backbone, fine-tune entire model"""
        self.current_phase = 2
        self.frozen_backbone = False
        
        # Unfreeze all parameters
        if hasattr(self.model, 'parameters'):
            for param in self.model.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
            for param in self.model.model.parameters():
                param.requires_grad = True
                
        self.logger.info("🔓 Phase 2: Full model trainable")
    
    def forward(self, x: torch.Tensor, training: bool = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, channels, height, width]
            training: Override training mode
            
        Returns:
            Raw YOLOv5 output during training, processed output during inference
        """
        if training is None:
            training = self.training
            
        # Forward through YOLOv5
        output = self.model(x)
        
        if training:
            # During training, return raw YOLOv5 output for loss computation
            return output
        else:
            # During inference, apply post-processing and class mapping
            return self._process_inference_output(output, x.shape)
    
    def _process_inference_output(
        self, 
        output: torch.Tensor, 
        input_shape: Tuple[int, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process inference output with class mapping and confidence adjustment
        
        Args:
            output: Raw YOLOv5 output
            input_shape: Input tensor shape
            
        Returns:
            List of detection dictionaries per image
        """
        # Apply NMS
        predictions = non_max_suppression(
            output,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False
        )
        
        processed_results = []
        
        for pred in predictions:
            if pred is None or len(pred) == 0:
                processed_results.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'denomination_scores': torch.zeros(7)  # 7 denomination scores
                })
                continue
                
            # Extract predictions
            boxes = pred[:, :4]
            scores = pred[:, 4]
            classes = pred[:, 5].long()
            
            # Apply class mapping and confidence adjustment
            mapped_results = self._map_classes_and_adjust_confidence(
                boxes, scores, classes
            )
            
            processed_results.append(mapped_results)
            
        return processed_results
    
    def _map_classes_and_adjust_confidence(
        self, 
        boxes: torch.Tensor,
        scores: torch.Tensor, 
        classes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Map 17 classes to 7 denominations and adjust confidence
        
        Args:
            boxes: Detection boxes [N, 4]
            scores: Confidence scores [N]
            classes: Class predictions [N]
            
        Returns:
            Dictionary with mapped results
        """
        # Initialize denomination tracking
        denomination_detections = {i: [] for i in range(7)}
        authenticity_scores = []
        
        # Group detections by denomination
        for i in range(len(classes)):
            cls = classes[i].item()
            box = boxes[i]
            score = scores[i].item()
            
            mapped_cls = self.class_mapping.get(cls, -1)
            
            if mapped_cls >= 0:  # Valid denomination (0-6)
                denomination_detections[mapped_cls].append({
                    'box': box,
                    'score': score,
                    'source_class': cls
                })
            elif cls >= 14:  # Authenticity features (14-16)
                authenticity_scores.append(score)
        
        # Process denominations and apply confidence adjustment
        final_boxes = []
        final_scores = []
        final_labels = []
        denomination_scores = torch.zeros(7)
        
        for denom_id in range(7):
            detections = denomination_detections[denom_id]
            
            if not detections:
                continue
                
            # Find best detection for this denomination
            best_det = max(detections, key=lambda x: x['score'])
            
            # Apply confidence adjustment based on supporting evidence
            adjusted_score = self._adjust_confidence_with_supporting_evidence(
                best_det, detections, authenticity_scores
            )
            
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
    
    def _adjust_confidence_with_supporting_evidence(
        self,
        primary_detection: Dict,
        all_detections: List[Dict],
        authenticity_scores: List[float]
    ) -> float:
        """
        Adjust confidence based on supporting evidence from other layers
        
        Args:
            primary_detection: Primary detection (Layer 1)
            all_detections: All detections for this denomination
            authenticity_scores: Authenticity feature scores (Layer 3)
            
        Returns:
            Adjusted confidence score
        """
        base_confidence = primary_detection['score']
        
        # Boost from Layer 2 (denomination-specific features)
        layer2_boost = 0.0
        for det in all_detections:
            if det['source_class'] >= 7 and det['source_class'] <= 13:
                layer2_boost = max(layer2_boost, det['score'] * 0.1)
                
        # Boost from Layer 3 (authenticity features)
        auth_boost = 0.0
        if authenticity_scores:
            avg_auth_score = sum(authenticity_scores) / len(authenticity_scores)
            auth_boost = avg_auth_score * 0.15
            
        # Apply adjustments
        adjusted = base_confidence + layer2_boost + auth_boost
        
        # Penalty if no authenticity features detected (possible fake)
        if not authenticity_scores and base_confidence > 0.5:
            adjusted *= 0.8  # Reduce confidence by 20%
            
        return min(adjusted, 1.0)
    
    def get_phase_info(self) -> Dict:
        """Get current training phase information"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'phase': self.current_phase,
            'backbone_frozen': self.frozen_backbone,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params
        }
    
    def get_model_config(self) -> Dict:
        """Get model configuration for saving"""
        return {
            'backbone': self.backbone,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'phase': self.current_phase,
            'class_mapping': self.class_mapping,
            'architecture': 'SmartCashYOLOv5Model'
        }


class SmartCashDirectYOLO:
    """
    Factory class for creating SmartCash YOLOv5 models
    """
    
    @staticmethod
    def create_model(
        backbone: str = "yolov5s",
        pretrained: bool = True,
        device: str = "auto"
    ) -> SmartCashYOLOv5Model:
        """
        Create SmartCash YOLOv5 model
        
        Args:
            backbone: YOLOv5 variant
            pretrained: Use pretrained weights
            device: Target device
            
        Returns:
            SmartCashYOLOv5Model instance
        """
        return SmartCashYOLOv5Model(
            backbone=backbone,
            num_classes=17,  # Fixed for SmartCash
            img_size=640,
            pretrained=pretrained,
            device=device
        )
    
    @staticmethod
    def get_supported_backbones() -> List[str]:
        """Get list of supported YOLOv5 backbones"""
        return ["yolov5s", "yolov5m", "yolov5l", "yolov5x", "efficientnet_b4"]
    
    @staticmethod
    def get_class_names() -> Dict[str, List[str]]:
        """Get class name mappings"""
        return {
            'training_classes': [
                # Layer 1: Main denominations (0-6)
                '001', '002', '005', '010', '020', '050', '100',
                # Layer 2: Denomination features (7-13) 
                'l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100',
                # Layer 3: Authenticity features (14-16)
                'l3_sign', 'l3_text', 'l3_thread'
            ],
            'inference_classes': [
                '001', '002', '005', '010', '020', '050', '100'
            ]
        }


# Export key components
__all__ = [
    'SmartCashYOLOv5Model',
    'SmartCashDirectYOLO'
]