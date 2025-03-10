"""
File: smartcash/utils/dataset/dataset_validator_core.py
Author: Alfrida Sabar
Deskripsi: Modul inti untuk validasi dataset dengan fokus pada fungsionalitas validasi utama
"""

import cv2
import threading
from pathlib import Path
from typing import Dict, Optional, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.dataset.dataset_utils import DatasetUtils, IMG_EXTENSIONS

class DatasetValidatorCore:
    """Kelas inti untuk validasi dataset dengan fokus pada fungsionalitas validasi utama"""
    
    def __init__(self, config: Dict, data_dir: Optional[Union[str, Path]] = None, logger: Optional[SmartCashLogger] = None):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        self.invalid_dir = self.data_dir / 'invalid'
        self.utils = DatasetUtils(config=config, data_dir=str(self.data_dir), logger=logger)
        self._lock = threading.RLock()

    def validate_image_label_pair(self, img_path, labels_dir):
        """Validasi satu pasang gambar dan label."""
        result = {
            'image_path': str(img_path),
            'label_path': str(labels_dir / f"{img_path.stem}.txt"),
            'status': 'invalid',
            'issues': [],
            'layer_stats': {layer: 0 for layer in self.layer_config_manager.get_layer_names()},
            'class_stats': {},
            'fixed': False,
            'visualized': False
        }
        
        # Validasi gambar
        try:
            img = self.utils.load_image(img_path)
            if img is None or img.size == 0:
                result['issues'].append(f"Gambar tidak dapat dibaca atau kosong: {img_path.name}")
                result['corrupt'] = True
                return result
                
            result['image_size'] = (img.shape[1], img.shape[0])
            result['image_valid'] = True
        except Exception as e:
            result['issues'].append(f"Error saat membaca gambar: {str(e)}")
            result['corrupt'] = True
            return result
        
        # Validasi label
        label_path = labels_dir / f"{img_path.stem}.txt"
        result['label_exists'] = label_path.exists()
        
        if not result['label_exists']:
            result['issues'].append(f"File label tidak ditemukan")
            result['missing_label'] = True
            return result
        
        # Validasi isi label dengan callback
        try:
            def validate_label_line(bbox_data, line, line_idx):
                cls_id = bbox_data['class_id']
                bbox = bbox_data['bbox']
                
                line_result = {
                    'text': line.strip(),
                    'valid': True,
                    'class_id': cls_id,
                    'bbox': bbox,
                    'issues': [],
                    'fixed': False
                }
                
                layer_name = bbox_data.get('layer', self.layer_config_manager.get_layer_for_class_id(cls_id))
                if layer_name:
                    line_result['layer'] = layer_name
                    if layer_name in result['layer_stats']:
                        result['layer_stats'][layer_name] += 1
                    
                    class_name = bbox_data.get('class_name', self.utils.get_class_name(cls_id))
                    if class_name:
                        if class_name not in result['class_stats']:
                            result['class_stats'][class_name] = 0
                        result['class_stats'][class_name] += 1
                else:
                    line_result['issues'].append(f"Class ID tidak valid: {cls_id}")
                    line_result['valid'] = False
                
                return line_result
            
            bboxes = self.utils.parse_yolo_label(label_path, parse_callback=validate_label_line)
            
            if not bboxes:
                result['issues'].append(f"File label kosong atau format tidak valid")
                result['empty_label'] = True
                return result
            
            result['label_lines'] = bboxes
            result['has_active_layer'] = any(layer in result['layer_stats'] and result['layer_stats'][layer] > 0 
                                          for layer in self.active_layers)
            valid_lines = sum(1 for box in bboxes if box.get('valid', False))
            result['label_valid'] = valid_lines > 0
            
            if result['label_valid'] and result['image_valid']:
                result['status'] = 'valid'
            
        except Exception as e:
            result['issues'].append(f"Error saat membaca label: {str(e)}")
        
        return result
    
    def visualize_issues(self, img_path: Path, result: Dict, vis_dir: Path) -> bool:
        """Visualisasikan masalah dalam gambar dan label."""
        try:
            img = self.utils.load_image(img_path)
            if img is None: return False
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vis_path = vis_dir / f"{img_path.stem}_issues.jpg"
            h, w = img.shape[:2]
            
            for line_result in result.get('label_lines', []):
                if not line_result.get('bbox'): continue
                
                bbox = line_result['bbox']
                cls_id = line_result.get('class_id')
                
                # Convert YOLO format ke pixel
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                color = (0, 255, 0) if line_result.get('valid', False) else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                class_name = self.utils.get_class_name(cls_id) if cls_id is not None else "Unknown"
                label_text = f"ID: {cls_id}, Class: {class_name}"
                if 'issues' in line_result and line_result['issues']:
                    label_text += f", Issues: {len(line_result['issues'])}"
                
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.putText(img, f"Issues: {len(result.get('issues', []))}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(str(vis_path), img)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat visualisasi: {str(e)}")
            return False