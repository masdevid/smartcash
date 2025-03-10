# File: smartcash/handlers/dataset/multilayer/multilayer_label_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pengelolaan label multilayer (versi ringkas)

import torch
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class MultilayerLabelHandler:
    """Handler untuk pengelolaan dan parsing label multilayer."""
    
    def __init__(self, layers: Optional[List[str]] = None, logger: Optional[SmartCashLogger] = None):
        """Inisialisasi MultilayerLabelHandler."""
        self.logger = logger or SmartCashLogger("multilayer_label_handler")
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Mapping class_id ke layer untuk lookup cepat
        self.class_to_layer = {}
        self.class_to_name = {}
        
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                self.class_to_layer[cls_id] = layer
                if i < len(layer_config['classes']):
                    self.class_to_name[cls_id] = layer_config['classes'][i]
    
    def parse_label_file(self, label_path: Path) -> List[int]:
        """Parse file label untuk mendapatkan class ID."""
        class_ids = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Format YOLO: class_id x y w h
                        try:
                            class_ids.append(int(float(parts[0])))
                        except ValueError:
                            continue
        except Exception as e:
            self.logger.debug(f"⚠️ Error parsing label {label_path}: {str(e)}")
        return class_ids
    
    def parse_label_by_layer(self, label_path: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
        """Parse file label dan kelompokkan berdasarkan layer."""
        layer_labels = {layer: [] for layer in self.layers}
        
        if not label_path.exists():
            return layer_labels
            
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            coords = [float(x) for x in parts[1:5]]
                            
                            if cls_id in self.class_to_layer and self.class_to_layer[cls_id] in self.layers:
                                layer = self.class_to_layer[cls_id]
                                layer_labels[layer].append((cls_id, coords))
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return layer_labels
    
    def get_available_layers(self, label_path: Path) -> List[str]:
        """Mendapatkan layer yang tersedia dalam file label."""
        available_layers = []
        class_ids = self.parse_label_file(label_path)
        
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            class_ids_set = set(layer_config['class_ids'])
            if any(cls_id in class_ids_set for cls_id in class_ids):
                available_layers.append(layer)
                
        return available_layers
    
    def create_layer_tensor(self, layer: str, layer_data: List[Tuple[int, List[float]]]) -> torch.Tensor:
        """Membuat tensor untuk layer tertentu dari data label."""
        layer_config = self.layer_config_manager.get_layer_config(layer)
        num_classes = len(layer_config['classes'])
        class_ids = layer_config['class_ids']
        
        # Inisialisasi tensor kosong
        layer_tensor = torch.zeros((num_classes, 5))  # [x, y, w, h, conf]
        
        # Isi dengan data label
        for cls_id, bbox in layer_data:
            if cls_id in class_ids:
                local_idx = class_ids.index(cls_id)
                if 0 <= local_idx < num_classes:
                    x_center, y_center, width, height = bbox
                    layer_tensor[local_idx, 0] = x_center
                    layer_tensor[local_idx, 1] = y_center
                    layer_tensor[local_idx, 2] = width
                    layer_tensor[local_idx, 3] = height
                    layer_tensor[local_idx, 4] = 1.0  # Confidence
                    
        return layer_tensor
    
    def get_class_name(self, cls_id: int) -> str:
        """Mendapatkan nama kelas dari class ID."""
        if cls_id in self.class_to_name:
            return self.class_to_name[cls_id]
        return f"Class-{cls_id}"