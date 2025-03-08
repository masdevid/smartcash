# File: smartcash/handlers/dataset/multilayer_label_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pengelolaan label multilayer

import torch
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.coordinate_utils import CoordinateUtils

class MultilayerLabelHandler:
    """Handler untuk pengelolaan dan parsing label multilayer."""
    
    def __init__(
        self,
        layers: Optional[List[str]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi MultilayerLabelHandler.
        
        Args:
            layers: Daftar layer yang akan diproses
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger("multilayer_label_handler")
        
        # Dapatkan konfigurasi layer
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Mapping class_id ke layer untuk pencarian cepat
        self.class_to_layer = {}
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for cls_id in layer_config['class_ids']:
                self.class_to_layer[cls_id] = layer
    
    def parse_label_file(self, label_path: Path) -> List[int]:
        """
        Parse file label untuk mendapatkan class ID.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List class ID yang ditemukan
        """
        class_ids = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Format YOLO: class_id x y w h
                        try:
                            cls_id = int(float(parts[0]))
                            class_ids.append(cls_id)
                        except ValueError:
                            continue
        except Exception as e:
            self.logger.debug(f"⚠️ Error parsing label {label_path}: {str(e)}")
            
        return class_ids
    
    def parse_label_by_layer(self, label_path: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
        """
        Parse file label dan kelompokkan berdasarkan layer.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            Dict label per layer: {layer_name: [(class_id, [x, y, w, h]), ...]}
        """
        layer_labels = {layer: [] for layer in self.layers}
        
        # Parse label
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                coords = [float(x) for x in parts[1:5]]
                                
                                # Cek apakah class ID termasuk dalam layer yang diaktifkan
                                if cls_id in self.class_to_layer and self.class_to_layer[cls_id] in self.layers:
                                    layer = self.class_to_layer[cls_id]
                                    normalized_coords = coords  # YOLO sudah dalam format normalized
                                    layer_labels[layer].append((cls_id, normalized_coords))
                            except (ValueError, IndexError) as e:
                                self.logger.debug(f"⚠️ Format label tidak valid: {line.strip()} - {str(e)}")
                                continue
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return layer_labels
    
    def get_available_layers(self, label_path: Path) -> List[str]:
        """
        Mendapatkan layer yang tersedia dalam file label.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List nama layer yang tersedia
        """
        available_layers = []
        class_ids = self.parse_label_file(label_path)
        
        # Kelompokkan berdasarkan layer
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            class_ids_set = set(layer_config['class_ids'])
            if any(cls_id in class_ids_set for cls_id in class_ids):
                available_layers.append(layer)
                
        return available_layers
    
    def create_layer_tensor(self, layer: str, layer_data: List[Tuple[int, List[float]]]) -> torch.Tensor:
        """
        Membuat tensor untuk layer tertentu dari data label.
        
        Args:
            layer: Nama layer
            layer_data: Data label layer [(class_id, [x, y, w, h]), ...]
            
        Returns:
            Tensor layer dengan format [num_classes, 5] di mana setiap baris adalah [x, y, w, h, conf]
        """
        layer_config = self.layer_config_manager.get_layer_config(layer)
        num_classes = len(layer_config['classes'])
        class_ids = layer_config['class_ids']
        
        # Inisialisasi tensor kosong
        layer_tensor = torch.zeros((num_classes, 5))  # [x, y, w, h, conf]
        
        # Isi dengan data label
        for cls_id, bbox in layer_data:
            # Konversi global class_id ke indeks lokal dalam layer
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
    
    def get_class_statistics(self, labels_dir: Path, label_paths: List[Path]) -> Dict[str, int]:
        """
        Mendapatkan statistik distribusi kelas dalam dataset.
        
        Args:
            labels_dir: Direktori label (untuk log)
            label_paths: List path file label
            
        Returns:
            Dict berisi jumlah sampel per kelas
        """
        # Inisialisasi counter untuk semua kelas dari semua layer
        class_counts = {}
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                if i < len(layer_config['classes']):
                    cls_name = layer_config['classes'][i]
                    class_counts[cls_name] = 0
        
        # Counter untuk layer total
        class_counter = collections.Counter()
        
        # Hitung kelas untuk setiap file label
        for label_path in label_paths:
            if not Path(label_path).exists():
                continue
                
            try:
                class_ids = self.parse_label_file(Path(label_path))
                class_counter.update(class_ids)
            except Exception as e:
                self.logger.debug(f"⚠️ Error menghitung statistik kelas: {str(e)}")
        
        # Konversi dari class_id ke nama kelas
        for cls_id, count in class_counter.items():
            if cls_id in self.class_to_layer:
                layer = self.class_to_layer[cls_id]
                layer_config = self.layer_config_manager.get_layer_config(layer)
                
                if cls_id in layer_config['class_ids']:
                    idx = layer_config['class_ids'].index(cls_id)
                    if idx < len(layer_config['classes']):
                        cls_name = layer_config['classes'][idx]
                        class_counts[cls_name] = count
        
        return class_counts
    
    def get_class_name(self, cls_id: int) -> str:
        """
        Mendapatkan nama kelas dari class ID.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama kelas
        """
        if cls_id in self.class_to_layer:
            layer = self.class_to_layer[cls_id]
            layer_config = self.layer_config_manager.get_layer_config(layer)
            class_ids = layer_config['class_ids']
            classes = layer_config['classes']
            
            if cls_id in class_ids:
                idx = class_ids.index(cls_id)
                if idx < len(classes):
                    return classes[idx]
                    
        return f"Class-{cls_id}"