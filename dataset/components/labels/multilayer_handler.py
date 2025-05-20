"""
File: smartcash/dataset/components/labels/multilayer_handler.py
Deskripsi: Utilitas untuk penanganan label multilayer dalam format YOLO
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.components.labels.label_handler import LabelHandler


class MultilayerLabelHandler:
    """Kelas untuk penanganan label multilayer dalam format YOLO."""
    
    def __init__(self, label_dir: Union[str, Path], logger=None, config: Optional[Dict] = None):
        """
        Inisialisasi MultilayerLabelHandler.
        
        Args:
            label_dir: Direktori label
            logger: Logger kustom (opsional)
            config: Konfigurasi aplikasi (opsional)
        """
        self.label_dir = Path(label_dir)
        self.logger = logger or get_logger()
        self.config = config or {}
        
        # Setup layer config
        self.layer_config = get_layer_config()
        self.active_layers = self.config.get('layers', self.layer_config.get_layer_names())
        
        # Membangun mapping class ID ke layer
        self.class_to_layer = {}
        self.class_to_name = {}
        
        for layer in self.active_layers:
            layer_config = self.layer_config.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                self.class_to_layer[cls_id] = layer
                if i < len(layer_config['classes']):
                    self.class_to_name[cls_id] = layer_config['classes'][i]
        
        # Setup base label handler
        self.label_handler = LabelHandler(label_dir, logger)
        
        # Pastikan direktori label ada
        if not self.label_dir.exists():
            os.makedirs(self.label_dir, exist_ok=True)
    
    def load_multilayer_label(self, image_id: str) -> Dict[str, List[Dict]]:
        """
        Load label dan kelompokkan berdasarkan layer.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            
        Returns:
            Dictionary berisi list label untuk setiap layer
        """
        # Load semua label
        all_labels = self.label_handler.load_yolo_label(image_id)
        
        # Kelompokkan berdasarkan layer
        layer_labels = {layer: [] for layer in self.active_layers}
        unknown_layer = []
        
        for label in all_labels:
            cls_id = label['class_id']
            
            # Tentukan layer berdasarkan class_id
            if cls_id in self.class_to_layer:
                layer = self.class_to_layer[cls_id]
                
                # Tambahkan info layer dan class_name
                label['layer'] = layer
                label['class_name'] = self.class_to_name.get(cls_id, f"Class-{cls_id}")
                
                layer_labels[layer].append(label)
            else:
                # Class ID tidak dikenal
                label['layer'] = 'unknown'
                unknown_layer.append(label)
        
        # Tambahkan layer unknown jika ada
        if unknown_layer:
            layer_labels['unknown'] = unknown_layer
        
        return layer_labels
    
    def save_multilayer_label(self, image_id: str, layer_labels: Dict[str, List[Dict]]) -> bool:
        """
        Simpan label multilayer.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            layer_labels: Dictionary berisi list label untuk setiap layer
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Gabungkan semua label
        all_labels = []
        
        for layer, labels in layer_labels.items():
            for label in labels:
                # Pastikan hanya menyimpan data yang diperlukan
                clean_label = {
                    'class_id': label['class_id'],
                    'bbox': label['bbox']
                }
                
                # Tambahkan data tambahan jika ada
                if 'extra' in label:
                    clean_label['extra'] = label['extra']
                
                all_labels.append(clean_label)
        
        # Simpan semua label
        return self.label_handler.save_yolo_label(image_id, all_labels)
    
    def get_layer_annotation(self, image_id: str, layer: str) -> List[Dict]:
        """
        Dapatkan anotasi untuk layer tertentu.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            layer: Nama layer
            
        Returns:
            List label untuk layer yang ditentukan
        """
        layer_labels = self.load_multilayer_label(image_id)
        return layer_labels.get(layer, [])
    
    def get_available_layers(self, image_id: str) -> List[str]:
        """
        Dapatkan daftar layer yang tersedia dalam file label.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            
        Returns:
            List layer yang tersedia
        """
        layer_labels = self.load_multilayer_label(image_id)
        
        # Filter layer yang memiliki anotasi
        available_layers = [layer for layer, labels in layer_labels.items() 
                          if labels and layer != 'unknown']
        
        return available_layers
    
    def add_bbox_to_layer(self, image_id: str, layer: str, class_id: int, 
                         bbox: List[float], extra: List[str] = None) -> bool:
        """
        Tambahkan bbox ke layer tertentu.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            layer: Nama layer
            class_id: ID kelas
            bbox: Koordinat bbox [x_center, y_center, width, height]
            extra: Data tambahan (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Validasi class_id untuk layer
        layer_config = self.layer_config.get_layer_config(layer)
        if class_id not in layer_config['class_ids']:
            self.logger.warning(f"⚠️ Class ID {class_id} tidak valid untuk layer {layer}")
            return False
        
        # Load label multilayer yang ada
        layer_labels = self.load_multilayer_label(image_id)
        
        # Tambahkan bbox baru
        new_label = {
            'class_id': class_id,
            'bbox': bbox,
            'layer': layer,
            'class_name': self.class_to_name.get(class_id, f"Class-{class_id}")
        }
        
        if extra:
            new_label['extra'] = extra
        
        # Tambahkan ke layer yang sesuai
        if layer not in layer_labels:
            layer_labels[layer] = []
        
        layer_labels[layer].append(new_label)
        
        # Simpan kembali
        return self.save_multilayer_label(image_id, layer_labels)
    
    def update_bbox_in_layer(self, image_id: str, layer: str, bbox_idx: int, 
                            class_id: int = None, bbox: List[float] = None, 
                            extra: List[str] = None) -> bool:
        """
        Update bbox dalam layer tertentu.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            layer: Nama layer
            bbox_idx: Indeks bbox dalam layer
            class_id: ID kelas baru (opsional)
            bbox: Koordinat bbox baru [x_center, y_center, width, height] (opsional)
            extra: Data tambahan baru (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load label multilayer yang ada
        layer_labels = self.load_multilayer_label(image_id)
        
        # Periksa apakah layer ada
        if layer not in layer_labels:
            self.logger.warning(f"⚠️ Layer {layer} tidak ditemukan di {image_id}")
            return False
        
        # Periksa apakah indeks valid
        if bbox_idx >= len(layer_labels[layer]):
            self.logger.warning(f"⚠️ Indeks bbox {bbox_idx} tidak valid untuk layer {layer}")
            return False
        
        # Validasi class_id untuk layer jika ada
        if class_id is not None:
            layer_config = self.layer_config.get_layer_config(layer)
            if class_id not in layer_config['class_ids']:
                self.logger.warning(f"⚠️ Class ID {class_id} tidak valid untuk layer {layer}")
                return False
            
            # Update class_id
            layer_labels[layer][bbox_idx]['class_id'] = class_id
            layer_labels[layer][bbox_idx]['class_name'] = self.class_to_name.get(class_id, f"Class-{class_id}")
        
        # Update bbox jika ada
        if bbox is not None:
            layer_labels[layer][bbox_idx]['bbox'] = bbox
        
        # Update extra jika ada
        if extra is not None:
            layer_labels[layer][bbox_idx]['extra'] = extra
        
        # Simpan kembali
        return self.save_multilayer_label(image_id, layer_labels)
    
    def delete_bbox_from_layer(self, image_id: str, layer: str, bbox_idx: int) -> bool:
        """
        Hapus bbox dari layer tertentu.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            layer: Nama layer
            bbox_idx: Indeks bbox dalam layer
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load label multilayer yang ada
        layer_labels = self.load_multilayer_label(image_id)
        
        # Periksa apakah layer ada
        if layer not in layer_labels:
            self.logger.warning(f"⚠️ Layer {layer} tidak ditemukan di {image_id}")
            return False
        
        # Periksa apakah indeks valid
        if bbox_idx >= len(layer_labels[layer]):
            self.logger.warning(f"⚠️ Indeks bbox {bbox_idx} tidak valid untuk layer {layer}")
            return False
        
        # Hapus bbox
        layer_labels[layer].pop(bbox_idx)
        
        # Hapus layer jika kosong
        if not layer_labels[layer]:
            del layer_labels[layer]
        
        # Hapus file jika tidak ada label
        if not any(labels for labels in layer_labels.values()):
            return self.label_handler.delete_label(image_id)
        
        # Simpan kembali
        return self.save_multilayer_label(image_id, layer_labels)
    
    def get_layer_statistics(self, image_ids: List[str] = None) -> Dict[str, int]:
        """
        Dapatkan statistik jumlah objek per layer.
        
        Args:
            image_ids: List ID gambar (opsional)
            
        Returns:
            Dictionary berisi jumlah objek per layer
        """
        layer_counts = {layer: 0 for layer in self.active_layers}
        
        # Jika tidak ada image_ids, proses semua file label
        if not image_ids:
            for label_file in self.label_dir.glob('*.txt'):
                image_id = label_file.stem
                layer_labels = self.load_multilayer_label(image_id)
                
                for layer, labels in layer_labels.items():
                    if layer in layer_counts:
                        layer_counts[layer] += len(labels)
        else:
            # Proses hanya image_ids yang diberikan
            for image_id in image_ids:
                layer_labels = self.load_multilayer_label(image_id)
                
                for layer, labels in layer_labels.items():
                    if layer in layer_counts:
                        layer_counts[layer] += len(labels)
        
        return layer_counts
    
    def get_class_statistics(self, image_ids: List[str] = None) -> Dict[int, int]:
        """
        Dapatkan statistik jumlah objek per kelas.
        
        Args:
            image_ids: List ID gambar (opsional)
            
        Returns:
            Dictionary berisi jumlah objek per kelas
        """
        class_counts = {}
        
        # Jika tidak ada image_ids, proses semua file label
        if not image_ids:
            for label_file in self.label_dir.glob('*.txt'):
                image_id = label_file.stem
                layer_labels = self.load_multilayer_label(image_id)
                
                for layer, labels in layer_labels.items():
                    for label in labels:
                        cls_id = label['class_id']
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        else:
            # Proses hanya image_ids yang diberikan
            for image_id in image_ids:
                layer_labels = self.load_multilayer_label(image_id)
                
                for layer, labels in layer_labels.items():
                    for label in labels:
                        cls_id = label['class_id']
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        
        return class_counts