"""
File: smartcash/utils/dataset/dataset_validator_core.py
Author: Alfrida Sabar
Deskripsi: Modul inti untuk validasi dataset dengan fokus pada fungsionalitas validasi utama
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import random
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class DatasetValidatorCore:
    """
    Kelas inti untuk validasi dataset dengan fokus pada fungsionalitas validasi utama
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi validator dataset inti.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Setup direktori untuk file tidak valid
        self.invalid_dir = self.data_dir / 'invalid'
        
        # Lock untuk thread safety
        self._lock = threading.RLock()

    def validate_image_label_pair(self, img_path, labels_dir):
        """
        Validasi satu pasang gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori yang berisi file label
            
        Returns:
            Dict hasil validasi
        """
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
            img = cv2.imread(str(img_path))
            if img is None:
                result['issues'].append(f"Gambar tidak dapat dibaca: {img_path.name}")
                result['corrupt'] = True
                return result
                
            if img.size == 0:
                result['issues'].append(f"Gambar kosong: {img_path.name}")
                result['corrupt'] = True
                return result
                
            result['image_size'] = (img.shape[1], img.shape[0])  # (width, height)
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
        
        # Validasi isi label
        try:
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
            
            if not label_lines:
                result['issues'].append(f"File label kosong")
                result['empty_label'] = True
                return result
            
            result['label_lines'] = []
            valid_label_lines = []
            has_issue = False
            fixed_something = False
            
            # Mengumpulkan statistik per layer dan kelas
            layer_counts = {layer: 0 for layer in self.layer_config_manager.get_layer_names()}
            class_counts = {}
            
            # Validasi setiap baris label
            for i, line in enumerate(label_lines):
                line_result = self._validate_label_line(line, i, layer_counts, class_counts)
                
                if line_result['issues']:
                    has_issue = True
                
                if line_result.get('fixed', False):
                    fixed_something = True
                    valid_label_lines.append(
                        f"{line_result['class_id']} {' '.join(map(str, line_result['bbox']))}\n"
                    )
                elif line_result['valid']:
                    valid_label_lines.append(
                        f"{line_result['class_id']} {' '.join(map(str, line_result['bbox']))}\n"
                    )
                
                result['label_lines'].append(line_result)
            
            # Simpan statistik layer dan kelas
            result['layer_stats'] = layer_counts
            result['class_stats'] = class_counts
            
            # Cek apakah ada label yang valid
            result['label_valid'] = any(line.get('valid', False) for line in result['label_lines'])
            
            # Cek apakah ada layer yang aktif
            result['has_active_layer'] = False
            for layer in self.active_layers:
                if layer in layer_counts and layer_counts[layer] > 0:
                    result['has_active_layer'] = True
                    break
                    
            # Status final
            if result['label_valid'] and result['image_valid']:
                result['status'] = 'valid'
            
            # Set fixed flag jika ada perbaikan
            if fixed_something:
                result['fixed'] = True
                result['fixed_bbox'] = valid_label_lines
            
        except Exception as e:
            result['issues'].append(f"Error saat membaca label: {str(e)}")
            
        return result
        
    def _validate_label_line(self, line: str, line_index: int, layer_counts: Dict[str, int], class_counts: Dict[str, int]) -> Dict:
        """
        Validasi satu baris label.
        
        Args:
            line: String baris label
            line_index: Indeks baris
            layer_counts: Dictionary untuk menghitung jumlah objek per layer
            class_counts: Dictionary untuk menghitung jumlah objek per kelas
            
        Returns:
            Dict hasil validasi baris
        """
        line_result = {
            'text': line.strip(),
            'valid': False,
            'class_id': None,
            'layer': None,
            'bbox': None,
            'issues': [],
            'fixed': False
        }
        
        parts = line.strip().split()
        if len(parts) < 5:
            line_result['issues'].append(f"Label tidak lengkap pada baris {line_index+1}")
            return line_result
        
        try:
            cls_id = int(float(parts[0]))
            bbox = [float(x) for x in parts[1:5]]
            
            # Cek apakah class ID valid
            layer_name = self.layer_config_manager.get_layer_for_class_id(cls_id)
            if layer_name:
                line_result['layer'] = layer_name
                layer_counts[layer_name] += 1
                
                class_name = self.layer_config_manager.get_class_name(cls_id)
                if class_name:
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
            else:
                line_result['issues'].append(f"Class ID tidak valid: {cls_id}")
                return line_result
            
            # Cek apakah koordinat valid (0-1)
            invalid_coords = [i for i, coord in enumerate(bbox) if not (0 <= coord <= 1)]
            if invalid_coords:
                coord_names = ['x_center', 'y_center', 'width', 'height']
                invalid_names = [coord_names[i] for i in invalid_coords]
                
                line_result['issues'].append(
                    f"Koordinat tidak valid: {', '.join(invalid_names)}"
                )
                
                # Fix koordinat jika diminta
                fixed_bbox = [max(0.001, min(0.999, coord)) for coord in bbox]
                line_result['bbox'] = fixed_bbox
                line_result['fixed'] = True
            else:
                line_result['bbox'] = bbox
            
            line_result['class_id'] = cls_id
            
            # Tandai sebagai valid jika tidak ada masalah atau telah diperbaiki
            if not line_result['issues'] or line_result['fixed']:
                line_result['valid'] = True
            
        except ValueError:
            line_result['issues'].append(f"Format tidak valid pada baris {line_index+1}")
        
        return line_result
    
    def visualize_issues(
        self,
        img_path: Path,
        result: Dict,
        vis_dir: Path
    ) -> bool:
        """
        Visualisasikan masalah dalam gambar dan label.
        
        Args:
            img_path: Path gambar
            result: Hasil validasi
            vis_dir: Direktori output visualisasi
            
        Returns:
            Boolean sukses/gagal
        """
        try:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                return False
                
            # Buat nama file output
            vis_path = vis_dir / f"{img_path.stem}_issues.jpg"
            
            # Dimensi gambar
            h, w = img.shape[:2]
            
            # Gambar kotak merah untuk label tidak valid, hijau untuk yang valid
            for line_result in result['label_lines']:
                if not line_result.get('bbox'):
                    continue
                    
                bbox = line_result['bbox']
                cls_id = line_result.get('class_id')
                
                # Convert YOLO format (x_center, y_center, width, height) ke pixel
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Warna berdasarkan valid/tidak
                color = (0, 255, 0) if line_result.get('valid', False) else (0, 0, 255)
                thickness = 2
                
                # Gambar bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Tambahkan label
                class_name = self.layer_config_manager.get_class_name(cls_id) if cls_id is not None else "Unknown"
                label_text = f"ID: {cls_id}, Class: {class_name}"
                
                if line_result.get('issues'):
                    label_text += f", Issues: {len(line_result['issues'])}"
                    
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1
                )
            
            # Tambahkan teks ringkasan masalah di bagian atas gambar
            issues_text = f"Issues: {len(result['issues'])}"
            cv2.putText(
                img,
                issues_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                thickness=2
            )
            
            # Simpan gambar
            cv2.imwrite(str(vis_path), img)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat visualisasi: {str(e)}")
            return False