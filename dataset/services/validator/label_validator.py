"""
File: smartcash/dataset/services/validator/label_validator.py
Deskripsi: Komponen untuk memvalidasi file label dalam format YOLO
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config


class LabelValidator:
    """Validator khusus untuk memeriksa dan memvalidasi file label dataset."""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi LabelValidator.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger()
        
        # Setup layer config
        self.layer_config = get_layer_config()
        self.valid_class_ids = set()
        self.class_to_layer = {}
        
        # Kumpulkan semua class ID valid
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            for cls_id in layer_config['class_ids']:
                self.valid_class_ids.add(cls_id)
                self.class_to_layer[cls_id] = layer_name
        
        # Setup parameter validasi
        validation_config = self.config.get('validation', {}).get('label', {})
        self.min_bbox_size = validation_config.get('min_bbox_size', 0.001)  # Minimum size (width/height) dari bbox (0-1)
        self.max_bbox_per_image = validation_config.get('max_bbox_per_image', 100)
        self.reject_incomplete_labels = validation_config.get('reject_incomplete_labels', False)
    
    def validate_label(self, label_path: Path, check_class_ids: bool = True) -> Tuple[bool, List[str], List[Dict]]:
        """
        Validasi satu file label dalam format YOLO.
        
        Args:
            label_path: Path ke file label
            check_class_ids: Apakah memeriksa ID kelas
            
        Returns:
            Tuple (valid, list_masalah, bbox_data)
        """
        issues = []
        bbox_data = []
        
        # Cek eksistensi file
        if not label_path.exists():
            issues.append("File label tidak ditemukan")
            return False, issues, bbox_data
            
        # Cek ukuran file
        if os.path.getsize(label_path) == 0:
            issues.append("File label kosong")
            return False, issues, bbox_data
            
        # Baca dan parse file
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                issues.append("File label tidak berisi data")
                return False, issues, bbox_data
                
            # Parse setiap baris
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    issues.append(f"Format tidak valid pada baris {line_idx+1}: jumlah kolom kurang dari 5")
                    continue
                    
                try:
                    cls_id = int(float(parts[0]))
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Validasi format bbox
                    bbox_issues = []
                    
                    # Cek class ID
                    if check_class_ids and cls_id not in self.valid_class_ids:
                        bbox_issues.append(f"Class ID tidak valid: {cls_id}")
                    
                    # Cek koordinat yang valid (0-1)
                    if not (0 <= x_center <= 1):
                        bbox_issues.append(f"x_center di luar batas: {x_center}")
                    if not (0 <= y_center <= 1):
                        bbox_issues.append(f"y_center di luar batas: {y_center}")
                    if not (0 < width <= 1):
                        bbox_issues.append(f"width di luar batas: {width}")
                    if not (0 < height <= 1):
                        bbox_issues.append(f"height di luar batas: {height}")
                    
                    # Cek bbox terlalu kecil
                    if width < self.min_bbox_size or height < self.min_bbox_size:
                        bbox_issues.append(f"Bounding box terlalu kecil: {width:.4f}x{height:.4f}")
                    
                    # Tambahkan ke bbox data
                    bbox = {
                        'class_id': cls_id,
                        'bbox': [x_center, y_center, width, height],
                        'line_idx': line_idx,
                        'issues': bbox_issues
                    }
                    
                    # Tambahkan layer jika class ID valid
                    if check_class_ids and cls_id in self.class_to_layer:
                        bbox['layer'] = self.class_to_layer[cls_id]
                    
                    bbox_data.append(bbox)
                    
                    # Tambahkan isu per bbox jika ada
                    if bbox_issues:
                        issues.append(f"Masalah pada bbox di baris {line_idx+1}: {', '.join(bbox_issues)}")
                        
                except ValueError as e:
                    issues.append(f"Nilai tidak valid pada baris {line_idx+1}: {str(e)}")
                except Exception as e:
                    issues.append(f"Error saat memproses baris {line_idx+1}: {str(e)}")
                    
            # Cek jumlah bbox
            if len(bbox_data) > self.max_bbox_per_image:
                issues.append(f"Jumlah bbox terlalu banyak: {len(bbox_data)} (maksimum: {self.max_bbox_per_image})")
                
            # Cek jika tidak ada bbox valid
            if not bbox_data and self.reject_incomplete_labels:
                issues.append("Tidak ada bbox valid dalam label")
                
        except Exception as e:
            issues.append(f"Error saat membaca file label: {str(e)}")
            return False, issues, bbox_data
            
        # Label valid jika tidak ada masalah
        is_valid = len(issues) == 0
        return is_valid, issues, bbox_data
    
    def fix_label(self, label_path: Path, fix_coordinates: bool = True, fix_format: bool = True) -> Tuple[bool, List[str]]:
        """
        Perbaiki masalah pada file label.
        
        Args:
            label_path: Path ke file label
            fix_coordinates: Apakah memperbaiki koordinat bbox
            fix_format: Apakah memperbaiki format file
            
        Returns:
            Tuple (berhasil_diperbaiki, perbaikan_yang_dilakukan)
        """
        fixes = []
        
        try:
            # Validasi label terlebih dahulu
            _, issues, bbox_data = self.validate_label(label_path, check_class_ids=True)
            
            if not issues and not bbox_data:
                return False, ["Tidak ada yang perlu diperbaiki"]
                
            # Baca file label
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Perbaiki setiap bbox
            fixed_lines = []
            fixed_count = 0
            
            for bbox in bbox_data:
                cls_id = bbox['class_id']
                x_center, y_center, width, height = bbox['bbox']
                
                # Skip jika class ID tidak valid
                if cls_id not in self.valid_class_ids:
                    continue
                    
                # Perbaiki koordinat jika diminta
                fixed_bbox = [x_center, y_center, width, height]
                if fix_coordinates:
                    if not (0 <= x_center <= 1): 
                        fixed_bbox[0] = max(0, min(1, x_center))
                        fixes.append(f"Perbaikan x_center: {x_center:.4f} → {fixed_bbox[0]:.4f}")
                        fixed_count += 1
                        
                    if not (0 <= y_center <= 1): 
                        fixed_bbox[1] = max(0, min(1, y_center))
                        fixes.append(f"Perbaikan y_center: {y_center:.4f} → {fixed_bbox[1]:.4f}")
                        fixed_count += 1
                        
                    if not (0 < width <= 1): 
                        fixed_bbox[2] = max(self.min_bbox_size, min(1, width))
                        fixes.append(f"Perbaikan width: {width:.4f} → {fixed_bbox[2]:.4f}")
                        fixed_count += 1
                        
                    if not (0 < height <= 1): 
                        fixed_bbox[3] = max(self.min_bbox_size, min(1, height))
                        fixes.append(f"Perbaikan height: {height:.4f} → {fixed_bbox[3]:.4f}")
                        fixed_count += 1
                
                # Tambahkan ke daftar baris yang diperbaiki
                fixed_lines.append(f"{cls_id} {' '.join(map(str, fixed_bbox))}")
            
            # Simpan hasil perbaikan jika ada yang berubah
            if fixed_count > 0 or len(fixed_lines) != len(lines):
                with open(label_path, 'w') as f:
                    for line in fixed_lines:
                        f.write(f"{line}\n")
                
                if len(fixed_lines) != len(lines):
                    fixes.append(f"Jumlah bbox berubah: {len(lines)} → {len(fixed_lines)}")
                    
                self.logger.info(f"🔧 Label {label_path.name} diperbaiki: {fixed_count} koordinat")
                return True, fixes
                
            return False, ["Tidak ada perubahan yang dilakukan"]
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memperbaiki label {label_path.name}: {str(e)}")
            return False, [f"Error: {str(e)}"]
    
    def get_label_metadata(self, label_path: Path) -> Dict[str, Any]:
        """
        Dapatkan metadata dari file label.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            Dictionary berisi metadata label
        """
        metadata = {
            'path': str(label_path),
            'filename': label_path.name,
            'size_bytes': 0,
            'bbox_count': 0,
            'classes': [],
            'layers': [],
            'is_valid': False
        }
        
        if not label_path.exists():
            return metadata
            
        # Tambahkan info ukuran file
        file_size = os.path.getsize(label_path)
        metadata['size_bytes'] = file_size
        
        # Validasi label untuk metadata
        is_valid, issues, bbox_data = self.validate_label(label_path)
        
        if bbox_data:
            # Hitung statistik
            class_counts = {}
            layer_counts = {}
            
            for bbox in bbox_data:
                cls_id = bbox['class_id']
                if 'layer' in bbox:
                    layer = bbox['layer']
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1
                
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            
            # Update metadata
            metadata.update({
                'bbox_count': len(bbox_data),
                'class_counts': class_counts,
                'classes': list(class_counts.keys()),
                'layer_counts': layer_counts,
                'layers': list(layer_counts.keys()),
                'is_valid': is_valid,
                'issues': issues if issues else []
            })
        
        return metadata
    
    def check_layers_coverage(self, label_path: Path, required_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Cek coverage layer dalam file label.
        
        Args:
            label_path: Path ke file label
            required_layers: Daftar layer yang diperlukan (opsional)
            
        Returns:
            Dictionary berisi info coverage layer
        """
        result = {
            'has_all_required_layers': False,
            'available_layers': [],
            'missing_layers': [],
            'layer_counts': {}
        }
        
        # Validasi label
        _, _, bbox_data = self.validate_label(label_path)
        
        if not bbox_data:
            return result
            
        # Dapatkan layer yang tersedia
        available_layers = set()
        layer_counts = {}
        
        for bbox in bbox_data:
            if 'layer' in bbox:
                layer = bbox['layer']
                available_layers.add(layer)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Tentukan required layers jika tidak disediakan
        if required_layers is None:
            required_layers = self.layer_config.get_layer_names()
            
        # Cek coverage
        missing_layers = [layer for layer in required_layers if layer not in available_layers]
        
        # Update result
        result.update({
            'has_all_required_layers': len(missing_layers) == 0,
            'available_layers': list(available_layers),
            'missing_layers': missing_layers,
            'layer_counts': layer_counts
        })
        
        return result