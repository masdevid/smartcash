"""
File: smartcash/dataset/components/labels/label_handler.py
Deskripsi: Utilitas untuk manipulasi dan manajemen file label
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.components.geometry.coord_converter import CoordinateConverter


class LabelHandler:
    """Kelas untuk manipulasi dan manajemen file label."""
    
    def __init__(self, label_dir: Union[str, Path], logger=None):
        """
        Inisialisasi LabelHandler.
        
        Args:
            label_dir: Direktori label
            logger: Logger kustom (opsional)
        """
        self.label_dir = Path(label_dir)
        self.logger = logger or get_logger("label_handler")
        
        # Pastikan direktori label ada
        if not self.label_dir.exists():
            os.makedirs(self.label_dir, exist_ok=True)
    
    def load_yolo_label(self, image_id: str) -> List[Dict]:
        """
        Load label dalam format YOLO.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            
        Returns:
            List dictionary dengan data label
        """
        label_path = self.label_dir / f"{image_id}.txt"
        
        if not label_path.exists():
            return []
        
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            bbox = list(map(float, parts[1:5]))  # [x_center, y_center, width, height]
                            
                            # Validasi koordinat
                            if not all(0 <= coord <= 1 for coord in bbox[0:2]) or not all(0 < coord <= 1 for coord in bbox[2:4]):
                                continue
                            
                            label = {
                                'class_id': cls_id,
                                'bbox': bbox,
                                'line_idx': line_idx
                            }
                            
                            # Tambahkan data tambahan jika ada
                            if len(parts) > 5:
                                label['extra'] = parts[5:]
                            
                            labels.append(label)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return labels
    
    def save_yolo_label(self, image_id: str, labels: List[Dict]) -> bool:
        """
        Simpan label dalam format YOLO.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            labels: List dictionary dengan data label
            
        Returns:
            True jika berhasil, False jika gagal
        """
        label_path = self.label_dir / f"{image_id}.txt"
        
        try:
            with open(label_path, 'w') as f:
                for label in labels:
                    cls_id = label['class_id']
                    bbox = label['bbox']
                    
                    # Format YOLO: class_id x_center y_center width height
                    line = f"{cls_id} {' '.join(map(str, bbox))}"
                    
                    # Tambahkan data tambahan jika ada
                    if 'extra' in label:
                        line += f" {' '.join(map(str, label['extra']))}"
                    
                    f.write(line + '\n')
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saat menyimpan label {label_path}: {str(e)}")
            return False
    
    def create_empty_label(self, image_id: str) -> bool:
        """
        Buat file label kosong.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        label_path = self.label_dir / f"{image_id}.txt"
        
        try:
            with open(label_path, 'w') as f:
                pass
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat label kosong {label_path}: {str(e)}")
            return False
    
    def delete_label(self, image_id: str) -> bool:
        """
        Hapus file label.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        label_path = self.label_dir / f"{image_id}.txt"
        
        if not label_path.exists():
            return True
        
        try:
            label_path.unlink()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saat menghapus label {label_path}: {str(e)}")
            return False
    
    def add_bbox(self, image_id: str, class_id: int, bbox: List[float], extra: List[str] = None) -> bool:
        """
        Tambahkan bbox ke label yang ada.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            class_id: ID kelas
            bbox: Koordinat bbox [x_center, y_center, width, height]
            extra: Data tambahan (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load label yang ada
        labels = self.load_yolo_label(image_id)
        
        # Tambahkan bbox baru
        new_label = {
            'class_id': class_id,
            'bbox': bbox
        }
        
        if extra:
            new_label['extra'] = extra
        
        labels.append(new_label)
        
        # Simpan kembali
        return self.save_yolo_label(image_id, labels)
    
    def update_bbox(self, image_id: str, line_idx: int, class_id: int = None, 
                   bbox: List[float] = None, extra: List[str] = None) -> bool:
        """
        Update bbox yang ada di label.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            line_idx: Indeks baris bbox yang akan diupdate
            class_id: ID kelas baru (opsional)
            bbox: Koordinat bbox baru [x_center, y_center, width, height] (opsional)
            extra: Data tambahan baru (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load label yang ada
        labels = self.load_yolo_label(image_id)
        
        # Temukan bbox dengan line_idx yang sesuai
        for i, label in enumerate(labels):
            if label.get('line_idx') == line_idx:
                # Update data yang diberikan
                if class_id is not None:
                    labels[i]['class_id'] = class_id
                
                if bbox is not None:
                    labels[i]['bbox'] = bbox
                
                if extra is not None:
                    labels[i]['extra'] = extra
                
                # Simpan kembali
                return self.save_yolo_label(image_id, labels)
        
        self.logger.warning(f"⚠️ Tidak ditemukan bbox dengan line_idx {line_idx} di {image_id}")
        return False
    
    def delete_bbox(self, image_id: str, line_idx: int) -> bool:
        """
        Hapus bbox dari label.
        
        Args:
            image_id: ID gambar (stem filename tanpa ekstensi)
            line_idx: Indeks baris bbox yang akan dihapus
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load label yang ada
        labels = self.load_yolo_label(image_id)
        
        # Temukan bbox dengan line_idx yang sesuai
        for i, label in enumerate(labels):
            if label.get('line_idx') == line_idx:
                # Hapus bbox
                labels.pop(i)
                
                # Simpan kembali atau hapus file jika kosong
                if not labels:
                    return self.delete_label(image_id)
                else:
                    return self.save_yolo_label(image_id, labels)
        
        self.logger.warning(f"⚠️ Tidak ditemukan bbox dengan line_idx {line_idx} di {image_id}")
        return False
    
    def update_class_ids(self, mapping: Dict[int, int]) -> Dict[str, int]:
        """
        Update ID kelas di semua file label dengan mapping baru.
        
        Args:
            mapping: Mapping dari ID kelas lama ke ID kelas baru {old_id: new_id}
            
        Returns:
            Dictionary berisi jumlah file yang diupdate
        """
        stats = {'total': 0, 'updated': 0, 'errors': 0}
        
        # Proses semua file label
        for label_file in self.label_dir.glob('*.txt'):
            stats['total'] += 1
            image_id = label_file.stem
            
            try:
                # Load label
                labels = self.load_yolo_label(image_id)
                updated = False
                
                # Update class_id jika ada dalam mapping
                for label in labels:
                    old_id = label['class_id']
                    if old_id in mapping:
                        label['class_id'] = mapping[old_id]
                        updated = True
                
                # Simpan kembali jika ada perubahan
                if updated:
                    if self.save_yolo_label(image_id, labels):
                        stats['updated'] += 1
                    else:
                        stats['errors'] += 1
            except Exception as e:
                self.logger.error(f"❌ Error saat update class_id untuk {image_id}: {str(e)}")
                stats['errors'] += 1
        
        self.logger.info(
            f"🔄 Update class ID: {stats['updated']} dari {stats['total']} file diupdate, "
            f"{stats['errors']} error"
        )
        
        return stats