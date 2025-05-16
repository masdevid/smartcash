"""
File: smartcash/dataset/utils/bbox_utils.py
Deskripsi: Utilitas untuk manipulasi dan visualisasi bounding box
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any, Optional, Union


def load_yolo_labels(label_path: str) -> List[List[float]]:
    """
    Muat label YOLO dari file.
    
    Args:
        label_path: Path file label
        
    Returns:
        List label YOLO [class_id, x_center, y_center, width, height]
    """
    labels = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append([class_id, x_center, y_center, width, height])
    except Exception as e:
        print(f"⚠️ Error saat memuat label dari {label_path}: {str(e)}")
        
    return labels


def save_yolo_labels(labels: List[List[float]], output_path: str) -> bool:
    """
    Simpan label YOLO ke file.
    
    Args:
        labels: List label YOLO [class_id, x_center, y_center, width, height]
        output_path: Path file output
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        with open(output_path, 'w') as f:
            for label in labels:
                if len(label) >= 5:
                    class_id = int(label[0])
                    x_center = float(label[1])
                    y_center = float(label[2])
                    width = float(label[3])
                    height = float(label[4])
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        return True
    except Exception as e:
        print(f"⚠️ Error saat menyimpan label ke {output_path}: {str(e)}")
        return False


def draw_bboxes(ax, image, labels: List[List[float]], class_names: List[str] = None, colors: List[Tuple[float, float, float]] = None):
    """
    Gambar bounding box pada gambar.
    
    Args:
        ax: Matplotlib axis
        image: Gambar numpy array
        labels: List label YOLO [class_id, x_center, y_center, width, height]
        class_names: List nama kelas (opsional)
        colors: List warna untuk setiap kelas (opsional)
    """
    height, width = image.shape[:2]
    
    # Default colors jika tidak disediakan
    if colors is None:
        colors = [
            (1, 0, 0),      # Merah
            (0, 1, 0),      # Hijau
            (0, 0, 1),      # Biru
            (1, 1, 0),      # Kuning
            (1, 0, 1),      # Magenta
            (0, 1, 1),      # Cyan
            (0.5, 0, 0),    # Maroon
            (0, 0.5, 0),    # Dark Green
            (0, 0, 0.5),    # Navy
            (0.5, 0.5, 0)   # Olive
        ]
    
    for label in labels:
        if len(label) >= 5:
            class_id = int(label[0])
            x_center = float(label[1])
            y_center = float(label[2])
            bbox_width = float(label[3])
            bbox_height = float(label[4])
            
            # Konversi koordinat YOLO ke piksel
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            box_width = int(bbox_width * width)
            box_height = int(bbox_height * height)
            
            # Pilih warna berdasarkan class_id
            color = colors[class_id % len(colors)]
            
            # Buat rectangle patch
            rect = patches.Rectangle(
                (x1, y1), box_width, box_height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            
            # Tambahkan patch ke axis
            ax.add_patch(rect)
            
            # Tambahkan label kelas jika class_names disediakan
            if class_names is not None and class_id < len(class_names):
                class_name = class_names[class_id]
                ax.text(
                    x1, y1 - 5, class_name,
                    color=color, fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
                )
            else:
                ax.text(
                    x1, y1 - 5, f"Class {class_id}",
                    color=color, fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
                )


def convert_bbox_format(bbox: List[float], from_format: str, to_format: str, image_size: Tuple[int, int]) -> List[float]:
    """
    Konversi format bounding box.
    
    Args:
        bbox: Bounding box dalam format sumber
        from_format: Format sumber ('yolo', 'pascal_voc', 'coco')
        to_format: Format tujuan ('yolo', 'pascal_voc', 'coco')
        image_size: Ukuran gambar (width, height)
        
    Returns:
        Bounding box dalam format tujuan
    """
    width, height = image_size
    
    # Ekstrak koordinat sesuai format sumber
    if from_format == 'yolo':
        # YOLO: [x_center, y_center, width, height] (normalized)
        x_center, y_center, box_width, box_height = bbox
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
    elif from_format == 'pascal_voc':
        # Pascal VOC: [x1, y1, x2, y2] (absolute)
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
    elif from_format == 'coco':
        # COCO: [x1, y1, width, height] (absolute)
        x1, y1, box_width, box_height = bbox
        x2 = x1 + box_width
        y2 = y1 + box_height
        x_center = x1 + box_width / 2
        y_center = y1 + box_height / 2
    else:
        raise ValueError(f"Format sumber tidak didukung: {from_format}")
    
    # Konversi ke format tujuan
    if to_format == 'yolo':
        # YOLO: [x_center, y_center, width, height] (normalized)
        return [
            x_center / width,
            y_center / height,
            box_width / width,
            box_height / height
        ]
    elif to_format == 'pascal_voc':
        # Pascal VOC: [x1, y1, x2, y2] (absolute)
        return [x1, y1, x2, y2]
    elif to_format == 'coco':
        # COCO: [x1, y1, width, height] (absolute)
        return [x1, y1, box_width, box_height]
    else:
        raise ValueError(f"Format tujuan tidak didukung: {to_format}")


def get_bbox_area(bbox: List[float], format: str = 'yolo', image_size: Tuple[int, int] = None) -> float:
    """
    Hitung luas bounding box.
    
    Args:
        bbox: Bounding box
        format: Format bounding box ('yolo', 'pascal_voc', 'coco')
        image_size: Ukuran gambar (width, height), diperlukan untuk format 'yolo'
        
    Returns:
        Luas bounding box
    """
    if format == 'yolo':
        if image_size is None:
            raise ValueError("image_size diperlukan untuk format 'yolo'")
        width, height = image_size
        _, _, box_width, box_height = bbox
        return (box_width * width) * (box_height * height)
    elif format == 'pascal_voc':
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    elif format == 'coco':
        _, _, box_width, box_height = bbox
        return box_width * box_height
    else:
        raise ValueError(f"Format tidak didukung: {format}")


def calculate_iou(bbox1: List[float], bbox2: List[float], format: str = 'yolo', image_size: Tuple[int, int] = None) -> float:
    """
    Hitung Intersection over Union (IoU) antara dua bounding box.
    
    Args:
        bbox1: Bounding box pertama
        bbox2: Bounding box kedua
        format: Format bounding box ('yolo', 'pascal_voc', 'coco')
        image_size: Ukuran gambar (width, height), diperlukan untuk format 'yolo'
        
    Returns:
        Nilai IoU
    """
    # Konversi ke format pascal_voc untuk perhitungan IoU
    if format == 'yolo':
        if image_size is None:
            raise ValueError("image_size diperlukan untuk format 'yolo'")
        width, height = image_size
        
        # YOLO: [x_center, y_center, width, height] (normalized)
        x_center1, y_center1, box_width1, box_height1 = bbox1
        x1_1 = (x_center1 - box_width1 / 2) * width
        y1_1 = (y_center1 - box_height1 / 2) * height
        x2_1 = (x_center1 + box_width1 / 2) * width
        y2_1 = (y_center1 + box_height1 / 2) * height
        
        x_center2, y_center2, box_width2, box_height2 = bbox2
        x1_2 = (x_center2 - box_width2 / 2) * width
        y1_2 = (y_center2 - box_height2 / 2) * height
        x2_2 = (x_center2 + box_width2 / 2) * width
        y2_2 = (y_center2 + box_height2 / 2) * height
    elif format == 'pascal_voc':
        # Pascal VOC: [x1, y1, x2, y2] (absolute)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
    elif format == 'coco':
        # COCO: [x1, y1, width, height] (absolute)
        x1_1, y1_1, box_width1, box_height1 = bbox1
        x2_1 = x1_1 + box_width1
        y2_1 = y1_1 + box_height1
        
        x1_2, y1_2, box_width2, box_height2 = bbox2
        x2_2 = x1_2 + box_width2
        y2_2 = y1_2 + box_height2
    else:
        raise ValueError(f"Format tidak didukung: {format}")
    
    # Hitung koordinat intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Hitung luas intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # Tidak ada intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Hitung luas masing-masing bounding box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Hitung luas union
    union_area = area1 + area2 - intersection_area
    
    # Hitung IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou
