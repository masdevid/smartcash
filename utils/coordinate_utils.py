"""
File: smartcash/utils/coordinate_utils.py
Author: Alfrida Sabar
Deskripsi: Utilitas terintegrasi untuk operasi pada koordinat, bounding box, dan polygon
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm.auto import tqdm
from pathlib import Path

try:
    from shapely.geometry import Polygon, box
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from smartcash.utils.logger import get_logger

class CoordinateUtils:
    """
    Kelas utilitas untuk berbagai operasi pada koordinat, bounding box, dan polygon.
    Menggabungkan fungsionalitas dari CoordinateNormalizer dan PolygonMetrics.
    """
    
    def __init__(
        self, 
        logger: Optional = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi utilitas koordinat.
        
        Args:
            logger: Logger untuk logging
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.logger = logger or get_logger("coord_utils")
        self.num_workers = num_workers
        self._lock = threading.Lock()
    
    def normalize_bbox(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float]],
        image_size: Tuple[int, int],
        format: str = 'xyxy'
    ) -> Union[List[float], Tuple[float, float, float, float]]:
        """
        Normalisasi koordinat bounding box menjadi relatif (0-1).
        
        Args:
            bbox: Bounding box dalam format [x1, y1, x2, y2] (xyxy) atau [x, y, w, h] (xywh)
            image_size: Ukuran gambar (width, height)
            format: Format bbox ('xyxy' atau 'xywh')
            
        Returns:
            Koordinat bounding box ternormalisasi
        """
        width, height = image_size
        
        if format == 'xyxy':
            x1, y1, x2, y2 = bbox
            return [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height))
            ]
        elif format == 'xywh':
            x, y, w, h = bbox
            return [
                max(0.0, min(1.0, x / width)),
                max(0.0, min(1.0, y / height)),
                max(0.0, min(1.0, w / width)),
                max(0.0, min(1.0, h / height))
            ]
        elif format == 'yolo':
            # Format YOLO (x_center, y_center, width, height)
            x_center, y_center, w, h = bbox
            return [
                max(0.0, min(1.0, x_center / width)),
                max(0.0, min(1.0, y_center / height)),
                max(0.0, min(1.0, w / width)),
                max(0.0, min(1.0, h / height))
            ]
        else:
            self.logger.warning(f"⚠️ Format bbox tidak dikenal: {format}")
            return bbox
    
    def denormalize_bbox(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float]],
        image_size: Tuple[int, int],
        format: str = 'xyxy'
    ) -> Union[List[float], Tuple[float, float, float, float]]:
        """
        Denormalisasi koordinat bounding box relatif (0-1) menjadi absolut.
        
        Args:
            bbox: Bounding box ternormalisasi
            image_size: Ukuran gambar (width, height)
            format: Format bbox ('xyxy' atau 'xywh')
            
        Returns:
            Koordinat bounding box absolut
        """
        width, height = image_size
        
        if format == 'xyxy':
            x1, y1, x2, y2 = bbox
            return [
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height
            ]
        elif format == 'xywh':
            x, y, w, h = bbox
            return [
                x * width,
                y * height,
                w * width,
                h * height
            ]
        elif format == 'yolo':
            # Format YOLO (x_center, y_center, width, height)
            x_center, y_center, w, h = bbox
            return [
                x_center * width,
                y_center * height,
                w * width,
                h * height
            ]
        else:
            self.logger.warning(f"⚠️ Format bbox tidak dikenal: {format}")
            return bbox
    
    def normalize_polygon(
        self,
        points: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> List[float]:
        """
        Normalisasi koordinat polygon menjadi relatif (0-1).
        
        Args:
            points: List koordinat [(x1,y1), (x2,y2), ...]
            image_size: Ukuran gambar (width, height)
            
        Returns:
            List koordinat ternormalisasi [x1,y1,x2,y2,...]
        """
        width, height = image_size
        
        # Flatten dan normalisasi koordinat
        normalized = []
        for x, y in points:
            normalized.extend([
                max(0.0, min(1.0, x / width)),
                max(0.0, min(1.0, y / height))
            ])
            
        return normalized
    
    def denormalize_polygon(
        self,
        normalized: List[float],
        image_size: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Denormalisasi koordinat polygon relatif (0-1) menjadi absolut.
        
        Args:
            normalized: List koordinat ternormalisasi [x1,y1,x2,y2,...]
            image_size: Ukuran gambar (width, height)
            
        Returns:
            List koordinat absolut [(x1,y1), (x2,y2), ...]
        """
        width, height = image_size
        points = []
        
        # Reconstruct koordinat
        for i in range(0, len(normalized), 2):
            if i + 1 < len(normalized):
                points.append((
                    normalized[i] * width,
                    normalized[i+1] * height
                ))
            
        return points
    
    def convert_bbox_format(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float]],
        from_format: str,
        to_format: str
    ) -> Union[List[float], Tuple[float, float, float, float]]:
        """
        Konversi antar format bounding box.
        
        Args:
            bbox: Bounding box
            from_format: Format asal ('xyxy', 'xywh', 'yolo')
            to_format: Format tujuan ('xyxy', 'xywh', 'yolo')
            
        Returns:
            Bounding box dalam format tujuan
        """
        # Jika format sama, kembalikan langsung
        if from_format == to_format:
            return bbox
        
        # Konversi ke xyxy sebagai format perantara
        if from_format == 'xywh':
            x, y, w, h = bbox
            xyxy = [x, y, x + w, y + h]
        elif from_format == 'yolo':
            # YOLO format: x_center, y_center, width, height
            x_center, y_center, w, h = bbox
            x1 = x_center - w/2
            y1 = y_center - h/2
            x2 = x_center + w/2
            y2 = y_center + h/2
            xyxy = [x1, y1, x2, y2]
        else:  # from_format == 'xyxy'
            xyxy = bbox
        
        # Konversi dari xyxy ke format tujuan
        if to_format == 'xywh':
            x1, y1, x2, y2 = xyxy
            return [x1, y1, x2 - x1, y2 - y1]
        elif to_format == 'yolo':
            # YOLO format: x_center, y_center, width, height
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width/2
            y_center = y1 + height/2
            return [x_center, y_center, width, height]
        else:  # to_format == 'xyxy'
            return xyxy
    
    def calculate_iou(
        self, 
        box1: Union[List[float], np.ndarray], 
        box2: Union[List[float], np.ndarray],
        format: str = 'xyxy'
    ) -> float:
        """
        Hitung IoU (Intersection over Union) antara dua bounding box.
        
        Args:
            box1: Bounding box pertama
            box2: Bounding box kedua
            format: Format box ('xyxy', 'xywh', 'yolo')
            
        Returns:
            Nilai IoU (0-1)
        """
        # Konversi ke format xyxy
        if format != 'xyxy':
            box1 = self.convert_bbox_format(box1, format, 'xyxy')
            box2 = self.convert_bbox_format(box2, format, 'xyxy')
        
        # Pastikan box dalam format yang benar
        if len(box1) != 4 or len(box2) != 4:
            self.logger.warning("⚠️ Format box tidak valid, harus [x1, y1, x2, y2]")
            return 0.0
        
        # Ekstrak koordinat
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Cek validitas box
        if x1_1 > x2_1 or y1_1 > y2_1 or x1_2 > x2_2 or y1_2 > y2_2:
            self.logger.warning("⚠️ Koordinat box tidak valid (x1 > x2 or y1 > y2)")
            return 0.0
        
        # Hitung koordinat intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Cek apakah ada intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Hitung area intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Hitung area masing-masing box
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Hitung area union
        union_area = box1_area + box2_area - intersection_area
        
        # Hitung IoU
        iou = intersection_area / union_area
        
        return iou
    
    def calculate_poly_iou(
        self, 
        poly1: List[Tuple[float, float]], 
        poly2: List[Tuple[float, float]]
    ) -> float:
        """
        Hitung IoU (Intersection over Union) antara dua polygon.
        
        Args:
            poly1: Polygon pertama sebagai list koordinat [(x1, y1), (x2, y2), ...]
            poly2: Polygon kedua sebagai list koordinat [(x1, y1), (x2, y2), ...]
            
        Returns:
            Nilai IoU (0-1)
        """
        if not HAS_SHAPELY:
            self.logger.warning("⚠️ Shapely library tidak tersedia untuk operasi polygon")
            return self._fallback_poly_iou(poly1, poly2)
        
        try:
            # Konversi ke objek Polygon
            polygon1 = Polygon(poly1)
            polygon2 = Polygon(poly2)
            
            # Cek validitas polygon
            if not polygon1.is_valid or not polygon2.is_valid:
                self.logger.warning("⚠️ Polygon tidak valid")
                return 0.0
                
            # Hitung intersection
            intersection = polygon1.intersection(polygon2).area
            
            # Hitung union
            union = polygon1.union(polygon2).area
            
            # Handle division by zero
            if union == 0:
                return 0.0
                
            # Hitung IoU
            iou = intersection / union
            
            return iou
            
        except Exception as e:
            self.logger.error(f"❌ Error menghitung Polygon IoU: {str(e)}")
            return self._fallback_poly_iou(poly1, poly2)
    
    def _fallback_poly_iou(
        self, 
        poly1: List[Tuple[float, float]], 
        poly2: List[Tuple[float, float]]
    ) -> float:
        """
        Fallback sederhana untuk menghitung IoU polygon tanpa shapely.
        Menggunakan bounding box sebagai pendekatan.
        
        Args:
            poly1: Polygon pertama sebagai list koordinat
            poly2: Polygon kedua sebagai list koordinat
            
        Returns:
            Nilai IoU approksimasi
        """
        # Ekstrak bounding box dari polygon
        x1_1 = min(p[0] for p in poly1)
        y1_1 = min(p[1] for p in poly1)
        x2_1 = max(p[0] for p in poly1)
        y2_1 = max(p[1] for p in poly1)
        
        x1_2 = min(p[0] for p in poly2)
        y1_2 = min(p[1] for p in poly2)
        x2_2 = max(p[0] for p in poly2)
        y2_2 = max(p[1] for p in poly2)
        
        # Hitung IoU dengan bounding box
        bb1 = [x1_1, y1_1, x2_1, y2_1]
        bb2 = [x1_2, y1_2, x2_2, y2_2]
        
        return self.calculate_iou(bb1, bb2)
    
    def box_to_polygon(
        self, 
        box: List[float], 
        format: str = 'xyxy'
    ) -> List[Tuple[float, float]]:
        """
        Konversi bounding box ke format polygon.
        
        Args:
            box: Bounding box dalam format [x1, y1, x2, y2] (xyxy) atau [x, y, w, h] (xywh)
            format: Format box ('xyxy' atau 'xywh')
            
        Returns:
            List koordinat polygon [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        # Konversi ke format xyxy
        if format != 'xyxy':
            box = self.convert_bbox_format(box, format, 'xyxy')
        
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def calculate_mean_iou(
        self, 
        boxes_true: List[List[float]], 
        boxes_pred: List[List[float]], 
        threshold: float = 0.5,
        format: str = 'xyxy'
    ) -> Dict[str, float]:
        """
        Hitung mIoU (mean Intersection over Union) untuk evaluasi deteksi.
        
        Args:
            boxes_true: List box ground truth
            boxes_pred: List box prediksi
            threshold: Threshold IoU untuk true positive
            format: Format box ('xyxy', 'xywh', 'yolo')
            
        Returns:
            Dictionary berisi mIoU dan metrik terkait
        """
        if not boxes_true or not boxes_pred:
            return {
                'miou': 0.0,
                'tp': 0,
                'fp': len(boxes_pred),
                'fn': len(boxes_true),
                'precision': 0.0,
                'recall': 0.0
            }
            
        # Hitung IoU untuk semua kombinasi box
        ious = np.zeros((len(boxes_true), len(boxes_pred)))
        for i, box_true in enumerate(boxes_true):
            for j, box_pred in enumerate(boxes_pred):
                ious[i, j] = self.calculate_iou(box_true, box_pred, format)
        
        # Temukan matching dengan IoU tertinggi
        matched_indices = []
        unmatched_true = list(range(len(boxes_true)))
        unmatched_pred = list(range(len(boxes_pred)))
        
        # Untuk setiap ground truth, temukan prediksi dengan IoU tertinggi
        while unmatched_true and unmatched_pred:
            # Temukan pasangan dengan IoU tertinggi
            if len(unmatched_true) == 1 and len(unmatched_pred) == 1:
                i, j = unmatched_true[0], unmatched_pred[0]
                if ious[i, j] >= threshold:
                    matched_indices.append((i, j))
                break
            
            # Temukan pasangan dengan IoU tertinggi
            max_iou = 0
            max_pair = None
            
            for i in unmatched_true:
                for j in unmatched_pred:
                    if ious[i, j] > max_iou:
                        max_iou = ious[i, j]
                        max_pair = (i, j)
            
            # Jika tidak ada match yang memenuhi threshold, keluar
            if max_pair is None or max_iou < threshold:
                break
                
            # Tambahkan ke matches dan hapus dari unmatched
            i, j = max_pair
            matched_indices.append((i, j))
            unmatched_true.remove(i)
            unmatched_pred.remove(j)
        
        # Hitung true positives, false positives, false negatives
        tp = len(matched_indices)
        fp = len(boxes_pred) - tp
        fn = len(boxes_true) - tp
        
        # Hitung precision, recall
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        
        # Hitung mean IoU dari matched pairs
        if tp > 0:
            total_iou = sum(ious[i, j] for i, j in matched_indices)
            miou = total_iou / tp
        else:
            miou = 0.0
            
        return {
            'miou': miou,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall
        }
    
    def process_label_file(
        self,
        label_path: Union[str, Path],
        image_size: Tuple[int, int],
        save_path: Optional[Union[str, Path]] = None,
        normalize: bool = True
    ) -> List[Dict]:
        """
        Proses file label untuk normalisasi/denormalisasi koordinat.
        
        Args:
            label_path: Path ke file label
            image_size: Ukuran gambar (width, height)
            save_path: Path untuk menyimpan hasil (optional)
            normalize: Normalisasi jika True, denormalisasi jika False
            
        Returns:
            List label yang diproses dalam format dictionary
        """
        label_path = Path(label_path)
        if save_path:
            save_path = Path(save_path)
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            processed_lines = []
            processed_labels = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:  # Minimal class_id dan 4 koordinat
                    continue
                    
                class_id = int(parts[0])
                
                # Convert koordinat ke float
                coords = [float(x) for x in parts[1:]]
                
                # Check format koordinat (YOLO atau deteksi lain)
                if len(coords) == 4 and all(0 <= c <= 1 for c in coords[:4]):
                    # Format YOLO (x_center, y_center, width, height) ternormalisasi
                    if normalize:
                        # Sudah ternormalisasi
                        normalized = coords[:4]
                    else:
                        # Denormalisasi
                        x_center, y_center, width, height = coords[:4]
                        x_center = x_center * image_size[0]
                        y_center = y_center * image_size[1]
                        width = width * image_size[0]
                        height = height * image_size[1]
                        normalized = [x_center, y_center, width, height]
                elif len(coords) >= 4:
                    # Format polygon atau custom
                    if normalize:
                        if len(coords) % 2 == 0:
                            # Polygon (x1, y1, x2, y2, ...)
                            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                            normalized = self.normalize_polygon(points, image_size)
                        else:
                            # Format tidak dikenal
                            self.logger.warning(f"⚠️ Format koordinat tidak dikenal pada {label_path}")
                            continue
                    else:
                        if len(coords) % 2 == 0:
                            # Polygon (x1, y1, x2, y2, ...)
                            normalized = self.denormalize_polygon(coords, image_size)
                            normalized = [coord for point in normalized for coord in point]
                        else:
                            # Format tidak dikenal
                            self.logger.warning(f"⚠️ Format koordinat tidak dikenal pada {label_path}")
                            continue
                        
                # Format output
                if isinstance(normalized, list) and len(normalized) >= 4:
                    coords_str = ' '.join(map(str, normalized))
                    processed_lines.append(f"{class_id} {coords_str}\n")
                    
                    # Tambahkan ke hasil dalam format dictionary
                    processed_labels.append({
                        'class_id': class_id,
                        'coordinates': normalized
                    })
            
            # Simpan hasil
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.writelines(processed_lines)
                    
            return processed_labels
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memproses {label_path}: {str(e)}")
            raise e
    
    def process_dataset(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        normalize: bool = True
    ) -> Dict[str, int]:
        """
        Proses seluruh dataset untuk normalisasi/denormalisasi koordinat.
        
        Args:
            label_dir: Direktori file label
            image_dir: Direktori gambar
            output_dir: Direktori output (optional)
            normalize: Normalisasi jika True, denormalisasi jika False
            
        Returns:
            Dictionary statistik pemrosesan
        """
        operation = "normalisasi" if normalize else "denormalisasi"
        self.logger.start(f"🎯 Memulai {operation} koordinat di {label_dir}")
        
        stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        try:
            label_dir = Path(label_dir)
            image_dir = Path(image_dir)
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
            # Collect file paths
            label_files = list(label_dir.glob("*.txt"))
            
            if not label_files:
                self.logger.warning(f"⚠️ Tidak ada file label ditemukan di {label_dir}")
                return stats
            
            # Setup progress bar
            pbar = tqdm(
                total=len(label_files),
                desc=f"💫 {operation.capitalize()} koordinat"
            )
            
            def process_file(label_file: Path):
                try:
                    with self._lock:
                        # Get corresponding image size
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            image_file = image_dir / (label_file.stem + ext)
                            if image_file.exists():
                                break
                        else:
                            self.logger.warning(f"⚠️ Gambar tidak ditemukan untuk {label_file.name}")
                            stats['skipped'] += 1
                            pbar.update(1)
                            return
                            
                        # Get image size
                        img = cv2.imread(str(image_file))
                        if img is None:
                            self.logger.warning(f"⚠️ Gagal membaca gambar {image_file.name}")
                            stats['skipped'] += 1
                            pbar.update(1)
                            return
                            
                        image_size = (img.shape[1], img.shape[0])  # width, height
                    
                    # Process label
                    if output_dir:
                        save_path = output_dir / label_file.name
                    else:
                        save_path = None
                        
                    self.process_label_file(
                        label_file, 
                        image_size, 
                        save_path,
                        normalize
                    )
                    
                    with self._lock:
                        stats['processed'] += 1
                        pbar.update(1)
                    
                except Exception as e:
                    with self._lock:
                        self.logger.error(f"❌ Gagal memproses {label_file.name}: {str(e)}")
                        stats['failed'] += 1
                        pbar.update(1)
            
            # Process dengan multi-threading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                executor.map(process_file, label_files)
                
            pbar.close()
            self.logger.success(f"✨ {operation.capitalize()} koordinat selesai!")
            
            # Log statistik
            self.logger.info(
                f"📊 Statistik {operation}:\n"
                f"   • Berhasil: {stats['processed']}\n"
                f"   • Gagal: {stats['failed']}\n"
                f"   • Dilewati: {stats['skipped']}"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ {operation.capitalize()} dataset gagal: {str(e)}")
            raise e


# Fungsi helper untuk operasi koordinat umum
def calculate_iou(
    box1: List[float], 
    box2: List[float], 
    format: str = 'xyxy'
) -> float:
    """
    Fungsi helper untuk menghitung IoU antara dua box.
    
    Args:
        box1: Bounding box pertama
        box2: Bounding box kedua
        format: Format box ('xyxy', 'xywh', 'yolo')
        
    Returns:
        Nilai IoU (0-1)
    """
    coord_utils = CoordinateUtils()
    return coord_utils.calculate_iou(box1, box2, format)