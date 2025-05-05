"""
File: smartcash/dataset/components/labels/format_converter.py
Deskripsi: Utilitas untuk konversi antar format label (YOLO, COCO, VOC, dll)
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.components.geometry.coord_converter import CoordinateConverter


class LabelFormatConverter:
    """Kelas untuk konversi antar format label deteksi objek."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi LabelFormatConverter.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("label_format_converter")
    
    def yolo_to_coco(self, yolo_dir: Union[str, Path], img_dir: Union[str, Path], 
                   class_names: List[str], output_file: Union[str, Path]) -> Dict:
        """
        Konversi format YOLO ke format COCO JSON.
        
        Args:
            yolo_dir: Direktori file label YOLO
            img_dir: Direktori file gambar
            class_names: List nama kelas
            output_file: Path file output JSON COCO
            
        Returns:
            Dict berisi statistik konversi
        """
        import cv2
        
        yolo_dir = Path(yolo_dir)
        img_dir = Path(img_dir)
        output_file = Path(output_file)
        
        # Inisialisasi statistik
        stats = {'labels': 0, 'images': 0, 'annotations': 0, 'errors': 0}
        
        # Inisialisasi struktur COCO
        coco_json = {
            "info": {
                "description": "Dataset SmartCash",
                "version": "1.0",
                "contributor": "SmartCash"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": "https://smartcash.com"
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # Setup kategori
        for i, name in enumerate(class_names):
            coco_json["categories"].append({
                "id": i,
                "name": name,
                "supercategory": "none"
            })
        
        # Proses semua gambar
        image_id = 0
        annotation_id = 0
        
        # Temukan semua file gambar
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        
        for img_path in image_files:
            # Baca ukuran gambar
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.warning(f"⚠️ Gambar tidak dapat dibaca: {img_path}")
                    stats['errors'] += 1
                    continue
                
                height, width, _ = img.shape
            except Exception as e:
                self.logger.error(f"❌ Error saat membaca gambar {img_path}: {str(e)}")
                stats['errors'] += 1
                continue
            
            # Tambahkan entry gambar
            coco_json["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": img_path.name,
                "height": height,
                "width": width,
                "date_captured": ""
            })
            
            # Periksa file label
            label_path = yolo_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                bbox = list(map(float, parts[1:5]))  # [x_center, y_center, width, height]
                                
                                # Konversi ke format COCO
                                x, y, w, h = CoordinateConverter.yolo_to_coco(bbox, width, height)
                                
                                # Tambahkan anotasi
                                coco_json["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": cls_id,
                                    "bbox": [x, y, w, h],
                                    "area": w * h,
                                    "segmentation": [],
                                    "iscrowd": 0
                                })
                                
                                annotation_id += 1
                                stats['annotations'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error saat membaca label {label_path}: {str(e)}")
                    stats['errors'] += 1
            
            image_id += 1
            stats['images'] += 1
        
        # Simpan file JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(coco_json, f, indent=2)
            
            stats['labels'] = len(coco_json["annotations"])
            self.logger.success(
                f"✅ Konversi YOLO ke COCO selesai: {stats['images']} gambar, "
                f"{stats['annotations']} anotasi, {stats['errors']} error"
            )
        except Exception as e:
            self.logger.error(f"❌ Error saat menyimpan file JSON: {str(e)}")
            stats['errors'] += 1
        
        return stats
    
    def coco_to_yolo(self, coco_file: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """
        Konversi format COCO JSON ke format YOLO.
        
        Args:
            coco_file: Path ke file JSON COCO
            output_dir: Direktori output label YOLO
            
        Returns:
            Dict berisi statistik konversi
        """
        coco_file = Path(coco_file)
        output_dir = Path(output_dir)
        
        # Inisialisasi statistik
        stats = {'labels': 0, 'images': 0, 'annotations': 0, 'errors': 0}
        
        # Buat direktori output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Baca file COCO
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Proses setiap gambar
            image_bbox = {}  # {image_id: [(category_id, bbox), ...]}
            
            # Mapping image_id ke file_name dan dimensi
            image_info = {}  # {image_id: (file_name, width, height)}
            
            for img in coco_data.get('images', []):
                image_id = img['id']
                file_name = img['file_name']
                width = img['width']
                height = img['height']
                
                image_info[image_id] = (file_name, width, height)
                image_bbox[image_id] = []
            
            # Proses semua anotasi
            for ann in coco_data.get('annotations', []):
                image_id = ann['image_id']
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height] format COCO
                
                if image_id in image_bbox:
                    image_bbox[image_id].append((category_id, bbox))
                    stats['annotations'] += 1
            
            # Buat file YOLO
            for image_id, bboxes in image_bbox.items():
                if not bboxes:
                    continue
                
                file_name, width, height = image_info[image_id]
                base_name = Path(file_name).stem
                output_path = output_dir / f"{base_name}.txt"
                
                try:
                    with open(output_path, 'w') as f:
                        for category_id, bbox in bboxes:
                            # Konversi ke format YOLO
                            yolo_bbox = CoordinateConverter.coco_to_yolo(bbox, width, height)
                            
                            # Format YOLO: class_id x_center y_center width height
                            line = f"{category_id} {' '.join(map(str, yolo_bbox))}\n"
                            f.write(line)
                    
                    stats['labels'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error saat menulis label {output_path}: {str(e)}")
                    stats['errors'] += 1
            
            stats['images'] = len(image_info)
            self.logger.success(
                f"✅ Konversi COCO ke YOLO selesai: {stats['images']} gambar, "
                f"{stats['annotations']} anotasi, {stats['errors']} error"
            )
        except Exception as e:
            self.logger.error(f"❌ Error saat konversi COCO ke YOLO: {str(e)}")
            stats['errors'] += 1
        
        return stats
    
    def yolo_to_voc(self, yolo_dir: Union[str, Path], img_dir: Union[str, Path], 
                   class_names: List[str], output_dir: Union[str, Path]) -> Dict:
        """
        Konversi format YOLO ke format Pascal VOC XML.
        
        Args:
            yolo_dir: Direktori file label YOLO
            img_dir: Direktori file gambar
            class_names: List nama kelas
            output_dir: Direktori output XML VOC
            
        Returns:
            Dict berisi statistik konversi
        """
        import cv2
        
        yolo_dir = Path(yolo_dir)
        img_dir = Path(img_dir)
        output_dir = Path(output_dir)
        
        # Inisialisasi statistik
        stats = {'labels': 0, 'images': 0, 'annotations': 0, 'errors': 0}
        
        # Buat direktori output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temukan semua file gambar
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        
        # Proses setiap gambar
        for img_path in image_files:
            # Baca ukuran gambar
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.warning(f"⚠️ Gambar tidak dapat dibaca: {img_path}")
                    stats['errors'] += 1
                    continue
                
                height, width, _ = img.shape
            except Exception as e:
                self.logger.error(f"❌ Error saat membaca gambar {img_path}: {str(e)}")
                stats['errors'] += 1
                continue
            
            # Periksa file label
            label_path = yolo_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                # Inisialisasi XML
                annotation = ET.Element("annotation")
                
                # Metadata gambar
                ET.SubElement(annotation, "folder").text = str(img_dir.name)
                ET.SubElement(annotation, "filename").text = img_path.name
                ET.SubElement(annotation, "path").text = str(img_path)
                
                source = ET.SubElement(annotation, "source")
                ET.SubElement(source, "database").text = "SmartCash"
                
                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = str(width)
                ET.SubElement(size, "height").text = str(height)
                ET.SubElement(size, "depth").text = "3"
                
                ET.SubElement(annotation, "segmented").text = "0"
                
                # Baca dan konversi anotasi
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                bbox = list(map(float, parts[1:5]))  # [x_center, y_center, width, height]
                                
                                # Konversi ke format corners
                                x_min, y_min, x_max, y_max = CoordinateConverter.yolo_to_corners(bbox, width, height)
                                
                                # Class name
                                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                                
                                # Tambahkan objek
                                obj = ET.SubElement(annotation, "object")
                                ET.SubElement(obj, "name").text = class_name
                                ET.SubElement(obj, "pose").text = "Unspecified"
                                ET.SubElement(obj, "truncated").text = "0"
                                ET.SubElement(obj, "difficult").text = "0"
                                
                                bndbox = ET.SubElement(obj, "bndbox")
                                ET.SubElement(bndbox, "xmin").text = str(x_min)
                                ET.SubElement(bndbox, "ymin").text = str(y_min)
                                ET.SubElement(bndbox, "xmax").text = str(x_max)
                                ET.SubElement(bndbox, "ymax").text = str(y_max)
                                
                                stats['annotations'] += 1
                    
                    # Simpan XML
                    output_path = output_dir / f"{img_path.stem}.xml"
                    
                    tree = ET.ElementTree(annotation)
                    tree.write(output_path)
                    
                    stats['labels'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error saat proses label {label_path}: {str(e)}")
                    stats['errors'] += 1
            
            stats['images'] += 1
        
        self.logger.success(
            f"✅ Konversi YOLO ke VOC selesai: {stats['images']} gambar, "
            f"{stats['annotations']} anotasi, {stats['errors']} error"
        )
        
        return stats
    
    def voc_to_yolo(self, voc_dir: Union[str, Path], class_names: List[str], 
                   output_dir: Union[str, Path]) -> Dict:
        """
        Konversi format Pascal VOC XML ke format YOLO.
        
        Args:
            voc_dir: Direktori file XML VOC
            class_names: List nama kelas
            output_dir: Direktori output label YOLO
            
        Returns:
            Dict berisi statistik konversi
        """
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        
        # Inisialisasi statistik
        stats = {'labels': 0, 'images': 0, 'annotations': 0, 'errors': 0}
        
        # Buat direktori output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buat mapping nama kelas ke indeks
        class_map = {name: i for i, name in enumerate(class_names)}
        
        # Temukan semua file XML
        xml_files = list(voc_dir.glob('*.xml'))
        
        # Proses setiap file XML
        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Ambil ukuran gambar
                size_elem = root.find('size')
                if size_elem is None:
                    self.logger.warning(f"⚠️ Tidak ada elemen size di {xml_path}")
                    stats['errors'] += 1
                    continue
                
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
                
                # Hasil anotasi untuk file ini
                yolo_lines = []
                
                # Proses semua objek
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    
                    # Map nama kelas ke ID
                    if name in class_map:
                        cls_id = class_map[name]
                    else:
                        self.logger.warning(f"⚠️ Kelas tidak dikenal: {name} di {xml_path}")
                        continue
                    
                    # Ambil bounding box
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    
                    x_min = float(bbox.find('xmin').text)
                    y_min = float(bbox.find('ymin').text)
                    x_max = float(bbox.find('xmax').text)
                    y_max = float(bbox.find('ymax').text)
                    
                    # Konversi ke format YOLO
                    yolo_bbox = CoordinateConverter.corners_to_yolo([x_min, y_min, x_max, y_max], width, height)
                    
                    # Format YOLO: class_id x_center y_center width height
                    line = f"{cls_id} {' '.join(map(str, yolo_bbox))}"
                    yolo_lines.append(line)
                    
                    stats['annotations'] += 1
                
                # Simpan ke file YOLO jika ada anotasi
                if yolo_lines:
                    output_path = output_dir / f"{xml_path.stem}.txt"
                    
                    with open(output_path, 'w') as f:
                        for line in yolo_lines:
                            f.write(line + '\n')
                    
                    stats['labels'] += 1
                
                stats['images'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error saat konversi {xml_path}: {str(e)}")
                stats['errors'] += 1
        
        self.logger.success(
            f"✅ Konversi VOC ke YOLO selesai: {stats['images']} gambar, "
            f"{stats['annotations']} anotasi, {stats['errors']} error"
        )
        
        return stats