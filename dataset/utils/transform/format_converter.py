"""
File: smartcash/dataset/utils/transform/format_converter.py
Deskripsi: Konversi format dataset antara YOLO, COCO, Pascal VOC, dan format lainnya
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.transform.bbox_transform import BBoxTransformer


class FormatConverter:
    """Utilitas untuk konversi format dataset antar format umum."""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi FormatConverter.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("format_converter")
        self.transformer = BBoxTransformer()
    
    def yolo_to_coco(
        self, 
        source_dir: Union[str, Path], 
        target_path: Union[str, Path],
        class_map: Dict[int, str] = None,
        img_ext: str = '.jpg'
    ) -> Dict[str, Any]:
        """
        Konversi dataset dari format YOLO ke format COCO.
        
        Args:
            source_dir: Path ke direktori dataset YOLO
            target_path: Path target untuk file JSON COCO
            class_map: Mapping dari ID kelas ke nama kelas
            img_ext: Ekstensi file gambar
            
        Returns:
            Dictionary hasil konversi
        """
        source_path = Path(source_dir)
        images_dir = source_path / 'images'
        labels_dir = source_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori tidak lengkap: {source_path}")
            return {'status': 'error', 'message': 'Direktori tidak lengkap'}
            
        # Buat struktur dasar COCO
        coco_data = {
            "info": {
                "description": "Converted from YOLO format",
                "version": "1.0",
                "year": 2023,
                "contributor": "SmartCash FormatConverter"
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Buat kategori
        if class_map:
            for class_id, class_name in class_map.items():
                coco_data["categories"].append({
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "none"
                })
        
        # Proses gambar dan anotasi
        image_id = 0
        annotation_id = 0
        
        # Daftar file gambar
        image_files = list(images_dir.glob(f'*{img_ext}'))
        self.logger.info(f"🔄 Mengkonversi {len(image_files)} gambar dari YOLO ke COCO")
        
        for img_file in image_files:
            # Load gambar untuk mendapatkan dimensi
            import cv2
            img = cv2.imread(str(img_file))
            if img is None:
                self.logger.warning(f"⚠️ Gagal membaca gambar: {img_file}")
                continue
                
            h, w = img.shape[:2]
            
            # Tambahkan info gambar
            image_id += 1
            coco_data["images"].append({
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": img_file.name,
                "license": 1
            })
            
            # Proses label jika ada
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        # Parse YOLO format: class_id, x_center, y_center, width, height
                        class_id = int(float(parts[0]))
                        bbox_yolo = [float(x) for x in parts[1:5]]
                        
                        # Konversi ke format COCO: [x, y, width, height]
                        bbox_coco = self.transformer.yolo_to_coco(bbox_yolo, w, h)
                        
                        # Tambahkan anotasi
                        annotation_id += 1
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": bbox_coco,
                            "area": bbox_coco[2] * bbox_coco[3],
                            "segmentation": [],
                            "iscrowd": 0
                        })
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat memproses label {label_file}: {str(e)}")
        
        # Simpan ke file JSON
        with open(target_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        self.logger.success(
            f"✅ Konversi selesai:\n"
            f"   • Gambar: {len(coco_data['images'])}\n"
            f"   • Anotasi: {len(coco_data['annotations'])}\n"
            f"   • Kategori: {len(coco_data['categories'])}\n"
            f"   • Output: {target_path}"
        )
        
        return {
            'status': 'success',
            'images': len(coco_data['images']),
            'annotations': len(coco_data['annotations']),
            'categories': len(coco_data['categories']),
            'output_path': str(target_path)
        }
    
    def coco_to_yolo(
        self, 
        source_path: Union[str, Path], 
        target_dir: Union[str, Path],
        images_src: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Konversi dataset dari format COCO ke format YOLO.
        
        Args:
            source_path: Path ke file JSON COCO
            target_dir: Direktori target untuk format YOLO
            images_src: Path untuk gambar sumber (jika berbeda dari path JSON)
            
        Returns:
            Dictionary hasil konversi
        """
        source_path = Path(source_path)
        target_dir = Path(target_dir)
        
        # Struktur output YOLO
        images_dir = target_dir / 'images'
        labels_dir = target_dir / 'labels'
        
        # Buat direktori output
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load file COCO
        try:
            with open(source_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Gagal membaca file COCO: {str(e)}")
            return {'status': 'error', 'message': f'Gagal membaca file COCO: {str(e)}'}
            
        # Buat mapping image_id ke info gambar
        images_map = {img['id']: img for img in coco_data['images']}
        
        # Buat mapping kategori
        categories_map = {cat['id']: cat for cat in coco_data['categories']}
        
        # Buat direktori source gambar
        if images_src is None:
            images_src = source_path.parent / 'images'
        else:
            images_src = Path(images_src)
            
        if not images_src.exists():
            self.logger.warning(f"⚠️ Direktori gambar sumber tidak ditemukan: {images_src}")
            
        # Kelompokkan anotasi berdasarkan image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
            
        # Proses setiap gambar
        processed_count = 0
        for image_id, annotations in annotations_by_image.items():
            if image_id not in images_map:
                continue
                
            image_info = images_map[image_id]
            img_filename = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']
            
            # Salin gambar jika ada
            src_img_path = images_src / img_filename
            if src_img_path.exists():
                import shutil
                shutil.copy2(src_img_path, images_dir / img_filename)
            
            # Buat file label YOLO
            label_lines = []
            for ann in annotations:
                category_id = ann['category_id']
                bbox_coco = ann['bbox']  # [x, y, width, height]
                
                # Konversi ke format YOLO
                bbox_yolo = self.transformer.coco_to_yolo(bbox_coco, img_width, img_height)
                
                # Format baris YOLO: class_id x_center y_center width height
                label_lines.append(f"{category_id} {' '.join(map(str, bbox_yolo))}")
                
            # Simpan file label
            label_path = labels_dir / f"{Path(img_filename).stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
                
            processed_count += 1
                
        self.logger.success(
            f"✅ Konversi selesai:\n"
            f"   • Gambar diproses: {processed_count}\n"
            f"   • Output direktori: {target_dir}"
        )
        
        return {
            'status': 'success',
            'processed_images': processed_count,
            'output_dir': str(target_dir)
        }
    
    def yolo_to_voc(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        class_map: Dict[int, str] = None,
        img_ext: str = '.jpg'
    ) -> Dict[str, Any]:
        """
        Konversi dataset dari format YOLO ke format Pascal VOC.
        
        Args:
            source_dir: Path ke direktori dataset YOLO
            target_dir: Direktori target untuk format VOC
            class_map: Mapping dari ID kelas ke nama kelas
            img_ext: Ekstensi file gambar
            
        Returns:
            Dictionary hasil konversi
        """
        source_path = Path(source_dir)
        target_dir = Path(target_dir)
        
        images_dir = source_path / 'images'
        labels_dir = source_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori tidak lengkap: {source_path}")
            return {'status': 'error', 'message': 'Direktori tidak lengkap'}
            
        # Buat direktori output VOC
        voc_annotations_dir = target_dir / 'Annotations'
        voc_images_dir = target_dir / 'JPEGImages'
        voc_imagesets_dir = target_dir / 'ImageSets' / 'Main'
        
        voc_annotations_dir.mkdir(parents=True, exist_ok=True)
        voc_images_dir.mkdir(parents=True, exist_ok=True)
        voc_imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # Daftar file gambar
        image_files = list(images_dir.glob(f'*{img_ext}'))
        self.logger.info(f"🔄 Mengkonversi {len(image_files)} gambar dari YOLO ke VOC")
        
        # Default class map jika tidak disediakan
        if class_map is None:
            class_map = {}
            
        processed_count = 0
        image_names = []
        
        for img_file in image_files:
            # Load gambar untuk mendapatkan dimensi
            import cv2
            img = cv2.imread(str(img_file))
            if img is None:
                self.logger.warning(f"⚠️ Gagal membaca gambar: {img_file}")
                continue
                
            h, w = img.shape[:2]
            img_name = img_file.stem
            image_names.append(img_name)
            
            # Salin gambar
            import shutil
            shutil.copy2(img_file, voc_images_dir / img_file.name)
            
            # Buat XML annotation
            annotation = ET.Element('annotation')
            
            # Tambahkan file info
            ET.SubElement(annotation, 'folder').text = 'JPEGImages'
            ET.SubElement(annotation, 'filename').text = img_file.name
            
            # Tambahkan size info
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(w)
            ET.SubElement(size, 'height').text = str(h)
            ET.SubElement(size, 'depth').text = str(3)
            
            # Tambahkan info tambahan
            ET.SubElement(annotation, 'segmented').text = '0'
            
            # Proses label jika ada
            label_file = labels_dir / f"{img_name}.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                                
                            # Parse YOLO format: class_id, x_center, y_center, width, height
                            class_id = int(float(parts[0]))
                            bbox_yolo = [float(x) for x in parts[1:5]]
                            
                            # Konversi ke format XYXY: [x1, y1, x2, y2]
                            bbox_xyxy = self.transformer.yolo_to_xyxy(bbox_yolo, w, h)
                            
                            # Tambahkan object
                            obj = ET.SubElement(annotation, 'object')
                            ET.SubElement(obj, 'name').text = class_map.get(class_id, f'class_{class_id}')
                            ET.SubElement(obj, 'pose').text = 'Unspecified'
                            ET.SubElement(obj, 'truncated').text = '0'
                            ET.SubElement(obj, 'difficult').text = '0'
                            
                            # Tambahkan bounding box
                            bndbox = ET.SubElement(obj, 'bndbox')
                            ET.SubElement(bndbox, 'xmin').text = str(bbox_xyxy[0])
                            ET.SubElement(bndbox, 'ymin').text = str(bbox_xyxy[1])
                            ET.SubElement(bndbox, 'xmax').text = str(bbox_xyxy[2])
                            ET.SubElement(bndbox, 'ymax').text = str(bbox_xyxy[3])
                except Exception as e:
                    self.logger.warning(f"⚠️ Error saat memproses label {label_file}: {str(e)}")
            
            # Simpan file XML
            tree = ET.ElementTree(annotation)
            xml_path = voc_annotations_dir / f"{img_name}.xml"
            
            tree.write(xml_path)
            processed_count += 1
            
        # Buat file ImageSets
        with open(voc_imagesets_dir / 'default.txt', 'w') as f:
            f.write('\n'.join(image_names))
            
        self.logger.success(
            f"✅ Konversi selesai:\n"
            f"   • Gambar diproses: {processed_count}\n"
            f"   • Output direktori: {target_dir}"
        )
        
        return {
            'status': 'success',
            'processed_images': processed_count,
            'output_dir': str(target_dir)
        }
    
    def voc_to_yolo(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        class_map: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Konversi dataset dari format Pascal VOC ke format YOLO.
        
        Args:
            source_dir: Path ke direktori dataset VOC
            target_dir: Direktori target untuk format YOLO
            class_map: Mapping dari nama kelas ke ID kelas
            
        Returns:
            Dictionary hasil konversi
        """
        source_path = Path(source_dir)
        target_dir = Path(target_dir)
        
        # Jalur VOC
        voc_annotations_dir = source_path / 'Annotations'
        voc_images_dir = source_path / 'JPEGImages'
        
        if not voc_annotations_dir.exists() or not voc_images_dir.exists():
            self.logger.error(f"❌ Direktori VOC tidak lengkap: {source_path}")
            return {'status': 'error', 'message': 'Direktori VOC tidak lengkap'}
            
        # Buat direktori output YOLO
        yolo_images_dir = target_dir / 'images'
        yolo_labels_dir = target_dir / 'labels'
        
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Default class map jika tidak disediakan
        if class_map is None:
            class_map = {}
            
        # Daftar file XML
        xml_files = list(voc_annotations_dir.glob('*.xml'))
        self.logger.info(f"🔄 Mengkonversi {len(xml_files)} anotasi dari VOC ke YOLO")
        
        processed_count = 0
        class_counts = {}
        
        for xml_file in xml_files:
            try:
                # Parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Dapatkan info gambar
                filename = root.find('filename').text
                img_path = voc_images_dir / filename
                
                # Dapatkan dimensi gambar
                size_elem = root.find('size')
                img_width = int(size_elem.find('width').text)
                img_height = int(size_elem.find('height').text)
                
                # Salin gambar jika ada
                if img_path.exists():
                    import shutil
                    shutil.copy2(img_path, yolo_images_dir / filename)
                
                # Proses setiap objek
                label_lines = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    
                    # Dapatkan class_id dari class_map
                    if class_name in class_map:
                        class_id = class_map[class_name]
                    else:
                        # Auto-increment class_id jika tidak ada di map
                        if class_name not in class_counts:
                            class_counts[class_name] = len(class_counts)
                        class_id = class_counts[class_name]
                    
                    # Dapatkan bbox
                    bbox_elem = obj.find('bndbox')
                    x1 = int(float(bbox_elem.find('xmin').text))
                    y1 = int(float(bbox_elem.find('ymin').text))
                    x2 = int(float(bbox_elem.find('xmax').text))
                    y2 = int(float(bbox_elem.find('ymax').text))
                    
                    # Konversi ke format YOLO
                    bbox_yolo = self.transformer.xyxy_to_yolo([x1, y1, x2, y2], img_width, img_height)
                    
                    # Format baris YOLO: class_id x_center y_center width height
                    label_lines.append(f"{class_id} {' '.join(map(str, bbox_yolo))}")
                
                # Simpan file label
                label_path = yolo_labels_dir / f"{xml_file.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(label_lines))
                    
                processed_count += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat memproses XML {xml_file}: {str(e)}")
                
        # Simpan class map
        if class_counts:
            class_map_path = target_dir / 'class_map.json'
            with open(class_map_path, 'w') as f:
                json.dump({name: id for name, id in class_counts.items()}, f, indent=2)
            
        self.logger.success(
            f"✅ Konversi selesai:\n"
            f"   • Gambar diproses: {processed_count}\n"
            f"   • Kelas terdeteksi: {len(class_counts)}\n"
            f"   • Output direktori: {target_dir}"
        )
        
        return {
            'status': 'success',
            'processed_images': processed_count,
            'classes': len(class_counts),
            'output_dir': str(target_dir)
        }