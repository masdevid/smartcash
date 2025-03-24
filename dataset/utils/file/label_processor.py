"""
File: smartcash/dataset/utils/file/label_processor.py
Deskripsi: Utilitas untuk memproses dan memanipulasi file label dalam dataset dengan pendekatan DRY
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from tqdm.auto import tqdm
import re
import json

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.common.threadpools import process_in_parallel
from smartcash.dataset.utils.file_wrapper import ensure_dir


class LabelProcessor:
    """Utilitas untuk pemrosesan dan manipulasi file label dataset."""
    
    def __init__(self, config: Dict = None, data_dir: Optional[str] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi LabelProcessor.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config or {}
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.logger = logger or get_logger("label_processor")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(self.config, str(self.data_dir), self.logger)
        self.layer_config = get_layer_config()
        
        self.logger.info(f"🏷️ LabelProcessor diinisialisasi dengan data_dir: {self.data_dir}")
    
    def fix_labels(
        self, 
        directory: Union[str, Path],
        fix_coordinates: bool = True,
        fix_class_ids: bool = True,
        recursive: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Perbaiki masalah pada file label.
        
        Args:
            directory: Direktori yang berisi label
            fix_coordinates: Apakah memperbaiki koordinat bbox yang di luar range
            fix_class_ids: Apakah memperbaiki class ID yang tidak valid
            recursive: Apakah memeriksa subdirektori secara rekursif
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik perbaikan
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'fixed': 0, 'errors': 0}
            
        # Tentukan pola pencarian
        search_pattern = '**/*.txt' if recursive else '*.txt'
            
        # Dapatkan semua file label
        label_files = list(directory.glob(search_pattern))
        if not label_files:
            self.logger.warning(f"⚠️ Tidak ada file label ditemukan di {directory}")
            return {'fixed': 0, 'errors': 0}
            
        self.logger.info(f"🔧 Memperbaiki {len(label_files)} file label")
        
        # Dapatkan valid class_ids jika diperlukan
        valid_ids = []
        if fix_class_ids:
            valid_ids = [cls_id for layer in self.layer_config.get_layer_names() 
                       for cls_id in self.layer_config.get_layer_config(layer)['class_ids']]
        
        # Fungsi untuk memperbaiki satu file label
        def fix_label(label_path):
            try:
                # Baca file label
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if not lines:
                    return 'empty'
                
                new_lines = []
                fixed_something = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    
                    # Cek apakah format valid
                    if len(parts) < 5:
                        continue
                        
                    try:
                        # Parse data
                        cls_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Fix class ID
                        if fix_class_ids and valid_ids and cls_id not in valid_ids:
                            fixed_something = True
                            continue
                        
                        # Fix koordinat
                        fixed_bbox = bbox.copy()
                        
                        if fix_coordinates:
                            for i, coord in enumerate(bbox):
                                if not (0 <= coord <= 1):
                                    fixed_bbox[i] = max(0.001, min(0.999, coord))
                                    fixed_something = True
                        
                        # Tambahkan ke hasil
                        new_line = f"{cls_id} {' '.join(map(str, fixed_bbox))}"
                        new_lines.append(new_line)
                    except ValueError:
                        # Skip baris dengan format yang tidak valid
                        fixed_something = True
                        continue
                
                # Simpan hasil jika ada yang diperbaiki
                if fixed_something or len(new_lines) != len(lines):
                    with open(label_path, 'w') as f:
                        for line in new_lines:
                            f.write(f"{line}\n")
                    return 'fixed'
                
                return 'unchanged'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal memperbaiki {label_path}: {str(e)}")
                return 'error'
        
        # Perbaiki label secara paralel
        results = process_in_parallel(
            label_files,
            fix_label,
            max_workers=self.num_workers,
            desc="🔧 Memperbaiki label",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {
            'fixed': results.count('fixed'),
            'unchanged': results.count('unchanged'),
            'empty': results.count('empty'),
            'errors': results.count('error')
        }
        
        self.logger.success(
            f"✅ Perbaikan label selesai:\n"
            f"   • Fixed: {stats['fixed']}\n"
            f"   • Unchanged: {stats['unchanged']}\n"
            f"   • Empty: {stats['empty']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def filter_classes(
        self, 
        directory: Union[str, Path],
        keep_classes: Optional[List[int]] = None,
        remove_classes: Optional[List[int]] = None,
        recursive: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Filter kelas dalam file label.
        
        Args:
            directory: Direktori yang berisi label
            keep_classes: Daftar class_id yang akan dipertahankan (opsional)
            remove_classes: Daftar class_id yang akan dihapus (opsional)
            recursive: Apakah memeriksa subdirektori secara rekursif
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik filtering
        """
        if keep_classes is None and remove_classes is None:
            self.logger.error("❌ Harus menentukan keep_classes atau remove_classes")
            return {'filtered': 0, 'errors': 0}
            
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'filtered': 0, 'errors': 0}
            
        # Tentukan pola pencarian
        search_pattern = '**/*.txt' if recursive else '*.txt'
            
        # Dapatkan semua file label
        label_files = list(directory.glob(search_pattern))
        if not label_files:
            self.logger.warning(f"⚠️ Tidak ada file label ditemukan di {directory}")
            return {'filtered': 0, 'errors': 0}
            
        self.logger.info(f"🔍 Memfilter {len(label_files)} file label")
        
        # Fungsi untuk memfilter satu file label
        def filter_label(label_path):
            try:
                # Baca file label
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if not lines:
                    return 'empty'
                
                new_lines = []
                filtered_something = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    
                    # Cek apakah format valid
                    if len(parts) < 5:
                        continue
                        
                    try:
                        # Parse class ID
                        cls_id = int(float(parts[0]))
                        
                        # Tentukan apakah baris ini dipertahankan
                        keep_line = (cls_id in keep_classes if keep_classes is not None else cls_id not in remove_classes)
                        
                        if keep_line:
                            new_lines.append(line)
                        else:
                            filtered_something = True
                    except ValueError:
                        # Skip baris dengan format yang tidak valid
                        continue
                
                # Simpan hasil jika ada yang difilter
                if filtered_something or len(new_lines) != len(lines):
                    with open(label_path, 'w') as f:
                        for line in new_lines:
                            f.write(f"{line}\n")
                    
                    if len(new_lines) == 0:
                        return 'empty_after_filter'
                    
                    return 'filtered'
                
                return 'unchanged'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal memfilter {label_path}: {str(e)}")
                return 'error'
        
        # Filter label secara paralel
        results = process_in_parallel(
            label_files,
            filter_label,
            max_workers=self.num_workers,
            desc="🔍 Memfilter label",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {
            'filtered': results.count('filtered'),
            'empty_after_filter': results.count('empty_after_filter'),
            'unchanged': results.count('unchanged'),
            'empty': results.count('empty'),
            'errors': results.count('error')
        }
        
        self.logger.success(
            f"✅ Filtering label selesai:\n"
            f"   • Filtered: {stats['filtered']}\n"
            f"   • Empty after filter: {stats['empty_after_filter']}\n"
            f"   • Unchanged: {stats['unchanged']}\n"
            f"   • Empty: {stats['empty']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def extract_layer(
        self, 
        directory: Union[str, Path],
        layer_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Ekstrak satu layer dari file label multi-layer.
        
        Args:
            directory: Direktori yang berisi label
            layer_name: Nama layer yang akan diekstrak
            output_dir: Direktori output (opsional, default: {directory}_{layer_name})
            recursive: Apakah memeriksa subdirektori secara rekursif
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik ekstraksi
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'extracted': 0, 'errors': 0}
            
        # Validasi layer
        if layer_name not in self.layer_config.get_layer_names():
            self.logger.error(f"❌ Layer '{layer_name}' tidak valid")
            return {'extracted': 0, 'errors': 0}
            
        # Setup direktori output
        output_dir = Path(output_dir) if output_dir else Path(f"{directory}_{layer_name}")
        ensure_dir(output_dir)
        
        # Dapatkan class_ids untuk layer
        layer_config = self.layer_config.get_layer_config(layer_name)
        layer_class_ids = set(layer_config['class_ids'])
        
        # Tentukan pola pencarian
        search_pattern = '**/*.txt' if recursive else '*.txt'
            
        # Dapatkan semua file label
        label_files = list(directory.glob(search_pattern))
        if not label_files:
            self.logger.warning(f"⚠️ Tidak ada file label ditemukan di {directory}")
            return {'extracted': 0, 'errors': 0}
            
        self.logger.info(f"🔍 Mengekstrak layer '{layer_name}' dari {len(label_files)} file label")
        
        # Fungsi untuk mengekstrak layer dari satu file label
        def extract_layer_from_file(label_path):
            try:
                # Baca file label
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if not lines:
                    return 'empty'
                
                new_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    
                    # Cek apakah format valid
                    if len(parts) < 5:
                        continue
                        
                    try:
                        # Parse class ID
                        cls_id = int(float(parts[0]))
                        
                        # Cek apakah class_id masuk dalam layer
                        if cls_id in layer_class_ids:
                            new_lines.append(line)
                    except ValueError:
                        # Skip baris dengan format yang tidak valid
                        continue
                
                # Tentukan path output
                rel_path = label_path.relative_to(directory) if recursive else label_path.name
                output_path = output_dir / rel_path
                ensure_dir(output_path.parent)
                
                # Simpan hasil
                with open(output_path, 'w') as f:
                    for line in new_lines:
                        f.write(f"{line}\n")
                
                if len(new_lines) == 0:
                    return 'empty_after_extract'
                
                return 'extracted'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal mengekstrak layer dari {label_path}: {str(e)}")
                return 'error'
        
        # Ekstrak layer secara paralel
        results = process_in_parallel(
            label_files,
            extract_layer_from_file,
            max_workers=self.num_workers,
            desc=f"🔍 Mengekstrak layer '{layer_name}'",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {
            'extracted': results.count('extracted'),
            'empty_after_extract': results.count('empty_after_extract'),
            'empty': results.count('empty'),
            'errors': results.count('error')
        }
        
        self.logger.success(
            f"✅ Ekstraksi layer selesai:\n"
            f"   • Extracted: {stats['extracted']}\n"
            f"   • Empty after extract: {stats['empty_after_extract']}\n"
            f"   • Empty: {stats['empty']}\n"
            f"   • Errors: {stats['errors']}\n"
            f"   • Output dir: {output_dir}"
        )
        
        return stats
    
    def convert_to_coco(
        self, 
        dataset_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        split: str = 'train',
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Konversi dataset YOLO ke format COCO JSON.
        
        Args:
            dataset_dir: Direktori dataset
            output_file: File output (opsional, default: {dataset_dir}/{split}_coco.json)
            split: Split dataset ('train', 'valid', 'test')
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik konversi
        """
        import cv2
        from smartcash.dataset.utils.transform.bbox_transform import BBoxTransformer
        
        dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            self.logger.error(f"❌ Direktori dataset {dataset_dir} tidak ditemukan")
            return {'status': 'error', 'message': 'Direktori tidak ditemukan'}
            
        # Tentukan direktori images dan labels
        split_dir = dataset_dir / split if (dataset_dir / split).exists() else dataset_dir
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Struktur direktori tidak valid di {split_dir}")
            return {'status': 'error', 'message': 'Struktur direktori tidak valid'}
            
        # Setup file output
        output_file = Path(output_file) if output_file else dataset_dir / f"{split}_coco.json"
        ensure_dir(output_file.parent)
        
        # Dapatkan semua file gambar dan label
        image_files = list(images_dir.glob('*.*'))
        image_files = [img for img in image_files if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'error', 'message': 'Tidak ada gambar ditemukan'}
            
        self.logger.info(f"🔄 Mengkonversi {len(image_files)} gambar ke format COCO")
        
        # Inisialisasi struktur COCO
        coco_data = {
            "info": {
                "description": f"SmartCash Dataset - {split}",
                "url": "",
                "version": "1.0",
                "year": 2023,
                "contributor": "SmartCash",
                "date_created": ""
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Setup kategori dari layer config
        categories = []
        category_id_map = {}  # class_id -> category_id
        category_id = 1
        
        for layer in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                class_name = layer_config['classes'][i] if i < len(layer_config['classes']) else f"class-{cls_id}"
                
                categories.append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": layer
                })
                
                category_id_map[cls_id] = category_id
                category_id += 1
        
        coco_data["categories"] = categories
        
        # Fungsi untuk memproses satu gambar dan label
        def process_image_and_label(img_path, image_id):
            try:
                # Baca gambar untuk mendapatkan dimensi
                img = cv2.imread(str(img_path))
                if img is None:
                    return None
                    
                height, width = img.shape[:2]
                
                # Tambahkan info gambar
                image_info = {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "license": 1,
                    "date_captured": ""
                }
                
                # Cari file label
                label_path = labels_dir / f"{img_path.stem}.txt"
                annotations = []
                
                if label_path.exists():
                    # Parse label YOLO
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        
                    for ann_id, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        
                        # Cek apakah format valid
                        if len(parts) < 5:
                            continue
                            
                        try:
                            # Parse YOLO format: class_id, x_center, y_center, width, height
                            cls_id = int(float(parts[0]))
                            bbox_yolo = [float(x) for x in parts[1:5]]
                            
                            # Konversi ke format COCO: x, y, width, height (x,y adalah top-left)
                            bbox_coco = BBoxTransformer.yolo_to_coco(bbox_yolo, width, height)
                            
                            # Cek apakah class_id valid
                            if cls_id in category_id_map:
                                annotation = {
                                    "id": image_id * 100 + ann_id,
                                    "image_id": image_id,
                                    "category_id": category_id_map[cls_id],
                                    "bbox": bbox_coco,
                                    "area": bbox_coco[2] * bbox_coco[3],
                                    "segmentation": [],
                                    "iscrowd": 0
                                }
                                
                                annotations.append(annotation)
                        except ValueError:
                            continue
                
                return (image_info, annotations)
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal memproses {img_path}: {str(e)}")
                return None
        
        # Proses gambar dan label secara paralel
        all_images = []
        all_annotations = []
        
        # Proses secara terurut untuk memastikan ID yang konsisten
        with tqdm(enumerate(image_files), total=len(image_files), desc="🔄 Konversi ke COCO", unit="img", disable=not show_progress) as pbar:
            for i, img_path in pbar:
                image_id = i + 1
                result = process_image_and_label(img_path, image_id)
                if result:
                    image_info, annotations = result
                    all_images.append(image_info)
                    all_annotations.extend(annotations)
        
        # Update data COCO
        coco_data["images"] = all_images
        coco_data["annotations"] = all_annotations
        
        # Simpan file JSON
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        self.logger.success(
            f"✅ Konversi ke COCO selesai:\n"
            f"   • Images: {len(all_images)}\n"
            f"   • Annotations: {len(all_annotations)}\n"
            f"   • Categories: {len(coco_data['categories'])}\n"
            f"   • Output file: {output_file}"
        )
        
        return {
            'status': 'success',
            'output_file': str(output_file),
            'images': len(all_images),
            'annotations': len(all_annotations),
            'categories': len(coco_data['categories'])
        }