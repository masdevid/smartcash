# File: smartcash/utils/enhanced_dataset_validator.py
# Author: Alfrida Sabar
# Deskripsi: Modul validasi dataset yang ditingkatkan dengan kemampuan perbaikan otomatis

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class EnhancedDatasetValidator:
    """
    Validator dataset yang ditingkatkan dengan kemampuan:
    - Validasi label multilayer
    - Perbaikan otomatis untuk masalah umum
    - Visualisasi masalah
    - Analisis distribusi kelas/layer
    - Error recovery yang kuat
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi validator dataset yang ditingkatkan.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
            num_workers: Jumlah worker untuk paralelisasi
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.num_workers = num_workers
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Setup direktori untuk file tidak valid
        self.invalid_dir = self.data_dir / 'invalid'
        
        # Lock untuk thread safety
        self._lock = threading.RLock()
        
    def validate_dataset(
        self,
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = False,
        sample_size: int = 0
    ) -> Dict:
        """
        Validasi dataset untuk satu split.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_issues: Jika True, perbaiki masalah yang ditemukan
            move_invalid: Jika True, pindahkan file tidak valid
            visualize: Jika True, hasilkan visualisasi masalah
            sample_size: Jika > 0, gunakan subset untuk percepatan
            
        Returns:
            Dict hasil validasi
        """
        start_time = time.time()
        
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Pastikan direktori ada
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_dir}")
            return {
                'status': 'error',
                'message': f"Direktori dataset tidak lengkap: {split_dir}",
                'stats': {}
            }
        
        # Setup direktori untuk visualisasi dan file tidak valid jika diperlukan
        vis_dir = None
        if visualize:
            vis_dir = self.data_dir / 'visualizations' / split
            vis_dir.mkdir(parents=True, exist_ok=True)
            
        if move_invalid:
            (self.invalid_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.invalid_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Temukan semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {
                'status': 'warning',
                'message': f"Tidak ada gambar ditemukan di {images_dir}",
                'stats': {
                    'total_images': 0,
                    'total_labels': 0
                }
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            random.seed(42)  # Untuk hasil yang konsisten
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Persiapkan statistik untuk hasil validasi
        validation_stats = {
            'total_images': len(image_files),
            'valid_images': 0,
            'invalid_images': 0,
            'corrupt_images': 0,
            'total_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'missing_labels': 0,
            'empty_labels': 0,
            'fixed_labels': 0,
            'fixed_coordinates': 0,
            'layer_stats': {layer: 0 for layer in self.layer_config_manager.get_layer_names()},
            'class_stats': {},
            'issues': []
        }
        
        # Fungsi untuk memvalidasi satu pasang gambar dan label
        def validate_image_label_pair(img_path):
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
                        line_result['issues'].append(f"Label tidak lengkap pada baris {i+1}")
                        has_issue = True
                    else:
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
                                has_issue = True
                            
                            # Cek apakah koordinat valid (0-1)
                            invalid_coords = [i for i, coord in enumerate(bbox) if not (0 <= coord <= 1)]
                            if invalid_coords:
                                coord_names = ['x_center', 'y_center', 'width', 'height']
                                invalid_names = [coord_names[i] for i in invalid_coords]
                                
                                line_result['issues'].append(
                                    f"Koordinat tidak valid: {', '.join(invalid_names)}"
                                )
                                has_issue = True
                                
                                # Fix koordinat jika diminta
                                if fix_issues:
                                    fixed_bbox = [max(0.001, min(0.999, coord)) for coord in bbox]
                                    line_result['bbox'] = fixed_bbox
                                    line_result['fixed'] = True
                                    fixed_something = True
                                    
                                    # Ganti bbox original dengan yang diperbaiki
                                    for j, coord in enumerate(fixed_bbox):
                                        if j in invalid_coords:
                                            result['fixed_coordinates'] = result.get('fixed_coordinates', 0) + 1
                            else:
                                line_result['bbox'] = bbox
                            
                            line_result['class_id'] = cls_id
                            
                            # Tandai sebagai valid jika tidak ada masalah
                            if not line_result['issues']:
                                line_result['valid'] = True
                                valid_label_lines.append(
                                    f"{cls_id} {' '.join(map(str, bbox))}\n"
                                )
                            elif line_result.get('fixed', False):
                                # Jika berhasil diperbaiki, tambahkan ke valid
                                line_result['valid'] = True
                                valid_label_lines.append(
                                    f"{cls_id} {' '.join(map(str, line_result['bbox']))}\n"
                                )
                            
                        except ValueError:
                            line_result['issues'].append(f"Format tidak valid pada baris {i+1}")
                            has_issue = True
                    
                    result['label_lines'].append(line_result)
                
                # Simpan statistik layer dan kelas
                result['layer_stats'] = layer_counts
                result['class_stats'] = class_counts
                
                # Cek apakah ada label yang valid
                result['label_valid'] = any(line.get('valid', False) for line in result['label_lines'])
                
                # Simpan label yang diperbaiki jika ada
                if fix_issues and fixed_something and valid_label_lines:
                    with open(label_path, 'w') as f:
                        f.writelines(valid_label_lines)
                    result['fixed'] = True
                
                # Visualisasi jika diminta dan ada masalah
                if visualize and has_issue and vis_dir:
                    self._visualize_issues(img_path, result, vis_dir)
                    result['visualized'] = True
                
                # Cek apakah ada layer yang aktif
                result['has_active_layer'] = False
                for layer in self.active_layers:
                    if layer in layer_counts and layer_counts[layer] > 0:
                        result['has_active_layer'] = True
                        break
                        
                # Status final
                if result['label_valid'] and result['image_valid']:
                    result['status'] = 'valid'
                
            except Exception as e:
                result['issues'].append(f"Error saat membaca label: {str(e)}")
                
            return result
        
        # Jalankan validasi secara paralel
        self.logger.info(f"🔍 Memvalidasi dataset {split}: {len(image_files)} gambar")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(validate_image_label_pair, image_files),
                total=len(image_files),
                desc="🔍 Validasi Dataset"
            ))
            
            # Agregasi hasil validasi
            for result in results:
                # Validasi gambar
                if result.get('image_valid', False):
                    validation_stats['valid_images'] += 1
                else:
                    validation_stats['invalid_images'] += 1
                    
                if result.get('corrupt', False):
                    validation_stats['corrupt_images'] += 1
                
                # Validasi label
                if result.get('label_exists', False):
                    validation_stats['total_labels'] += 1
                    
                    if result.get('label_valid', False):
                        validation_stats['valid_labels'] += 1
                        
                        # Update statistik layer
                        for layer, count in result.get('layer_stats', {}).items():
                            validation_stats['layer_stats'][layer] += count
                            
                        # Update statistik kelas
                        for cls, count in result.get('class_stats', {}).items():
                            if cls not in validation_stats['class_stats']:
                                validation_stats['class_stats'][cls] = 0
                            validation_stats['class_stats'][cls] += count
                    else:
                        validation_stats['invalid_labels'] += 1
                        
                    if result.get('empty_label', False):
                        validation_stats['empty_labels'] += 1
                    
                    # Track label yang diperbaiki
                    if result.get('fixed', False):
                        validation_stats['fixed_labels'] += 1
                        
                    if result.get('fixed_coordinates', 0) > 0:
                        validation_stats['fixed_coordinates'] += result.get('fixed_coordinates', 0)
                else:
                    validation_stats['missing_labels'] += 1
                
                # Kumpulkan masalah
                if result.get('issues'):
                    for issue in result['issues']:
                        validation_stats['issues'].append(
                            f"{Path(result['image_path']).name}: {issue}"
                        )
            
            # Pindahkan file tidak valid jika diminta
            if move_invalid:
                self._move_invalid_files(split, results)
            
            # Durasi validasi
            validation_stats['duration'] = time.time() - start_time
            
            # Log ringkasan validasi
            self.logger.info(
                f"✅ Ringkasan Validasi Dataset {split} ({validation_stats['duration']:.1f} detik):\n"
                f"📸 Total Gambar: {validation_stats['total_images']}\n"
                f"   • Valid: {validation_stats['valid_images']}\n"
                f"   • Tidak Valid: {validation_stats['invalid_images']}\n"
                f"   • Corrupt: {validation_stats['corrupt_images']}\n"
                f"📋 Total Label: {validation_stats['total_labels']}\n"
                f"   • Valid: {validation_stats['valid_labels']}\n"
                f"   • Tidak Valid: {validation_stats['invalid_labels']}\n"
                f"   • Label Hilang: {validation_stats['missing_labels']}\n"
                f"   • Label Kosong: {validation_stats['empty_labels']}\n"
                f"🔧 Perbaikan:\n"
                f"   • Label Diperbaiki: {validation_stats['fixed_labels']}\n"
                f"   • Koordinat Diperbaiki: {validation_stats['fixed_coordinates']}"
            )
            
            # Log statistik per layer
            self.logger.info("📊 Distribusi Layer:")
            for layer, count in validation_stats['layer_stats'].items():
                if count > 0:
                    percentage = 0
                    if validation_stats['valid_labels'] > 0:
                        percentage = (count / validation_stats['valid_labels']) * 100
                    self.logger.info(f"   • {layer}: {count} objek ({percentage:.1f}%)")
            
            # Log statistik kelas (top 10)
            if validation_stats['class_stats']:
                top_classes = sorted(
                    validation_stats['class_stats'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                self.logger.info("📊 Top 10 Kelas:")
                for cls, count in top_classes:
                    percentage = 0
                    if validation_stats['valid_labels'] > 0:
                        percentage = (count / validation_stats['valid_labels']) * 100
                    self.logger.info(f"   • {cls}: {count} ({percentage:.1f}%)")
            
            return validation_stats
    
    def _move_invalid_files(
        self,
        split: str,
        validation_results: List[Dict]
    ) -> Dict[str, int]:
        """
        Pindahkan file tidak valid ke direktori terpisah.
        
        Args:
            split: Split dataset
            validation_results: Hasil validasi
            
        Returns:
            Dict statistik pemindahan
        """
        self.logger.info(f"🔄 Memindahkan file tidak valid ke {self.invalid_dir}...")
        
        move_stats = {
            'moved_images': 0,
            'moved_labels': 0,
            'errors': 0
        }
        
        # Setup direktori target
        target_images_dir = self.invalid_dir / split / 'images'
        target_labels_dir = self.invalid_dir / split / 'labels'
        
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Pindahkan file tidak valid
        for result in validation_results:
            # Skip yang valid
            if result.get('status') == 'valid':
                continue
                
            try:
                # Pindahkan gambar
                if not result.get('image_valid', False) or result.get('corrupt', False):
                    img_path = Path(result['image_path'])
                    target_path = target_images_dir / img_path.name
                    
                    # Salin, bukan pindah (untuk keamanan)
                    shutil.copy2(img_path, target_path)
                    move_stats['moved_images'] += 1
                
                # Pindahkan label
                if result.get('label_exists', False) and not result.get('label_valid', False):
                    label_path = Path(result['label_path'])
                    target_path = target_labels_dir / label_path.name
                    
                    # Salin, bukan pindah (untuk keamanan)
                    shutil.copy2(label_path, target_path)
                    move_stats['moved_labels'] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Error saat memindahkan file: {str(e)}")
                move_stats['errors'] += 1
        
        self.logger.success(
            f"✅ Pemindahan file tidak valid selesai:\n"
            f"   • Gambar dipindahkan: {move_stats['moved_images']}\n"
            f"   • Label dipindahkan: {move_stats['moved_labels']}\n"
            f"   • Error: {move_stats['errors']}"
        )
        
        return move_stats
    
    def _visualize_issues(
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
    
    def analyze_dataset(
        self,
        split: str = 'train',
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis mendalam tentang dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict hasil analisis
        """
        # Jalankan validasi tanpa perbaikan atau pemindahan
        validation_results = self.validate_dataset(
            split=split,
            fix_issues=False,
            move_invalid=False,
            visualize=False,
            sample_size=sample_size
        )
        
        # Tambahkan analisis lebih lanjut
        analysis = {
            'validation': validation_results,
            'image_size_distribution': self._analyze_image_sizes(split, sample_size),
            'class_balance': self._analyze_class_balance(validation_results),
            'layer_balance': self._analyze_layer_balance(validation_results),
            'bbox_statistics': self._analyze_bbox_statistics(split, sample_size)
        }
        
        # Log hasil analisis
        self.logger.info(
            f"📊 Analisis Dataset {split}:\n"
            f"   • Ketidakseimbangan kelas: {analysis['class_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ketidakseimbangan layer: {analysis['layer_balance']['imbalance_score']:.2f}/10\n"
            f"   • Ukuran gambar yang dominan: {analysis['image_size_distribution']['dominant_size']}\n"
            f"   • Rasio aspek dominan: {analysis['image_size_distribution']['dominant_aspect_ratio']}\n"
            f"   • Ukuran bbox rata-rata: {analysis['bbox_statistics']['mean_width']:.2f}x{analysis['bbox_statistics']['mean_height']:.2f}"
        )
        
        return analysis
    
    def _analyze_image_sizes(
        self,
        split: str,
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict statistik ukuran gambar
        """
        images_dir = self.data_dir / split / 'images'
        
        if not images_dir.exists():
            return {
                'status': 'error',
                'message': f"Direktori gambar tidak ditemukan: {images_dir}",
                'sizes': {}
            }
        
        # Temukan semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            return {
                'status': 'error',
                'message': f"Tidak ada gambar ditemukan di {images_dir}",
                'sizes': {}
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            random.seed(42)  # Untuk hasil yang konsisten
            image_files = random.sample(image_files, sample_size)
        
        # Analisis ukuran
        size_counts = {}
        aspect_ratios = {}
        width_sum = 0
        height_sum = 0
        
        for img_path in tqdm(image_files, desc="Analisis Ukuran Gambar"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                size_key = f"{w}x{h}"
                
                # Track ukuran
                if size_key not in size_counts:
                    size_counts[size_key] = 0
                size_counts[size_key] += 1
                
                # Track rasio aspek (dibulatkan ke 2 desimal)
                aspect = round(w / h, 2)
                if aspect not in aspect_ratios:
                    aspect_ratios[aspect] = 0
                aspect_ratios[aspect] += 1
                
                # Akumulasi untuk rata-rata
                width_sum += w
                height_sum += h
                
            except Exception:
                continue
        
        # Tentukan ukuran dominan
        if size_counts:
            dominant_size = max(size_counts.items(), key=lambda x: x[1])
            dominant_pct = (dominant_size[1] / len(image_files)) * 100
        else:
            dominant_size = ("Unknown", 0)
            dominant_pct = 0
            
        # Tentukan rasio aspek dominan
        if aspect_ratios:
            dominant_aspect = max(aspect_ratios.items(), key=lambda x: x[1])
            dominant_aspect_pct = (dominant_aspect[1] / len(image_files)) * 100
        else:
            dominant_aspect = (0, 0)
            dominant_aspect_pct = 0
            
        # Hitung rata-rata
        avg_width = width_sum / max(1, len(image_files))
        avg_height = height_sum / max(1, len(image_files))
        
        return {
            'status': 'success',
            'total_images': len(image_files),
            'sizes': dict(sorted(size_counts.items(), key=lambda x: x[1], reverse=True)),
            'aspect_ratios': dict(sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)),
            'dominant_size': dominant_size[0],
            'dominant_size_count': dominant_size[1],
            'dominant_size_percent': dominant_pct,
            'dominant_aspect_ratio': dominant_aspect[0],
            'dominant_aspect_ratio_count': dominant_aspect[1],
            'dominant_aspect_ratio_percent': dominant_aspect_pct,
            'mean_width': avg_width,
            'mean_height': avg_height
        }
    
    def _analyze_class_balance(self, validation_results: Dict) -> Dict:
        """
        Analisis keseimbangan kelas dalam dataset.
        
        Args:
            validation_results: Hasil validasi dataset
            
        Returns:
            Dict statistik keseimbangan kelas
        """
        class_stats = validation_results.get('class_stats', {})
        
        if not class_stats:
            return {
                'status': 'error',
                'message': "Tidak ada statistik kelas dalam hasil validasi",
                'imbalance_score': 10.0
            }
            
        # Hitung statistik dasar
        total_objects = sum(class_stats.values())
        if total_objects == 0:
            return {
                'status': 'error',
                'message': "Tidak ada objek yang valid",
                'imbalance_score': 10.0
            }
            
        # Hitung persentase per kelas
        class_percentages = {
            cls: (count / total_objects) * 100
            for cls, count in class_stats.items()
        }
        
        # Hitung ketidakseimbangan
        mean_percentage = 100 / len(class_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in class_percentages.values())
        
        # Normalisasi ketidakseimbangan ke skala 0-10
        # 0 = seimbang sempurna, 10 = sangat tidak seimbang
        imbalance_score = min(10.0, (max_deviation / mean_percentage) * 5)
        
        # Identifikasi kelas yang kurang terwakili
        underrepresented = [
            cls for cls, pct in class_percentages.items()
            if pct < mean_percentage * 0.5
        ]
        
        # Identifikasi kelas yang terlalu terwakili
        overrepresented = [
            cls for cls, pct in class_percentages.items()
            if pct > mean_percentage * 2
        ]
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'class_count': len(class_stats),
            'class_percentages': dict(sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)),
            'mean_percentage': mean_percentage,
            'max_deviation': max_deviation,
            'imbalance_score': imbalance_score,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented
        }
    
    def _analyze_layer_balance(self, validation_results: Dict) -> Dict:
        """
        Analisis keseimbangan layer dalam dataset.
        
        Args:
            validation_results: Hasil validasi dataset
            
        Returns:
            Dict statistik keseimbangan layer
        """
        layer_stats = validation_results.get('layer_stats', {})
        
        # Filter hanya layer yang aktif
        active_layer_stats = {
            layer: count for layer, count in layer_stats.items()
            if layer in self.active_layers
        }
        
        if not active_layer_stats:
            return {
                'status': 'error',
                'message': "Tidak ada statistik layer aktif",
                'imbalance_score': 10.0
            }
            
        # Hitung statistik dasar
        total_objects = sum(active_layer_stats.values())
        if total_objects == 0:
            return {
                'status': 'error',
                'message': "Tidak ada objek layer aktif",
                'imbalance_score': 10.0
            }
            
        # Hitung persentase per layer
        layer_percentages = {
            layer: (count / total_objects) * 100
            for layer, count in active_layer_stats.items()
        }
        
        # Hitung ketidakseimbangan
        mean_percentage = 100 / len(active_layer_stats)
        max_deviation = max(abs(pct - mean_percentage) for pct in layer_percentages.values())
        
        # Normalisasi ketidakseimbangan ke skala 0-10
        imbalance_score = min(10.0, (max_deviation / mean_percentage) * 5)
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'layer_count': len(active_layer_stats),
            'layer_percentages': dict(sorted(layer_percentages.items(), key=lambda x: x[1], reverse=True)),
            'mean_percentage': mean_percentage,
            'max_deviation': max_deviation,
            'imbalance_score': imbalance_score
        }
    
    def _analyze_bbox_statistics(
        self,
        split: str,
        sample_size: int = 0
    ) -> Dict:
        """
        Analisis statistik bbox dalam dataset.
        
        Args:
            split: Split dataset
            sample_size: Jika > 0, gunakan sampel
            
        Returns:
            Dict statistik bounding box
        """
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            return {
                'status': 'error',
                'message': f"Direktori tidak lengkap",
                'bbox_count': 0
            }
        
        # Temukan semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            return {
                'status': 'error',
                'message': f"Tidak ada gambar ditemukan",
                'bbox_count': 0
            }
            
        # Jika sample_size ditentukan, ambil sampel acak
        if 0 < sample_size < len(image_files):
            random.seed(42)  # Untuk hasil yang konsisten
            image_files = random.sample(image_files, sample_size)
        
        # Analisis bbox
        width_vals = []
        height_vals = []
        area_vals = []
        aspect_ratio_vals = []
        class_bbox_sizes = {}
        
        for img_path in tqdm(image_files, desc="Analisis Bounding Box"):
            # Baca gambar untuk mendapatkan dimensi
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img_h, img_w = img.shape[:2]
                
                # Baca label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                    
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        try:
                            cls_id = int(float(parts[0]))
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Skip koordinat tidak valid
                            if not all(0 <= x <= 1 for x in [x_center, y_center, width, height]):
                                continue
                                
                            # Convert ke pixel
                            pixel_width = width * img_w
                            pixel_height = height * img_h
                            pixel_area = pixel_width * pixel_height
                            
                            # Hitung rasio aspek
                            aspect_ratio = pixel_width / max(1, pixel_height)
                            
                            # Tambahkan ke statistik
                            width_vals.append(pixel_width)
                            height_vals.append(pixel_height)
                            area_vals.append(pixel_area)
                            aspect_ratio_vals.append(aspect_ratio)
                            
                            # Track per kelas
                            class_name = self.layer_config_manager.get_class_name(cls_id) or f"Unknown_{cls_id}"
                            if class_name not in class_bbox_sizes:
                                class_bbox_sizes[class_name] = {
                                    'widths': [],
                                    'heights': [],
                                    'areas': [],
                                    'count': 0
                                }
                                
                            class_bbox_sizes[class_name]['widths'].append(pixel_width)
                            class_bbox_sizes[class_name]['heights'].append(pixel_height)
                            class_bbox_sizes[class_name]['areas'].append(pixel_area)
                            class_bbox_sizes[class_name]['count'] += 1
                            
                        except (ValueError, IndexError, ZeroDivisionError):
                            continue
                            
            except Exception:
                continue
        
        # Hitung statistik
        if not width_vals:
            return {
                'status': 'error',
                'message': "Tidak ada bounding box valid ditemukan",
                'bbox_count': 0
            }
            
        # Statistik global
        bbox_stats = {
            'status': 'success',
            'bbox_count': len(width_vals),
            'mean_width': np.mean(width_vals),
            'mean_height': np.mean(height_vals),
            'mean_area': np.mean(area_vals),
            'mean_aspect_ratio': np.mean(aspect_ratio_vals),
            'min_width': np.min(width_vals),
            'max_width': np.max(width_vals),
            'min_height': np.min(height_vals),
            'max_height': np.max(height_vals),
            'min_area': np.min(area_vals),
            'max_area': np.max(area_vals)
        }
        
        # Statistik per kelas
        class_stats = {}
        for cls, stats in class_bbox_sizes.items():
            if stats['count'] > 0:
                class_stats[cls] = {
                    'count': stats['count'],
                    'mean_width': np.mean(stats['widths']),
                    'mean_height': np.mean(stats['heights']),
                    'mean_area': np.mean(stats['areas']),
                    'min_area': np.min(stats['areas']),
                    'max_area': np.max(stats['areas'])
                }
        
        bbox_stats['class_stats'] = class_stats
        
        return bbox_stats
    
    def fix_dataset(
        self,
        split: str = 'train',
        fix_coordinates: bool = True,
        fix_labels: bool = True,
        fix_images: bool = False,
        backup: bool = True
    ) -> Dict:
        """
        Perbaiki masalah umum dalam dataset.
        
        Args:
            split: Split dataset
            fix_coordinates: Perbaiki koordinat yang tidak valid
            fix_labels: Perbaiki format label yang tidak valid
            fix_images: Coba perbaiki gambar yang rusak
            backup: Buat backup sebelum perbaikan
            
        Returns:
            Dict statistik perbaikan
        """
        # Buat backup jika diminta
        if backup:
            backup_success = self._backup_split(split)
            if not backup_success:
                self.logger.error(f"❌ Gagal membuat backup, membatalkan perbaikan")
                return {
                    'status': 'error',
                    'message': 'Backup gagal'
                }
        
        # Jalankan validasi dengan perbaikan
        fix_results = self.validate_dataset(
            split=split,
            fix_issues=True,
            move_invalid=False
        )
        
        # Tambahkan statistik tambahan
        fix_results['backup_created'] = backup
        
        self.logger.success(
            f"✅ Perbaikan dataset {split} selesai:\n"
            f"   • Label diperbaiki: {fix_results['fixed_labels']}\n"
            f"   • Koordinat diperbaiki: {fix_results['fixed_coordinates']}"
        )
        
        return fix_results
    
    def _backup_split(self, split: str) -> bool:
        """
        Buat backup split dataset.
        
        Args:
            split: Split dataset
            
        Returns:
            Boolean sukses/gagal
        """
        split_dir = self.data_dir / split
        backup_dir = self.data_dir / f"{split}_backup_{int(time.time())}"
        
        self.logger.info(f"🔄 Membuat backup {split} ke {backup_dir}...")
        
        try:
            # Buat direktori backup
            if backup_dir.exists():
                self.logger.warning(f"⚠️ Direktori backup sudah ada: {backup_dir}")
                # Buat nama unik
                backup_dir = self.data_dir / f"{split}_backup_{int(time.time())}"
            
            # Salin direktori
            shutil.copytree(split_dir, backup_dir)
            
            self.logger.success(f"✅ Backup berhasil dibuat: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup: {str(e)}")
            return False