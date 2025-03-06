# File: smartcash/utils/dataset_validator.py
# Author: Alfrida Sabar
# Deskripsi: Modul untuk validasi dan perbaikan dataset multilayer

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import shutil

from smartcash.utils.logger import SmartCashLogger

class DatasetValidator:
    """Kelas untuk validasi dan perbaikan dataset multilayer."""
    
    def __init__(
        self, 
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetValidator.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset (jika None, gunakan dari config)
            logger: Logger kustom
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Konfigurasi layer
        self.layer_config = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'class_ids': list(range(7))  # 0-6
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'class_ids': list(range(7, 14))  # 7-13
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'class_ids': list(range(14, 17))  # 14-16
            }
        }
        
        # Layer yang diaktifkan
        self.active_layers = config.get('layers', ['banknote'])
        
        # Setup direktori untuk file tidak valid
        self.invalid_dir = self.data_dir / 'invalid'
        self.invalid_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset(
        self, 
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        num_workers: int = 4
    ) -> Dict:
        """
        Validasi dataset untuk satu split.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_issues: Jika True, mencoba memperbaiki masalah yang ditemukan
            move_invalid: Jika True, pindahkan file tidak valid ke direktori terpisah
            num_workers: Jumlah worker untuk paralelisasi
            
        Returns:
            Dict berisi hasil validasi
        """
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
        
        # Setup direktori untuk file tidak valid jika diperlukan
        if move_invalid:
            (self.invalid_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.invalid_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Temukan semua file gambar
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + \
                     list(images_dir.glob('*.png')) + list(images_dir.glob('*.JPG')) + \
                     list(images_dir.glob('*.PNG'))
        
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
        
        # Persiapkan statistik untuk hasil validasi
        validation_stats = {
            'total_images': len(image_files),
            'valid_images': 0,
            'invalid_images': 0,
            'empty_images': 0,
            'total_labels': 0,
            'valid_labels': 0,
            'invalid_labels': 0,
            'missing_labels': 0,
            'layer_stats': {layer: 0 for layer in self.layer_config},
            'fixed_labels': 0,
            'issues': []
        }
        
        # Fungsi untuk memvalidasi satu pasang gambar dan label
        def validate_image_label_pair(img_path):
            result = {
                'image_path': str(img_path),
                'image_valid': False,
                'label_path': str(labels_dir / f"{img_path.stem}.txt"),
                'label_exists': False,
                'label_valid': False,
                'layer_stats': {layer: 0 for layer in self.layer_config},
                'issues': [],
                'fixed': False
            }
            
            # Validasi gambar
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    result['issues'].append(f"Gambar kosong atau tidak dapat dibaca: {img_path.name}")
                    return result
                
                result['image_valid'] = True
                result['image_size'] = (img.shape[1], img.shape[0])  # (width, height)
            except Exception as e:
                result['issues'].append(f"Error saat membaca gambar: {str(e)}")
                return result
            
            # Validasi label
            label_path = labels_dir / f"{img_path.stem}.txt"
            result['label_exists'] = label_path.exists()
            
            if not result['label_exists']:
                result['issues'].append(f"File label tidak ditemukan untuk {img_path.name}")
                return result
            
            # Validasi isi label
            try:
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                
                result['label_lines'] = []
                valid_label_lines = []
                has_issue = False
                layer_counts = {layer: 0 for layer in self.layer_config}
                
                # Validasi setiap baris label
                for i, line in enumerate(label_lines):
                    line_result = {
                        'text': line.strip(),
                        'valid': False,
                        'class_id': None,
                        'layer': None,
                        'bbox': None,
                        'issues': []
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
                            valid_class = False
                            for layer, config in self.layer_config.items():
                                if cls_id in config['class_ids']:
                                    valid_class = True
                                    line_result['layer'] = layer
                                    layer_counts[layer] += 1
                                    break
                                    
                            if not valid_class:
                                line_result['issues'].append(f"Class ID tidak valid: {cls_id}")
                                has_issue = True
                            
                            # Cek apakah koordinat valid (0-1)
                            if any(not (0 <= coord <= 1) for coord in bbox):
                                line_result['issues'].append(f"Koordinat tidak valid: {bbox}")
                                has_issue = True
                                
                                # Fix koordinat jika diminta
                                if fix_issues:
                                    fixed_bbox = [max(0, min(1, coord)) for coord in bbox]
                                    line_result['bbox'] = fixed_bbox
                                    line_result['fixed'] = True
                                    result['fixed'] = True
                            else:
                                line_result['bbox'] = bbox
                            
                            line_result['class_id'] = cls_id
                            
                            # Tandai sebagai valid jika tidak ada masalah
                            if not line_result['issues']:
                                line_result['valid'] = True
                                valid_label_lines.append(
                                    f"{cls_id} {' '.join(map(str, line_result.get('bbox', bbox)))}\n"
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
                
                # Cek apakah ada label yang valid
                result['label_valid'] = any(line.get('valid', False) for line in result['label_lines'])
                result['layer_stats'] = layer_counts
                
                # Simpan label yang diperbaiki jika ada
                if fix_issues and result['fixed'] and valid_label_lines:
                    with open(label_path, 'w') as f:
                        f.writelines(valid_label_lines)
                
                # Cek apakah ada layer yang aktif
                active_layer_present = False
                for layer in self.active_layers:
                    if layer_counts[layer] > 0:
                        active_layer_present = True
                        break
                        
                if not active_layer_present and self.active_layers:
                    result['issues'].append(f"Tidak ada layer aktif yang ditemukan dalam label")
                
            except Exception as e:
                result['issues'].append(f"Error saat membaca label: {str(e)}")
                return result
                
            return result
        
        # Jalankan validasi secara paralel
        self.logger.info(f"🔍 Memvalidasi dataset {split}: {len(image_files)} gambar")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(validate_image_label_pair, image_files),total=len(image_files), desc="🔍 Validasi Dataset"
            ))
            
            # Inisialisasi statistik validasi
            validation_stats = {
                'total_images': len(image_files),
                'valid_images': 0,
                'invalid_images': 0,
                'empty_images': 0,
                'total_labels': 0,
                'valid_labels': 0,
                'invalid_labels': 0,
                'missing_labels': 0,
                'layer_stats': {layer: 0 for layer in self.layer_config},
                'fixed_labels': 0,
                'issues': []
            }
            
            for result in results:
                # Validasi gambar
                if result.get('image_valid', False):
                    validation_stats['valid_images'] += 1
                else:
                    validation_stats['invalid_images'] += 1
                    validation_stats['issues'].extend(result.get('issues', []))
                
                # Validasi label
                if result.get('label_exists', False):
                    validation_stats['total_labels'] += 1
                    
                    if result.get('label_valid', False):
                        validation_stats['valid_labels'] += 1
                        
                        # Update statistik layer
                        for layer, count in result.get('layer_stats', {}).items():
                            validation_stats['layer_stats'][layer] += count
                    else:
                        validation_stats['invalid_labels'] += 1
                        validation_stats['issues'].extend(result.get('issues', []))
                    
                    # Track jumlah label yang diperbaiki
                    if result.get('fixed', False):
                        validation_stats['fixed_labels'] += 1
                else:
                    validation_stats['missing_labels'] += 1
                    validation_stats['issues'].append(f"Label tidak ditemukan: {result.get('image_path', 'Unknown')}")
            
            # Pindahkan file tidak valid jika diminta
            if move_invalid and (validation_stats['invalid_images'] > 0 or 
                                 validation_stats['invalid_labels'] > 0 or 
                                 validation_stats['missing_labels'] > 0):
                try:
                    self._move_invalid_files(split, validation_stats)
                except Exception as e:
                    self.logger.error(f"❌ Gagal memindahkan file tidak valid: {str(e)}")
            
            # Log ringkasan validasi
            self.logger.info(
                f"🔍 Ringkasan Validasi Dataset {split}:\n"
                f"📸 Total Gambar: {validation_stats['total_images']}\n"
                f"   • Valid: {validation_stats['valid_images']}\n"
                f"   • Tidak Valid: {validation_stats['invalid_images']}\n"
                f"📋 Total Label: {validation_stats['total_labels']}\n"
                f"   • Valid: {validation_stats['valid_labels']}\n"
                f"   • Tidak Valid: {validation_stats['invalid_labels']}\n"
                f"   • Label Hilang: {validation_stats['missing_labels']}\n"
                f"   • Label Diperbaiki: {validation_stats['fixed_labels']}"
            )
            
            # Log statistik per layer
            self.logger.info("📊 Distribusi Layer:")
            for layer, count in validation_stats['layer_stats'].items():
                self.logger.info(f"   • {layer}: {count} objek")
            
            return validation_stats
    
    def _move_invalid_files(self, split: str, validation_stats: Dict):
        """
        Pindahkan file gambar dan label yang tidak valid ke direktori invalid.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            validation_stats: Statistik validasi dataset
        """
        # Path direktori sumber dan tujuan
        src_images_dir = self.data_dir / split / 'images'
        src_labels_dir = self.data_dir / split / 'labels'
        
        dst_images_dir = self.invalid_dir / split / 'images'
        dst_labels_dir = self.invalid_dir / split / 'labels'
        
        # Buat direktori tujuan jika belum ada
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Cari file yang tidak valid
        for img_path in src_images_dir.glob('*.*'):
            label_path = src_labels_dir / f"{img_path.stem}.txt"
            
            # Kriteria untuk memindahkan:
            # 1. Gambar tidak valid
            # 2. Label tidak valid
            # 3. Label hilang
            move_file = False
            
            try:
                # Baca gambar
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    move_file = True
                
                # Periksa label
                if not label_path.exists():
                    move_file = True
                else:
                    # Validasi isi label
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Validasi baris label
                    valid_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        
                        # Invalid jika kurang dari 5 bagian
                        if len(parts) < 5:
                            move_file = True
                            break
                        
                        try:
                            # Validasi koordinat
                            cls_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            
                            # Invalid jika koordinat di luar range
                            if any(not (0 <= coord <= 1) for coord in bbox):
                                move_file = True
                                break
                            
                            # Invalid jika class ID tidak valid
                            valid_class = False
                            for layer_config in self.layer_config.values():
                                if cls_id in layer_config['class_ids']:
                                    valid_class = True
                                    break
                            
                            if not valid_class:
                                move_file = True
                                break
                            
                            # Jika valid, tambahkan ke baris valid
                            valid_lines.append(line)
                        
                        except (ValueError, IndexError):
                            move_file = True
                            break
                    
                    # Jika ada baris yang tidak valid, pindahkan
                    if move_file:
                        print(f"❌ File tidak valid: {img_path}")
                    
                # Pindahkan file jika invalid
                if move_file:
                    # Salin gambar ke direktori invalid
                    dst_img_path = dst_images_dir / img_path.name
                    shutil.copy2(img_path, dst_img_path)
                    
                    # Jika ada label, salin ke direktori invalid
                    if label_path.exists():
                        dst_label_path = dst_labels_dir / label_path.name
                        shutil.copy2(label_path, dst_label_path)
                        
                        # Optional: Hapus file sumber jika ingin memisahkan dataset
                        # Gunakan dengan hati-hati!
                        # img_path.unlink()
                        # label_path.unlink()
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memvalidasi {img_path}: {str(e)}")
        
        self.logger.success(
            f"✅ Pemindahan file tidak valid selesai:\n"
            f"   • Direktori tujuan: {self.invalid_dir}\n"
            f"   • Split: {split}"
        )
    
    def fix_labels(self, split: str = 'train', max_workers: int = 4) -> Dict:
        """
        Perbaiki label yang tidak valid.
        
        Args:
            split: Split dataset yang akan diperbaiki
            max_workers: Jumlah worker untuk paralelisasi
            
        Returns:
            Dict statistik perbaikan label
        """
        # Setup direktori
        images_dir = self.data_dir / split / 'images'
        labels_dir = self.data_dir / split / 'labels'
        
        # Cari semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Statistik perbaikan
        fix_stats = {
            'processed': 0,
            'fixed': 0,
            'skipped': 0,
            'errors': 0,
            'layer_adjustments': {layer: 0 for layer in self.layer_config}
        }
        
        # Fungsi untuk memperbaiki satu file label
        def fix_label_file(label_path):
            result = {
                'fixed': False,
                'issues': [],
                'layer_adjustments': {layer: 0 for layer in self.layer_config}
            }
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Simpan baris yang valid
                valid_lines = []
                
                for line in lines:
                    line_parts = line.strip().split()
                    
                    # Validasi struktur baris
                    if len(line_parts) < 5:
                        result['issues'].append(f"Baris tidak lengkap: {line}")
                        continue
                    
                    try:
                        # Konversi dan validasi
                        cls_id = int(float(line_parts[0]))
                        coords = [float(x) for x in line_parts[1:5]]
                        
                        # Koreksi koordinat
                        corrected_coords = [
                            max(0.0, min(1.0, coord)) for coord in coords
                        ]
                        
                        # Validasi kelas
                        valid_class = False
                        for layer, config in self.layer_config.items():
                            if cls_id in config['class_ids']:
                                valid_class = True
                                result['layer_adjustments'][layer] += 1
                                break
                        
                        if not valid_class:
                            result['issues'].append(f"Class ID tidak valid: {cls_id}")
                            continue
                        
                        # Tulis ulang baris dengan koordinat yang dikoreksi
                        corrected_line = f"{cls_id} {' '.join(map(str, corrected_coords))}\n"
                        valid_lines.append(corrected_line)
                        
                        # Tandai ada perbaikan jika koordinat berbeda
                        if corrected_coords != coords:
                            result['fixed'] = True
                    
                    except (ValueError, IndexError) as e:
                        result['issues'].append(f"Error parsing baris: {line}, {str(e)}")
                
                # Simpan kembali label yang diperbaiki
                if result['fixed']:
                    with open(label_path, 'w') as f:
                        f.writelines(valid_lines)
                
                return result
            
            except Exception as e:
                result['errors'] = str(e)
                return result
        
        # Proses label secara paralel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(fix_label_file, label_files),
                total=len(label_files),
                desc="🔧 Memperbaiki Label"
            ))
        
        # Agregasi statistik
        for result in results:
            fix_stats['processed'] += 1
            
            if result['fixed']:
                fix_stats['fixed'] += 1
            
            if result.get('errors'):
                fix_stats['errors'] += 1
            
            # Akumulasi penyesuaian layer
            for layer, count in result['layer_adjustments'].items():
                fix_stats['layer_adjustments'][layer] += count
        
        # Log statistik perbaikan
        self.logger.success(
            f"✅ Perbaikan Label Selesai:\n"
            f"   • Total Diproses: {fix_stats['processed']}\n"
            f"   • Label Diperbaiki: {fix_stats['fixed']}\n"
            f"   • Errors: {fix_stats['errors']}"
        )
        
        # Log penyesuaian layer
        self.logger.info("📊 Penyesuaian Layer:")
        for layer, count in fix_stats['layer_adjustments'].items():
            self.logger.info(f"   • {layer}: {count} label")
        
        return fix_stats