# File: src/interfaces/handlers/preparation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk persiapan dan validasi dataset dengan dukungan paralel dan progress tracking

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import numpy as np

from interfaces.handlers.base_handler import BaseHandler
from utils.logging import ColoredLogger

class DataPreparationHandler(BaseHandler):
    """Handler untuk persiapan dataset dengan validasi paralel"""
    
    def __init__(self, config):
        super().__init__(config)
        self.required_structure = {
            'train': ['images', 'labels'],
            'val': ['images', 'labels'],
            'test': ['images', 'labels']
        }
        self.validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'issues': []
        }
        
    def prepare_dataset(self) -> bool:
        """
        Persiapkan struktur dataset dan validasi file
        
        Returns:
            bool: Status keberhasilan persiapan
        """
        # self.logger.info("🚀 Memulai persiapan dataset...")
        
        try:
            # Buat struktur direktori
            if not self._create_directories():
                return False
                
            # Buat konfigurasi dataset
            if not self._create_config():
                return False
            
            # Validasi dataset secara paralel
            self._validate_dataset_parallel()
            
            # Tampilkan ringkasan
            self._display_validation_summary()
            
            # self.log_operation("Persiapan dataset", "success")
            return True
            
        except Exception as e:
            self.log_operation("Persiapan dataset", "failed", str(e))
            return False
            
    def _create_directories(self) -> bool:
        """
        Buat struktur direktori yang diperlukan
        
        Returns:
            bool: Status pembuatan direktori
        """
        # self.logger.info("📁 Membuat struktur direktori...")
        
        with tqdm(total=len(self.required_structure) * 2, 
                 desc="Mempersiapkan direktori") as pbar:
            try:
                for split, subdirs in self.required_structure.items():
                    for subdir in subdirs:
                        path = self.rupiah_dir / split / subdir
                        if not self.validate_directory(path):
                            return False
                        pbar.update(1)
                        
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Gagal membuat direktori: {str(e)}")
                return False
            
    def _create_config(self) -> bool:
        """
        Buat file konfigurasi dataset
        
        Returns:
            bool: Status pembuatan konfigurasi
        """
        try:
            # self.logger.info("⚙️ Membuat konfigurasi dataset...")
            
            config = {
                'path': str(self.rupiah_dir),
                'train': str(self.rupiah_dir / 'train'),
                'val': str(self.rupiah_dir / 'val'),
                'test': str(self.rupiah_dir / 'test'),
                'nc': 7,  # jumlah kelas
                'names': ['100k', '10k', '1k', '20k', '2k', '50k', '5k']
            }
            
            config_path = self.rupiah_dir / 'rupiah.yaml'
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat konfigurasi: {str(e)}")
            return False
            
    def _validate_dataset_parallel(self):
        """Validasi dataset menggunakan pemrosesan paralel"""
        # self.logger.info("🔍 Memvalidasi dataset...")
        
        tasks = []
        for split in ['train', 'val', 'test']:
            img_dir = self.rupiah_dir / split / 'images'
            label_dir = self.rupiah_dir / split / 'labels'
            
            if img_dir.exists():
                for img_path in img_dir.glob('*.[jJ][pP][gG]') or img_dir.glob('*.[jJ][pP][eE][gG]'):
                    tasks.append((img_path, label_dir / f"{img_path.stem}.txt"))
                    
        with ThreadPoolExecutor() as executor:
            futures = []
            for img_path, label_path in tasks:
                future = executor.submit(self._validate_pair, img_path, label_path)
                futures.append(future)
                
            # Track progress
            with tqdm(total=len(tasks), desc="Validasi file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    self._update_validation_stats(result)
                    pbar.update(1)
                    
    def _validate_pair(self, img_path: Path, label_path: Path) -> Dict:
        """
        Validasi satu pasang file gambar dan label
        
        Args:
            img_path: Path file gambar
            label_path: Path file label
            
        Returns:
            Dict: Hasil validasi
        """
        result = {
            'path': str(img_path),
            'valid': True,
            'issues': []
        }
        
        # Validasi gambar
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                result['valid'] = False
                result['issues'].append('Gambar rusak atau tidak valid')
                return result
                
            # Periksa dimensi
            h, w = img.shape[:2]
            if h < 32 or w < 32:
                result['issues'].append('Dimensi gambar terlalu kecil')
                
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'Error membaca gambar: {str(e)}')
            return result
            
        # Validasi label
        if not label_path.exists():
            result['valid'] = False
            result['issues'].append('File label tidak ditemukan')
            return result
            
        try:
            with open(label_path) as f:
                for i, line in enumerate(f, 1):
                    try:
                        values = list(map(float, line.strip().split()))
                        
                        # Validasi format
                        if len(values) != 5:
                            result['issues'].append(
                                f'Format label tidak valid pada baris {i}'
                            )
                            continue
                            
                        # Validasi kelas
                        if values[0] not in range(7):  # 7 kelas
                            result['issues'].append(
                                f'Indeks kelas tidak valid pada baris {i}'
                            )
                            
                        # Validasi koordinat
                        if not all(0 <= v <= 1 for v in values[1:]):
                            result['issues'].append(
                                f'Koordinat tidak valid pada baris {i}'
                            )
                            
                    except ValueError:
                        result['issues'].append(f'Format angka tidak valid pada baris {i}')
                        
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f'Error membaca label: {str(e)}')
            
        result['valid'] = len(result['issues']) == 0
        return result
        
    def _update_validation_stats(self, result: Dict):
        """
        Update statistik validasi
        
        Args:
            result: Hasil validasi satu pasang file
        """
        self.validation_results['total_files'] += 1
        if result['valid']:
            self.validation_results['valid_files'] += 1
        if result['issues']:
            self.validation_results['issues'].append({
                'file': result['path'],
                'issues': result['issues']
            })
            
    def _display_validation_summary(self):
        """Tampilkan ringkasan hasil validasi dengan agregasi masalah"""
        total = self.validation_results['total_files']
        valid = self.validation_results['valid_files']
        issues = len(self.validation_results['issues'])
        
        # Agregasi masalah berdasarkan tipe
        issue_types = {}
        affected_files = {}
        
        for item in self.validation_results['issues']:
            for issue in item['issues']:
                if issue not in issue_types:
                    issue_types[issue] = 0
                    affected_files[issue] = set()
                issue_types[issue] += 1
                affected_files[issue].add(item['file'])
        
        # Tampilkan ringkasan
        print("\n📊 Ringkasan Validasi Dataset:")
        print(f"  • Total File: {total:,}")
        print(f"  • File Valid: {valid:,}")
        print(f"  • File Bermasalah: {issues:,}")
        
        if issues > 0:
            print("\n⚠️ Ringkasan Masalah:")
            for issue, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                file_count = len(affected_files[issue])
                print(f"\n  • {issue}")
                print(f"    Ditemukan di {count:,} lokasi ({file_count:,} file)")