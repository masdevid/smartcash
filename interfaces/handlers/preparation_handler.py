# File: src/interfaces/handlers/preparation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk persiapan dan validasi dataset dengan dukungan paralel dan progress tracking

import os
from pathlib import Path
import shutil
import time
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
        try:
            print(f"\n📂 Project path: {self.project_root}")
            
            # Check data source, excluding system folders
            excluded_dirs = {'rupiah', '__pycache__', 'logs', 'runs', 'weights'}
            data_sources = [p for p in self.data_dir.iterdir() 
                          if p.is_dir() and p.name not in excluded_dirs]
            
            if not data_sources:
                print("\n❌ Tidak ada folder dataset ditemukan di /src/data/")
                print("ℹ️  Silakan tambahkan folder dataset di /src/data/ lalu tekan Enter untuk melanjutkan...")
                input()
                # Recheck after user input
                data_sources = [p for p in self.data_dir.iterdir() 
                              if p.is_dir() and p.name not in excluded_dirs]
                if not data_sources:
                    self.logger.error("❌ Dataset masih belum ditemukan setelah pengecekan ulang")
                    return False
            
            # Validate structure and let user choose if multiple valid sources found
            valid_sources = []
            for source in data_sources:
                if self._validate_source_structure(source):
                    valid_sources.append(source)
            
            if not valid_sources:
                print("\n❌ Tidak ada folder dengan struktur yang valid ditemukan")
                print("\nℹ️  Struktur yang dibutuhkan:")
                self._display_required_structure()
                print("\nStruktur yang ditemukan:")
                for source in data_sources:
                    print(f"\n📁 {source.name}")
                    self._display_found_structure(source)
                
                # Offer retry option
                retry = input("\nIngin mengulangi persiapan data? (y/N): ").strip().lower()
                if retry == 'y':
                    return self.prepare_dataset()
                return False
            
            # Let user choose if multiple valid sources
            selected_source = valid_sources[0]
            if len(valid_sources) > 1:
                print("\n📦 Ditemukan beberapa folder dataset yang valid:")
                for idx, source in enumerate(valid_sources, 1):
                    print(f"{idx}. {source.name}")
                
                # Show structure for each valid source
                for idx, source in enumerate(valid_sources, 1):
                    print(f"\nStruktur folder {idx}:")
                    self._display_found_structure(source)
                print("\n")
                
                while True:
                    choice = input("Pilih nomor folder yang akan digunakan: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(valid_sources):
                        selected_source = valid_sources[int(choice)-1]
                        break
                    print("❌ Pilihan tidak valid!")
            
            # Copy selected data
            print(f"\nℹ️  Menggunakan dataset dari folder: {selected_source.name}")
            with tqdm(total=1, desc="Menyalin dataset") as pbar:
                try:
                    if self.rupiah_dir.exists():
                        shutil.rmtree(self.rupiah_dir)
                    shutil.copytree(selected_source, self.rupiah_dir)
                    pbar.update(1)
                except Exception as e:
                    self.logger.error(f"❌ Gagal menyalin {selected_source.name}: {str(e)}")
                    return False
            
            # Create directory structure if needed
            if not self._create_directories():
                return False
                
            # Create dataset config
            if not self._create_config():
                return False
            
            # Validate dataset in parallel
            self._validate_dataset_parallel()
            
            # Show summary
            self._display_validation_summary()
            
            return True
            
        except Exception as e:
            self.log_operation("Persiapan dataset", "failed", str(e))
            return False
    
    def _display_required_structure(self):
        """Tampilkan struktur folder yang dibutuhkan"""
        print("data/")
        for split in self.required_structure:
            print(f"  ├─ {split}/")
            for subdir in self.required_structure[split]:
                prefix = "  │  ├─" if subdir != self.required_structure[split][-1] else "  │  └─"
                print(f"{prefix} {subdir}/")
                
    def _get_folder_summary(self, path: Path) -> Tuple[int, int]:
        """
        Dapatkan jumlah file dalam folder
        
        Args:
            path: Path folder
            
        Returns:
            Tuple[int, int]: (jumlah file jpg, jumlah file txt)
        """
        jpg_count = len(list(path.glob('*.jpg')))
        txt_count = len(list(path.glob('*.txt')))
        return jpg_count, txt_count
        
    def _display_found_structure(self, path: Path, level: int = 0):
        """
        Tampilkan struktur folder yang ditemukan
        
        Args:
            path: Path folder yang akan ditampilkan
            level: Level kedalaman untuk indentasi
        """
        indent = "  " * level
        prefix = "├─" if level > 0 else ""
        
        if level == 0:
            print(f"{path.name}/")
            level += 1
            indent = "  " * level
        
        try:
            dirs = [x for x in path.iterdir() if x.is_dir() and x.name != '__pycache__']
            dirs = sorted(dirs, key=lambda x: x.name)
            
            for i, item in enumerate(dirs):
                is_last = i == len(dirs) - 1
                marker = "└─" if is_last else "├─"
                
                # Get file counts for images and labels folders
                if item.name in ['images', 'labels']:
                    jpg_count, txt_count = self._get_folder_summary(item)
                    count_str = f"({jpg_count if item.name == 'images' else txt_count} files)"
                    print(f"{indent}{marker} {item.name}/ {count_str}")
                else:
                    print(f"{indent}{marker} {item.name}/")
                    next_indent = "  " if is_last else "│ "
                    self._display_found_structure(item, level + 1)
                    
        except Exception:
            print(f"{indent}└─ ⚠️  Error reading directory")
                
    def _validate_source_structure(self, source_path: Path) -> bool:
        """
        Validasi struktur folder sumber dataset
        
        Args:
            source_path: Path ke folder sumber
            
        Returns:
            bool: True jika struktur valid
        """
        try:
            for split, subdirs in self.required_structure.items():
                split_path = source_path / split
                if not split_path.exists():
                    return False
                
                for subdir in subdirs:
                    if not (split_path / subdir).exists():
                        return False
            return True
        except Exception:
            return False
            
    def _create_directories(self) -> bool:
        """
        Buat struktur direktori yang diperlukan
        
        Returns:
            bool: Status pembuatan direktori
        """
        with tqdm(total=len(self.required_structure) * 2, 
                 desc="Mempersiapkan direktori", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            try:
                for split, subdirs in self.required_structure.items():
                    for subdir in subdirs:
                        path = self.rupiah_dir / split / subdir
                        if not self.validate_directory(path):
                            return False
                        pbar.update(1)
                        time.sleep(0.1)  # Add small delay for visibility
                        
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
        tasks = []
        
        # Collect all tasks first
        for split in ['train', 'val', 'test']:
            img_dir = self.rupiah_dir / split / 'images'
            label_dir = self.rupiah_dir / split / 'labels'
            
            if img_dir.exists():
                for img_path in img_dir.glob('*.[jJ][pP][gG]'):
                    tasks.append((img_path, label_dir / f"{img_path.stem}.txt"))
                    
        if not tasks:
            print("\n⚠️ Tidak ada file untuk divalidasi")
            return
            
        with ThreadPoolExecutor() as executor:
            futures = []
            for img_path, label_path in tasks:
                future = executor.submit(self._validate_pair, img_path, label_path)
                futures.append(future)
                
            # Track progress with a single progress bar
            with tqdm(total=len(tasks), 
                     desc="Validasi dataset", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
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