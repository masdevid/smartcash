"""
File: smartcash/utils/dataset/dataset_fixer.py
Author: Alfrida Sabar
Deskripsi: Modul untuk perbaikan dan pembersihan dataset dengan kemampuan auto-fixing berbagai masalah
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.dataset.dataset_utils import DatasetUtils

class DatasetFixer:
    """
    Kelas untuk perbaikan dan pembersihan dataset.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi dataset fixer.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Setup utils
        self.utils = DatasetUtils(logger)
        
        # Lock untuk thread safety
        self._lock = threading.RLock()
    
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
            split: Split dataset ('train', 'valid', 'test')
            fix_coordinates: Perbaiki koordinat yang tidak valid
            fix_labels: Perbaiki format label yang tidak valid
            fix_images: Coba perbaiki gambar yang rusak
            backup: Buat backup sebelum perbaikan
            
        Returns:
            Dict statistik perbaikan
        """
        # Buat backup jika diminta
        if backup:
            backup_dir = self.utils.backup_directory(self.data_dir / split)
            if backup_dir is None:
                self.logger.error(f"❌ Gagal membuat backup, membatalkan perbaikan")
                return {
                    'status': 'error',
                    'message': 'Backup gagal'
                }
        
        # Setup direktori
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_dir}")
            return {
                'status': 'error',
                'message': f'Direktori tidak lengkap: {split_dir}'
            }
        
        # Statistik perbaikan
        fix_stats = {
            'processed': 0,
            'fixed_labels': 0,
            'fixed_coordinates': 0,
            'fixed_images': 0,
            'skipped': 0,
            'errors': 0,
            'backup_created': backup
        }
        
        # Temukan semua file gambar
        image_files = self.utils.find_image_files(images_dir)
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return fix_stats
        
        # Proses setiap gambar
        self.logger.info(f"🔧 Memperbaiki dataset {split}: {len(image_files)} gambar")
        
        for img_path in image_files:
            try:
                # Proses gambar
                fixed_image = False
                
                if fix_images:
                    fixed_image = self._fix_image(img_path)
                    if fixed_image:
                        fix_stats['fixed_images'] += 1
                
                # Proses label
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists() and (fix_coordinates or fix_labels):
                    fixed_label, fixed_coords = self._fix_label(
                        label_path,
                        fix_coordinates=fix_coordinates,
                        fix_format=fix_labels
                    )
                    
                    if fixed_label:
                        fix_stats['fixed_labels'] += 1
                    
                    if fixed_coords:
                        fix_stats['fixed_coordinates'] += fixed_coords
                
                fix_stats['processed'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error saat memperbaiki {img_path.name}: {str(e)}")
                fix_stats['errors'] += 1
        
        self.logger.success(
            f"✅ Perbaikan dataset {split} selesai:\n"
            f"   • Gambar diproses: {fix_stats['processed']}\n"
            f"   • Label diperbaiki: {fix_stats['fixed_labels']}\n"
            f"   • Koordinat diperbaiki: {fix_stats['fixed_coordinates']}\n"
            f"   • Gambar diperbaiki: {fix_stats['fixed_images']}\n"
            f"   • Error: {fix_stats['errors']}"
        )
        
        return fix_stats
    
    def _fix_image(self, img_path: Path) -> bool:
        """
        Perbaiki gambar yang rusak.
        
        Args:
            img_path: Path ke file gambar
            
        Returns:
            Boolean yang menunjukkan keberhasilan perbaikan
        """
        try:
            import cv2
            
            # Baca gambar
            img = cv2.imread(str(img_path))
            
            # Jika gambar tidak bisa dibaca, tidak bisa diperbaiki
            if img is None:
                return False
            
            # Cek kualitas gambar
            issues = []
            
            # Cek resolusi (terlalu kecil?)
            h, w = img.shape[:2]
            if h < 100 or w < 100:
                issues.append("resolusi rendah")
                
                # Resize gambar
                target_size = (max(w, 100), max(h, 100))
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Cek kontras
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            
            if contrast < 20:
                issues.append("kontras rendah")
                
                # Perbaiki kontras
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                # Convert kembali ke BGR jika diperlukan
                if len(img.shape) == 3:
                    # Ini adalah pendekatan sederhana, ideal menggunakan transformasi di ruang warna lain
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Cek blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:
                issues.append("blur")
                
                # Tidak banyak yang bisa dilakukan untuk gambar blur,
                # namun kita bisa mencoba sharpening sederhana
                kernel = np.array([[-1,-1,-1], 
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
            
            # Jika ada isu yang diperbaiki, simpan gambar
            if issues:
                cv2.imwrite(str(img_path), img)
                self.logger.info(f"🔧 Gambar {img_path.name} diperbaiki: {', '.join(issues)}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memperbaiki gambar {img_path.name}: {str(e)}")
            return False
    
    def _fix_label(
        self,
        label_path: Path,
        fix_coordinates: bool = True,
        fix_format: bool = True
    ) -> Tuple[bool, int]:
        """
        Perbaiki file label.
        
        Args:
            label_path: Path ke file label
            fix_coordinates: Perbaiki koordinat yang tidak valid
            fix_format: Perbaiki format label yang tidak valid
            
        Returns:
            Tuple (label_fixed, coordinates_fixed_count)
        """
        try:
            # Baca file label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return False, 0
            
            fixed_lines = []
            fixed_something = False
            fixed_coords_count = 0
            
            for line in lines:
                line = line.strip()
                parts = line.split()
                
                # Skip baris kosong
                if not parts:
                    continue
                
                # Perbaiki format
                if fix_format and len(parts) != 5:
                    # Tidak bisa memperbaiki jika format terlalu rusak
                    if len(parts) < 5:
                        continue
                    
                    # Ambil lima bagian pertama saja
                    parts = parts[:5]
                    fixed_something = True
                
                try:
                    # Parse data
                    cls_id = int(float(parts[0]))
                    bbox = [float(x) for x in parts[1:5]]
                    
                    # Validasi class ID
                    if self.layer_config_manager.get_layer_for_class_id(cls_id) is None:
                        # Tidak bisa memperbaiki class ID yang tidak valid
                        continue
                    
                    # Perbaiki koordinat
                    fixed_bbox = bbox.copy()
                    
                    if fix_coordinates:
                        # Cek koordinat di luar range [0,1]
                        for i, coord in enumerate(bbox):
                            if not (0 <= coord <= 1):
                                fixed_bbox[i] = max(0.001, min(0.999, coord))
                                fixed_something = True
                                fixed_coords_count += 1
                    
                    # Tambahkan ke hasil
                    fixed_lines.append(f"{cls_id} {' '.join(map(str, fixed_bbox))}")
                    
                except ValueError:
                    # Skip baris dengan format yang tidak valid
                    continue
            
            # Simpan hasil jika ada yang diperbaiki
            if fixed_something and fixed_lines:
                with open(label_path, 'w') as f:
                    for line in fixed_lines:
                        f.write(f"{line}\n")
                
                return True, fixed_coords_count
            
            return False, 0
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memperbaiki label {label_path.name}: {str(e)}")
            return False, 0