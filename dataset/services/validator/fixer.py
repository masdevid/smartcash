"""
File: smartcash/dataset/services/validator/fixer.py
Deskripsi: Komponen untuk memperbaiki dataset secara otomatis
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.dataset.services.validator.image_validator import ImageValidator
from smartcash.dataset.services.validator.label_validator import LabelValidator


class DatasetFixer:
    """
    Komponen untuk memperbaiki berbagai masalah dalam dataset secara otomatis.
    Menggunakan validator tertentu untuk mengidentifikasi dan memperbaiki masalah.
    """
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi DatasetFixer.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("dataset_fixer")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Setup validator
        self.image_validator = ImageValidator(config, logger)
        self.label_validator = LabelValidator(config, logger)
        
        # Setup direktori untuk file yang diperbaiki
        self.fixed_dir = self.data_dir / 'fixed'
        
        self.logger.info(f"🔧 DatasetFixer diinisialisasi dengan {num_workers} workers")
    
    def fix_dataset(
        self, 
        split: str, 
        fix_images: bool = True, 
        fix_labels: bool = True,
        fix_coordinates: bool = True,
        create_backup: bool = True,
        move_fixed: bool = False
    ) -> Dict[str, Any]:
        """
        Perbaiki berbagai masalah dalam dataset.
        
        Args:
            split: Split dataset yang akan diperbaiki
            fix_images: Apakah memperbaiki gambar
            fix_labels: Apakah memperbaiki label
            fix_coordinates: Apakah memperbaiki koordinat bbox
            create_backup: Apakah membuat backup sebelum perbaikan
            move_fixed: Apakah memindahkan file yang diperbaiki ke direktori terpisah
            
        Returns:
            Statistik perbaikan
        """
        # Setup path
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Cek direktori
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap"}
        
        # Buat backup jika diminta
        backup_dir = None
        if create_backup:
            backup_dir = self.utils.backup_directory(split_path)
            if backup_dir is None:
                self.logger.error(f"❌ Gagal membuat backup, membatalkan perbaikan")
                return {'status': 'error', 'message': 'Backup gagal'}
        
        # Setup direktori untuk file yang diperbaiki jika diminta
        if move_fixed:
            (self.fixed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.fixed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Temukan semua file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'warning', 'message': 'Tidak ada gambar ditemukan', 'processed': 0}
        
        # Inisialisasi statistik
        stats = {
            'total_processed': 0,
            'image_fixed': 0,
            'label_fixed': 0,
            'coordinates_fixed': 0,
            'skipped': 0,
            'errors': 0,
            'backup_created': backup_dir is not None,
            'backup_path': str(backup_dir) if backup_dir else None
        }
        
        # Proses setiap file
        self.logger.info(f"🔧 Memperbaiki dataset {split}: {len(image_files)} gambar")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for img_path in image_files:
                futures.append(executor.submit(
                    self._fix_image_label_pair,
                    img_path,
                    labels_dir,
                    fix_images,
                    fix_labels,
                    fix_coordinates,
                    self.fixed_dir / split if move_fixed else None
                ))
            
            # Process results dengan progress bar
            for future in tqdm(futures, desc=f"🔧 Fixing {split}", unit="img"):
                result = future.result()
                
                stats['total_processed'] += 1
                if result.get('image_fixed', False): stats['image_fixed'] += 1
                if result.get('label_fixed', False): stats['label_fixed'] += 1
                if result.get('coordinates_fixed', 0) > 0: stats['coordinates_fixed'] += result['coordinates_fixed']
                if result.get('error', False): stats['errors'] += 1
                if result.get('skipped', False): stats['skipped'] += 1
        
        self.logger.success(
            f"✅ Perbaikan dataset {split} selesai:\n"
            f"   • Total diproses: {stats['total_processed']}\n"
            f"   • Gambar diperbaiki: {stats['image_fixed']}\n"
            f"   • Label diperbaiki: {stats['label_fixed']}\n"
            f"   • Koordinat diperbaiki: {stats['coordinates_fixed']}\n"
            f"   • Error: {stats['errors']}"
        )
        
        return stats
    
    def _fix_image_label_pair(
        self,
        img_path: Path,
        labels_dir: Path,
        fix_images: bool,
        fix_labels: bool,
        fix_coordinates: bool,
        fixed_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perbaiki satu pasangan gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori label
            fix_images: Apakah memperbaiki gambar
            fix_labels: Apakah memperbaiki label
            fix_coordinates: Apakah memperbaiki koordinat bbox
            fixed_dir: Direktori untuk menyimpan file yang diperbaiki (opsional)
            
        Returns:
            Hasil perbaikan
        """
        result = {
            'image_fixed': False,
            'label_fixed': False,
            'coordinates_fixed': 0,
            'skipped': False,
            'error': False
        }
        
        try:
            # Perbaiki gambar jika diminta
            if fix_images:
                fixed_image, image_fixes = self.image_validator.fix_image(img_path)
                result['image_fixed'] = fixed_image
                
                if fixed_image and fixed_dir is not None:
                    # Salin gambar yang diperbaiki ke direktori fixed
                    shutil.copy2(img_path, fixed_dir / 'images' / img_path.name)
            
            # Perbaiki label jika diminta
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists() and (fix_labels or fix_coordinates):
                fixed_label, label_fixes = self.label_validator.fix_label(
                    label_path,
                    fix_coordinates=fix_coordinates,
                    fix_format=fix_labels
                )
                
                result['label_fixed'] = fixed_label
                # Hitung jumlah koordinat yang diperbaiki
                if fixed_label:
                    result['coordinates_fixed'] = sum('Perbaikan' in fix for fix in label_fixes)
                    
                    if fixed_dir is not None:
                        # Salin label yang diperbaiki ke direktori fixed
                        shutil.copy2(label_path, fixed_dir / 'labels' / label_path.name)
            else:
                result['skipped'] = True
                
        except Exception as e:
            self.logger.error(f"❌ Error saat memperbaiki {img_path.name}: {str(e)}")
            result['error'] = True
            
        return result
    
    def fix_orphaned_files(self, split: str, create_empty_labels: bool = False, remove_orphaned_images: bool = False) -> Dict[str, Any]:
        """
        Perbaiki file yang tidak memiliki pasangan (orphaned).
        
        Args:
            split: Split dataset
            create_empty_labels: Apakah membuat label kosong untuk gambar tanpa label
            remove_orphaned_images: Apakah menghapus gambar tanpa label
            
        Returns:
            Statistik perbaikan
        """
        # Setup path
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Cek direktori
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap"}
        
        # Temukan semua file
        all_image_files = set(img_path.stem for img_path in self.utils.find_image_files(images_dir, with_labels=False))
        all_label_files = set(label_path.stem for label_path in labels_dir.glob('*.txt'))
        
        # Identifikasi file orphaned
        orphaned_images = all_image_files - all_label_files
        orphaned_labels = all_label_files - all_image_files
        
        # Inisialisasi statistik
        stats = {
            'orphaned_images': len(orphaned_images),
            'orphaned_labels': len(orphaned_labels),
            'created_labels': 0,
            'removed_images': 0,
            'removed_labels': 0,
            'errors': 0
        }
        
        # Log hasil analisis
        self.logger.info(
            f"📊 Analisis file orphaned di {split}:\n"
            f"   • Gambar tanpa label: {stats['orphaned_images']}\n"
            f"   • Label tanpa gambar: {stats['orphaned_labels']}"
        )
        
        # Tangani gambar orphaned
        if orphaned_images:
            if create_empty_labels:
                # Buat label kosong
                for img_stem in orphaned_images:
                    label_path = labels_dir / f"{img_stem}.txt"
                    try:
                        with open(label_path, 'w') as f:
                            # File kosong
                            pass
                        stats['created_labels'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal membuat label untuk {img_stem}: {str(e)}")
                        stats['errors'] += 1
                        
                self.logger.info(f"✅ Dibuat {stats['created_labels']} label kosong")
                
            elif remove_orphaned_images:
                # Hapus gambar tanpa label
                for img_stem in orphaned_images:
                    # Cari file gambar dengan berbagai ekstensi
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        img_path = images_dir / f"{img_stem}{ext}"
                        if img_path.exists():
                            try:
                                img_path.unlink()
                                stats['removed_images'] += 1
                            except Exception as e:
                                self.logger.error(f"❌ Gagal menghapus gambar {img_path.name}: {str(e)}")
                                stats['errors'] += 1
                
                self.logger.info(f"✅ Dihapus {stats['removed_images']} gambar tanpa label")
        
        # Hapus label orphaned (tanpa gambar)
        if orphaned_labels:
            for label_stem in orphaned_labels:
                label_path = labels_dir / f"{label_stem}.txt"
                try:
                    label_path.unlink()
                    stats['removed_labels'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal menghapus label {label_path.name}: {str(e)}")
                    stats['errors'] += 1
                    
            self.logger.info(f"✅ Dihapus {stats['removed_labels']} label tanpa gambar")
        
        self.logger.success(
            f"✅ Perbaikan file orphaned selesai:\n"
            f"   • Label dibuat: {stats['created_labels']}\n"
            f"   • Gambar dihapus: {stats['removed_images']}\n"
            f"   • Label dihapus: {stats['removed_labels']}\n"
            f"   • Error: {stats['errors']}"
        )
        
        return stats