"""
File: smartcash/dataset/services/validator/dataset_validator.py
Deskripsi: Layanan untuk validasi dan perbaikan dataset
"""

import cv2
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class DatasetValidatorService:
    """Service untuk validasi dan perbaikan dataset."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi DatasetValidatorService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("dataset_validator")
        self.num_workers = num_workers
        self.invalid_dir = self.data_dir / 'invalid'
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Lock untuk operasi thread-safe
        self._lock = threading.RLock()
        
        self.logger.info(f"🔍 DatasetValidatorService diinisialisasi dengan {num_workers} workers")
    
    def validate_dataset(
        self, 
        split: str = 'train', 
        fix_issues: bool = False,
        move_invalid: bool = False, 
        visualize: bool = False, 
        sample_size: int = 0
    ) -> Dict[str, Any]:
        """
        Validasi dataset untuk satu split.
        
        Args:
            split: Split dataset yang akan divalidasi
            fix_issues: Apakah langsung memperbaiki masalah yang ditemukan
            move_invalid: Apakah memindahkan file tidak valid ke direktori terpisah
            visualize: Apakah membuat visualisasi untuk file dengan masalah
            sample_size: Jumlah sampel yang akan divalidasi (0 = semua)
            
        Returns:
            Hasil validasi
        """
        start_time = time.time()
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Cek direktori
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap: {split_path}"}
        
        # Setup direktori
        vis_dir = None
        if visualize:
            vis_dir = self.data_dir / 'visualizations' / split
            vis_dir.mkdir(parents=True, exist_ok=True)
            
        if move_invalid:
            (self.invalid_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.invalid_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Cari file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'warning', 'message': f"Tidak ada gambar ditemukan", 'total_images': 0}
        
        # Ambil sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Inisialisasi statistik validasi
        validation_stats = self._init_validation_stats()
        validation_stats['total_images'] = len(image_files)
        
        self.logger.info(f"🔍 Validasi dataset {split}: {len(image_files)} gambar")
        
        # Validasi paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in image_files:
                futures.append(executor.submit(self._validate_image_label_pair, img_path, labels_dir))
            
            # Collect results dengan progress bar
            results = []
            for future in tqdm(futures, desc="🔍 Validating dataset", unit="img"):
                results.append(future.result())
        
        # Agregasi hasil
        self._aggregate_validation_results(
            results, validation_stats, fix_issues, labels_dir, visualize, vis_dir
        )
        
        # Pindahkan file tidak valid jika diminta
        if move_invalid:
            self._move_invalid_files(split, results)
        
        # Catat durasi dan log hasil
        validation_stats['duration'] = time.time() - start_time
        self._log_validation_summary(split, validation_stats)
        
        return validation_stats
    
    def fix_dataset(
        self, 
        split: str = 'train', 
        fix_coordinates: bool = True,
        fix_labels: bool = True, 
        fix_images: bool = False, 
        backup: bool = True
    ) -> Dict[str, Any]:
        """
        Perbaiki masalah dataset.
        
        Args:
            split: Split dataset yang akan diperbaiki
            fix_coordinates: Apakah memperbaiki koordinat bbox yang di luar range
            fix_labels: Apakah memperbaiki format label
            fix_images: Apakah memperbaiki gambar dengan kualitas rendah
            backup: Apakah membuat backup sebelum perbaikan
            
        Returns:
            Hasil perbaikan
        """
        # Buat backup jika diminta
        backup_dir = None
        if backup:
            backup_dir = self.utils.backup_directory(self.utils.get_split_path(split))
            if backup_dir is None:
                self.logger.error(f"❌ Gagal membuat backup, membatalkan perbaikan")
                return {'status': 'error', 'message': 'Backup gagal'}
        
        # Setup direktori
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f'Direktori tidak lengkap: {split_path}'}
        
        # Statistik perbaikan
        fix_stats = {
            'processed': 0, 'fixed_labels': 0, 'fixed_coordinates': 0,
            'fixed_images': 0, 'skipped': 0, 'errors': 0, 
            'backup_created': backup, 'backup_dir': str(backup_dir) if backup_dir else None
        }
        
        # Temukan semua file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return fix_stats
        
        # Proses setiap gambar
        self.logger.info(f"🔧 Memperbaiki dataset {split}: {len(image_files)} gambar")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in image_files:
                futures.append(executor.submit(
                    self._fix_image_label_pair, 
                    img_path, 
                    labels_dir,
                    fix_coordinates,
                    fix_labels,
                    fix_images
                ))
            
            # Process results dengan progress bar
            for future in tqdm(futures, desc=f"🔧 Fixing {split}", unit="img"):
                result = future.result()
                
                with self._lock:
                    fix_stats['processed'] += 1
                    if result.get('fixed_label', False): fix_stats['fixed_labels'] += 1
                    if result.get('fixed_coordinates', 0) > 0: fix_stats['fixed_coordinates'] += result['fixed_coordinates']
                    if result.get('fixed_image', False): fix_stats['fixed_images'] += 1
                    if result.get('error', False): fix_stats['errors'] += 1
                    if result.get('skipped', False): fix_stats['skipped'] += 1
        
        self.logger.success(
            f"✅ Perbaikan dataset {split} selesai:\n"
            f"   • Gambar diproses: {fix_stats['processed']}\n"
            f"   • Label diperbaiki: {fix_stats['fixed_labels']}\n"
            f"   • Koordinat diperbaiki: {fix_stats['fixed_coordinates']}\n"
            f"   • Gambar diperbaiki: {fix_stats['fixed_images']}\n"
            f"   • Error: {fix_stats['errors']}"
        )
        
        return fix_stats
    
    def _init_validation_stats(self) -> Dict[str, Any]:
        """Inisialisasi statistik validasi."""
        return {
            'total_images': 0, 'valid_images': 0, 'invalid_images': 0, 'corrupt_images': 0,
            'total_labels': 0, 'valid_labels': 0, 'invalid_labels': 0, 'missing_labels': 0, 'empty_labels': 0,
            'fixed_labels': 0, 'fixed_coordinates': 0,
            'layer_stats': {layer: 0 for layer in get_layer_config().get_layer_names()},
            'class_stats': {}, 'issues': []
        }
    
    def _validate_image_label_pair(self, img_path: Path, labels_dir: Path) -> Dict[str, Any]:
        """
        Validasi satu pasang gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori label
            
        Returns:
            Hasil validasi
        """
        result = {
            'image_path': str(img_path),
            'label_path': str(labels_dir / f"{img_path.stem}.txt"),
            'status': 'invalid',
            'issues': [],
            'layer_stats': {layer: 0 for layer in get_layer_config().get_layer_names()},
            'class_stats': {},
            'fixed': False,
            'visualized': False
        }
        
        # Validasi gambar
        try:
            img = self.utils.load_image(img_path)
            if img is None or img.size == 0:
                result['issues'].append(f"Gambar tidak dapat dibaca atau kosong: {img_path.name}")
                result['corrupt'] = True
                return result
                
            result['image_size'] = (img.shape[1], img.shape[0])
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
            bbox_data = self.utils.parse_yolo_label(label_path)
            
            if not bbox_data:
                result['issues'].append(f"File label kosong atau format tidak valid")
                result['empty_label'] = True
                return result
            
            # Cek layer dan kelas
            for box in bbox_data:
                cls_id = box['class_id']
                layer = box.get('layer')
                
                if layer and layer in result['layer_stats']:
                    result['layer_stats'][layer] += 1
                
                if 'class_name' in box:
                    class_name = box['class_name']
                    result['class_stats'][class_name] = result['class_stats'].get(class_name, 0) + 1
            
            # Set status valid jika tidak ada masalah
            result['has_active_layer'] = any(result['layer_stats'][layer] > 0 for layer in self.utils.active_layers)
            result['label_valid'] = result['has_active_layer']
            
            if result['label_valid'] and result['image_valid']:
                result['status'] = 'valid'
            
        except Exception as e:
            result['issues'].append(f"Error saat membaca label: {str(e)}")
        
        return result
    
    def _aggregate_validation_results(
        self, 
        results: List[Dict], 
        validation_stats: Dict[str, Any],
        fix_issues: bool,
        labels_dir: Path, 
        visualize: bool, 
        vis_dir: Optional[Path]
    ) -> None:
        """
        Agregasi hasil validasi dan update statistik.
        
        Args:
            results: Hasil validasi untuk setiap gambar
            validation_stats: Statistik validasi yang akan diupdate
            fix_issues: Apakah memperbaiki masalah
            labels_dir: Direktori label
            visualize: Apakah membuat visualisasi
            vis_dir: Direktori untuk visualisasi
        """
        for result in results:
            # Update statistik gambar
            if result.get('image_valid', False): 
                validation_stats['valid_images'] += 1
            else: 
                validation_stats['invalid_images'] += 1
                
            if result.get('corrupt', False): 
                validation_stats['corrupt_images'] += 1
            
            # Update statistik label
            if result.get('label_exists', False):
                validation_stats['total_labels'] += 1
                
                if result.get('label_valid', False):
                    validation_stats['valid_labels'] += 1
                    
                    # Update statistik layer & kelas
                    for layer, count in result.get('layer_stats', {}).items():
                        if layer in validation_stats['layer_stats']: 
                            validation_stats['layer_stats'][layer] += count
                        
                    for cls, count in result.get('class_stats', {}).items():
                        validation_stats['class_stats'][cls] = validation_stats['class_stats'].get(cls, 0) + count
                else:
                    validation_stats['invalid_labels'] += 1
                    
                if result.get('empty_label', False): 
                    validation_stats['empty_labels'] += 1
                
                # Perbaiki label jika diminta
                if fix_issues and result.get('fixed_label', False):
                    label_path = Path(result['label_path'])
                    if 'fixed_bbox' in result:
                        with open(label_path, 'w') as f:
                            for line in result['fixed_bbox']: 
                                f.write(line + '\n')
                        validation_stats['fixed_labels'] += 1
                    
                    # Hitung koordinat yang diperbaiki
                    if 'fixed_coordinates' in result:
                        validation_stats['fixed_coordinates'] += result['fixed_coordinates']
            else:
                validation_stats['missing_labels'] += 1
            
            # Kumpulkan masalah
            if result.get('issues'):
                for issue in result['issues']:
                    validation_stats['issues'].append(f"{Path(result['image_path']).name}: {issue}")
            
            # Visualisasi jika diminta
            if visualize and vis_dir and result.get('issues') and not result.get('visualized', False):
                self._visualize_issues(Path(result['image_path']), result, vis_dir)
                result['visualized'] = True
    
    def _move_invalid_files(self, split: str, validation_results: List[Dict]) -> Dict[str, int]:
        """
        Pindahkan file tidak valid ke direktori terpisah.
        
        Args:
            split: Split dataset
            validation_results: Hasil validasi untuk setiap gambar
            
        Returns:
            Statistik pemindahan
        """
        self.logger.info(f"🔄 Memindahkan file tidak valid ke {self.invalid_dir}...")
        
        # Filter file tidak valid
        invalid_images = [Path(r['image_path']) for r in validation_results 
                        if r.get('status') != 'valid' and 
                           (not r.get('image_valid', False) or r.get('corrupt', False))]
                           
        invalid_labels = [Path(r['label_path']) for r in validation_results 
                        if r.get('status') != 'valid' and 
                           r.get('label_exists', False) and not r.get('label_valid', False)]
        
        # Pindahkan file
        split_path = self.utils.get_split_path(split)
        img_stats = self.utils.move_invalid_files(
            split_path / 'images', self.invalid_dir / split / 'images', invalid_images)
        
        label_stats = self.utils.move_invalid_files(
            split_path / 'labels', self.invalid_dir / split / 'labels', invalid_labels)
        
        # Statistik gabungan
        stats = {
            'moved_images': img_stats['moved'], 
            'moved_labels': label_stats['moved'],
            'errors': img_stats['errors'] + label_stats['errors']
        }
        
        self.logger.success(
            f"✅ Pemindahan file tidak valid selesai:\n"
            f"   • Gambar dipindahkan: {stats['moved_images']}\n"
            f"   • Label dipindahkan: {stats['moved_labels']}\n"
            f"   • Error: {stats['errors']}"
        )
        
        return stats
    
    def _log_validation_summary(self, split: str, validation_stats: Dict[str, Any]) -> None:
        """
        Log ringkasan hasil validasi.
        
        Args:
            split: Split dataset
            validation_stats: Statistik validasi
        """
        self.logger.info(
            f"\n"
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
                pct = (count / max(1, validation_stats['valid_labels'])) * 100
                self.logger.info(f"   • {layer}: {count} objek ({pct:.1f}%)")
        
        # Log statistik kelas (top 10)
        if validation_stats['class_stats']:
            top_classes = sorted(validation_stats['class_stats'].items(), key=lambda x: x[1], reverse=True)[:10]
            
            self.logger.info("📊 Top 10 Kelas:")
            for cls, count in top_classes:
                pct = (count / max(1, validation_stats['valid_labels'])) * 100
                self.logger.info(f"   • {cls}: {count} ({pct:.1f}%)")
    
    def _fix_image_label_pair(
        self,
        img_path: Path,
        labels_dir: Path,
        fix_coordinates: bool,
        fix_labels: bool,
        fix_images: bool
    ) -> Dict[str, Any]:
        """
        Perbaiki masalah dalam satu pasang gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            labels_dir: Direktori label
            fix_coordinates: Apakah memperbaiki koordinat bbox
            fix_labels: Apakah memperbaiki format label
            fix_images: Apakah memperbaiki gambar dengan kualitas rendah
            
        Returns:
            Hasil perbaikan
        """
        result = {
            'fixed_label': False,
            'fixed_coordinates': 0,
            'fixed_image': False,
            'skipped': False,
            'error': False
        }
        
        # Proses gambar jika diminta
        if fix_images:
            try:
                result['fixed_image'] = self._fix_image(img_path)
            except Exception as e:
                self.logger.error(f"❌ Error saat memperbaiki gambar {img_path}: {str(e)}")
                result['error'] = True
        
        # Proses label jika diminta
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and (fix_coordinates or fix_labels):
            try:
                fixed_label, fixed_coords = self._fix_label(
                    label_path, 
                    fix_coordinates=fix_coordinates, 
                    fix_format=fix_labels
                )
                
                result['fixed_label'] = fixed_label
                result['fixed_coordinates'] = fixed_coords
            except Exception as e:
                self.logger.error(f"❌ Error saat memperbaiki label {label_path}: {str(e)}")
                result['error'] = True
        else:
            result['skipped'] = True
            
        return result
    
    def _fix_image(self, img_path: Path) -> bool:
        """
        Perbaiki gambar yang rusak.
        
        Args:
            img_path: Path ke file gambar
            
        Returns:
            True jika gambar berhasil diperbaiki
        """
        try:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Cek kualitas gambar
            issues = []
            h, w = img.shape[:2]
            
            # Cek resolusi (terlalu kecil?)
            if h < 100 or w < 100:
                issues.append("resolusi rendah")
                # Resize gambar
                img = cv2.resize(img, (max(w, 100), max(h, 100)), interpolation=cv2.INTER_CUBIC)
            
            # Cek kontras
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            
            if contrast < 20:
                issues.append("kontras rendah")
                # Perbaiki kontras
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Cek blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                issues.append("blur")
                # Sharpening
                kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
            
            # Simpan gambar jika ada isu yang diperbaiki
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
    ) -> tuple:
        """
        Perbaiki file label.
        
        Args:
            label_path: Path ke file label
            fix_coordinates: Apakah memperbaiki koordinat bbox
            fix_format: Apakah memperbaiki format label
            
        Returns:
            (fixed_label, fixed_coords_count)
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
                    if len(parts) < 5:
                        continue  # Format terlalu rusak
                    parts = parts[:5]  # Ambil 5 bagian pertama
                    fixed_something = True
                
                try:
                    # Parse data
                    cls_id = int(float(parts[0]))
                    bbox = [float(x) for x in parts[1:5]]
                    
                    # Validasi class ID
                    layer_config = get_layer_config()
                    if layer_config.get_layer_for_class_id(cls_id) is None:
                        continue  # Class ID tidak valid
                    
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
            
    def _visualize_issues(self, img_path: Path, result: Dict, vis_dir: Path) -> bool:
        """
        Visualisasikan masalah dalam gambar dan label.
        
        Args:
            img_path: Path ke file gambar
            result: Hasil validasi
            vis_dir: Direktori untuk visualisasi
            
        Returns:
            True jika visualisasi berhasil dibuat
        """
        try:
            img = self.utils.load_image(img_path)
            if img is None:
                return False
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vis_path = vis_dir / f"{img_path.stem}_issues.jpg"
            h, w = img.shape[:2]
            
            # Dapatkan bbox dari label
            label_path = Path(result['label_path'])
            if label_path.exists():
                bbox_data = self.utils.parse_yolo_label(label_path)
                
                for box in bbox_data:
                    bbox = box['bbox']
                    cls_id = box['class_id']
                    
                    # Konversi YOLO format ke pixel
                    x_center, y_center, width, height = bbox
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Tentukan warna berdasarkan layer
                    if 'layer' in box and box['layer'] in self.utils.active_layers:
                        color = (0, 255, 0)  # Hijau jika layer aktif
                    else:
                        color = (0, 0, 255)  # Merah jika bukan layer aktif
                    
                    # Gambar rectangle dan label
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    class_name = box.get('class_name', f"ID: {cls_id}")
                    cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Tambahkan informasi masalah
            cv2.putText(img, f"Issues: {len(result.get('issues', []))}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite(str(vis_path), img)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat visualisasi: {str(e)}")
            return False