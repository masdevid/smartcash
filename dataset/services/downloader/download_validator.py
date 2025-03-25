"""
File: smartcash/dataset/services/downloader/download_validator.py
Deskripsi: Komponen teroptimasi untuk memvalidasi integritas dan struktur dataset yang didownload
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.dataset.utils.file_wrapper import file_exists


class DownloadValidator:
    """
    Validator untuk memeriksa integritas dan struktur dataset yang didownload.
    Memastikan dataset lengkap dan sesuai dengan format yang diharapkan.
    """
    
    def __init__(self, logger=None, num_workers: int = None):
        """
        Inisialisasi DownloadValidator dengan performa teroptimasi.
        
        Args:
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel (opsional)
        """
        self.logger = logger or get_logger("download_validator")
        # Optimasi: gunakan CPU count sebagai default tapi batasi maksimum
        self.num_workers = min(num_workers or (os.cpu_count() or 2), 8)
    
    def verify_download(self, download_dir: Union[str, Path], metadata: Dict) -> bool:
        """
        Verifikasi integritas hasil download berdasarkan metadata.
        
        Args:
            download_dir: Direktori hasil download
            metadata: Metadata dataset dari Roboflow
            
        Returns:
            Boolean yang menunjukkan keberhasilan verifikasi
        """
        download_path = Path(download_dir)
        
        if not download_path.exists():
            self.logger.error(f"❌ Direktori download tidak ditemukan: {download_path}")
            return False
        
        try:
            # Cek struktur minimal
            all_splits = self._get_valid_splits(download_path)
            if not all_splits:
                self.logger.error("❌ Tidak ada split yang valid dalam dataset")
                return False
                
            # Optimasi: periksa split secara paralel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = {}
                for split in all_splits:
                    results[split] = executor.submit(
                        self._verify_split, 
                        download_path / split, 
                        metadata.get('version', {}).get('splits', {}).get(split)
                    )
                
                # Collect results
                split_results = {split: result.result() for split, result in results.items()}
            
            # Log overall stats and results
            for split, result in split_results.items():
                if not result.get('valid', False):
                    self.logger.warning(f"⚠️ Split {split} tidak valid: {result.get('reason', 'unknown')}")
                
                self.logger.info(
                    f"📊 Statistik {split}: {result.get('image_count', 0)} gambar, "
                    f"{result.get('label_count', 0)} label, "
                    f"orphan images: {result.get('orphan_images', 0)}, "
                    f"orphan labels: {result.get('orphan_labels', 0)}"
                )
            
            valid_splits = [split for split, result in split_results.items() if result.get('valid', False)]
            if not valid_splits:
                self.logger.error("❌ Tidak ada split yang valid dalam dataset")
                return False
                
            self.logger.success(f"✅ Verifikasi download berhasil, split valid: {', '.join(valid_splits)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat verifikasi download: {str(e)}")
            return False
    
    def _verify_split(self, split_dir: Path, expected_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Verifikasi satu split dataset.
        
        Args:
            split_dir: Path ke direktori split
            expected_count: Jumlah file yang diharapkan (dari metadata)
            
        Returns:
            Dictionary berisi hasil verifikasi
        """
        result = {
            'valid': False, 
            'image_count': 0, 
            'label_count': 0,
            'orphan_images': 0,
            'orphan_labels': 0
        }
        
        if not split_dir.exists():
            result['reason'] = f"Split direktori tidak ditemukan: {split_dir}"
            return result
            
        # Cek struktur dasar
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            result['reason'] = "Direktori images atau labels tidak ditemukan"
            return result
            
        # Hitung jumlah file
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        result['image_count'] = len(image_files)
        result['label_count'] = len(label_files)
        
        # Validasi jumlah gambar dan label
        if result['image_count'] == 0 or result['label_count'] == 0:
            result['reason'] = f"Split kosong (images: {result['image_count']}, labels: {result['label_count']})"
            return result
            
        # Check expected count
        if expected_count is not None and result['image_count'] < expected_count:
            self.logger.warning(
                f"⚠️ Jumlah gambar di {split_dir.name} ({result['image_count']}) "
                f"kurang dari yang diharapkan ({expected_count})"
            )
        
        # Optimasi: Sampling jika jumlah file terlalu banyak
        max_sample = 100
        orphan_images, orphan_labels = 0, 0
        
        # Periksa sampel gambar tanpa label
        sample_images = image_files[:max_sample] if len(image_files) > max_sample else image_files
        for img_file in sample_images:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                orphan_images += 1
                
        # Periksa sampel label tanpa gambar
        sample_labels = label_files[:max_sample] if len(label_files) > max_sample else label_files
        for label_file in sample_labels:
            found_image = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_file = images_dir / f"{label_file.stem}{ext}"
                if img_file.exists():
                    found_image = True
                    break
            
            if not found_image:
                orphan_labels += 1
        
        # Extrapolate untuk seluruh dataset jika sampling
        if len(image_files) > max_sample:
            orphan_images = int(orphan_images * (len(image_files) / len(sample_images)))
        if len(label_files) > max_sample:
            orphan_labels = int(orphan_labels * (len(label_files) / len(sample_labels)))
            
        result['orphan_images'] = orphan_images
        result['orphan_labels'] = orphan_labels
        
        # Validasi berhasil jika tidak ada masalah serius
        result['valid'] = True
        
        # Tapi masih warning jika banyak orphan files
        if orphan_images > 0 or orphan_labels > 0:
            threshold = 0.1  # 10% maksimum orphan files yang dianggap masih valid
            if (orphan_images / max(1, len(image_files)) > threshold or 
                orphan_labels / max(1, len(label_files)) > threshold):
                self.logger.warning(
                    f"⚠️ Terlalu banyak file orphan di {split_dir.name}: "
                    f"{orphan_images} gambar tanpa label ({orphan_images / max(1, len(image_files)):.1%}), "
                    f"{orphan_labels} label tanpa gambar ({orphan_labels / max(1, len(label_files)):.1%})"
                )
        
        return result
    
    def _get_valid_splits(self, dataset_dir: Path) -> List[str]:
        """Dapatkan daftar split yang ada di direktori dataset."""
        return [split for split in DEFAULT_SPLITS if (dataset_dir / split).exists()]
    
    def verify_local_dataset(self, data_dir: Union[str, Path]) -> bool:
        """
        Verifikasi struktur dataset lokal dengan optimasi.
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Boolean yang menunjukkan hasil verifikasi
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            self.logger.error(f"❌ Direktori dataset tidak ditemukan: {data_path}")
            return False
        
        try:
            # Optimasi: periksa split secara paralel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = {}
                for split in DEFAULT_SPLITS:
                    split_dir = data_path / split
                    results[split] = executor.submit(self._verify_local_split, split_dir)
                
                # Collect results
                valid_splits = [split for split, result in results.items() 
                               if result.result().get('valid', False)]
            
            # Cek minimal satu split harus ada dan valid
            if not valid_splits:
                self.logger.error("❌ Tidak ada split yang valid dalam dataset")
                return False
                
            self.logger.success(f"✅ Verifikasi dataset lokal berhasil, split valid: {', '.join(valid_splits)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat verifikasi dataset lokal: {str(e)}")
            return False
    
    def _verify_local_split(self, split_dir: Path) -> Dict[str, Any]:
        """Verifikasi satu split dataset lokal."""
        result = {
            'valid': False, 
            'image_count': 0, 
            'label_count': 0
        }
        
        if not split_dir.exists():
            return result
            
        # Cek struktur dasar
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            return result
            
        # Hitung jumlah file
        img_count = len(list(images_dir.glob('*.*')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        result['image_count'] = img_count
        result['label_count'] = label_count
        
        # Validasi jumlah gambar dan label
        if img_count > 0 and label_count > 0:
            result['valid'] = True
            
        return result
    
    def is_dataset_available(
        self,
        data_dir: Union[str, Path],
        verify_content: bool = False
    ) -> bool:
        """
        Cek apakah dataset tersedia di direktori.
        
        Args:
            data_dir: Direktori dataset
            verify_content: Apakah memverifikasi konten dataset
            
        Returns:
            Boolean yang menunjukkan ketersediaan dataset
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return False
        
        # Quick check struktur dasar
        for split in DEFAULT_SPLITS:
            split_dir = data_path / split
            if not split_dir.exists():
                return False
                
            # Cek subdirektori dengan operator and untuk short-circuit
            if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                return False
        
        # Jika tidak perlu verifikasi konten lebih lanjut
        if not verify_content:
            return True
            
        # Verifikasi konten (minimal ada file di train)
        train_images = list((data_path / 'train' / 'images').glob('*.*'))
        train_labels = list((data_path / 'train' / 'labels').glob('*.txt'))
        
        return bool(train_images) and bool(train_labels)
    
    def get_dataset_stats(self, dataset_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Dapatkan statistik dataset dengan teroptimasi untuk performa.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dictionary berisi statistik dataset
        """
        dataset_path = Path(dataset_dir)
        stats = {'total_images': 0, 'total_labels': 0, 'splits': {}}
        
        if not dataset_path.exists():
            return stats
        
        # Optimasi: hitung secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for split in DEFAULT_SPLITS:
                split_dir = dataset_path / split
                
                if not split_dir.exists():
                    continue
                    
                futures[split] = executor.submit(self._get_split_stats, split_dir)
            
            # Collect results
            for split, future in futures.items():
                split_stats = future.result()
                stats['splits'][split] = split_stats
                stats['total_images'] += split_stats.get('images', 0)
                stats['total_labels'] += split_stats.get('labels', 0)
        
        return stats
    
    def _get_split_stats(self, split_dir: Path) -> Dict[str, int]:
        """Hitung statistik untuk satu split."""
        stats = {'images': 0, 'labels': 0}
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if images_dir.exists():
            stats['images'] = len(list(images_dir.glob('*.*')))
            
        if labels_dir.exists():
            stats['labels'] = len(list(labels_dir.glob('*.txt')))
            
        return stats
    
    def get_local_stats(self, data_dir: Union[str, Path]) -> Dict[str, int]:
        """
        Dapatkan statistik dataset lokal (jumlah sampel per split).
        
        Args:
            data_dir: Direktori dataset
            
        Returns:
            Dictionary berisi jumlah sampel per split
        """
        data_path = Path(data_dir)
        stats = {}
        
        if not data_path.exists():
            return stats
        
        # Optimasi: hitung secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for split in DEFAULT_SPLITS:
                images_dir = data_path / split / 'images'
                if not images_dir.exists():
                    stats[split] = 0
                    continue
                    
                futures[split] = executor.submit(len, list(images_dir.glob('*.*')))
            
            # Collect results
            for split, future in futures.items():
                if split in futures:  # Skip yang sudah di-set = 0
                    stats[split] = future.result()
        
        return stats