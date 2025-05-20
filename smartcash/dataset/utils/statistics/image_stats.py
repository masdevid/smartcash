"""
File: smartcash/dataset/utils/statistics/image_stats.py
Deskripsi: Utilitas untuk menganalisis statistik gambar dalam dataset
"""

import os
import cv2
import collections
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class ImageStatistics:
    """Utilitas untuk analisis statistik gambar dalam dataset."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi ImageStatistics.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        self.logger.info(f"📊 ImageStatistics diinisialisasi dengan data_dir: {self.data_dir}")
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dictionary dengan hasil analisis ukuran gambar
        """
        self.logger.info(f"📐 Analisis ukuran gambar untuk split {split}")
        split_path = self.utils.get_split_path(split)
        images_dir = split_path / 'images'
        
        # Validasi direktori
        if not images_dir.exists():
            self.logger.error(f"❌ Direktori gambar tidak ditemukan: {images_dir}")
            return {'status': 'error', 'message': f"Direktori gambar tidak ditemukan"}
            
        # Cari semua file gambar
        image_files = self.utils.find_image_files(images_dir, with_labels=False)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'error', 'message': f"Tidak ada gambar ditemukan"}
            
        # Ambil sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Analisis ukuran gambar
        return self._analyze_images(image_files)
    
    def analyze_image_quality(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis kualitas gambar (blur, kontras, noise) dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dictionary dengan hasil analisis kualitas gambar
        """
        self.logger.info(f"🔍 Analisis kualitas gambar untuk split {split}")
        split_path = self.utils.get_split_path(split)
        images_dir = split_path / 'images'
        
        # Validasi direktori
        if not images_dir.exists():
            self.logger.error(f"❌ Direktori gambar tidak ditemukan: {images_dir}")
            return {'status': 'error', 'message': f"Direktori gambar tidak ditemukan"}
            
        # Cari semua file gambar
        image_files = self.utils.find_image_files(images_dir, with_labels=False)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'error', 'message': f"Tidak ada gambar ditemukan"}
            
        # Ambil sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            image_files = self.utils.get_random_sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan sampel {sample_size} gambar dari total {len(image_files)}")
        
        # Analisis kualitas gambar
        return self._analyze_image_quality(image_files)
    
    def find_problematic_images(self, split: str, threshold: Dict[str, float] = None) -> Dict[str, List[str]]:
        """
        Temukan gambar bermasalah dalam dataset berdasarkan threshold kualitas.
        
        Args:
            split: Split dataset yang akan dianalisis
            threshold: Dict threshold untuk setiap metrik (blur, contrast, noise)
            
        Returns:
            Dictionary dengan daftar gambar bermasalah per kategori
        """
        # Default threshold
        if threshold is None:
            threshold = {
                'blur': 50.0,       # Nilai variasi Laplacian di bawah ini dianggap blur
                'contrast': 40.0,    # Nilai standar deviasi kontras di bawah ini dianggap kontras rendah
                'noise': 5.0         # Nilai estimasi noise di atas ini dianggap noise tinggi
            }
        
        # Dapatkan hasil analisis kualitas
        result = self.analyze_image_quality(split)
        if result['status'] != 'success':
            return {'blur': [], 'contrast': [], 'noise': []}
        
        problematic_images = {
            'blur': [],
            'contrast': [],
            'noise': []
        }
        
        # Filter gambar bermasalah
        for img_path, metrics in result['image_metrics'].items():
            if metrics['blur_score'] < threshold['blur']:
                problematic_images['blur'].append(img_path)
                
            if metrics['contrast_score'] < threshold['contrast']:
                problematic_images['contrast'].append(img_path)
                
            if metrics['noise_score'] > threshold['noise']:
                problematic_images['noise'].append(img_path)
        
        # Log hasil
        self.logger.info(f"📑 Gambar bermasalah ditemukan:")
        for category, images in problematic_images.items():
            self.logger.info(f"   • {category}: {len(images)} gambar")
            
        return problematic_images
    
    def _analyze_images(self, image_files: List[Path]) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dari daftar file.
        
        Args:
            image_files: Daftar file gambar
            
        Returns:
            Dictionary dengan hasil analisis ukuran
        """
        # Hitung frekuensi ukuran gambar
        size_counts = collections.Counter()
        width_list, height_list, aspect_ratios = [], [], []
        
        # Fungsi untuk memproses satu gambar
        def process_image(img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    return None
                
                h, w = img.shape[:2]
                return (w, h)
            except Exception:
                return None
        
        # Proses gambar secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in image_files:
                futures.append(executor.submit(process_image, img_path))
            
            # Collect results dengan progress bar
            for future in tqdm(futures, desc="📐 Menganalisis ukuran gambar", unit="img"):
                result = future.result()
                if result:
                    w, h = result
                    size_counts[(w, h)] += 1
                    width_list.append(w)
                    height_list.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)
        
        # Jika tidak ada gambar valid
        if not width_list:
            self.logger.warning(f"⚠️ Tidak ada gambar yang dapat dibaca")
            return {'status': 'error', 'message': f"Tidak ada gambar yang dapat dibaca"}
        
        # Ukuran dominan
        dominant_size = size_counts.most_common(1)[0][0]
        dominant_percentage = (size_counts[dominant_size] / len(width_list)) * 100
        
        # Statistik dimensi
        width_stats = {
            'min': min(width_list),
            'max': max(width_list),
            'mean': np.mean(width_list),
            'median': np.median(width_list),
            'std': np.std(width_list)
        }
        
        height_stats = {
            'min': min(height_list),
            'max': max(height_list),
            'mean': np.mean(height_list),
            'median': np.median(height_list),
            'std': np.std(height_list)
        }
        
        aspect_ratio_stats = {
            'min': min(aspect_ratios),
            'max': max(aspect_ratios),
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'std': np.std(aspect_ratios)
        }
        
        # Kategorikan ukuran (kecil, sedang, besar)
        size_categories = {'small': 0, 'medium': 0, 'large': 0}
        
        for (w, h), count in size_counts.items():
            if w < 640 or h < 640:
                size_categories['small'] += count
            elif w > 1280 or h > 1280:
                size_categories['large'] += count
            else:
                size_categories['medium'] += count
        
        # Rekomendasi ukuran optimal (misalnya, kelipatkan 32 untuk model YOLO)
        recommended_w = round(width_stats['median'] / 32) * 32
        recommended_h = round(height_stats['median'] / 32) * 32
        recommended_size = f"{recommended_w}x{recommended_h}"
        
        # Log hasil
        self.logger.info(
            f"📊 Statistik ukuran gambar:\n"
            f"   • Total gambar valid: {len(width_list)}\n"
            f"   • Ukuran dominan: {dominant_size[0]}x{dominant_size[1]} ({dominant_percentage:.1f}%)\n"
            f"   • Rekomendasi ukuran: {recommended_size}"
        )
        
        # Kompilasi hasil
        result = {
            'status': 'success',
            'total_analyzed': len(width_list),
            'dominant_size': f"{dominant_size[0]}x{dominant_size[1]}",
            'dominant_percentage': dominant_percentage,
            'width_stats': width_stats,
            'height_stats': height_stats,
            'aspect_ratio_stats': aspect_ratio_stats,
            'size_categories': size_categories,
            'recommended_size': recommended_size
        }
        
        return result
    
    def _analyze_image_quality(self, image_files: List[Path]) -> Dict[str, Any]:
        """
        Analisis kualitas gambar dari daftar file.
        
        Args:
            image_files: Daftar file gambar
            
        Returns:
            Dictionary dengan hasil analisis kualitas
        """
        # Fungsi untuk memproses satu gambar
        def process_image_quality(img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    return None
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Blur score (variasi Laplacian - semakin tinggi, semakin tajam)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Contrast score (standar deviasi - semakin tinggi, semakin kontras)
                contrast_score = gray.std()
                
                # Noise score (estimasi sederhana - semakin rendah, semakin baik)
                # Menggunakan filter median untuk mengurangi noise
                denoised = cv2.medianBlur(gray, 5)
                noise_score = np.mean(np.abs(gray.astype(np.float32) - denoised.astype(np.float32)))
                
                return {
                    'path': str(img_path),
                    'blur_score': blur_score,
                    'contrast_score': contrast_score,
                    'noise_score': noise_score
                }
            except Exception as e:
                return None
        
        # Proses gambar secara paralel
        metrics = []
        image_metrics = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for img_path in image_files:
                futures.append(executor.submit(process_image_quality, img_path))
            
            # Collect results dengan progress bar
            for future in tqdm(futures, desc="🔍 Menganalisis kualitas gambar", unit="img"):
                result = future.result()
                if result:
                    metrics.append(result)
                    image_metrics[result['path']] = result
        
        # Jika tidak ada gambar valid
        if not metrics:
            self.logger.warning(f"⚠️ Tidak ada gambar yang dapat dianalisis")
            return {'status': 'error', 'message': f"Tidak ada gambar yang dapat dianalisis"}
        
        # Statistik blur
        blur_scores = [m['blur_score'] for m in metrics]
        blur_stats = {
            'min': min(blur_scores),
            'max': max(blur_scores),
            'mean': np.mean(blur_scores),
            'median': np.median(blur_scores),
            'std': np.std(blur_scores)
        }
        
        # Statistik kontras
        contrast_scores = [m['contrast_score'] for m in metrics]
        contrast_stats = {
            'min': min(contrast_scores),
            'max': max(contrast_scores),
            'mean': np.mean(contrast_scores),
            'median': np.median(contrast_scores),
            'std': np.std(contrast_scores)
        }
        
        # Statistik noise
        noise_scores = [m['noise_score'] for m in metrics]
        noise_stats = {
            'min': min(noise_scores),
            'max': max(noise_scores),
            'mean': np.mean(noise_scores),
            'median': np.median(noise_scores),
            'std': np.std(noise_scores)
        }
        
        # Kategorikan kualitas
        quality_categories = {
            'high': 0,    # Blur_score tinggi, contrast tinggi, noise rendah
            'medium': 0,  # Menengah
            'low': 0      # Blur_score rendah, contrast rendah, noise tinggi
        }
        
        for m in metrics:
            # Klasifikasi sederhana berdasarkan median
            if (m['blur_score'] > blur_stats['median'] and 
                m['contrast_score'] > contrast_stats['median'] and 
                m['noise_score'] < noise_stats['median']):
                quality_categories['high'] += 1
            elif (m['blur_score'] < blur_stats['median'] * 0.5 or 
                 m['contrast_score'] < contrast_stats['median'] * 0.5 or 
                 m['noise_score'] > noise_stats['median'] * 1.5):
                quality_categories['low'] += 1
            else:
                quality_categories['medium'] += 1
        
        # Log hasil
        self.logger.info(
            f"📊 Statistik kualitas gambar:\n"
            f"   • Total gambar yang dianalisis: {len(metrics)}\n"
            f"   • Blur score (tinggi = tajam): median = {blur_stats['median']:.2f}\n"
            f"   • Contrast score: median = {contrast_stats['median']:.2f}\n"
            f"   • Noise score (rendah = baik): median = {noise_stats['median']:.2f}\n"
            f"   • Distribusi kualitas: high = {quality_categories['high']}, "
            f"medium = {quality_categories['medium']}, low = {quality_categories['low']}"
        )
        
        # Kompilasi hasil
        result = {
            'status': 'success',
            'total_analyzed': len(metrics),
            'blur_stats': blur_stats,
            'contrast_stats': contrast_stats,
            'noise_stats': noise_stats,
            'quality_categories': quality_categories,
            'image_metrics': image_metrics
        }
        
        return result