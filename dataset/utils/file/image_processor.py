"""
File: smartcash/dataset/utils/file/image_processor.py
Deskripsi: Utilitas untuk memproses dan memanipulasi gambar dalam dataset dengan pendekatan DRY
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.common.logger import get_logger
from smartcash.common.file_utils import get_file_utils
from smartcash.dataset.utils import find_image_files, ensure_dir
from smartcash.dataset.utils.progress_wrapper import update_progress
from smartcash.common.threadpools import process_in_parallel


class ImageProcessor:
    """Utilitas untuk pemrosesan dan manipulasi gambar dataset."""
    
    def __init__(self, config: Dict = None, data_dir: Optional[str] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi ImageProcessor.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config or {}
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.logger = logger or get_logger("image_processor")
        self.num_workers = num_workers
        self.file_utils = get_file_utils(config, logger, num_workers)
        
        self.logger.info(f"🖼️ ImageProcessor diinisialisasi dengan data_dir: {self.data_dir}")
    
    def resize_images(
        self, 
        directory: Union[str, Path], 
        target_size: Tuple[int, int],
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        interpolation: int = cv2.INTER_AREA,
        keep_aspect_ratio: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Resize gambar dalam direktori.
        
        Args:
            directory: Direktori gambar
            target_size: Ukuran target (width, height)
            output_dir: Direktori output (opsional, default: sama dengan input)
            recursive: Apakah memeriksa subdirektori secara rekursif
            extensions: Daftar ekstensi file gambar (default: ['.jpg', '.jpeg', '.png'])
            interpolation: Metode interpolasi CV2 (default: INTER_AREA)
            keep_aspect_ratio: Apakah mempertahankan aspek rasio
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik resize
        """
        directory = Path(directory)
        output_dir = Path(output_dir) if output_dir else directory
        ensure_dir(output_dir)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'resized': 0, 'errors': 0}
            
        # Cari file gambar
        image_files = find_image_files(directory, recursive)
                
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
            return {'resized': 0, 'errors': 0}
            
        self.logger.info(f"🔄 Resize {len(image_files)} gambar ke {target_size[0]}x{target_size[1]}")
        
        # Fungsi untuk resize satu gambar
        def resize_image(img_path):
            try:
                # Baca gambar
                img = cv2.imread(str(img_path))
                if img is None:
                    return 'error'
                
                # Hitung dimensi target dengan mempertahankan aspek rasio jika diminta
                if keep_aspect_ratio:
                    h, w = img.shape[:2]
                    aspect = w / h
                    
                    if aspect > 1:  # Landscape
                        new_w, new_h = target_size[0], int(target_size[0] / aspect)
                        new_h = target_size[1] if new_h > target_size[1] else new_h
                    else:  # Portrait
                        new_h, new_w = target_size[1], int(target_size[1] * aspect)
                        new_w = target_size[0] if new_w > target_size[0] else new_w
                    
                    # Resize dan letakkan di canvas kosong
                    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
                    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    x_offset, y_offset = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    resized = canvas
                else:
                    # Resize langsung tanpa mempertahankan aspek rasio
                    resized = cv2.resize(img, target_size, interpolation=interpolation)
                
                # Tentukan path output
                rel_path = img_path.relative_to(directory) if directory != output_dir else Path(img_path.name)
                output_path = output_dir / rel_path
                ensure_dir(output_path.parent)
                
                # Simpan gambar
                cv2.imwrite(str(output_path), resized)
                return 'resized'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal resize {img_path}: {str(e)}")
                return 'error'
        
        # Resize gambar secara paralel
        results = process_in_parallel(
            image_files, 
            resize_image, 
            max_workers=self.num_workers,
            desc="🔄 Resize gambar",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {'resized': results.count('resized'), 'errors': results.count('error')}
        
        self.logger.success(
            f"✅ Resize gambar selesai:\n"
            f"   • Resized: {stats['resized']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def enhance_images(
        self, 
        directory: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        enhance_contrast: bool = True,
        sharpen: bool = True,
        denoise: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Tingkatkan kualitas gambar dengan berbagai metode.
        
        Args:
            directory: Direktori gambar
            output_dir: Direktori output (opsional, default: sama dengan input)
            recursive: Apakah memeriksa subdirektori secara rekursif
            extensions: Daftar ekstensi file gambar (default: ['.jpg', '.jpeg', '.png'])
            enhance_contrast: Apakah meningkatkan kontras
            sharpen: Apakah mempertajam gambar
            denoise: Apakah mengurangi noise
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik enhancement
        """
        directory = Path(directory)
        output_dir = Path(output_dir) if output_dir else directory
        ensure_dir(output_dir)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'enhanced': 0, 'errors': 0}
            
        # Cari file gambar
        image_files = find_image_files(directory, recursive)
                
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
            return {'enhanced': 0, 'errors': 0}
            
        self.logger.info(f"🔍 Meningkatkan kualitas {len(image_files)} gambar")
        
        # Fungsi untuk enhance satu gambar
        def enhance_image(img_path):
            try:
                # Baca gambar
                img = cv2.imread(str(img_path))
                if img is None:
                    return 'error'
                
                enhanced = img.copy()
                
                # Konversi ke Lab color space untuk enhancement yang lebih baik
                if enhance_contrast:
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    l_channel, a, b = cv2.split(lab)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l_channel)
                    
                    # Gabungkan channel
                    updated_lab = cv2.merge((cl, a, b))
                    
                    # Konversi kembali ke BGR
                    enhanced = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
                
                # Sharpen gambar
                if sharpen:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    enhanced = cv2.filter2D(enhanced, -1, kernel)
                
                # Denoise gambar
                if denoise:
                    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
                
                # Tentukan path output
                rel_path = img_path.relative_to(directory) if directory != output_dir else Path(img_path.name)
                output_path = output_dir / rel_path
                ensure_dir(output_path.parent)
                
                # Simpan gambar
                cv2.imwrite(str(output_path), enhanced)
                return 'enhanced'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal meningkatkan kualitas {img_path}: {str(e)}")
                return 'error'
        
        # Enhance gambar secara paralel
        results = process_in_parallel(
            image_files, 
            enhance_image, 
            max_workers=self.num_workers,
            desc="🔍 Meningkatkan kualitas gambar",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {'enhanced': results.count('enhanced'), 'errors': results.count('error')}
        
        self.logger.success(
            f"✅ Peningkatan kualitas gambar selesai:\n"
            f"   • Enhanced: {stats['enhanced']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def convert_format(
        self, 
        directory: Union[str, Path], 
        target_format: str,
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        quality: int = 95,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Konversi format gambar dalam direktori.
        
        Args:
            directory: Direktori gambar
            target_format: Format target ('.jpg', '.png', '.webp', etc.)
            output_dir: Direktori output (opsional, default: sama dengan input)
            recursive: Apakah memeriksa subdirektori secara rekursif
            extensions: Daftar ekstensi file gambar (default: ['.jpg', '.jpeg', '.png'])
            quality: Kualitas kompresi (0-100)
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik konversi
        """
        directory = Path(directory)
        output_dir = Path(output_dir) if output_dir else directory
        ensure_dir(output_dir)
        
        # Validasi format target
        target_format = target_format.lower()
        if not target_format.startswith('.'):
            target_format = f'.{target_format}'
            
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'converted': 0, 'errors': 0}
            
        # Cari file gambar
        image_files = find_image_files(directory, recursive)
                
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
            return {'converted': 0, 'errors': 0}
        
        # Filter gambar yang sudah dalam format target
        image_files = [img for img in image_files if img.suffix.lower() != target_format]
            
        self.logger.info(f"🔄 Konversi {len(image_files)} gambar ke format {target_format}")
        
        # Fungsi untuk konversi satu gambar
        def convert_image(img_path):
            try:
                # Baca gambar
                img = cv2.imread(str(img_path))
                if img is None:
                    return 'error'
                
                # Tentukan path output
                rel_path = img_path.relative_to(directory) if directory != output_dir else Path(img_path.name)
                output_path = output_dir / rel_path.with_suffix(target_format)
                ensure_dir(output_path.parent)
                
                # Setup parameter kualitas
                if target_format in ['.jpg', '.jpeg']:
                    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif target_format == '.png':
                    params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, int(9 - (quality / 11.1)))]
                elif target_format == '.webp':
                    params = [cv2.IMWRITE_WEBP_QUALITY, quality]
                else:
                    params = []
                
                # Simpan gambar
                cv2.imwrite(str(output_path), img, params)
                return 'converted'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal konversi {img_path}: {str(e)}")
                return 'error'
        
        # Konversi gambar secara paralel
        results = process_in_parallel(
            image_files, 
            convert_image, 
            max_workers=self.num_workers,
            desc=f"🔄 Konversi ke {target_format}",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {'converted': results.count('converted'), 'errors': results.count('error')}
        
        self.logger.success(
            f"✅ Konversi format gambar selesai:\n"
            f"   • Converted: {stats['converted']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def create_thumbnails(
        self, 
        directory: Union[str, Path], 
        thumbnail_size: Tuple[int, int],
        output_dir: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        keep_aspect_ratio: bool = True,
        quality: int = 90,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Buat thumbnail untuk gambar dalam direktori.
        
        Args:
            directory: Direktori gambar
            thumbnail_size: Ukuran thumbnail (width, height)
            output_dir: Direktori output (opsional, default: {directory}/thumbnails)
            recursive: Apakah memeriksa subdirektori secara rekursif
            extensions: Daftar ekstensi file gambar (default: ['.jpg', '.jpeg', '.png'])
            keep_aspect_ratio: Apakah mempertahankan aspek rasio
            quality: Kualitas kompresi (0-100)
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik pembuatan thumbnail
        """
        directory = Path(directory)
        output_dir = Path(output_dir) if output_dir else directory / 'thumbnails'
        ensure_dir(output_dir)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return {'created': 0, 'errors': 0}
            
        # Cari file gambar
        image_files = find_image_files(directory, recursive)
                
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
            return {'created': 0, 'errors': 0}
            
        self.logger.info(f"🖼️ Membuat thumbnail untuk {len(image_files)} gambar")
        
        # Fungsi untuk membuat thumbnail satu gambar
        def create_thumbnail(img_path):
            try:
                # Baca gambar
                img = cv2.imread(str(img_path))
                if img is None:
                    return 'error'
                
                # Buat thumbnail dengan mempertahankan aspek rasio jika diminta
                if keep_aspect_ratio:
                    h, w = img.shape[:2]
                    aspect = w / h
                    
                    if aspect > 1:  # Landscape
                        new_w = thumbnail_size[0]
                        new_h = int(new_w / aspect)
                    else:  # Portrait
                        new_h = thumbnail_size[1]
                        new_w = int(new_h * aspect)
                        
                    thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    # Resize langsung tanpa mempertahankan aspek rasio
                    thumbnail = cv2.resize(img, thumbnail_size, interpolation=cv2.INTER_AREA)
                
                # Tentukan path output
                rel_path = img_path.relative_to(directory) if recursive else img_path.name
                output_path = output_dir / rel_path
                ensure_dir(output_path.parent)
                
                # Setup parameter kualitas
                params = ([cv2.IMWRITE_JPEG_QUALITY, quality] if output_path.suffix.lower() in ['.jpg', '.jpeg'] else 
                          [cv2.IMWRITE_PNG_COMPRESSION, min(9, int(9 - (quality / 11.1)))] if output_path.suffix.lower() == '.png' else [])
                
                # Simpan thumbnail
                cv2.imwrite(str(output_path), thumbnail, params)
                return 'created'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal membuat thumbnail {img_path}: {str(e)}")
                return 'error'
        
        # Buat thumbnail secara paralel
        results = process_in_parallel(
            image_files, 
            create_thumbnail, 
            max_workers=self.num_workers,
            desc="🖼️ Membuat thumbnail",
            show_progress=show_progress
        )
        
        # Hitung statistik
        stats = {'created': results.count('created'), 'errors': results.count('error')}
        
        self.logger.success(
            f"✅ Pembuatan thumbnail selesai:\n"
            f"   • Created: {stats['created']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats