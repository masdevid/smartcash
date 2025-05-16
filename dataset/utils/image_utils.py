"""
File: smartcash/dataset/utils/image_utils.py
Deskripsi: Utilitas untuk manipulasi dan pengolahan gambar
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Muat gambar dari file.
    
    Args:
        image_path: Path file gambar
        
    Returns:
        Gambar dalam format numpy array (RGB) atau None jika gagal
    """
    try:
        # Baca gambar dengan OpenCV (BGR)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Gagal memuat gambar: {image_path}")
            return None
            
        # Konversi BGR ke RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        print(f"⚠️ Error saat memuat gambar dari {image_path}: {str(e)}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Simpan gambar ke file.
    
    Args:
        image: Gambar dalam format numpy array (RGB)
        output_path: Path file output
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Buat direktori output jika belum ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Konversi RGB ke BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Simpan gambar
        result = cv2.imwrite(output_path, image_bgr)
        
        if not result:
            print(f"⚠️ Gagal menyimpan gambar: {output_path}")
            return False
            
        return True
    except Exception as e:
        print(f"⚠️ Error saat menyimpan gambar ke {output_path}: {str(e)}")
        return False


def resize_image(image: np.ndarray, size: Tuple[int, int], keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Ubah ukuran gambar.
    
    Args:
        image: Gambar dalam format numpy array
        size: Ukuran target (width, height)
        keep_aspect_ratio: Jika True, pertahankan rasio aspek gambar
        
    Returns:
        Gambar yang sudah diubah ukurannya
    """
    width, height = size
    
    if keep_aspect_ratio:
        # Hitung rasio aspek
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Tentukan ukuran baru dengan mempertahankan rasio aspek
        if width / height > aspect_ratio:
            # Width yang membatasi
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            # Height yang membatasi
            new_width = width
            new_height = int(width / aspect_ratio)
            
        # Ubah ukuran gambar
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Buat canvas kosong dengan ukuran target
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Letakkan gambar yang sudah diubah ukurannya di tengah canvas
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return result
    else:
        # Ubah ukuran gambar tanpa mempertahankan rasio aspek
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_brightness_contrast(image: np.ndarray, brightness: float = 0, contrast: float = 0) -> np.ndarray:
    """
    Terapkan perubahan brightness dan contrast pada gambar.
    
    Args:
        image: Gambar dalam format numpy array
        brightness: Nilai brightness (-1.0 hingga 1.0)
        contrast: Nilai contrast (-1.0 hingga 1.0)
        
    Returns:
        Gambar yang sudah dimodifikasi
    """
    # Konversi nilai brightness dan contrast ke range yang digunakan oleh OpenCV
    brightness = int(brightness * 255)
    contrast = contrast * 127 + 127
    
    # Buat lookup table untuk brightness dan contrast
    brightness_table = np.array([max(0, min(255, i + brightness)) for i in range(256)], dtype=np.uint8)
    contrast_table = np.array([max(0, min(255, (i - 127) * contrast / 127 + 127)) for i in range(256)], dtype=np.uint8)
    
    # Terapkan brightness dan contrast
    result = cv2.LUT(image, brightness_table)
    result = cv2.LUT(result, contrast_table)
    
    return result


def apply_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Terapkan blur pada gambar.
    
    Args:
        image: Gambar dalam format numpy array
        kernel_size: Ukuran kernel blur (harus ganjil)
        
    Returns:
        Gambar yang sudah diblur
    """
    # Pastikan kernel_size ganjil
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Terapkan Gaussian blur
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_noise(image: np.ndarray, noise_type: str = 'gaussian', amount: float = 0.05) -> np.ndarray:
    """
    Terapkan noise pada gambar.
    
    Args:
        image: Gambar dalam format numpy array
        noise_type: Jenis noise ('gaussian', 'salt_pepper')
        amount: Jumlah noise (0.0 hingga 1.0)
        
    Returns:
        Gambar dengan noise
    """
    result = image.copy()
    
    if noise_type == 'gaussian':
        # Gaussian noise
        row, col, ch = image.shape
        mean = 0
        sigma = amount * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        result = cv2.add(result, gauss.astype(np.uint8))
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = min(amount, 0.5)  # Batasi amount untuk salt_pepper
        
        # Salt (white) noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        result[coords[0], coords[1], :] = 255
        
        # Pepper (black) noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        result[coords[0], coords[1], :] = 0
    
    return result


def apply_hsv_shift(image: np.ndarray, hue_shift: float = 0, saturation_shift: float = 0, value_shift: float = 0) -> np.ndarray:
    """
    Terapkan pergeseran HSV pada gambar.
    
    Args:
        image: Gambar dalam format numpy array (RGB)
        hue_shift: Pergeseran hue (-1.0 hingga 1.0)
        saturation_shift: Pergeseran saturation (-1.0 hingga 1.0)
        value_shift: Pergeseran value (-1.0 hingga 1.0)
        
    Returns:
        Gambar yang sudah dimodifikasi
    """
    # Konversi RGB ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Terapkan pergeseran
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180  # Hue (0-180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_shift), 0, 255)  # Saturation (0-255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + value_shift), 0, 255)  # Value (0-255)
    
    # Konversi kembali ke uint8
    hsv = hsv.astype(np.uint8)
    
    # Konversi HSV ke RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop gambar.
    
    Args:
        image: Gambar dalam format numpy array
        x: Koordinat x (dari kiri)
        y: Koordinat y (dari atas)
        width: Lebar crop
        height: Tinggi crop
        
    Returns:
        Gambar yang sudah di-crop
    """
    # Pastikan koordinat dan ukuran valid
    h, w = image.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = max(1, min(width, w - x))
    height = max(1, min(height, h - y))
    
    # Crop gambar
    return image[y:y+height, x:x+width].copy()


def get_image_stats(image: np.ndarray) -> Dict[str, Any]:
    """
    Dapatkan statistik gambar.
    
    Args:
        image: Gambar dalam format numpy array
        
    Returns:
        Dictionary berisi statistik gambar
    """
    # Hitung statistik dasar
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    # Hitung histogram untuk setiap channel
    histograms = []
    for i in range(channels):
        if channels == 1:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        else:
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.append(hist)
    
    # Hitung mean dan std untuk setiap channel
    means = []
    stds = []
    for i in range(channels):
        if channels == 1:
            means.append(np.mean(image))
            stds.append(np.std(image))
        else:
            means.append(np.mean(image[:, :, i]))
            stds.append(np.std(image[:, :, i]))
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'aspect_ratio': w / h,
        'size_bytes': image.nbytes,
        'means': means,
        'stds': stds,
        'histograms': histograms
    }
