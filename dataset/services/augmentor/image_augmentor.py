"""
File: smartcash/dataset/services/augmentor/image_augmentor.py
Deskripsi: Komponen untuk augmentasi gambar dengan metode advanced
"""

import cv2
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any

from smartcash.common.logger import get_logger


class ImageAugmentor:
    """
    Komponen untuk augmentasi khusus gambar dengan metode yang tidak tersedia di Albumentations.
    Seperti cutmix, mixup, mosaic, dan cutout.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi ImageAugmentor.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger()
        
        # Setup parameter augmentasi
        aug_config = self.config.get('augmentation', {}).get('image', {})
        self.cutout_size = aug_config.get('cutout_size', 0.1)
        self.mixup_alpha = aug_config.get('mixup_alpha', 0.5)
        self.blend_alpha = aug_config.get('blend_alpha', 0.5)
        self.cutmix_ratio = aug_config.get('cutmix_ratio', 0.5)
        
        self.logger.info(f"🖼️ ImageAugmentor diinisialisasi dengan parameter augmentasi custom")
    
    def cutout(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Terapkan augmentasi cutout (hapus bagian gambar dan ganti dengan noise/black).
        
        Args:
            image: Gambar input dalam format NumPy array
            p: Probabilitas penerapan
            
        Returns:
            Gambar yang sudah diaugmentasi
        """
        if random.random() > p:
            return image
            
        # Clone dulu untuk hindari modifikasi pada input asli
        augmented = image.copy()
        h, w = image.shape[:2]
        
        # Random ukuran cutout (5-20% dari gambar)
        cutout_size = random.uniform(0.05, 0.2)
        
        # Hitung ukuran kotak cutout
        cutout_width = int(w * cutout_size)
        cutout_height = int(h * cutout_size)
        
        # Posisi random
        x = random.randint(0, w - cutout_width)
        y = random.randint(0, h - cutout_height)
        
        # Terapkan cutout (fill dengan black)
        augmented[y:y+cutout_height, x:x+cutout_width, :] = 0
        
        return augmented
    
    def cutmix(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray, 
        ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Terapkan augmentasi cutmix (gabungkan bagian dari dua gambar).
        
        Args:
            image1: Gambar pertama
            image2: Gambar kedua (harus sama ukurannya dengan image1)
            ratio: Rasio cutmix (jika None, akan diambil dari parameter kelas)
            
        Returns:
            Gambar gabungan
        """
        # Validasi ukuran
        if image1.shape != image2.shape:
            self.logger.warning("⚠️ Cutmix memerlukan dua gambar dengan ukuran sama")
            return image1
            
        # Gunakan ratio default jika tidak disediakan
        if ratio is None:
            ratio = self.cutmix_ratio
            
        # Buat hasil
        result = image1.copy()
        h, w = image1.shape[:2]
        
        # Hitung ukuran kotak cutmix
        cut_width = int(w * ratio)
        cut_height = int(h * ratio)
        
        # Posisi random
        cx = random.randint(0, w - cut_width)
        cy = random.randint(0, h - cut_height)
        
        # Terapkan cutmix
        result[cy:cy+cut_height, cx:cx+cut_width, :] = image2[cy:cy+cut_height, cx:cx+cut_width, :]
        
        return result
    
    def mixup(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray, 
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Terapkan augmentasi mixup (blend dua gambar dengan alpha).
        
        Args:
            image1: Gambar pertama
            image2: Gambar kedua (harus sama ukurannya dengan image1)
            alpha: Parameter mixup (jika None, akan diambil dari parameter kelas)
            
        Returns:
            Gambar blend
        """
        # Validasi ukuran
        if image1.shape != image2.shape:
            self.logger.warning("⚠️ Mixup memerlukan dua gambar dengan ukuran sama")
            return image1
            
        # Gunakan alpha default jika tidak disediakan
        if alpha is None:
            alpha = self.mixup_alpha
            
        # Lakukan blending
        result = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
        
        return result
    
    def mosaic(self, images: List[np.ndarray], grid_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Terapkan augmentasi mosaic (gabungkan beberapa gambar dalam grid).
        
        Args:
            images: List gambar (minimal 4 untuk grid 2x2)
            grid_size: Ukuran grid (default 2x2)
            
        Returns:
            Gambar mosaic
        """
        rows, cols = grid_size
        required_images = rows * cols
        
        # Cek jumlah gambar
        if len(images) < required_images:
            self.logger.warning(f"⚠️ Mosaic memerlukan minimal {required_images} gambar untuk grid {rows}x{cols}")
            if len(images) > 0:
                return images[0]
            return None
            
        # Pilih sebanyak yang dibutuhkan
        selected_images = images[:required_images]
        
        # Normalisasi ukuran
        h, w = selected_images[0].shape[:2]
        for i in range(1, len(selected_images)):
            selected_images[i] = cv2.resize(selected_images[i], (w, h))
        
        # Buat gambar hasil
        result = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        # Gabungkan gambar
        for idx, img in enumerate(selected_images):
            r = idx // cols
            c = idx % cols
            
            result[r*h:(r+1)*h, c*w:(c+1)*w, :] = img
            
        return result
    
    def blend_with_gaussian_noise(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Tambahkan Gaussian noise ke gambar.
        
        Args:
            image: Gambar input
            p: Probabilitas penerapan
            
        Returns:
            Gambar dengan noise
        """
        if random.random() > p:
            return image
            
        # Clone dulu
        augmented = image.copy()
        
        # Random noise level
        noise_level = random.uniform(5, 20)
        
        # Generate noise dengan ukuran sama dengan gambar
        noise = np.random.normal(0, noise_level, augmented.shape).astype(np.uint8)
        
        # Tambahkan noise ke gambar, clipping untuk hindari overflow
        augmented = np.clip(augmented.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        return augmented
    
    def random_erase(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Terapkan random erase (seperti cutout, tapi bisa multiple region).
        
        Args:
            image: Gambar input
            p: Probabilitas penerapan
            
        Returns:
            Gambar yang sudah diaugmentasi
        """
        if random.random() > p:
            return image
            
        # Clone dulu
        augmented = image.copy()
        h, w = image.shape[:2]
        
        # Random jumlah area yang dihapus (1-3)
        num_areas = random.randint(1, 3)
        
        for _ in range(num_areas):
            # Random ukuran area (2-10% dari gambar)
            area_size = random.uniform(0.02, 0.1)
            
            # Hitung ukuran kotak
            area_width = int(w * area_size)
            area_height = int(h * area_size)
            
            # Posisi random
            x = random.randint(0, w - area_width)
            y = random.randint(0, h - area_height)
            
            # Random color untuk fill
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Terapkan erase
            augmented[y:y+area_height, x:x+area_width, :] = color
            
        return augmented
    
    def adjust_hue(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Adjust hue dari gambar.
        
        Args:
            image: Gambar input
            p: Probabilitas penerapan
            
        Returns:
            Gambar yang sudah diaugmentasi
        """
        if random.random() > p:
            return image
            
        # Clone dulu
        augmented = image.copy()
        
        # Konversi ke HSV
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV)
        
        # Random hue shift (-20 to 20)
        hue_shift = random.uniform(-20, 20)
        
        # Apply hue shift
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Konversi kembali ke RGB
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return augmented
    
    def simulate_shadow(self, image: np.ndarray, p: float = 0.3) -> np.ndarray:
        """
        Simulasikan bayangan pada gambar.
        
        Args:
            image: Gambar input
            p: Probabilitas penerapan
            
        Returns:
            Gambar dengan bayangan
        """
        if random.random() > p:
            return image
            
        # Clone dulu
        augmented = image.copy()
        h, w = augmented.shape[:2]
        
        # Buat mask untuk area bayangan
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Buat bayangan dengan bentuk poligon acak
        num_points = random.randint(3, 6)
        points = []
        
        for _ in range(num_points):
            points.append((random.randint(0, w), random.randint(0, h)))
            
        # Isi poligon pada mask
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # Konversi gambar ke HSV
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV)
        
        # Gelaptkan area dengan mask (kurangi brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * 0.7 * (mask / 255) + hsv[:, :, 2] * (1 - (mask / 255))
        
        # Konversi kembali ke RGB
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return augmented