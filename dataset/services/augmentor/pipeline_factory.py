"""
File: smartcash/dataset/services/augmentor/pipeline_factory.py
Deskripsi: Factory untuk membuat pipeline augmentasi dengan konfigurasi yang disesuaikan dan validasi parameter
"""

import cv2
import numpy as np
import albumentations as A
from typing import Dict, List, Optional, Union, Any

from smartcash.common.logger import get_logger


class AugmentationPipelineFactory:
    """Factory untuk membuat pipeline augmentasi dengan berbagai konfigurasi."""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi AugmentationPipelineFactory.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("pipeline_factory")
        
        # Ekstrak parameter dari config
        self.aug_config = self.config.get('augmentation', {})
        self.position_params = self.aug_config.get('position', {})
        self.lighting_params = self.aug_config.get('lighting', {})
        self.noise_params = self.aug_config.get('noise', {})
        
        # Parameter default
        self._setup_default_params()
        
        self.logger.info("🎨 AugmentationPipelineFactory siap membuat pipeline augmentasi")
    
    def _setup_default_params(self):
        """Setup parameter default jika tidak ada di config."""
        # Position defaults
        if not self.position_params:
            self.position_params = {
                'fliplr': 0.5,
                'flipud': 0.1,
                'translate': 0.1,
                'scale': 0.1,
                'degrees': 15,
                'shear': 5.0
            }
            
        # Lighting defaults
        if not self.lighting_params:
            self.lighting_params = {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4
            }
            
        # Noise defaults
        if not self.noise_params:
            self.noise_params = {
                'gaussian_prob': 0.1,
                'gaussian_limit': 10.0,
                'blur_prob': 0.1,
                'blur_limit': 3,
                'jpeg_prob': 0.1,
                'jpeg_quality': (80, 100)
            }
    
    def create_pipeline(
        self, 
        augmentation_types: List[str] = None,
        img_size: tuple = (640, 640),
        include_normalize: bool = True,
        intensity: float = 1.0,
        bbox_format: str = 'yolo',
        **kwargs
    ) -> A.Compose:
        """
        Buat pipeline augmentasi berdasarkan jenis yang dipilih.
        
        Args:
            augmentation_types: Jenis augmentasi yang akan diterapkan
            img_size: Ukuran target gambar
            include_normalize: Apakah menyertakan normalisasi di akhir
            intensity: Intensitas augmentasi (0.0-1.0)
            bbox_format: Format bbox ('yolo', 'pascal_voc', 'albumentations')
            **kwargs: Parameter kustom untuk override
            
        Returns:
            Pipeline augmentasi Albumentations
        """
        # Default augmentation jika tidak disediakan
        if not augmentation_types:
            augmentation_types = ['flip', 'rotate', 'brightness', 'contrast']
            
        # Buat list transformasi
        transforms = []
        
        # Dapatkan dimensi gambar
        height, width = int(img_size[1]), int(img_size[0])
        
        # Gunakan A.Resize sebagai pengganti RandomResizedCrop untuk menghindari masalah
        transforms.append(A.Resize(height=height, width=width, p=1.0))
        
        # Random crop jika diinginkan
        if 'crop' in augmentation_types:
            # Tambahkan transform untuk cropping
            scale = (0.9, 1.0)
            transforms.append(A.RandomCrop(height=int(height*0.9), width=int(width*0.9), p=0.5))
            # Tambahkan resize lagi untuk memastikan ukuran konsisten
            transforms.append(A.Resize(height=height, width=width, p=1.0))
        
        # Sesuaikan parameter berdasarkan intensitas
        params = self._adjust_params_by_intensity(intensity, **kwargs)
        
        # Tambahkan transformasi berdasarkan jenis augmentasi
        if 'flip' in augmentation_types:
            transforms.extend(self._get_flip_transforms(params))
            
        if 'rotate' in augmentation_types or 'position' in augmentation_types:
            transforms.extend(self._get_position_transforms(params))
            
        if 'lighting' in augmentation_types or 'brightness' in augmentation_types or 'contrast' in augmentation_types:
            transforms.extend(self._get_lighting_transforms(params))
            
        if 'noise' in augmentation_types or 'quality' in augmentation_types:
            transforms.extend(self._get_noise_transforms(params))
            
        if 'weather' in augmentation_types:
            transforms.extend(self._get_weather_transforms(params))
            
        if 'hsv' in augmentation_types:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(params['hue'] * 360),
                    sat_shift_limit=int(params['saturation'] * 255),
                    val_shift_limit=int(params['lightness'] * 255),
                    p=0.5 * params['p_factor']
                )
            )
        
        # Tambahkan normalisasi jika diminta
        if include_normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    p=1.0
                )
            )
            
        # Buat pipeline dengan parameter bbox
        pipeline = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format=bbox_format,
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )
        
        self.logger.info(
            f"🎨 Pipeline augmentasi dibuat: {len(augmentation_types)} jenis transformasi, "
            f"intensitas {intensity:.1f}, ukuran gambar {img_size}"
        )
        
        return pipeline
    
    def _adjust_params_by_intensity(self, intensity: float, **kwargs) -> Dict[str, Any]:
        """
        Sesuaikan parameter berdasarkan intensitas.
        
        Args:
            intensity: Intensitas augmentasi (0.0-1.0)
            **kwargs: Parameter kustom untuk override
            
        Returns:
            Parameter yang disesuaikan
        """
        intensity = max(0.0, min(1.0, intensity))
        params = {}
        
        # Parameter probabilitas
        params['p_factor'] = intensity
        
        # Position params
        params['fliplr'] = self.position_params.get('fliplr', 0.5) * intensity
        params['flipud'] = self.position_params.get('flipud', 0.1) * intensity
        params['translate'] = self.position_params.get('translate', 0.1) * intensity
        params['scale'] = self.position_params.get('scale', 0.1) * intensity
        params['degrees'] = self.position_params.get('degrees', 15) * intensity
        params['shear'] = self.position_params.get('shear', 5.0) * intensity
        
        # Lighting params
        params['brightness'] = self.lighting_params.get('brightness', 0.2) * intensity
        params['contrast'] = self.lighting_params.get('contrast', 0.2) * intensity
        params['saturation'] = self.lighting_params.get('saturation', 0.2) * intensity
        params['hue'] = self.lighting_params.get('hue', 0.015) * intensity
        params['lightness'] = self.lighting_params.get('hsv_v', 0.4) * intensity
        
        # Noise params
        params['gaussian_prob'] = self.noise_params.get('gaussian_prob', 0.1) * intensity
        params['gaussian_limit'] = self.noise_params.get('gaussian_limit', 10.0) * intensity
        params['blur_prob'] = self.noise_params.get('blur_prob', 0.1) * intensity
        params['blur_limit'] = int(self.noise_params.get('blur_limit', 3) * intensity) + 1
        params['jpeg_prob'] = self.noise_params.get('jpeg_prob', 0.1) * intensity
        
        # Weather params
        params['rain_prob'] = 0.1 * intensity
        params['fog_prob'] = 0.1 * intensity
        params['snow_prob'] = 0.05 * intensity
        params['sun_flare_prob'] = 0.05 * intensity
        
        # Override dengan kwargs yang disediakan
        params.update(kwargs)
        
        return params
    
    def _get_flip_transforms(self, params: Dict[str, Any]) -> List[A.BasicTransform]:
        """
        Dapatkan transformasi flip.
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        if params.get('fliplr', 0) > 0:
            transforms.append(
                A.HorizontalFlip(p=params['fliplr'])
            )
            
        if params.get('flipud', 0) > 0:
            transforms.append(
                A.VerticalFlip(p=params['flipud'])
            )
            
        return transforms
    
    def _get_position_transforms(self, params: Dict[str, Any]) -> List[A.BasicTransform]:
        """
        Dapatkan transformasi posisi (rotate, scale, translate).
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        # ShiftScaleRotate (kombinasi scale, rotate, translate)
        if any(params.get(key, 0) > 0 for key in ['translate', 'scale', 'degrees']):
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=params.get('translate', 0.1),
                    scale_limit=params.get('scale', 0.1),
                    rotate_limit=int(params.get('degrees', 15)),
                    p=0.7 * params.get('p_factor', 1.0),
                    border_mode=cv2.BORDER_CONSTANT
                )
            )
        
        # Affine untuk transformasi yang lebih kompleks
        if params.get('shear', 0) > 0:
            transforms.append(
                A.Affine(
                    shear=params.get('shear', 5.0),
                    p=0.3 * params.get('p_factor', 1.0),
                    cval=0,
                    mode=cv2.BORDER_CONSTANT
                )
            )
            
        return transforms
    
    def _get_lighting_transforms(self, params: Dict[str, Any]) -> List[A.BasicTransform]:
        """
        Dapatkan transformasi pencahayaan (brightness, contrast, color).
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        # RandomBrightnessContrast
        if params.get('brightness', 0) > 0 or params.get('contrast', 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=params.get('brightness', 0.2),
                    contrast_limit=params.get('contrast', 0.2),
                    p=0.8 * params.get('p_factor', 1.0)
                )
            )
            
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if params.get('contrast', 0) > 0.1:
            transforms.append(
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=0.2 * params.get('p_factor', 1.0)
                )
            )
            
        # RGB Shift
        if params.get('saturation', 0) > 0:
            transforms.append(
                A.RGBShift(
                    r_shift_limit=int(10 * params.get('saturation', 0.2)),
                    g_shift_limit=int(10 * params.get('saturation', 0.2)),
                    b_shift_limit=int(10 * params.get('saturation', 0.2)),
                    p=0.3 * params.get('p_factor', 1.0)
                )
            )
            
        return transforms
    
    def _get_noise_transforms(self, params: Dict[str, Any]) -> List[A.BasicTransform]:
        """
        Dapatkan transformasi noise dan kualitas gambar.
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        # GaussianNoise
        if params.get('gaussian_prob', 0) > 0:
            transforms.append(
                A.GaussianNoise(
                    var_limit=(5.0, params.get('gaussian_limit', 10.0)),
                    p=params.get('gaussian_prob', 0.1)
                )
            )
            
        # Blur
        if params.get('blur_prob', 0) > 0:
            blur_limit = max(1, int(params.get('blur_limit', 3)))
            transforms.append(
                A.Blur(
                    blur_limit=blur_limit,
                    p=params.get('blur_prob', 0.1)
                )
            )
            
        # JPEG Compression
        if params.get('jpeg_prob', 0) > 0:
            quality_lower = int(params.get('jpeg_quality', (80, 100))[0])
            quality_upper = int(params.get('jpeg_quality', (80, 100))[1])
            transforms.append(
                A.ImageCompression(
                    quality_lower=quality_lower,
                    quality_upper=quality_upper,
                    p=params.get('jpeg_prob', 0.1)
                )
            )
            
        return transforms
    
    def _get_weather_transforms(self, params: Dict[str, Any]) -> List[A.BasicTransform]:
        """
        Dapatkan transformasi simulasi cuaca.
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        # Rain
        if params.get('rain_prob', 0) > 0:
            transforms.append(
                A.RandomRain(
                    brightness_coefficient=0.9,
                    drop_width=1,
                    blur_value=3,
                    p=params.get('rain_prob', 0.1)
                )
            )
            
        # Fog
        if params.get('fog_prob', 0) > 0:
            transforms.append(
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.1,
                    p=params.get('fog_prob', 0.1)
                )
            )
            
        # Snow
        if params.get('snow_prob', 0) > 0:
            transforms.append(
                A.RandomSnow(
                    snow_point_lower=0.1,
                    snow_point_upper=0.3,
                    brightness_coeff=0.9,
                    p=params.get('snow_prob', 0.05)
                )
            )
            
        # Sun Flare
        if params.get('sun_flare_prob', 0) > 0:
            transforms.append(
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=3,
                    src_radius=100,
                    src_color=(255, 255, 255),
                    p=params.get('sun_flare_prob', 0.05)
                )
            )
            
        return transforms
    
    def create_train_pipeline(self, img_size: tuple = (640, 640)) -> A.Compose:
        """
        Buat pipeline untuk training (dengan augmentasi standar).
        
        Args:
            img_size: Ukuran target gambar
            
        Returns:
            Pipeline augmentasi untuk training
        """
        return self.create_pipeline(
            augmentation_types=['flip', 'rotate', 'brightness', 'contrast', 'hsv'],
            img_size=img_size,
            include_normalize=True,
            intensity=1.0
        )
    
    def create_validation_pipeline(self, img_size: tuple = (640, 640)) -> A.Compose:
        """
        Buat pipeline untuk validasi (tanpa augmentasi).
        
        Args:
            img_size: Ukuran target gambar
            
        Returns:
            Pipeline untuk validasi
        """
        return A.Compose(
            [
                A.Resize(height=int(img_size[1]), width=int(img_size[0]), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
    
    def create_light_augmentation_pipeline(self, img_size: tuple = (640, 640)) -> A.Compose:
        """
        Buat pipeline dengan augmentasi ringan (untuk dataset kecil).
        
        Args:
            img_size: Ukuran target gambar
            
        Returns:
            Pipeline augmentasi ringan
        """
        return self.create_pipeline(
            augmentation_types=['flip', 'brightness', 'contrast'],
            img_size=img_size,
            include_normalize=True,
            intensity=0.7
        )
    
    def create_heavy_augmentation_pipeline(self, img_size: tuple = (640, 640)) -> A.Compose:
        """
        Buat pipeline dengan augmentasi berat (untuk dataset sangat kecil atau imbalanced).
        
        Args:
            img_size: Ukuran target gambar
            
        Returns:
            Pipeline augmentasi berat
        """
        return self.create_pipeline(
            augmentation_types=['flip', 'rotate', 'brightness', 'contrast', 'hsv', 'noise', 'weather'],
            img_size=img_size,
            include_normalize=True,
            intensity=1.0
        )