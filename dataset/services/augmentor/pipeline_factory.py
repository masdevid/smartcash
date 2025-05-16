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
        bbox_format: str = 'yolo',
        include_normalize: bool = True,
        intensity: float = 0.5,
        **kwargs
    ) -> Any:
        """
        Buat pipeline augmentasi dengan konfigurasi yang disesuaikan.
        
        Args:
            augmentation_types: List jenis augmentasi yang akan digunakan
            img_size: Ukuran gambar output (width, height)
            bbox_format: Format bounding box ('yolo', 'coco', dll)
            include_normalize: Sertakan normalisasi
            intensity: Intensitas augmentasi (0.0-1.0)
            **kwargs: Parameter tambahan
            
        Returns:
            Pipeline augmentasi Albumentations
        """
        # Default ke semua jenis augmentasi jika tidak ditentukan
        if not augmentation_types:
            augmentation_types = ['combined']
            
        # Jika combined, gunakan semua jenis
        if 'combined' in augmentation_types:
            augmentation_types = ['flip', 'rotate', 'lighting', 'noise', 'hsv', 'weather']
            
        # Pastikan intensitas minimal 0.3 untuk memastikan perubahan terlihat
        intensity = max(0.3, intensity)
            
        # Validasi jenis augmentasi
        valid_types = {'flip', 'rotate', 'position', 'lighting', 'brightness', 'contrast', 
                      'noise', 'quality', 'hsv', 'weather', 'crop'}
        augmentation_types = [t for t in augmentation_types if t in valid_types]
        
        # Pastikan selalu ada minimal satu jenis augmentasi
        if not augmentation_types:
            augmentation_types = ['flip', 'rotate']  # Minimal dua jenis augmentasi
        
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
            
        # Buat pipeline dengan parameter bbox yang tepat
        if bbox_format:
            # Jika format bbox ditentukan, gunakan BboxParams
            pipeline = A.Compose(
                transforms,
                bbox_params=A.BboxParams(
                    format=bbox_format,
                    label_fields=['class_labels'],
                    min_visibility=0.3
                )
            )
            self.logger.info(f"🔍 Pipeline dibuat dengan dukungan bbox format: {bbox_format}")
        else:
            # Jika tidak ada format bbox, buat pipeline tanpa BboxParams
            pipeline = A.Compose(transforms)
            self.logger.info("🔍 Pipeline dibuat tanpa dukungan bbox")
        
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
        
        # Position params - pastikan nilai adalah float tunggal, bukan sequence
        fliplr_param = self.position_params.get('fliplr', 0.5)
        flipud_param = self.position_params.get('flipud', 0.1)
        translate_param = self.position_params.get('translate', 0.1)
        scale_param = self.position_params.get('scale', 0.1)
        degrees_param = self.position_params.get('degrees', 15)
        shear_param = self.position_params.get('shear', 5.0)
        
        # Jika parameter adalah tuple/list, gunakan nilai pertama
        if isinstance(fliplr_param, (list, tuple)):
            fliplr_param = fliplr_param[0] if len(fliplr_param) > 0 else 0.5
        if isinstance(flipud_param, (list, tuple)):
            flipud_param = flipud_param[0] if len(flipud_param) > 0 else 0.1
        if isinstance(translate_param, (list, tuple)):
            translate_param = translate_param[0] if len(translate_param) > 0 else 0.1
        if isinstance(scale_param, (list, tuple)):
            scale_param = scale_param[0] if len(scale_param) > 0 else 0.1
        if isinstance(degrees_param, (list, tuple)):
            degrees_param = degrees_param[0] if len(degrees_param) > 0 else 15
        if isinstance(shear_param, (list, tuple)):
            shear_param = shear_param[0] if len(shear_param) > 0 else 5.0
            
        # Kalikan dengan intensity
        params['fliplr'] = float(fliplr_param) * intensity
        params['flipud'] = float(flipud_param) * intensity
        params['translate'] = float(translate_param) * intensity
        params['scale'] = float(scale_param) * intensity
        params['degrees'] = float(degrees_param) * intensity
        params['shear'] = float(shear_param) * intensity
        
        # Lighting params - pastikan nilai adalah float tunggal, bukan sequence
        brightness_param = self.lighting_params.get('brightness', 0.2)
        contrast_param = self.lighting_params.get('contrast', 0.2)
        saturation_param = self.lighting_params.get('saturation', 0.2)
        hue_param = self.lighting_params.get('hue', 0.015)
        lightness_param = self.lighting_params.get('hsv_v', 0.4)
        
        # Jika parameter adalah tuple/list, gunakan nilai pertama
        if isinstance(brightness_param, (list, tuple)):
            brightness_param = brightness_param[0] if len(brightness_param) > 0 else 0.2
        if isinstance(contrast_param, (list, tuple)):
            contrast_param = contrast_param[0] if len(contrast_param) > 0 else 0.2
        if isinstance(saturation_param, (list, tuple)):
            saturation_param = saturation_param[0] if len(saturation_param) > 0 else 0.2
        if isinstance(hue_param, (list, tuple)):
            hue_param = hue_param[0] if len(hue_param) > 0 else 0.015
        if isinstance(lightness_param, (list, tuple)):
            lightness_param = lightness_param[0] if len(lightness_param) > 0 else 0.4
            
        # Kalikan dengan intensity
        params['brightness'] = float(brightness_param) * intensity
        params['contrast'] = float(contrast_param) * intensity
        params['saturation'] = float(saturation_param) * intensity
        params['hue'] = float(hue_param) * intensity
        params['lightness'] = float(lightness_param) * intensity
        
        # Noise params - pastikan nilai adalah float tunggal, bukan sequence
        gaussian_prob_param = self.noise_params.get('gaussian_prob', 0.1)
        gaussian_limit_param = self.noise_params.get('gaussian_limit', 10.0)
        blur_prob_param = self.noise_params.get('blur_prob', 0.1)
        blur_limit_param = self.noise_params.get('blur_limit', 3)
        jpeg_prob_param = self.noise_params.get('jpeg_prob', 0.1)
        
        # Jika parameter adalah tuple/list, gunakan nilai pertama
        if isinstance(gaussian_prob_param, (list, tuple)):
            gaussian_prob_param = gaussian_prob_param[0] if len(gaussian_prob_param) > 0 else 0.1
        if isinstance(gaussian_limit_param, (list, tuple)):
            gaussian_limit_param = gaussian_limit_param[0] if len(gaussian_limit_param) > 0 else 10.0
        if isinstance(blur_prob_param, (list, tuple)):
            blur_prob_param = blur_prob_param[0] if len(blur_prob_param) > 0 else 0.1
        if isinstance(blur_limit_param, (list, tuple)):
            blur_limit_param = blur_limit_param[0] if len(blur_limit_param) > 0 else 3
        if isinstance(jpeg_prob_param, (list, tuple)):
            jpeg_prob_param = jpeg_prob_param[0] if len(jpeg_prob_param) > 0 else 0.1
            
        # Kalikan dengan intensity
        params['gaussian_prob'] = float(gaussian_prob_param) * intensity
        params['gaussian_limit'] = float(gaussian_limit_param) * intensity
        params['blur_prob'] = float(blur_prob_param) * intensity
        params['blur_limit'] = int(float(blur_limit_param) * intensity) + 1
        params['jpeg_prob'] = float(jpeg_prob_param) * intensity
        
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
        Dapatkan transformasi posisi (rotate, scale, translate) dengan batasan scaling minimum.
        
        Args:
            params: Parameter untuk transformasi
            
        Returns:
            List transformasi
        """
        transforms = []
        
        # Gunakan Affine untuk semua transformasi posisi
        if any(params.get(key, 0) > 0 for key in ['translate', 'scale', 'degrees', 'shear']):
            # Batasi nilai scale untuk menghindari gambar menjadi terlalu kecil
            scale = params.get('scale', 0.1)
            # Ubah range scale dari simetris menjadi asimetris untuk mencegah pengecilan berlebihan
            # Misalnya dari (0.9, 1.1) menjadi (0.95, 1.1) 
            scale_range = (max(0.95, 1.0 - scale), 1.0 + scale)
            
            transforms.append(
                A.Affine(
                    scale=scale_range,  # Batasi pengecilan maksimal 5%
                    rotate=int(params.get('degrees', 15)),
                    translate_percent={
                        'x': params.get('translate', 0.1),
                        'y': params.get('translate', 0.1)
                    },
                    shear=params.get('shear', 5.0),
                    p=0.7 * params.get('p_factor', 1.0),
                    # Hapus mode dan cval yang menyebabkan warning
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
        
        # Gaussian Noise (menggunakan GaussNoise sebagai pengganti GaussianNoise)
        if params.get('gaussian_prob', 0) > 0:
            try:
                # Gunakan RandomBrightnessContrast sebagai alternatif yang lebih kompatibel
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=params.get('gaussian_prob', 0.1)
                    )
                )
                self.logger.info("✅ Menggunakan RandomBrightnessContrast sebagai pengganti noise")
            except AttributeError:
                # Fallback ke transformasi lain jika GaussNoise tidak tersedia
                self.logger.warning("⚠️ GaussNoise tidak tersedia, menggunakan RandomBrightnessContrast sebagai alternatif")
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
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
            
        # JPEG Compression - gunakan parameter yang kompatibel dengan berbagai versi
        if params.get('jpeg_prob', 0) > 0:
            quality_range = (int(params.get('jpeg_quality', (80, 100))[0]), int(params.get('jpeg_quality', (80, 100))[1]))
            try:
                # Coba dengan parameter sederhana yang lebih kompatibel
                transforms.append(
                    A.ImageCompression(
                        quality_lower=quality_range[0],
                        quality_upper=quality_range[1],
                        p=params.get('jpeg_prob', 0.1)
                    )
                )
                self.logger.info(f"✅ Menggunakan ImageCompression dengan quality range {quality_range}")
            except Exception as e:
                # Jika masih error, gunakan transformasi alternatif
                self.logger.warning(f"⚠️ Error pada ImageCompression: {str(e)}. Menggunakan Downscale sebagai alternatif")
                transforms.append(
                    A.Downscale(
                        scale_min=0.8,
                        scale_max=0.9,
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
            
        # Fog - gunakan parameter yang kompatibel
        if params.get('fog_prob', 0) > 0:
            try:
                # Coba dengan parameter yang disederhanakan
                transforms.append(
                    A.RandomFog(
                        fog_coef=0.2,  # Nilai tunggal lebih kompatibel
                        p=params.get('fog_prob', 0.1)
                    )
                )
                self.logger.info(f"✅ Menggunakan RandomFog dengan parameter yang disederhanakan")
            except Exception as e:
                self.logger.warning(f"⚠️ Error pada RandomFog: {str(e)}. Menggunakan Blur sebagai alternatif")
                transforms.append(
                    A.Blur(
                        blur_limit=3,
                        p=params.get('fog_prob', 0.1)
                    )
                )
            
        # Snow - gunakan parameter yang kompatibel
        if params.get('snow_prob', 0) > 0:
            try:
                # Coba dengan parameter yang disederhanakan
                transforms.append(
                    A.RandomSnow(
                        snow_point_lower=0.1,
                        snow_point_upper=0.3,
                        brightness_coeff=0.9,
                        p=params.get('snow_prob', 0.05)
                    )
                )
                self.logger.info(f"✅ Menggunakan RandomSnow dengan parameter yang disederhanakan")
            except Exception as e:
                self.logger.warning(f"⚠️ Error pada RandomSnow: {str(e)}. Menggunakan RandomBrightness sebagai alternatif")
                transforms.append(
                    A.RandomBrightness(
                        limit=0.1,
                        p=params.get('snow_prob', 0.05)
                    )
                )
            
        # Sun Flare - gunakan parameter yang kompatibel
        if params.get('sun_flare_prob', 0) > 0:
            try:
                # Coba dengan parameter yang disederhanakan
                transforms.append(
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        src_radius=100,
                        src_color=(255, 255, 255),
                        p=params.get('sun_flare_prob', 0.05)
                    )
                )
                self.logger.info(f"✅ Menggunakan RandomSunFlare dengan parameter yang disederhanakan")
            except Exception as e:
                self.logger.warning(f"⚠️ Error pada RandomSunFlare: {str(e)}. Menggunakan RandomBrightnessContrast sebagai alternatif")
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.1,
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