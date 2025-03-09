# File: smartcash/handlers/detection/detection_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama untuk deteksi objek mata uang Rupiah sebagai facade

import os
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
import torch

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import SmartCashError, ModelError, DataError
from smartcash.handlers.detection.core.detector import DefaultDetector
from smartcash.handlers.detection.core.preprocessor import ImagePreprocessor
from smartcash.handlers.detection.core.postprocessor import DetectionPostprocessor
from smartcash.handlers.detection.strategies.image_strategy import ImageDetectionStrategy
from smartcash.handlers.detection.strategies.directory_strategy import DirectoryDetectionStrategy
from smartcash.handlers.detection.output.output_manager import OutputManager
from smartcash.handlers.detection.integration.model_adapter import ModelAdapter
from smartcash.handlers.detection.integration.visualizer_adapter import VisualizerAdapter

class DetectionManager:
    """
    Manager utama deteksi yang berfungsi sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability dengan
    menyediakan antarmuka sederhana untuk deteksi objek mata uang.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger = None,
        colab_mode: Optional[bool] = None
    ):
        """
        Inisialisasi detection manager.
        
        Args:
            config: Konfigurasi deteksi
            logger: Logger kustom (opsional)
            colab_mode: Mode Google Colab (opsional, dideteksi otomatis jika None)
        """
        # Setup logger
        self.logger = logger or get_logger("detection_manager")
        self.config = config
        
        # Deteksi mode colab otomatis jika tidak diberikan
        self.colab_mode = self._is_running_in_colab() if colab_mode is None else colab_mode
        
        # Parameter dari konfigurasi
        inference_config = config.get('inference', {})
        self.conf_threshold = inference_config.get('conf_threshold', 0.25)
        self.iou_threshold = inference_config.get('iou_threshold', 0.45)
        self.output_dir = Path(inference_config.get('output_dir', 'results/detections'))
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponen akan di-lazy-load saat diperlukan
        self._components = {}
        
        self.logger.info(f"🔍 DetectionManager diinisialisasi dengan threshold {self.conf_threshold}")
        
    def detect(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari berbagai sumber (single image atau direktori).
        
        Args:
            source: Path ke gambar atau direktori
            **kwargs: Parameter tambahan untuk deteksi
                - conf_threshold: Threshold konfidiensi
                - iou_threshold: Threshold IOU untuk NMS
                - output_dir: Direktori output custom
                - visualize: Flag untuk visualisasi (default: True)
                
        Returns:
            Hasil deteksi
        """
        source_path = Path(source)
        
        # Validasi
        if not source_path.exists():
            raise DataError(f"🚫 Source tidak ditemukan: {source_path}")
        
        # Pilih strategi berdasarkan tipe source
        if source_path.is_file():
            return self.detect_image(source_path, **kwargs)
        elif source_path.is_dir():
            return self.detect_directory(source_path, **kwargs)
        else:
            raise DataError(f"🚫 Source tidak didukung: {source_path}")
    
    def detect_image(
        self,
        image_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari gambar tunggal.
        
        Args:
            image_path: Path ke gambar
            **kwargs: Parameter tambahan untuk deteksi
                
        Returns:
            Hasil deteksi
        """
        # Dapatkan parameter
        conf_threshold = kwargs.get('conf_threshold', self.conf_threshold)
        
        # Dapatkan atau inisialisasi komponen
        strategy = self._get_image_detection_strategy()
        
        self.logger.info(f"🔍 Mendeteksi objek dari gambar: {image_path}")
        
        # Jalankan deteksi
        results = strategy.detect(image_path, conf_threshold=conf_threshold, **kwargs)
        
        self.logger.info(f"✅ Deteksi selesai: {len(results.get('detections', []))} objek terdeteksi")
        
        return results
    
    def detect_directory(
        self,
        dir_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari direktori berisi gambar.
        
        Args:
            dir_path: Path ke direktori
            **kwargs: Parameter tambahan untuk deteksi
                - batch_size: Ukuran batch (default: 1)
                - recursive: Cari gambar secara rekursif (default: False)
                
        Returns:
            Hasil deteksi
        """
        # Dapatkan parameter
        batch_size = kwargs.get('batch_size', 1)
        recursive = kwargs.get('recursive', False)
        
        # Dapatkan atau inisialisasi komponen
        strategy = self._get_directory_detection_strategy()
        
        self.logger.info(f"🔍 Mendeteksi objek dari direktori: {dir_path}")
        
        # Jalankan deteksi
        results = strategy.detect(
            dir_path, 
            batch_size=batch_size, 
            recursive=recursive,
            **kwargs
        )
        
        self.logger.info(f"✅ Deteksi selesai: {results.get('processed_images', 0)} gambar diproses")
        
        return results
    
    def _get_model_adapter(self) -> ModelAdapter:
        """Dapatkan atau inisialisasi model adapter."""
        if 'model_adapter' not in self._components:
            self._components['model_adapter'] = ModelAdapter(
                self.config,
                logger=self.logger,
                colab_mode=self.colab_mode
            )
        return self._components['model_adapter']
    
    def _get_detector(self) -> DefaultDetector:
        """Dapatkan atau inisialisasi detector."""
        if 'detector' not in self._components:
            model_adapter = self._get_model_adapter()
            model = model_adapter.get_model()
            
            self._components['detector'] = DefaultDetector(
                model=model,
                config=self.config,
                logger=self.logger
            )
        return self._components['detector']
    
    def _get_preprocessor(self) -> ImagePreprocessor:
        """Dapatkan atau inisialisasi preprocessor."""
        if 'preprocessor' not in self._components:
            img_size = self.config.get('model', {}).get('img_size', (640, 640))
            self._components['preprocessor'] = ImagePreprocessor(
                config=self.config,
                img_size=img_size
            )
        return self._components['preprocessor']
    
    def _get_postprocessor(self) -> DetectionPostprocessor:
        """Dapatkan atau inisialisasi postprocessor."""
        if 'postprocessor' not in self._components:
            self._components['postprocessor'] = DetectionPostprocessor(
                config=self.config,
                logger=self.logger
            )
        return self._components['postprocessor']
    
    def _get_output_manager(self) -> OutputManager:
        """Dapatkan atau inisialisasi output manager."""
        if 'output_manager' not in self._components:
            visualizer_adapter = VisualizerAdapter(self.config, self.logger)
            
            self._components['output_manager'] = OutputManager(
                config=self.config,
                output_dir=self.output_dir,
                visualizer=visualizer_adapter,
                logger=self.logger,
                colab_mode=self.colab_mode
            )
        return self._components['output_manager']
    
    def _get_image_detection_strategy(self) -> ImageDetectionStrategy:
        """Dapatkan atau inisialisasi strategy deteksi gambar."""
        if 'image_strategy' not in self._components:
            self._components['image_strategy'] = ImageDetectionStrategy(
                config=self.config,
                detector=self._get_detector(),
                preprocessor=self._get_preprocessor(),
                postprocessor=self._get_postprocessor(),
                output_manager=self._get_output_manager(),
                logger=self.logger
            )
        return self._components['image_strategy']
    
    def _get_directory_detection_strategy(self) -> DirectoryDetectionStrategy:
        """Dapatkan atau inisialisasi strategy deteksi direktori."""
        if 'directory_strategy' not in self._components:
            self._components['directory_strategy'] = DirectoryDetectionStrategy(
                config=self.config,
                detector=self._get_detector(),
                preprocessor=self._get_preprocessor(),
                postprocessor=self._get_postprocessor(),
                output_manager=self._get_output_manager(),
                logger=self.logger
            )
        return self._components['directory_strategy']
    
    def _is_running_in_colab(self) -> bool:
        """Deteksi apakah kode berjalan dalam Google Colab."""
        try:
            import google.colab
            self.logger.info("🌐 Terdeteksi berjalan di Google Colab")
            return True
        except ImportError:
            return False