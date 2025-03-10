# File: smartcash/handlers/detection/detection_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama untuk deteksi mata uang Rupiah, berperan sebagai facade untuk komponen deteksi

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import torch

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import ModelError, DataError

from smartcash.handlers.detection.core import (
    DefaultDetector, ImagePreprocessor, DetectionPostprocessor
)
from smartcash.handlers.detection.integration import (
    ModelAdapter, VisualizerAdapter
)
from smartcash.handlers.detection.output import OutputManager
from smartcash.handlers.detection.strategies import (
    ImageDetectionStrategy, DirectoryDetectionStrategy
)
from smartcash.handlers.detection.pipeline import (
    DetectionPipeline, BatchDetectionPipeline
)


class DetectionManager:
    """
    Manager utama untuk deteksi mata uang Rupiah, berperan sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        logger = None,
        colab_mode: Optional[bool] = None
    ):
        """
        Inisialisasi detection manager.
        
        Args:
            config: Konfigurasi (opsional, gunakan ConfigManager.get_config() jika None)
            logger: Logger kustom (opsional)
            colab_mode: Flag untuk mode Google Colab (opsional, deteksi otomatis jika None)
        """
        # Setup logger
        self.logger = logger or get_logger("detection_manager")
        
        # Deteksi otomatis colab jika tidak diberikan
        self.colab_mode = self._is_running_in_colab() if colab_mode is None else colab_mode
        
        # Load konfigurasi dari ConfigManager jika tidak diberikan
        self.config = config or self._get_default_config()
        
        # Inisialisasi komponen lazily
        self._model_adapter = None
        self._model = None
        self._preprocessor = None
        self._detector = None
        self._postprocessor = None
        self._visualizer_adapter = None
        self._output_manager = None
        self._image_strategy = None
        self._directory_strategy = None
        self._detection_pipeline = None
        self._batch_pipeline = None
        
        self.logger.info(f"🚀 DetectionManager diinisialisasi {'(mode: Colab)' if self.colab_mode else ''}")
    
    def detect(
        self,
        source: Union[str, Path],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        use_pipeline: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari berbagai sumber (otomatis memilih strategi).
        
        Args:
            source: Path ke gambar atau direktori
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            use_pipeline: Gunakan pipeline vs strategi (default: True)
            **kwargs: Parameter tambahan untuk strategi
            
        Returns:
            Dictionary hasil deteksi
        """
        # Konversi ke Path untuk memudahkan validasi
        if isinstance(source, str):
            source = Path(source)
            
        # Validasi source
        if not source.exists():
            raise DataError(f"Source tidak ditemukan: {source}")
            
        # Tentukan berdasarkan tipe source
        if source.is_dir():
            return self.detect_directory(source, conf_threshold, visualize, use_pipeline=use_pipeline, **kwargs)
        else:
            return self.detect_image(source, conf_threshold, visualize, use_pipeline=use_pipeline, **kwargs)
    
    def detect_image(
        self,
        image_path: Union[str, Path],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        use_pipeline: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari gambar.
        
        Args:
            image_path: Path ke gambar
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            use_pipeline: Gunakan pipeline baru (default: False)
            **kwargs: Parameter tambahan untuk strategi
            
        Returns:
            Dictionary hasil deteksi
        """
        # Jika menggunakan pipeline baru
        if use_pipeline:
            pipeline = self._get_detection_pipeline()
            return pipeline.run(
                source=image_path,
                conf_threshold=conf_threshold,
                visualize=visualize,
                **kwargs
            )
        else:
            # Dapatkan strategi image detection
            strategy = self._get_image_strategy()
            
            # Jalankan deteksi
            return strategy.detect(
                source=image_path,
                conf_threshold=conf_threshold,
                visualize=visualize,
                **kwargs
            )
            
    def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari batch gambar.
        
        Args:
            image_paths: List path ke gambar
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            **kwargs: Parameter tambahan untuk pipeline
            
        Returns:
            Dictionary hasil deteksi
        """
        batch_pipeline = self._get_batch_pipeline()
        
        return batch_pipeline.run(
            sources=image_paths,
            conf_threshold=conf_threshold,
            visualize=visualize,
            **kwargs
        )
    
    def detect_directory(
        self,
        dir_path: Union[str, Path],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        recursive: bool = False,
        batch_size: int = 1,
        use_pipeline: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari direktori berisi gambar.
        
        Args:
            dir_path: Path ke direktori
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            recursive: Flag untuk pencarian rekursif (default: False)
            batch_size: Ukuran batch untuk proses paralel (default: 1)
            use_pipeline: Gunakan pipeline baru (default: False)
            **kwargs: Parameter tambahan untuk strategi
            
        Returns:
            Dictionary hasil deteksi
        """
        # Konversi ke Path untuk memudahkan validasi
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
            
        # Validasi direktori
        if not dir_path.exists() or not dir_path.is_dir():
            raise DataError(f"Direktori tidak ditemukan: {dir_path}")
            
        # Jika menggunakan pipeline baru
        if use_pipeline:
            # Cari semua file gambar
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            glob_func = dir_path.rglob if recursive else dir_path.glob
            
            for ext in extensions:
                image_files.extend(list(glob_func(f"*{ext}")))
                image_files.extend(list(glob_func(f"*{ext.upper()}")))
            
            # Sort agar urutan konsisten
            image_files.sort()
            
            # Gunakan batch pipeline
            batch_pipeline = self._get_batch_pipeline()
            return batch_pipeline.run(
                sources=image_files,
                conf_threshold=conf_threshold,
                visualize=visualize,
                **kwargs
            )
        else:
            # Gunakan strategi directory detection
            strategy = self._get_directory_strategy()
            
            # Jalankan deteksi
            return strategy.detect(
                source=dir_path,
                conf_threshold=conf_threshold,
                visualize=visualize,
                recursive=recursive,
                batch_size=batch_size,
                **kwargs
            )

    def _get_model_adapter(self) -> ModelAdapter:
        """Lazy-load model adapter."""
        if self._model_adapter is None:
            self._model_adapter = ModelAdapter(
                config=self.config,
                logger=self.logger,
                colab_mode=self.colab_mode
            )
        return self._model_adapter
    
    def _get_model(self) -> torch.nn.Module:
        """Lazy-load model."""
        if self._model is None:
            model_adapter = self._get_model_adapter()
            self._model = model_adapter.get_model()
        return self._model
    
    def _get_preprocessor(self) -> ImagePreprocessor:
        """Lazy-load preprocessor."""
        if self._preprocessor is None:
            # Dapatkan img_size dari konfigurasi
            img_size = self.config.get('model', {}).get('img_size', (640, 640))
            self._preprocessor = ImagePreprocessor(
                config=self.config,
                img_size=img_size
            )
        return self._preprocessor
    
    def _get_detector(self) -> DefaultDetector:
        """Lazy-load detector."""
        if self._detector is None:
            model = self._get_model()
            self._detector = DefaultDetector(
                model=model,
                config=self.config,
                logger=self.logger
            )
        return self._detector
    
    def _get_postprocessor(self) -> DetectionPostprocessor:
        """Lazy-load postprocessor."""
        if self._postprocessor is None:
            self._postprocessor = DetectionPostprocessor(
                config=self.config,
                logger=self.logger
            )
        return self._postprocessor
    
    def _get_visualizer_adapter(self) -> VisualizerAdapter:
        """Lazy-load visualizer adapter."""
        if self._visualizer_adapter is None:
            self._visualizer_adapter = VisualizerAdapter(
                config=self.config
            )
        return self._visualizer_adapter
    
    def _get_output_manager(self) -> OutputManager:
        """Lazy-load output manager."""
        if self._output_manager is None:
            output_dir = self.config.get('inference', {}).get('output_dir', 'results/detections')
            visualizer = self._get_visualizer_adapter()
            self._output_manager = OutputManager(
                config=self.config,
                output_dir=output_dir,
                visualizer=visualizer,
                logger=self.logger,
                colab_mode=self.colab_mode
            )
        return self._output_manager
    
    def _get_image_strategy(self) -> ImageDetectionStrategy:
        """Lazy-load image detection strategy."""
        if self._image_strategy is None:
            detector = self._get_detector()
            preprocessor = self._get_preprocessor()
            postprocessor = self._get_postprocessor()
            output_manager = self._get_output_manager()
            
            self._image_strategy = ImageDetectionStrategy(
                config=self.config,
                detector=detector,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                output_manager=output_manager,
                logger=self.logger
            )
        return self._image_strategy
    
    def _get_directory_strategy(self) -> DirectoryDetectionStrategy:
        """Lazy-load directory detection strategy."""
        if self._directory_strategy is None:
            detector = self._get_detector()
            preprocessor = self._get_preprocessor()
            postprocessor = self._get_postprocessor()
            output_manager = self._get_output_manager()
            
            self._directory_strategy = DirectoryDetectionStrategy(
                config=self.config,
                detector=detector,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                output_manager=output_manager,
                logger=self.logger
            )
        return self._directory_strategy
        
    def _get_detection_pipeline(self) -> DetectionPipeline:
        """Lazy-load detection pipeline."""
        if self._detection_pipeline is None:
            detector = self._get_detector()
            preprocessor = self._get_preprocessor()
            postprocessor = self._get_postprocessor()
            output_manager = self._get_output_manager()
            
            self._detection_pipeline = DetectionPipeline(
                config=self.config,
                preprocessor=preprocessor,
                detector=detector,
                postprocessor=postprocessor,
                output_manager=output_manager,
                logger=self.logger
            )
        return self._detection_pipeline
    
    def _get_batch_pipeline(self) -> BatchDetectionPipeline:
        """Lazy-load batch detection pipeline."""
        if self._batch_pipeline is None:
            single_pipeline = self._get_detection_pipeline()
            
            self._batch_pipeline = BatchDetectionPipeline(
                config=self.config,
                single_pipeline=single_pipeline,
                logger=self.logger
            )
        return self._batch_pipeline
    
    def _is_running_in_colab(self) -> bool:
        """Deteksi apakah kode berjalan dalam Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default dari ConfigManager."""
        try:
            from smartcash.configs import get_config_manager
            config_manager = get_config_manager()
            
            # Coba load dari file jika ada
            if os.path.exists('configs/base_config.yaml'):
                return config_manager.load_from_file('configs/base_config.yaml')
            
            # Jika tidak ada file, gunakan config default
            return config_manager.get_config()
            
        except ImportError as e:
            self.logger.error(f"❌ ConfigManager tidak tersedia: {str(e)}")
            raise ImportError(f"ConfigManager diperlukan untuk DetectionManager: {str(e)}")