# File: smartcash/handlers/detection/strategies/image_strategy.py
# Author: Alfrida Sabar
# Deskripsi: Strategi deteksi untuk gambar tunggal

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import torch
import time

from smartcash.handlers.detection.strategies.base_strategy import BaseDetectionStrategy
from smartcash.exceptions.base import DataError

class ImageDetectionStrategy(BaseDetectionStrategy):
    """
    Strategi untuk deteksi objek pada gambar tunggal.
    """
    
    def detect(
        self,
        source: Union[str, Path],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari gambar tunggal.
        
        Args:
            source: Path ke gambar
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi
        """
        # Tentukan path gambar
        if isinstance(source, str):
            source = Path(source)
            
        # Validasi
        if not source.exists():
            raise DataError(f"File gambar tidak ditemukan: {source}")
            
        # Notifikasi start
        self.notify_observers('start', {'source': str(source)})
        
        start_time = time.time()
        self.logger.info(f"🔍 Mendeteksi objek dari gambar: {source}")
        
        try:
            # Preprocessing
            preprocess_result = self.preprocessor.process(source)
            img_tensor = preprocess_result['tensor']
            original_shape = preprocess_result['original_shape']
            original_image = preprocess_result.get('original_image')
            
            # Deteksi
            detection_result = self.detector.detect(
                img_tensor, 
                conf_thres=conf_threshold,
                **kwargs
            )
            
            # Postprocessing
            result = self.postprocessor.process(
                detection_result,
                original_shape=original_shape
            )
            
            # Tambahkan informasi source
            result['source'] = str(source)
            result['execution_time'] = time.time() - start_time
            
            # Log hasil
            num_detections = result['num_detections']
            self.logger.info(f"✅ Deteksi selesai: {num_detections} objek terdeteksi " 
                           f"({result['execution_time']:.3f} detik)")
            
            # Visualisasi jika diminta
            if visualize and original_image is not None:
                output_path = self.output_manager.save_visualization(
                    source=source,
                    image=original_image,
                    detections=result['detections'],
                    **kwargs
                )
                result['visualization_path'] = output_path
            
            # Simpan hasil ke JSON jika diperlukan
            output_paths = self.output_manager.save_results(source, result, **kwargs)
            result['output_paths'] = output_paths
            
            # Notifikasi complete
            self.notify_observers('complete', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error saat deteksi gambar {source}: {str(e)}")
            # Notifikasi error
            self.notify_observers('error', {
                'source': str(source),
                'error': str(e)
            })
            raise