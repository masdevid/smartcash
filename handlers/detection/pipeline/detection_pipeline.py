# File: smartcash/handlers/detection/pipeline/detection_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline untuk proses deteksi mata uang pada gambar tunggal

import time
from pathlib import Path
from typing import Dict, Any, Union, Optional

from smartcash.handlers.detection.pipeline.base_pipeline import BasePipeline

class DetectionPipeline(BasePipeline):
    """
    Pipeline untuk deteksi objek pada gambar tunggal.
    Menggabungkan preprocessing, deteksi, dan postprocessing
    dalam satu alur kerja yang terstruktur.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        preprocessor,
        detector,
        postprocessor,
        output_manager,
        logger = None
    ):
        """
        Inisialisasi pipeline.
        
        Args:
            config: Konfigurasi
            preprocessor: Komponen preprocessor
            detector: Komponen detector
            postprocessor: Komponen postprocessor
            output_manager: Komponen output manager
            logger: Logger kustom (opsional)
        """
        super().__init__(config, logger, "detection_pipeline")
        self.preprocessor = preprocessor
        self.detector = detector
        self.postprocessor = postprocessor
        self.output_manager = output_manager
        
    def run(
        self,
        source: Union[str, Path],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        output_json: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline deteksi.
        
        Args:
            source: Path ke gambar
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            output_json: Flag untuk menyimpan hasil ke JSON (default: True)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi
        """
        self.logger.info(f"🚀 Menjalankan pipeline deteksi pada {source}")
        self.notify_observers('start', {'source': str(source)})
        
        start_time = time.time()
        
        try:
            # Preprocessing
            self.logger.debug(f"⏳ Preprocessing gambar...")
            preprocess_result = self.preprocessor.process(source)
            img_tensor = preprocess_result['tensor']
            original_shape = preprocess_result['original_shape']
            original_image = preprocess_result.get('original_image')
            
            # Deteksi
            self.logger.debug(f"🔍 Menjalankan model deteksi...")
            detection_result = self.detector.detect(
                img_tensor, 
                conf_thres=conf_threshold,
                **kwargs
            )
            
            # Postprocessing
            self.logger.debug(f"⚙️ Postprocessing hasil deteksi...")
            result = self.postprocessor.process(
                detection_result,
                original_shape=original_shape
            )
            
            # Tambahkan informasi source
            result['source'] = str(source)
            result['execution_time'] = time.time() - start_time
            
            # Visualisasi jika diminta
            if visualize and original_image is not None:
                self.logger.debug(f"🎨 Membuat visualisasi hasil...")
                output_path = self.output_manager.save_visualization(
                    source=source,
                    image=original_image,
                    detections=result['detections'],
                    **kwargs
                )
                result['visualization_path'] = output_path
            
            # Simpan hasil ke JSON jika diminta
            if output_json:
                self.logger.debug(f"💾 Menyimpan hasil deteksi...")
                output_paths = self.output_manager.save_results(source, result, **kwargs)
                result['output_paths'] = output_paths
            
            # Notifikasi complete
            self.notify_observers('complete', result)
            
            self.logger.info(
                f"✅ Pipeline deteksi selesai: {result.get('num_detections', 0)} objek terdeteksi "
                f"({result.get('execution_time', 0):.3f} detik)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam pipeline deteksi: {str(e)}")
            # Notifikasi error
            self.notify_observers('error', {
                'source': str(source),
                'error': str(e)
            })
            raise