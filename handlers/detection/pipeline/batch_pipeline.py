# File: smartcash/handlers/detection/pipeline/batch_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline untuk proses deteksi mata uang pada batch gambar

import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from tqdm import tqdm

from smartcash.handlers.detection.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.detection.pipeline.detection_pipeline import DetectionPipeline

class BatchDetectionPipeline(BasePipeline):
    """
    Pipeline untuk deteksi objek pada batch gambar.
    Menggunakan DetectionPipeline untuk setiap gambar.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        single_pipeline: DetectionPipeline,
        logger = None
    ):
        """
        Inisialisasi pipeline batch.
        
        Args:
            config: Konfigurasi
            single_pipeline: Pipeline untuk gambar tunggal
            logger: Logger kustom (opsional)
        """
        super().__init__(config, logger, "batch_detection_pipeline")
        self.single_pipeline = single_pipeline
        
    def run(
        self,
        sources: List[Union[str, Path]],
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        output_json: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline deteksi batch.
        
        Args:
            sources: List path ke gambar
            conf_threshold: Threshold konfidiensi (opsional)
            visualize: Flag untuk visualisasi hasil (default: True)
            output_json: Flag untuk menyimpan hasil ke JSON (default: True)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi batch
        """
        self.logger.info(f"🚀 Menjalankan pipeline deteksi batch pada {len(sources)} gambar")
        self.notify_observers('start', {'sources': [str(s) for s in sources]})
        
        start_time = time.time()
        
        # Inisialisasi hasil
        batch_results = {
            'total_images': len(sources),
            'processed_images': 0,
            'results': [],
            'errors': [],
            'detections_by_layer': {}
        }
        
        # Proses setiap gambar
        for source in tqdm(sources, desc="📷 Memproses gambar"):
            try:
                # Jalankan pipeline deteksi untuk satu gambar
                result = self.single_pipeline.run(
                    source=source,
                    conf_threshold=conf_threshold,
                    visualize=visualize,
                    output_json=output_json,
                    **kwargs
                )
                
                # Tambahkan ke hasil batch
                batch_results['results'].append(result)
                batch_results['processed_images'] += 1
                
                # Update detections_by_layer
                for layer, detections in result.get('detections_by_layer', {}).items():
                    if layer not in batch_results['detections_by_layer']:
                        batch_results['detections_by_layer'][layer] = []
                    batch_results['detections_by_layer'][layer].extend(detections)
                
                # Notifikasi progress
                self.notify_observers('progress', {
                    'current': batch_results['processed_images'],
                    'total': batch_results['total_images'],
                    'latest_result': result
                })
                
            except Exception as e:
                self.logger.error(f"❌ Error saat memproses {source}: {str(e)}")
                batch_results['errors'].append({
                    'source': str(source),
                    'error': str(e)
                })
        
        # Hitung statistik akhir
        batch_results['execution_time'] = time.time() - start_time
        batch_results['success_rate'] = batch_results['processed_images'] / batch_results['total_images'] if batch_results['total_images'] > 0 else 0
        batch_results['num_errors'] = len(batch_results['errors'])
        
        total_detections = sum(len(r.get('detections', [])) for r in batch_results['results'])
        batch_results['total_detections'] = total_detections
        
        # Notifikasi complete
        self.notify_observers('complete', batch_results)
        
        self.logger.info(
            f"✅ Pipeline deteksi batch selesai: {batch_results['processed_images']}/{batch_results['total_images']} "
            f"gambar sukses, {total_detections} objek terdeteksi ({batch_results['execution_time']:.3f} detik)"
        )
        
        return batch_results