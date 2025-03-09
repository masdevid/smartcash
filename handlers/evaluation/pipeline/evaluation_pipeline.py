# File: smartcash/handlers/evaluation/pipeline/evaluation_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline evaluasi model dengan penggunaan BasePipeline

from typing import Dict, List, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.evaluation.core.model_evaluator import ModelEvaluator
from smartcash.handlers.evaluation.integration.metrics_adapter import MetricsAdapter
from smartcash.handlers.evaluation.integration.model_manager_adapter import ModelManagerAdapter
from smartcash.handlers.evaluation.integration.dataset_adapter import DatasetAdapter

class EvaluationPipeline(BasePipeline):
    """
    Pipeline evaluasi dengan berbagai komponen yang dapat dikonfigurasi.
    Menggabungkan beberapa komponen evaluasi menjadi satu alur kerja.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "EvaluationPipeline"
    ):
        """
        Inisialisasi evaluation pipeline.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
        """
        super().__init__(config, logger, name)
    
    def run(
        self,
        model_path: str,
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline evaluasi.
        
        Args:
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Notifikasi observer tentang start
        self.notify_observers('pipeline_start', {
            'pipeline_name': self.name,
            'model_path': model_path,
            'dataset_path': dataset_path,
            'start_time': time.time()
        })
        
        self.logger.info(f"🚀 Menjalankan {self.name} untuk model: {Path(model_path).name}")
        
        try:
            # Dapatkan adapter, gunakan default jika tidak disediakan
            metrics_adapter = self.get_adapter(metrics_adapter, MetricsAdapter)
            model_adapter = self.get_adapter(model_adapter, ModelManagerAdapter)
            dataset_adapter = self.get_adapter(dataset_adapter, DatasetAdapter)
            
            # Jika tidak ada komponen, gunakan default (ModelEvaluator)
            if not self.components:
                self.logger.info("ℹ️ Menggunakan ModelEvaluator default")
                model_evaluator = ModelEvaluator(
                    config=self.config,
                    metrics_adapter=metrics_adapter,
                    model_adapter=model_adapter,
                    dataset_adapter=dataset_adapter,
                    logger=self.logger
                )
                
                # Jalankan evaluasi dengan penangkapan waktu eksekusi
                results, execution_time = self.execute_with_timing(
                    model_evaluator.process,
                    model_path=model_path,
                    dataset_path=dataset_path,
                    observers=self.observers,
                    **kwargs
                )
            else:
                # Jika ada komponen, jalankan secara berurutan
                self.logger.info(f"ℹ️ Menjalankan {len(self.components)} komponen")
                
                # Inisialisasi hasil
                results = {
                    'pipeline_name': self.name,
                    'model_path': model_path,
                    'dataset_path': dataset_path,
                    'component_results': {}
                }
                
                # Jalankan setiap komponen
                for component in self.components:
                    self.logger.info(f"▶️ Menjalankan komponen: {component.name}")
                    self.notify_observers('component_start', {
                        'component_name': component.name
                    })
                    
                    # Jalankan komponen dengan penangkapan waktu eksekusi
                    component_result, component_time = self.execute_with_timing(
                        component.process,
                        model_path=model_path,
                        dataset_path=dataset_path,
                        metrics_adapter=metrics_adapter,
                        model_adapter=model_adapter,
                        dataset_adapter=dataset_adapter,
                        observers=self.observers,
                        **kwargs
                    )
                    
                    # Simpan hasil
                    results['component_results'][component.name] = component_result
                    
                    # Notifikasi observer
                    self.notify_observers('component_complete', {
                        'component_name': component.name,
                        'execution_time': component_time
                    })
                    
                    self.logger.info(f"✅ Komponen {component.name} selesai dalam {component_time:.2f}s")
                
                # Gabungkan hasil metrik jika ada
                self._merge_component_metrics(results)
                
                # Tambahkan waktu total eksekusi dari semua komponen
                execution_time = sum(
                    component_result.get('execution_time', 0) 
                    for component_result in results['component_results'].values()
                )
                results['execution_time'] = execution_time
            
            # Notifikasi observer tentang completion
            self.notify_observers('pipeline_complete', {
                'pipeline_name': self.name,
                'execution_time': execution_time,
                'results': results
            })
            
            self.logger.success(f"✅ Pipeline {self.name} selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            # Notifikasi observer tentang error
            self.notify_observers('pipeline_error', {
                'pipeline_name': self.name,
                'error': str(e),
            })
            
            self.logger.error(f"❌ Pipeline {self.name} gagal: {str(e)}")
            raise
    
    def _merge_component_metrics(self, results: Dict[str, Any]):
        """
        Gabungkan metrik dari berbagai komponen.
        
        Args:
            results: Dictionary hasil yang akan diupdate
        """
        if any('metrics' in r for r in results['component_results'].values()):
            results['metrics'] = {}
            for component_name, component_result in results['component_results'].items():
                if 'metrics' in component_result:
                    results['metrics'][component_name] = component_result['metrics']