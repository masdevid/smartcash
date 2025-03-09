# File: smartcash/handlers/evaluation/pipeline/batch_evaluation_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline untuk evaluasi batch model dengan penggunaan BasePipeline

import os
import pandas as pd
from typing import Dict, List, Optional, Any
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from smartcash.handlers.evaluation.integration.visualization_adapter import VisualizationAdapter
from smartcash.handlers.evaluation.integration.metrics_adapter import MetricsAdapter
from smartcash.handlers.evaluation.integration.model_manager_adapter import ModelManagerAdapter
from smartcash.handlers.evaluation.integration.dataset_adapter import DatasetAdapter

class BatchEvaluationPipeline(BasePipeline):
    """
    Pipeline untuk evaluasi batch model.
    Evaluasi beberapa model dengan dataset yang sama secara paralel.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "BatchEvaluationPipeline"
    ):
        """
        Inisialisasi batch evaluation pipeline.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
        """
        super().__init__(config, logger, name)
        
        # Konfigurasi batch
        self.batch_config = self.config.get('evaluation', {}).get('batch', {})
        self.max_workers = self.batch_config.get('max_workers', 1)
        self.timeout = self.batch_config.get('timeout', 3600)  # Default 1 jam
        
        # Setup plot directory
        self.plot_dir = Path(self.config.get('output_dir', 'results/evaluation/batch')) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization adapter
        self.visualization_adapter = VisualizationAdapter(
            config=self.config,
            output_dir=str(self.plot_dir),
            logger=self.logger
        )
        
        self.logger.debug(f"🔧 {name} diinisialisasi (max_workers={self.max_workers})")
    
    def run(
        self,
        model_paths: List[str],
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi batch untuk beberapa model.
        
        Args:
            model_paths: List path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            parallel: Evaluasi secara paralel
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap model
        """
        # Notifikasi observer tentang start
        self.notify_observers('batch_start', {
            'pipeline_name': self.name,
            'num_models': len(model_paths),
            'dataset_path': dataset_path
        })
        
        self.logger.info(
            f"🚀 Menjalankan evaluasi batch untuk {len(model_paths)} model "
            f"dengan dataset: {Path(dataset_path).name}"
        )
        
        # Inisialisasi hasil
        results = {
            'pipeline_name': self.name,
            'dataset_path': dataset_path,
            'num_models': len(model_paths),
            'model_results': {},
            'summary': {},
            'plots': {}
        }
        
        try:
            # Dapatkan adapter, gunakan default jika tidak disediakan
            metrics_adapter = self.get_adapter(metrics_adapter, MetricsAdapter)
            model_adapter = self.get_adapter(model_adapter, ModelManagerAdapter)
            dataset_adapter = self.get_adapter(dataset_adapter, DatasetAdapter)
            
            # Verifikasi dataset terlebih dahulu
            if dataset_adapter:
                dataset_adapter.verify_dataset(dataset_path)
            
            # Jalankan evaluasi batch
            eval_func = self._run_parallel if parallel and self.max_workers > 1 else self._run_sequential
            model_results, execution_time = self.execute_with_timing(
                eval_func,
                model_paths=model_paths,
                dataset_path=dataset_path,
                metrics_adapter=metrics_adapter,
                model_adapter=model_adapter,
                dataset_adapter=dataset_adapter,
                **kwargs
            )
            
            # Simpan hasil
            results['model_results'] = model_results
            results['execution_time'] = execution_time
            
            # Buat summary
            summary = self._create_summary(model_results)
            results['summary'] = summary
            
            # Buat visualisasi dengan adapter
            plots = self.visualization_adapter.generate_batch_plots(results)
            results['plots'] = plots
            
            # Notifikasi observer tentang completion
            self.notify_observers('batch_complete', {
                'pipeline_name': self.name,
                'execution_time': execution_time,
                'num_models': len(model_paths),
                'summary': summary
            })
            
            self.logger.success(f"✅ Evaluasi batch selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            # Notifikasi observer tentang error
            self.notify_observers('batch_error', {
                'pipeline_name': self.name,
                'error': str(e)
            })
            
            self.logger.error(f"❌ Evaluasi batch gagal: {str(e)}")
            raise
    
    def _run_sequential(
        self,
        model_paths: List[str],
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi secara sekuensial.
        
        Args:
            model_paths: List path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap model
        """
        # Inisialisasi pipeline evaluasi
        evaluation_pipeline = EvaluationPipeline(
            config=self.config,
            logger=self.logger,
            name="SingleModelEvaluation"
        )
        
        # Tambahkan observer
        for observer in self.observers:
            evaluation_pipeline.add_observer(observer)
        
        # Inisialisasi hasil
        model_results = {}
        
        # Jalankan evaluasi untuk setiap model
        for model_idx, model_path in enumerate(tqdm(model_paths, desc="Evaluasi Model")):
            model_name = Path(model_path).stem
            self.logger.info(f"🔄 Evaluasi model {model_idx+1}/{len(model_paths)}: {model_name}")
            
            try:
                # Jalankan evaluasi
                result = evaluation_pipeline.run(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    metrics_adapter=metrics_adapter,
                    model_adapter=model_adapter,
                    dataset_adapter=dataset_adapter,
                    **kwargs
                )
                
                # Simpan hasil
                model_results[model_name] = result
                
                # Log metrik utama
                if 'metrics' in result:
                    metrics = result['metrics']
                    self.logger.info(
                        f"📊 Hasil {model_name}: "
                        f"mAP={metrics.get('mAP', 0):.4f}, "
                        f"F1={metrics.get('f1', 0):.4f}"
                    )
                
                # Notify observer setelah setiap model
                self.notify_observers('batch_process', {
                    'model_name': model_name,
                    'model_idx': model_idx,
                    'total_models': len(model_paths),
                    'metrics': result.get('metrics', {})
                })
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                model_results[model_name] = {'error': str(e)}
        
        return model_results
    
    def _run_parallel(
        self,
        model_paths: List[str],
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi secara paralel.
        
        Args:
            model_paths: List path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap model
        """
        # Inisialisasi hasil
        model_results = {}
        
        # Fungsi helper untuk evaluasi satu model
        def evaluate_model(model_path):
            model_name = Path(model_path).stem
            individual_logger = self.logger.get_child(f"eval_{model_name}")
            
            try:
                # Buat pipeline baru untuk setiap model
                pipeline = EvaluationPipeline(
                    config=self.config,
                    logger=individual_logger,
                    name=f"Eval_{model_name}"
                )
                
                # Jalankan evaluasi
                result = pipeline.run(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    metrics_adapter=metrics_adapter,
                    model_adapter=model_adapter,
                    dataset_adapter=dataset_adapter,
                    **kwargs
                )
                
                return model_name, result
                
            except Exception as e:
                individual_logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                return model_name, {'error': str(e)}
        
        # Jalankan evaluasi paralel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(evaluate_model, model_path): model_path for model_path in model_paths}
            
            for i, future in enumerate(tqdm(as_completed(futures), total=len(model_paths), desc="Evaluasi Model")):
                model_path = futures[future]
                model_name = Path(model_path).stem
                
                try:
                    model_name, result = future.result(timeout=self.timeout)
                    model_results[model_name] = result
                    
                    # Log metrik utama
                    if 'metrics' in result:
                        metrics = result['metrics']
                        self.logger.info(
                            f"📊 Hasil {model_name}: "
                            f"mAP={metrics.get('mAP', 0):.4f}, "
                            f"F1={metrics.get('f1', 0):.4f}"
                        )
                    
                    # Notify observer setelah setiap model
                    self.notify_observers('batch_process', {
                        'model_name': model_name,
                        'model_idx': i,
                        'total_models': len(model_paths),
                        'metrics': result.get('metrics', {})
                    })
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                    model_results[model_name] = {'error': str(e)}
        
        return model_results
    
    def _create_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Buat summary dari hasil evaluasi batch.
        
        Args:
            model_results: Dictionary hasil evaluasi untuk setiap model
            
        Returns:
            Dictionary berisi summary evaluasi
        """
        # Inisialisasi summary
        summary = {
            'num_models': len(model_results),
            'successful_models': 0,
            'failed_models': 0,
            'best_model': None,
            'best_map': 0,
            'best_f1': 0,
            'average_map': 0,
            'average_f1': 0,
            'metrics_table': None,
            'performance_comparison': {}
        }
        
        # Ambil metrik dari setiap model
        metrics_list = []
        map_values = []
        f1_values = []
        
        for model_name, result in model_results.items():
            # Cek apakah evaluasi berhasil
            if 'error' in result:
                summary['failed_models'] += 1
                continue
                
            summary['successful_models'] += 1
            
            # Ambil metrik
            if 'metrics' in result:
                metrics = result['metrics']
                map_value = metrics.get('mAP', 0)
                f1_value = metrics.get('f1', 0)
                
                # Simpan untuk perhitungan rata-rata
                map_values.append(map_value)
                f1_values.append(f1_value)
                
                # Update best_model jika lebih baik
                if map_value > summary['best_map']:
                    summary['best_map'] = map_value
                    summary['best_model'] = model_name
                    
                # Simpan untuk tabel metrik
                metrics_list.append({
                    'model': model_name,
                    'mAP': map_value,
                    'F1': f1_value,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'inference_time': metrics.get('inference_time', 0) * 1000  # ms
                })
        
        # Hitung rata-rata
        if map_values:
            summary['average_map'] = sum(map_values) / len(map_values)
        if f1_values:
            summary['average_f1'] = sum(f1_values) / len(f1_values)
        
        # Buat tabel metrik
        if metrics_list:
            try:
                metrics_df = pd.DataFrame(metrics_list)
                # Sort berdasarkan mAP (descending)
                metrics_df = metrics_df.sort_values('mAP', ascending=False)
                summary['metrics_table'] = metrics_df
                
                # Performance comparison
                for metric in ['mAP', 'F1', 'precision', 'recall', 'inference_time']:
                    if metric in metrics_df.columns:
                        best_model = metrics_df.iloc[0]['model']
                        best_value = metrics_df.iloc[0][metric]
                        
                        summary['performance_comparison'][metric] = {
                            'best_model': best_model,
                            'best_value': best_value,
                            'average': metrics_df[metric].mean()
                        }
            
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat tabel metrik: {str(e)}")
        
        return summary