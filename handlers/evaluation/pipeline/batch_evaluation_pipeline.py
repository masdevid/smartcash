# File: smartcash/handlers/evaluation/pipeline/batch_evaluation_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline untuk evaluasi batch model yang diringkas

import os
import pandas as pd
from typing import Dict, List, Optional, Any
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline

class BatchEvaluationPipeline(BasePipeline):
    """Pipeline untuk evaluasi batch model secara paralel."""
    
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
        self.max_workers = self.config.get('evaluation', {}).get('batch', {}).get('max_workers', 1)
        self.timeout = self.config.get('evaluation', {}).get('batch', {}).get('timeout', 3600)
        self.logger.debug(f"🔧 {name} diinisialisasi (max_workers={self.max_workers})")
    
    def run(
        self,
        model_paths: List[str],
        dataset_path: str,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi batch untuk beberapa model.
        
        Args:
            model_paths: List path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            parallel: Evaluasi secara paralel
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap model
        """
        self.notify('batch_start', {'num_models': len(model_paths), 'dataset_path': dataset_path})
        self.logger.info(f"🚀 Menjalankan evaluasi batch untuk {len(model_paths)} model dengan dataset: {Path(dataset_path).name}")
        
        try:
            # Verifikasi dataset
            self._verify_dataset(dataset_path)
            
            # Jalankan evaluasi
            eval_func = self._run_parallel if parallel and self.max_workers > 1 else self._run_sequential
            model_results, execution_time = self.execute_with_timing(
                eval_func, model_paths=model_paths, dataset_path=dataset_path, **kwargs
            )
            
            # Buat summary dan hasil akhir
            summary = self._create_summary(model_results)
            result = {
                'pipeline_name': self.name,
                'dataset_path': dataset_path,
                'num_models': len(model_paths),
                'model_results': model_results,
                'summary': summary,
                'execution_time': execution_time
            }
            
            self.notify('batch_complete', {
                'execution_time': execution_time,
                'num_models': len(model_paths),
                'summary': summary
            })
            
            self.logger.success(f"✅ Evaluasi batch selesai dalam {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.notify('batch_error', {'error': str(e)})
            self.logger.error(f"❌ Evaluasi batch gagal: {str(e)}")
            raise
    
    def _verify_dataset(self, dataset_path: str) -> bool:
        """Verifikasi dataset."""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset tidak ditemukan: {dataset_path}")
            
        images_dir = os.path.join(dataset_path, 'images')
        labels_dir = os.path.join(dataset_path, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise ValueError(f"Struktur dataset tidak valid: {dataset_path}")
            
        image_files = os.listdir(images_dir)
        label_files = os.listdir(labels_dir)
        
        if not image_files or not label_files:
            raise ValueError(f"Dataset kosong: {dataset_path}")
            
        self.logger.info(f"✅ Dataset valid: {len(image_files)} gambar, {len(label_files)} label")
        return True
    
    def _run_sequential(self, model_paths: List[str], dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Jalankan evaluasi secara sekuensial."""
        model_results = {}
        
        for model_idx, model_path in enumerate(tqdm(model_paths, desc="Evaluasi Model")):
            model_name = Path(model_path).stem
            self.logger.info(f"🔄 Evaluasi model {model_idx+1}/{len(model_paths)}: {model_name}")
            
            try:
                # Evaluasi model
                pipeline = EvaluationPipeline(self.config, self.logger, f"Eval_{model_name}")
                result = pipeline.run(model_path=model_path, dataset_path=dataset_path, **kwargs)
                model_results[model_name] = result
                
                # Log metrik
                if 'metrics' in result:
                    metrics = result['metrics']
                    self.logger.info(f"📊 Hasil {model_name}: mAP={metrics.get('mAP', 0):.4f}, F1={metrics.get('f1', 0):.4f}")
                
                # Notifikasi
                self.notify('batch_progress', {
                    'model_name': model_name,
                    'model_idx': model_idx,
                    'total_models': len(model_paths),
                    'metrics': result.get('metrics', {})
                })
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                model_results[model_name] = {'error': str(e)}
        
        return model_results
    
    def _run_parallel(self, model_paths: List[str], dataset_path: str, **kwargs) -> Dict[str, Any]:
        """Jalankan evaluasi secara paralel."""
        model_results = {}
        
        def evaluate_model(model_path):
            model_name = Path(model_path).stem
            logger = self.logger.get_child(f"eval_{model_name}") if hasattr(self.logger, 'get_child') else self.logger
            
            try:
                pipeline = EvaluationPipeline(self.config, logger, f"Eval_{model_name}")
                result = pipeline.run(model_path=model_path, dataset_path=dataset_path, **kwargs)
                return model_name, result
            except Exception as e:
                logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                return model_name, {'error': str(e)}
        
        # Eksekusi paralel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(evaluate_model, path): path for path in model_paths}
            
            for i, future in enumerate(tqdm(as_completed(futures), total=len(model_paths), desc="Evaluasi Model")):
                try:
                    model_name, result = future.result(timeout=self.timeout)
                    model_results[model_name] = result
                    
                    # Log metrik jika ada
                    if 'metrics' in result:
                        metrics = result['metrics']
                        self.logger.info(f"📊 Hasil {model_name}: mAP={metrics.get('mAP', 0):.4f}, F1={metrics.get('f1', 0):.4f}")
                    
                    # Notifikasi progress
                    self.notify('batch_progress', {
                        'model_name': model_name,
                        'model_idx': i,
                        'total_models': len(model_paths),
                        'metrics': result.get('metrics', {})
                    })
                        
                except Exception as e:
                    model_path = futures[future]
                    model_name = Path(model_path).stem
                    self.logger.warning(f"⚠️ Gagal evaluasi {model_name}: {str(e)}")
                    model_results[model_name] = {'error': str(e)}
        
        return model_results
    
    def _create_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Buat summary dari hasil evaluasi batch."""
        summary = {
            'num_models': len(model_results),
            'successful_models': 0,
            'failed_models': 0,
            'best_model': None,
            'best_map': 0,
            'average_map': 0,
            'average_f1': 0
        }
        
        # Kumpulkan metrik
        metrics_list = []
        map_values = []
        f1_values = []
        
        for model_name, result in model_results.items():
            if 'error' in result:
                summary['failed_models'] += 1
                continue
                
            summary['successful_models'] += 1
            
            if 'metrics' in result:
                metrics = result['metrics']
                map_value = metrics.get('mAP', 0)
                f1_value = metrics.get('f1', 0)
                
                map_values.append(map_value)
                f1_values.append(f1_value)
                
                # Cek best model
                if map_value > summary['best_map']:
                    summary['best_map'] = map_value
                    summary['best_model'] = model_name
                    
                # Tambahkan ke list metrik
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
                metrics_df = pd.DataFrame(metrics_list).sort_values('mAP', ascending=False)
                summary['metrics_table'] = metrics_df
                
                # Buat perbandingan performa
                summary['performance_comparison'] = {}
                for metric in ['mAP', 'F1', 'precision', 'recall', 'inference_time']:
                    if metric in metrics_df.columns:
                        summary['performance_comparison'][metric] = {
                            'best_model': metrics_df.iloc[0]['model'],
                            'best_value': metrics_df.iloc[0][metric],
                            'average': metrics_df[metric].mean()
                        }
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat tabel metrik: {str(e)}")
        
        return summary