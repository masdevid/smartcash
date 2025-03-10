# File: smartcash/handlers/evaluation/pipeline/research_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline evaluasi skenario penelitian yang diringkas

import time
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline

class ResearchPipeline(BasePipeline):
    """Pipeline untuk evaluasi skenario penelitian."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "ResearchPipeline"
    ):
        """
        Inisialisasi research pipeline.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            name: Nama pipeline
        """
        super().__init__(config, logger, name)
        
        # Konfigurasi research
        research_config = self.config.get('evaluation', {}).get('research', {})
        self.num_runs = research_config.get('num_runs', 3)
        self.parallel_scenarios = research_config.get('parallel_scenarios', False)
        self.max_workers = research_config.get('max_workers', 2)
        
        # Cache untuk path yang sudah di-resolve
        self._path_cache = {'model': {}, 'dataset': {}}
        
        self.logger.debug(f"🔧 {name} diinisialisasi (num_runs={self.num_runs}, workers={self.max_workers})")
    
    def run(
        self,
        scenarios: Dict[str, Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi skenario penelitian.
        
        Args:
            scenarios: Dictionary skenario penelitian
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap skenario
        """
        # Notifikasi dan timing
        start_time = time.time()
        self.notify('research_start', {'num_scenarios': len(scenarios)})
        self.logger.info(f"🔬 Memulai evaluasi {len(scenarios)} skenario penelitian")
        
        try:
            # Jalankan skenario
            if self.parallel_scenarios and self.max_workers > 1 and len(scenarios) > 1:
                scenario_results = self._run_scenarios_parallel(scenarios, **kwargs)
            else:
                scenario_results = self._run_scenarios_sequential(scenarios, **kwargs)
            
            # Buat summary dan hasil akhir
            summary = self._create_summary(scenario_results)
            execution_time = time.time() - start_time
            
            result = {
                'pipeline_name': self.name,
                'num_scenarios': len(scenarios),
                'scenario_results': scenario_results,
                'summary': summary,
                'execution_time': execution_time
            }
            
            # Notifikasi selesai
            self.notify('research_complete', {
                'execution_time': execution_time,
                'num_scenarios': len(scenarios),
                'summary': summary
            })
            
            self.logger.success(f"✅ Evaluasi penelitian selesai dalam {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Notifikasi error
            self.notify('research_error', {'error': str(e)})
            self.logger.error(f"❌ Evaluasi penelitian gagal: {str(e)}")
            raise
    
    def _run_scenarios_sequential(self, scenarios: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
        """Jalankan skenario secara sekuensial."""
        scenario_results = {}
        
        for scenario_name, scenario_config in tqdm(scenarios.items(), desc="Evaluasi Skenario"):
            result = self._evaluate_scenario(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                **kwargs
            )
            scenario_results[scenario_name] = result
        
        return scenario_results
    
    def _run_scenarios_parallel(self, scenarios: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
        """Jalankan skenario secara paralel."""
        scenario_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit semua job
            futures = {
                executor.submit(self._evaluate_scenario, scenario_name, scenario_config, **kwargs): scenario_name
                for scenario_name, scenario_config in scenarios.items()
            }
            
            # Proses hasil
            for future in tqdm(as_completed(futures), total=len(scenarios), desc="Evaluasi Skenario"):
                scenario_name = futures[future]
                try:
                    result = future.result()
                    scenario_results[scenario_name] = result
                except Exception as e:
                    self.logger.error(f"❌ Skenario {scenario_name} gagal: {str(e)}")
                    scenario_results[scenario_name] = {
                        'error': str(e),
                        'config': scenarios[scenario_name]
                    }
        
        return scenario_results
    
    def _evaluate_scenario(self, scenario_name: str, scenario_config: Dict, **kwargs) -> Dict[str, Any]:
        """Evaluasi satu skenario."""
        self.logger.info(f"🔄 Evaluasi skenario: {scenario_name} - {scenario_config['desc']}")
        
        # Resolve paths
        model_path = self._resolve_model_path(scenario_config['model'])
        dataset_path = self._resolve_dataset_path(scenario_config['data'])
        
        # Verifikasi file
        if not os.path.exists(model_path):
            self.logger.warning(f"⚠️ Model tidak ditemukan: {model_path}")
            return {'error': 'Model tidak ditemukan', 'config': scenario_config}
            
        if not dataset_path or not os.path.exists(dataset_path):
            self.logger.warning(f"⚠️ Dataset tidak ditemukan: {dataset_path}")
            return {'error': 'Dataset tidak ditemukan', 'config': scenario_config}
        
        # Jalankan evaluasi berulang kali
        scenario_result = self._run_multiple_evaluations(
            scenario_name=scenario_name,
            model_path=model_path,
            dataset_path=dataset_path,
            **kwargs
        )
        
        # Log hasil
        if 'avg_metrics' in scenario_result:
            metrics = scenario_result['avg_metrics']
            self.logger.info(
                f"📊 Hasil {scenario_name}: "
                f"mAP={metrics.get('mAP', 0):.4f}, "
                f"F1={metrics.get('f1', 0):.4f}, "
                f"Inference={metrics.get('inference_time', 0)*1000:.2f}ms"
            )
        
        return {
            'config': scenario_config,
            'results': scenario_result,
            'model_path': model_path,
            'dataset_path': dataset_path
        }
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve path model."""
        # Cek cache
        if model_path in self._path_cache['model']:
            return self._path_cache['model'][model_path]
            
        # Path absolute
        if os.path.isabs(model_path) and os.path.exists(model_path):
            self._path_cache['model'][model_path] = model_path
            return model_path
            
        # Direktori checkpoint
        checkpoints_dir = Path(self.config.get('checkpoints_dir', 'runs/train/weights'))
        
        # Path relatif terhadap checkpoint dir
        checkpoint_path = checkpoints_dir / model_path
        if checkpoint_path.exists():
            self._path_cache['model'][model_path] = str(checkpoint_path)
            return str(checkpoint_path)
            
        # Cari file dengan nama yang cocok
        if checkpoints_dir.exists():
            matches = list(checkpoints_dir.glob(f"*{model_path}*"))
            if matches:
                self._path_cache['model'][model_path] = str(matches[0])
                return str(matches[0])
            
        # Default
        self._path_cache['model'][model_path] = model_path
        return model_path
    
    def _resolve_dataset_path(self, dataset_path: str) -> str:
        """Resolve path dataset."""
        # Cek cache
        if dataset_path in self._path_cache['dataset']:
            return self._path_cache['dataset'][dataset_path]
            
        # Path absolute
        if os.path.isabs(dataset_path) and os.path.exists(dataset_path):
            self._path_cache['dataset'][dataset_path] = dataset_path
            return dataset_path
            
        # Direktori data
        data_dir = Path(self.config.get('data_dir', 'data'))
        
        # Path relatif terhadap data dir
        dataset_full_path = data_dir / dataset_path
        if dataset_full_path.exists():
            self._path_cache['dataset'][dataset_path] = str(dataset_full_path)
            return str(dataset_full_path)
            
        # Cari direktori dengan nama yang cocok
        if data_dir.exists():
            matches = list(data_dir.glob(f"*{dataset_path}*"))
            if matches:
                self._path_cache['dataset'][dataset_path] = str(matches[0])
                return str(matches[0])
            
        # Default
        self._path_cache['dataset'][dataset_path] = dataset_path
        return dataset_path
    
    def _run_multiple_evaluations(
        self,
        scenario_name: str,
        model_path: str,
        dataset_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Jalankan evaluasi berulang kali."""
        # Inisialisasi
        evaluation_pipeline = EvaluationPipeline(
            config=self.config,
            logger=self.logger.get_child(f"eval_{scenario_name}") if hasattr(self.logger, 'get_child') else self.logger,
            name=f"Eval_{scenario_name}"
        )
        
        # Evaluasi multiple kali
        run_results = []
        all_metrics = {}
        
        for run in range(self.num_runs):
            self.logger.debug(f"🔄 Run {run+1}/{self.num_runs} untuk {scenario_name}")
            
            try:
                # Evaluasi model
                result = evaluation_pipeline.run(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    **kwargs
                )
                
                # Simpan hasil
                run_results.append(result)
                
                # Kumpulkan metrik untuk rata-rata
                if 'metrics' in result:
                    metrics = result['metrics']
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Run {run+1} gagal: {str(e)}")
        
        # Hitung rata-rata dan standar deviasi
        avg_metrics = {}
        std_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
                # Standar deviasi
                if len(values) > 1:
                    mean = avg_metrics[key]
                    std_metrics[key] = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        return {
            'run_results': run_results,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'num_successful_runs': len(run_results)
        }
    
    def _create_summary(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Buat summary hasil evaluasi."""
        # Inisialisasi
        summary = {
            'num_scenarios': len(scenario_results),
            'successful_scenarios': 0,
            'failed_scenarios': 0,
            'best_scenario': None,
            'best_map': 0
        }
        
        # Kumpulkan metrik
        metrics_list = []
        
        for scenario_name, result in scenario_results.items():
            # Periksa status
            if 'error' in result:
                summary['failed_scenarios'] += 1
                continue
                
            summary['successful_scenarios'] += 1
            
            # Ambil metrik
            if 'results' in result and 'avg_metrics' in result['results']:
                metrics = result['results']['avg_metrics']
                map_value = metrics.get('mAP', 0)
                f1_value = metrics.get('f1', 0)
                
                # Periksa best scenario
                if map_value > summary['best_map']:
                    summary['best_map'] = map_value
                    summary['best_scenario'] = scenario_name
                    
                # Tambahkan ke metrics list
                metrics_entry = {
                    'scenario': scenario_name,
                    'description': result['config']['desc'],
                    'mAP': map_value,
                    'F1': f1_value,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'inference_time': metrics.get('inference_time', 0) * 1000  # ms
                }
                
                # Tambahkan std jika ada
                if 'std_metrics' in result['results']:
                    std_metrics = result['results']['std_metrics']
                    for key, value in std_metrics.items():
                        metrics_entry[f'{key}_std'] = value
                
                metrics_list.append(metrics_entry)
        
        # Buat tabel metrik dan analisis
        if metrics_list:
            try:
                # Buat DataFrame untuk analisis
                metrics_df = pd.DataFrame(metrics_list).sort_values('mAP', ascending=False)
                summary['metrics_table'] = metrics_df
                
                # Analisis backbone
                self._analyze_backbone_performance(metrics_df, summary)
                
                # Analisis kondisi
                self._analyze_condition_performance(metrics_df, summary)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat tabel metrik: {str(e)}")
        
        return summary
    
    def _analyze_backbone_performance(self, metrics_df: pd.DataFrame, summary: Dict):
        """Analisis performa backbone."""
        backbone_data = {'efficientnet': [], 'cspdarknet': []}
        
        for _, row in metrics_df.iterrows():
            desc = row['description'].lower()
            if 'efficientnet' in desc:
                backbone_data['efficientnet'].append({
                    'mAP': row['mAP'],
                    'F1': row['F1'],
                    'inference_time': row['inference_time']
                })
            elif 'cspdarknet' in desc or 'default' in desc:
                backbone_data['cspdarknet'].append({
                    'mAP': row['mAP'],
                    'F1': row['F1'],
                    'inference_time': row['inference_time']
                })
        
        # Hitung rata-rata per backbone
        summary['backbone_comparison'] = {}
        for backbone, data in backbone_data.items():
            if data:
                summary['backbone_comparison'][backbone] = {
                    'mAP': sum(item['mAP'] for item in data) / len(data),
                    'F1': sum(item['F1'] for item in data) / len(data),
                    'inference_time': sum(item['inference_time'] for item in data) / len(data),
                    'count': len(data)
                }
    
    def _analyze_condition_performance(self, metrics_df: pd.DataFrame, summary: Dict):
        """Analisis performa kondisi pengujian."""
        condition_data = {'Posisi Bervariasi': [], 'Pencahayaan Bervariasi': []}
        
        for _, row in metrics_df.iterrows():
            desc = row['description'].lower()
            if 'posisi' in desc:
                condition_data['Posisi Bervariasi'].append({
                    'mAP': row['mAP'],
                    'F1': row['F1']
                })
            elif 'pencahayaan' in desc or 'lighting' in desc:
                condition_data['Pencahayaan Bervariasi'].append({
                    'mAP': row['mAP'],
                    'F1': row['F1']
                })
        
        # Hitung rata-rata per kondisi
        summary['condition_comparison'] = {}
        for condition, data in condition_data.items():
            if data:
                summary['condition_comparison'][condition] = {
                    'mAP': sum(item['mAP'] for item in data) / len(data),
                    'F1': sum(item['F1'] for item in data) / len(data),
                    'count': len(data)
                }