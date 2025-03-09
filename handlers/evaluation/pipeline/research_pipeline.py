# File: smartcash/handlers/evaluation/pipeline/research_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline evaluasi skenario penelitian untuk perbandingan model yang dioptimasi

import time
import os
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from tqdm.auto import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from smartcash.handlers.evaluation.observers.progress_observer import ProgressObserver
from smartcash.handlers.evaluation.integration.visualization_adapter import VisualizationAdapter

class ResearchPipeline(BasePipeline):
    """
    Pipeline untuk evaluasi skenario penelitian.
    Evaluasi model dalam konteks perbandingan skenario penelitian dengan visualisasi hasil.
    """
    
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
        self.generate_plots = research_config.get('generate_plots', True)
        self.parallel_scenarios = research_config.get('parallel_scenarios', False)
        self.max_workers = research_config.get('max_workers', 2)
        
        # Setup direktori output
        self.output_dir = Path(self.config.get('output_dir', 'results/evaluation/research'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot dir
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization adapter
        self.visualization_adapter = VisualizationAdapter(
            config=self.config, 
            output_dir=str(self.plot_dir),
            logger=self.logger
        )
        
        # Cache untuk path yang sudah di-resolve
        self._path_cache = {'model': {}, 'dataset': {}}
        
        self.logger.debug(f"🔧 {name} diinisialisasi (num_runs={self.num_runs}, workers={self.max_workers})")
    
    def run(
        self,
        scenarios: Dict[str, Dict],
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi skenario penelitian.
        
        Args:
            scenarios: Dictionary skenario penelitian
                Format: {
                    'Skenario-1': {
                        'desc': 'Deskripsi skenario',
                        'model': 'path/ke/model.pt',
                        'data': 'path/ke/dataset'
                    },
                    ...
                }
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap skenario
        """
        # Ukur waktu
        start_time = time.time()
        
        # Notifikasi observer tentang start
        self.notify_observers('research_start', {
            'pipeline_name': self.name,
            'num_scenarios': len(scenarios),
            'start_time': start_time
        })
        
        self.logger.info(f"🔬 Memulai evaluasi {len(scenarios)} skenario penelitian")
        
        # Inisialisasi hasil
        results = {
            'pipeline_name': self.name,
            'num_scenarios': len(scenarios),
            'scenario_results': {},
            'summary': {},
            'plots': {}
        }
        
        try:
            # Pra-verifikasi dataset untuk mempercepat proses
            if dataset_adapter:
                self._prevalidate_datasets(scenarios, dataset_adapter)
            
            # Gunakan executor untuk memproses secara paralel jika dikonfigurasi
            if self.parallel_scenarios and self.max_workers > 1 and len(scenarios) > 1:
                scenario_results = self._run_scenarios_parallel(
                    scenarios, metrics_adapter, model_adapter, dataset_adapter, **kwargs
                )
            else:
                scenario_results = self._run_scenarios_sequential(
                    scenarios, metrics_adapter, model_adapter, dataset_adapter, **kwargs
                )
            
            # Simpan hasil
            results['scenario_results'] = scenario_results
            
            # Buat summary
            summary = self._create_summary(scenario_results)
            results['summary'] = summary
            
            # Buat visualisasi
            if self.generate_plots:
                plots = self.visualization_adapter.generate_research_plots(results)
                results['plots'] = plots
            
            # Hitung waktu total
            total_time = time.time() - start_time
            results['execution_time'] = total_time
            
            # Notifikasi observer tentang completion
            self.notify_observers('research_complete', {
                'pipeline_name': self.name,
                'execution_time': total_time,
                'num_scenarios': len(scenarios),
                'summary': summary
            })
            
            self.logger.success(f"✅ Evaluasi penelitian selesai dalam {total_time:.2f}s")
            return results
            
        except Exception as e:
            # Hitung waktu execution meskipun error
            total_time = time.time() - start_time
            
            # Notifikasi observer tentang error
            self.notify_observers('research_error', {
                'pipeline_name': self.name,
                'error': str(e),
                'execution_time': total_time
            })
            
            self.logger.error(f"❌ Evaluasi penelitian gagal: {str(e)}")
            raise
    
    def _run_scenarios_sequential(
        self,
        scenarios: Dict[str, Dict],
        metrics_adapter,
        model_adapter,
        dataset_adapter,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan skenario secara sekuensial.
        
        Args:
            scenarios: Dictionary skenario penelitian
            metrics_adapter: Adapter untuk metrik
            model_adapter: Adapter untuk model
            dataset_adapter: Adapter untuk dataset
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap skenario
        """
        scenario_results = {}
        
        for scenario_name, scenario_config in tqdm(scenarios.items(), desc="Evaluasi Skenario"):
            result = self._evaluate_scenario(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                metrics_adapter=metrics_adapter,
                model_adapter=model_adapter,
                dataset_adapter=dataset_adapter,
                **kwargs
            )
            scenario_results[scenario_name] = result
        
        return scenario_results
    
    def _run_scenarios_parallel(
        self,
        scenarios: Dict[str, Dict],
        metrics_adapter,
        model_adapter,
        dataset_adapter,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan skenario secara paralel menggunakan ThreadPoolExecutor.
        
        Args:
            scenarios: Dictionary skenario penelitian
            metrics_adapter: Adapter untuk metrik
            model_adapter: Adapter untuk model
            dataset_adapter: Adapter untuk dataset
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap skenario
        """
        scenario_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit semua job
            future_to_scenario = {
                executor.submit(
                    self._evaluate_scenario,
                    scenario_name=scenario_name,
                    scenario_config=scenario_config,
                    metrics_adapter=metrics_adapter,
                    model_adapter=model_adapter,
                    dataset_adapter=dataset_adapter,
                    **kwargs
                ): scenario_name
                for scenario_name, scenario_config in scenarios.items()
            }
            
            # Proses hasil seperti yang mereka selesai
            for future in tqdm(as_completed(future_to_scenario), total=len(scenarios), desc="Evaluasi Skenario"):
                scenario_name = future_to_scenario[future]
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
    
    def _evaluate_scenario(
        self,
        scenario_name: str,
        scenario_config: Dict,
        metrics_adapter,
        model_adapter,
        dataset_adapter,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi satu skenario.
        
        Args:
            scenario_name: Nama skenario
            scenario_config: Konfigurasi skenario
            metrics_adapter: Adapter untuk metrik
            model_adapter: Adapter untuk model
            dataset_adapter: Adapter untuk dataset
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi skenario
        """
        self.logger.info(f"🔄 Evaluasi skenario: {scenario_name} - {scenario_config['desc']}")
        
        # Ambil path model dan dataset
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
            metrics_adapter=metrics_adapter,
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
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
    
    def _prevalidate_datasets(self, scenarios: Dict[str, Dict], dataset_adapter):
        """
        Pra-validasi dataset untuk mempercepat proses evaluasi.
        
        Args:
            scenarios: Dictionary skenario penelitian
            dataset_adapter: Adapter untuk dataset
        """
        # Ambil semua dataset path unik
        dataset_paths = set()
        for scenario_config in scenarios.values():
            dataset_path = self._resolve_dataset_path(scenario_config['data'])
            if dataset_path:
                dataset_paths.add(dataset_path)
        
        # Validasi semua dataset sekaligus
        for dataset_path in dataset_paths:
            try:
                if os.path.exists(dataset_path):
                    dataset_adapter.verify_dataset(dataset_path)
                    self.logger.debug(f"✅ Dataset tervalidasi: {dataset_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Validasi dataset gagal: {dataset_path} - {str(e)}")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve path model relatif dengan caching.
        
        Args:
            model_path: Path model (relatif atau absolut)
            
        Returns:
            Path absolut ke model
        """
        # Cek cache
        if model_path in self._path_cache['model']:
            return self._path_cache['model'][model_path]
            
        # Cek apakah path absolut
        if os.path.isabs(model_path) and os.path.exists(model_path):
            self._path_cache['model'][model_path] = model_path
            return model_path
            
        # Coba cari di direktori checkpoint
        checkpoints_dir = Path(self.config.get('checkpoints_dir', 'runs/train/weights'))
        
        # Coba path relatif terhadap checkpoint dir
        checkpoint_path = checkpoints_dir / model_path
        if checkpoint_path.exists():
            self._path_cache['model'][model_path] = str(checkpoint_path)
            return str(checkpoint_path)
            
        # Coba cari file dengan nama yang cocok di checkpoint dir
        if checkpoints_dir.exists():
            matches = list(checkpoints_dir.glob(f"*{model_path}*"))
            if matches:
                self._path_cache['model'][model_path] = str(matches[0])
                return str(matches[0])
            
        # Return path asli jika tidak bisa di-resolve
        self._path_cache['model'][model_path] = model_path
        return model_path
    
    def _resolve_dataset_path(self, dataset_path: str) -> str:
        """
        Resolve path dataset relatif dengan caching.
        
        Args:
            dataset_path: Path dataset (relatif atau absolut)
            
        Returns:
            Path absolut ke dataset
        """
        # Cek cache
        if dataset_path in self._path_cache['dataset']:
            return self._path_cache['dataset'][dataset_path]
            
        # Cek apakah path absolut
        if os.path.isabs(dataset_path) and os.path.exists(dataset_path):
            self._path_cache['dataset'][dataset_path] = dataset_path
            return dataset_path
            
        # Coba cari di direktori data
        data_dir = Path(self.config.get('data_dir', 'data'))
        
        # Coba path relatif terhadap data dir
        dataset_full_path = data_dir / dataset_path
        if dataset_full_path.exists():
            self._path_cache['dataset'][dataset_path] = str(dataset_full_path)
            return str(dataset_full_path)
            
        # Coba cari direktori dengan nama yang cocok di data dir
        if data_dir.exists():
            matches = list(data_dir.glob(f"*{dataset_path}*"))
            if matches:
                self._path_cache['dataset'][dataset_path] = str(matches[0])
                return str(matches[0])
            
        # Return path asli jika tidak bisa di-resolve
        self._path_cache['dataset'][dataset_path] = dataset_path
        return dataset_path
    
    def _run_multiple_evaluations(
        self,
        scenario_name: str,
        model_path: str,
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi berulang kali untuk mendapatkan hasil yang stabil.
        
        Args:
            scenario_name: Nama skenario
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi rata-rata
        """
        # Inisialisasi pipeline evaluasi
        evaluation_pipeline = EvaluationPipeline(
            config=self.config,
            logger=self.logger.get_child(f"eval_{scenario_name}") if hasattr(self.logger, 'get_child') else self.logger,
            name=f"Eval_{scenario_name}"
        )
        
        # Tambahkan observer
        for observer in self.observers:
            evaluation_pipeline.add_observer(observer)
        
        # Inisialisasi hasil
        run_results = []
        all_metrics = {}
        
        # Jalankan evaluasi multiple kali
        for run in range(self.num_runs):
            self.logger.debug(f"🔄 Run {run+1}/{self.num_runs} untuk {scenario_name}")
            
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
                run_results.append(result)
                
                # Simpan metrics untuk perhitungan rata-rata
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
                # Hitung standar deviasi jika ada lebih dari 1 nilai
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
        """
        Buat summary dari hasil evaluasi skenario.
        
        Args:
            scenario_results: Dictionary hasil evaluasi untuk setiap skenario
            
        Returns:
            Dictionary berisi summary evaluasi
        """
        # Inisialisasi summary
        summary = {
            'num_scenarios': len(scenario_results),
            'successful_scenarios': 0,
            'failed_scenarios': 0,
            'best_scenario': None,
            'best_map': 0,
            'best_f1': 0,
            'metrics_table': None,
            'backbone_comparison': {},
            'condition_comparison': {}
        }
        
        # Ambil metrik dari setiap skenario
        metrics_list = []
        
        for scenario_name, result in scenario_results.items():
            # Cek apakah evaluasi berhasil
            if 'error' in result:
                summary['failed_scenarios'] += 1
                continue
                
            summary['successful_scenarios'] += 1
            
            # Ambil metrik
            if 'results' in result and 'avg_metrics' in result['results']:
                metrics = result['results']['avg_metrics']
                map_value = metrics.get('mAP', 0)
                f1_value = metrics.get('f1', 0)
                
                # Update best_scenario jika lebih baik
                if map_value > summary['best_map']:
                    summary['best_map'] = map_value
                    summary['best_scenario'] = scenario_name
                    
                # Simpan untuk tabel metrik
                metrics_entry = {
                    'scenario': scenario_name,
                    'description': result['config']['desc'],
                    'mAP': map_value,
                    'F1': f1_value,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'inference_time': metrics.get('inference_time', 0) * 1000  # ms
                }
                
                # Tambahkan std deviation jika tersedia
                if 'std_metrics' in result['results']:
                    std_metrics = result['results']['std_metrics']
                    metrics_entry.update({
                        'mAP_std': std_metrics.get('mAP', 0),
                        'F1_std': std_metrics.get('f1', 0),
                        'inference_time_std': std_metrics.get('inference_time', 0) * 1000  # ms
                    })
                
                metrics_list.append(metrics_entry)
        
        # Buat tabel metrik
        if metrics_list:
            try:
                metrics_df = pd.DataFrame(metrics_list)
                # Sort berdasarkan mAP (descending)
                metrics_df = metrics_df.sort_values('mAP', ascending=False)
                summary['metrics_table'] = metrics_df
                
                # Analisis backbone (efficientnet vs cspdarknet)
                self._analyze_backbone_performance(metrics_df, summary)
                
                # Analisis kondisi (posisi vs pencahayaan)
                self._analyze_condition_performance(metrics_df, summary)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat tabel metrik: {str(e)}")
        
        return summary
    
    def _analyze_backbone_performance(self, metrics_df: pd.DataFrame, summary: Dict):
        """
        Analisis performa berdasarkan backbone.
        
        Args:
            metrics_df: DataFrame metrik
            summary: Dictionary summary untuk diupdate
        """
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
        for backbone, data in backbone_data.items():
            if data:
                summary['backbone_comparison'][backbone] = {
                    'mAP': sum(item['mAP'] for item in data) / len(data),
                    'F1': sum(item['F1'] for item in data) / len(data),
                    'inference_time': sum(item['inference_time'] for item in data) / len(data),
                    'count': len(data)
                }
    
    def _analyze_condition_performance(self, metrics_df: pd.DataFrame, summary: Dict):
        """
        Analisis performa berdasarkan kondisi test.
        
        Args:
            metrics_df: DataFrame metrik
            summary: Dictionary summary untuk diupdate
        """
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
        for condition, data in condition_data.items():
            if data:
                summary['condition_comparison'][condition] = {
                    'mAP': sum(item['mAP'] for item in data) / len(data),
                    'F1': sum(item['F1'] for item in data) / len(data),
                    'count': len(data)
                }