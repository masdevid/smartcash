"""
File: smartcash/model/evaluation/utils/inference_timer.py
Deskripsi: Timing measurements untuk inference performance evaluation
"""

import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from contextlib import contextmanager

from smartcash.common.logger import get_logger

class InferenceTimer:
    """Timer untuk measuring inference performance dengan detailed statistics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Ensure config is a dictionary
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        else:
            self.logger = get_logger('inference_timer')
            self.logger.warning(f"Config is not a dictionary (got {type(config).__name__}), converting to dict")
            self.config = {"evaluation": {"metrics": {"inference_time": config}}} if config is not None else {}
        
        if not hasattr(self, 'logger'):
            self.logger = get_logger('inference_timer')
            
        # Safely access nested config with proper defaults
        try:
            if not isinstance(self.config, dict):
                raise ValueError(f"Expected dict for config, got {type(self.config).__name__}")
                
            self.timing_config = {}
            if 'evaluation' in self.config and isinstance(self.config['evaluation'], dict):
                metrics = self.config['evaluation'].get('metrics', {})
                if isinstance(metrics, dict):
                    self.timing_config = metrics.get('inference_time', {})
        except Exception as e:
            self.logger.warning(f"Error processing timing config: {e}")
            self.timing_config = {}
        
        # Timing storage
        self.timings = defaultdict(list)
        self.batch_timings = defaultdict(list)
        self.current_timer = None
        
        # GPU memory tracking
        self.gpu_available = torch.cuda.is_available()
        
    @contextmanager
    def time_inference(self, batch_size: int = 1, operation: str = 'inference'):
        """🕒 Context manager untuk timing inference operations"""
        start_time = time.perf_counter()
        
        # GPU synchronization untuk accurate timing
        if self.gpu_available:
            torch.cuda.synchronize()
        
        memory_before = self._get_memory_usage()
        
        try:
            yield
        finally:
            # Ensure GPU operations complete
            if self.gpu_available:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before if memory_after and memory_before else 0
            
            # Store timing
            self.record_inference(inference_time, batch_size, operation, memory_used)
    
    def start_timer(self, operation: str = 'inference') -> None:
        """▶️ Start timing operation"""
        if self.gpu_available:
            torch.cuda.synchronize()
        
        self.current_timer = {
            'operation': operation,
            'start_time': time.perf_counter(),
            'memory_before': self._get_memory_usage()
        }
    
    def end_timer(self, batch_size: int = 1) -> float:
        """⏹️ End timing dan return elapsed time"""
        if not self.current_timer:
            self.logger.warning("⚠️ Timer belum dimulai")
            return 0.0
        
        if self.gpu_available:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_time = end_time - self.current_timer['start_time']
        
        memory_after = self._get_memory_usage()
        memory_used = (memory_after - self.current_timer['memory_before'] 
                      if memory_after and self.current_timer['memory_before'] else 0)
        
        # Record timing
        self.record_inference(
            inference_time, 
            batch_size, 
            self.current_timer['operation'], 
            memory_used
        )
        
        self.current_timer = None
        return inference_time
    
    def record_inference(self, inference_time: float, batch_size: int = 1, 
                        operation: str = 'inference', memory_used: float = 0) -> None:
        """📝 Record single inference timing"""
        timing_record = {
            'time': inference_time,
            'batch_size': batch_size,
            'per_image_time': inference_time / batch_size if batch_size > 0 else inference_time,
            'fps': batch_size / inference_time if inference_time > 0 else 0,
            'memory_used_mb': memory_used,
            'timestamp': time.time()
        }
        
        self.timings[operation].append(timing_record)
        self.batch_timings[batch_size].append(timing_record)
        
        self.logger.debug(f"⏱️ {operation}: {inference_time:.3f}s (batch={batch_size}, fps={timing_record['fps']:.1f})")
    
    def warmup_model(self, model, input_tensor: torch.Tensor, warmup_runs: int = None) -> Dict[str, Any]:
        """🔥 Warmup model untuk stable timing"""
        if warmup_runs is None:
            warmup_runs = self.timing_config.get('warmup_runs', 10)
        
        self.logger.info(f"🔥 Warming up model dengan {warmup_runs} runs")
        
        warmup_times = []
        
        with torch.no_grad():
            for i in range(warmup_runs):
                with self.time_inference(batch_size=input_tensor.shape[0], operation='warmup'):
                    _ = model(input_tensor)
                
                # Record warmup time
                if self.timings['warmup']:
                    warmup_times.append(self.timings['warmup'][-1]['time'])
        
        warmup_result = {
            'warmup_runs': warmup_runs,
            'avg_warmup_time': np.mean(warmup_times) if warmup_times else 0,
            'warmup_stable': self._is_timing_stable(warmup_times[-5:]) if len(warmup_times) >= 5 else False
        }
        
        self.logger.info(f"✅ Warmup complete: {warmup_result['avg_warmup_time']:.3f}s avg")
        return warmup_result
    
    def benchmark_inference(self, model, input_tensor: torch.Tensor, 
                           measurement_runs: int = None, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """📊 Comprehensive inference benchmarking"""
        if measurement_runs is None:
            measurement_runs = self.timing_config.get('measurement_runs', 100)
        if batch_sizes is None:
            batch_sizes = self.timing_config.get('batch_sizes', [1, 4, 8, 16])
        
        self.logger.info(f"📊 Benchmarking inference: {measurement_runs} runs, batch sizes: {batch_sizes}")
        
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            # Adjust input tensor untuk batch size
            if input_tensor.shape[0] != batch_size:
                # Repeat atau slice tensor untuk match batch size
                if batch_size > input_tensor.shape[0]:
                    repeats = (batch_size + input_tensor.shape[0] - 1) // input_tensor.shape[0]
                    batch_tensor = input_tensor.repeat(repeats, 1, 1, 1)[:batch_size]
                else:
                    batch_tensor = input_tensor[:batch_size]
            else:
                batch_tensor = input_tensor
            
            # Warmup untuk batch size ini
            self.warmup_model(model, batch_tensor, warmup_runs=5)
            
            # Measurement runs
            batch_times = []
            
            with torch.no_grad():
                for run in range(measurement_runs):
                    with self.time_inference(batch_size=batch_size, operation=f'benchmark_bs{batch_size}'):
                        _ = model(batch_tensor)
                    
                    if self.timings[f'benchmark_bs{batch_size}']:
                        batch_times.append(self.timings[f'benchmark_bs{batch_size}'][-1]['time'])
            
            # Calculate statistics untuk batch size ini
            if batch_times:
                benchmark_results[f'batch_size_{batch_size}'] = {
                    'batch_size': batch_size,
                    'runs': len(batch_times),
                    'avg_time': np.mean(batch_times),
                    'median_time': np.median(batch_times),
                    'min_time': np.min(batch_times),
                    'max_time': np.max(batch_times),
                    'std_time': np.std(batch_times),
                    'avg_per_image': np.mean(batch_times) / batch_size,
                    'fps': batch_size / np.mean(batch_times),
                    'throughput': measurement_runs * batch_size / np.sum(batch_times)
                }
        
        # Overall benchmark summary
        benchmark_results['summary'] = self._create_benchmark_summary(benchmark_results)
        
        self.logger.info(f"📊 Benchmark complete: {len(batch_sizes)} batch sizes tested")
        return benchmark_results
    
    def get_average_time(self, operation: str = 'inference') -> float:
        """⏱️ Get average inference time untuk operation"""
        if operation not in self.timings or not self.timings[operation]:
            return 0.0
        
        times = [record['time'] for record in self.timings[operation]]
        return np.mean(times)
    
    def get_timing_stats(self, operation: str = 'inference') -> Dict[str, Any]:
        """📈 Get comprehensive timing statistics"""
        if operation not in self.timings or not self.timings[operation]:
            return {'operation': operation, 'samples': 0}
        
        records = self.timings[operation]
        times = [record['time'] for record in records]
        per_image_times = [record['per_image_time'] for record in records]
        fps_values = [record['fps'] for record in records]
        memory_usage = [record['memory_used_mb'] for record in records if record['memory_used_mb'] > 0]
        
        stats = {
            'operation': operation,
            'samples': len(records),
            'total_time': np.sum(times),
            'avg_time': np.mean(times),
            'median_time': np.median(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_per_image': np.mean(per_image_times),
            'avg_fps': np.mean(fps_values),
            'max_fps': np.max(fps_values),
            'timing_stability': self._calculate_stability(times)
        }
        
        # Memory statistics jika tersedia
        if memory_usage:
            stats.update({
                'avg_memory_mb': np.mean(memory_usage),
                'max_memory_mb': np.max(memory_usage),
                'total_memory_mb': np.sum(memory_usage)
            })
        
        # Batch size analysis
        batch_analysis = defaultdict(list)
        for record in records:
            batch_analysis[record['batch_size']].append(record['time'])
        
        stats['batch_analysis'] = {
            str(bs): {
                'samples': len(times_list),
                'avg_time': np.mean(times_list),
                'avg_fps': bs / np.mean(times_list) if np.mean(times_list) > 0 else 0
            }
            for bs, times_list in batch_analysis.items()
        }
        
        return stats
    
    def export_timing_data(self, operation: str = None) -> Dict[str, Any]:
        """📤 Export timing data untuk analysis"""
        if operation:
            operations = [operation] if operation in self.timings else []
        else:
            operations = list(self.timings.keys())
        
        export_data = {
            'export_timestamp': time.time(),
            'gpu_available': self.gpu_available,
            'operations': {}
        }
        
        for op in operations:
            export_data['operations'][op] = {
                'records': self.timings[op],
                'statistics': self.get_timing_stats(op)
            }
        
        return export_data
    
    def reset_timings(self, operation: str = None) -> None:
        """🔄 Reset timing data"""
        if operation:
            if operation in self.timings:
                self.timings[operation].clear()
                self.logger.info(f"🔄 Reset timing data untuk {operation}")
        else:
            self.timings.clear()
            self.batch_timings.clear()
            self.logger.info("🔄 Reset semua timing data")
    
    def _get_memory_usage(self) -> Optional[float]:
        """💾 Get current GPU memory usage dalam MB"""
        if not self.gpu_available:
            return None
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        except Exception:
            return None
    
    def _is_timing_stable(self, times: List[float], threshold: float = 0.1) -> bool:
        """📊 Check if timing is stable (low variance)"""
        if len(times) < 3:
            return False
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 1.0
        
        return cv < threshold
    
    def _calculate_stability(self, times: List[float]) -> Dict[str, float]:
        """📊 Calculate timing stability metrics"""
        if len(times) < 2:
            return {'coefficient_of_variation': 1.0, 'stability_score': 0.0}
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 1.0
        
        # Stability score (0-1, higher is more stable)
        stability_score = max(0, 1 - cv)
        
        return {
            'coefficient_of_variation': cv,
            'stability_score': stability_score,
            'is_stable': cv < 0.1
        }
    
    def _create_benchmark_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """📋 Create benchmark summary dari batch results"""
        batch_results = [v for k, v in benchmark_results.items() if k.startswith('batch_size_')]
        
        if not batch_results:
            return {}
        
        return {
            'best_fps_batch': max(batch_results, key=lambda x: x['fps']),
            'best_latency_batch': min(batch_results, key=lambda x: x['avg_per_image']),
            'most_stable_batch': min(batch_results, key=lambda x: x['std_time']),
            'throughput_comparison': {
                f"batch_{r['batch_size']}": r['throughput'] for r in batch_results
            },
            'recommended_batch_size': self._recommend_batch_size(batch_results)
        }
    
    def _recommend_batch_size(self, batch_results: List[Dict[str, Any]]) -> int:
        """🎯 Recommend optimal batch size berdasarkan throughput dan stability"""
        if not batch_results:
            return 1
        
        # Score berdasarkan throughput dan stability
        best_batch = max(batch_results, key=lambda x: x['throughput'] * (1 - x['std_time'] / x['avg_time']))
        return best_batch['batch_size']


# Factory functions
def create_inference_timer(config: Dict[str, Any] = None) -> InferenceTimer:
    """🏭 Factory untuk InferenceTimer"""
    return InferenceTimer(config)

def benchmark_model_inference(model, input_tensor: torch.Tensor, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 One-liner untuk benchmark model inference"""
    timer = create_inference_timer(config)
    return timer.benchmark_inference(model, input_tensor)