"""
File: smartcash/model/service/metrics_tracker.py
Deskripsi: Implementasi metrics tracker untuk training model
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from collections import defaultdict, deque
from smartcash.common.logger import get_logger
from smartcash.model.service.callback_interfaces import MetricsCallback, MetricsCallbackFn

class MetricsTracker:
    """Metrics tracker untuk training model dengan dukungan callback UI"""
    
    def __init__(self, callback: Optional[Union[MetricsCallback, MetricsCallbackFn, Dict[str, Callable]]] = None, history_size: int = 100):
        self.logger = get_logger(__name__)
        self._callback = callback
        self._metrics_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=history_size)))
        self._current_metrics = defaultdict(dict)
        self._best_metrics = defaultdict(dict)
        self._metrics_improvement = defaultdict(dict)
        self._start_time = time.time()
        self._epoch_times = []
        self._batch_times = deque(maxlen=100)
        self._inference_times = deque(maxlen=100)  # Untuk tracking waktu inferensi
        self._last_time = time.time()
        
    def set_callback(self, callback: Union[MetricsCallback, MetricsCallbackFn, Dict[str, Callable]]) -> None:
        """Set callback untuk metrics tracking"""
        self._callback = callback
        self.logger.debug(f"🔄 Metrics callback diatur: {type(callback).__name__}")
        
    def update(self, metrics: Dict[str, float], phase: str = "train") -> None:
        """Update metrics untuk fase tertentu (train/val)"""
        # Update current metrics
        self._current_metrics[phase].update(metrics)
        
        # Update history
        for metric_name, value in metrics.items():
            self._metrics_history[phase][metric_name].append(value)
        
        # Update best metrics dan improvement
        for metric_name, value in metrics.items():
            # Inisialisasi best metric jika belum ada
            if metric_name not in self._best_metrics[phase]:
                self._best_metrics[phase][metric_name] = value
                self._metrics_improvement[phase][metric_name] = 0.0
                continue
                
            # Check apakah ini adalah metric yang lebih baik (lower is better untuk loss, higher untuk lainnya)
            is_better = False
            if metric_name.lower().endswith('loss'):
                is_better = value < self._best_metrics[phase][metric_name]
            else:
                is_better = value > self._best_metrics[phase][metric_name]
                
            # Update best dan improvement jika lebih baik
            if is_better:
                old_value = self._best_metrics[phase][metric_name]
                self._best_metrics[phase][metric_name] = value
                
                # Hitung improvement (persentase)
                if old_value != 0:
                    if metric_name.lower().endswith('loss'):
                        self._metrics_improvement[phase][metric_name] = ((old_value - value) / old_value) * 100
                    else:
                        self._metrics_improvement[phase][metric_name] = ((value - old_value) / old_value) * 100
        
        # Update timing metrics
        current_time = time.time()
        time_diff = current_time - self._last_time
        self._last_time = current_time
        
        if phase == "train_batch":
            self._batch_times.append(time_diff)
        elif phase == "train_epoch":
            self._epoch_times.append(time_diff)
        
        # Log metrics pada interval yang sesuai
        self._log_metrics(metrics, phase)
        
        # Panggil callback jika tersedia
        self._call_metrics_callback(metrics, phase)
        
    def update_learning_rate(self, lr: float) -> None:
        """Update learning rate saat ini"""
        self._current_metrics["train"]["learning_rate"] = lr
        self._metrics_history["train"]["learning_rate"].append(lr)
        
        # Log learning rate
        self.logger.info(f"📉 Learning rate: {lr:.6f}")
        
        # Panggil callback jika tersedia
        self._call_lr_callback(lr)
        
    def update_loss_breakdown(self, loss_components: Dict[str, float]) -> None:
        """Update breakdown komponen loss"""
        # Update current metrics dengan prefix "loss_"
        for name, value in loss_components.items():
            metric_name = f"loss_{name}" if not name.startswith("loss_") else name
            self._current_metrics["train"][metric_name] = value
            self._metrics_history["train"][metric_name].append(value)
        
        # Log loss breakdown
        components_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
        self.logger.debug(f"📊 Loss breakdown: {components_str}")
        
        # Panggil callback jika tersedia
        self._call_loss_breakdown_callback(loss_components)
        
    def update_prediction_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Update sample prediksi untuk visualisasi"""
        # Panggil callback jika tersedia
        self._call_prediction_samples_callback(samples)
        
    def update_inference_time(self, inference_time: float) -> None:
        """Update waktu inferensi (dalam detik)"""
        self._inference_times.append(inference_time)
        self._current_metrics["inference"]["time"] = inference_time
        self._metrics_history["inference"]["time"].append(inference_time)
        
        # Log inference time
        self.logger.debug(f"⏱️ Waktu inferensi: {inference_time:.6f}s")
        
        # Panggil callback jika tersedia
        self._call_inference_time_callback(inference_time)
        
    def get_current_metrics(self, phase: str = "train") -> Dict[str, float]:
        """Dapatkan metrics saat ini untuk fase tertentu"""
        return dict(self._current_metrics.get(phase, {}))
        
    def get_best_metrics(self, phase: str = "train") -> Dict[str, float]:
        """Dapatkan best metrics untuk fase tertentu"""
        return dict(self._best_metrics.get(phase, {}))
        
    def get_metrics_improvement(self, phase: str = "train") -> Dict[str, float]:
        """Dapatkan persentase improvement metrics untuk fase tertentu"""
        return dict(self._metrics_improvement.get(phase, {}))
        
    def get_metrics_history(self, phase: str = "train", metric_name: Optional[str] = None) -> Union[Dict[str, List[float]], List[float]]:
        """Dapatkan history metrics untuk fase dan metric tertentu"""
        if metric_name:
            return list(self._metrics_history.get(phase, {}).get(metric_name, []))
        return {k: list(v) for k, v in self._metrics_history.get(phase, {}).items()}
        
    def get_average_batch_time(self) -> float:
        """Dapatkan rata-rata waktu per batch"""
        return np.mean(self._batch_times) if self._batch_times else 0.0
        
    def get_average_inference_time(self) -> float:
        """Dapatkan rata-rata waktu inferensi"""
        return np.mean(self._inference_times) if self._inference_times else 0.0
        
    def get_average_epoch_time(self) -> float:
        """Dapatkan rata-rata waktu per epoch"""
        return np.mean(self._epoch_times) if self._epoch_times else 0.0
        
    def get_estimated_epoch_time(self) -> float:
        """Estimasi waktu untuk menyelesaikan satu epoch berdasarkan batch time"""
        return self.get_average_batch_time() * 100  # Asumsi 100 batch per epoch
        
    def get_estimated_remaining_time(self, current_epoch: int, total_epochs: int) -> float:
        """Estimasi waktu tersisa untuk menyelesaikan training"""
        avg_epoch_time = self.get_average_epoch_time()
        if avg_epoch_time <= 0: return 0.0
        return avg_epoch_time * (total_epochs - current_epoch)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Dapatkan ringkasan metrics untuk semua fase"""
        return {
            "current": dict(self._current_metrics),
            "best": dict(self._best_metrics),
            "improvement": dict(self._metrics_improvement),
            "timing": {
                "total_time": time.time() - self._start_time,
                "avg_batch_time": self.get_average_batch_time(),
                "avg_epoch_time": self.get_average_epoch_time(),
                "avg_inference_time": self.get_average_inference_time(),
            }
        }
        
    def reset(self) -> None:
        """Reset metrics tracker untuk digunakan kembali"""
        self._metrics_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self._metrics_history["train"]["loss"].maxlen if "train" in self._metrics_history and "loss" in self._metrics_history["train"] else 100)))
        self._current_metrics = defaultdict(dict)
        self._best_metrics = defaultdict(dict)
        self._metrics_improvement = defaultdict(dict)
        self._start_time = time.time()
        self._epoch_times = []
        self._batch_times = deque(maxlen=100)
        self._inference_times = deque(maxlen=100)
        self._last_time = time.time()
        self.logger.debug("🔄 Metrics tracker direset")
        
    def is_best_metric(self, metric_name: str, value: float, phase: str = "val") -> bool:
        """Check apakah nilai metric ini adalah yang terbaik sejauh ini"""
        if metric_name not in self._best_metrics[phase]: return True
        
        if metric_name.lower().endswith('loss'):
            return value < self._best_metrics[phase][metric_name]
        else:
            return value > self._best_metrics[phase][metric_name]
    
    # Helper methods untuk memanggil callback dengan berbagai format
    def _call_metrics_callback(self, metrics: Dict[str, float], phase: str) -> None:
        """Panggil callback metrics dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah MetricsCallback interface
            if hasattr(self._callback, 'update_metrics'):
                self._callback.update_metrics(metrics, phase)
                return
                
            # Jika callback adalah dict dengan fungsi metrics
            if isinstance(self._callback, dict) and 'metrics' in self._callback:
                self._callback['metrics'](metrics, phase)
                return
                
            # Jika callback adalah fungsi
            if callable(self._callback):
                self._callback(event="metrics", metrics=metrics, phase=phase)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil metrics callback: {str(e)}")
    
    def _call_lr_callback(self, lr: float) -> None:
        """Panggil callback learning rate dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah MetricsCallback interface
            if hasattr(self._callback, 'update_learning_rate'):
                self._callback.update_learning_rate(lr)
                return
                
            # Jika callback adalah dict dengan fungsi lr
            if isinstance(self._callback, dict) and 'learning_rate' in self._callback:
                self._callback['learning_rate'](lr)
                return
                
            # Jika callback adalah fungsi
            if callable(self._callback):
                self._callback(event="learning_rate", lr=lr)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil learning rate callback: {str(e)}")
    
    def _call_loss_breakdown_callback(self, loss_components: Dict[str, float]) -> None:
        """Panggil callback loss breakdown dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah MetricsCallback interface
            if hasattr(self._callback, 'update_loss_breakdown'):
                self._callback.update_loss_breakdown(loss_components)
                return
                
            # Jika callback adalah dict dengan fungsi loss_breakdown
            if isinstance(self._callback, dict) and 'loss_breakdown' in self._callback:
                self._callback['loss_breakdown'](loss_components)
                return
                
            # Jika callback adalah fungsi
            if callable(self._callback):
                self._callback(event="loss_breakdown", components=loss_components)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil loss breakdown callback: {str(e)}")
    
    def _call_prediction_samples_callback(self, samples: List[Dict[str, Any]]) -> None:
        """Panggil callback prediction samples dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah MetricsCallback interface
            if hasattr(self._callback, 'update_prediction_samples'):
                self._callback.update_prediction_samples(samples)
                return
                
            # Jika callback adalah dict dengan fungsi prediction_samples
            if isinstance(self._callback, dict) and 'prediction_samples' in self._callback:
                self._callback['prediction_samples'](samples)
                return
                
            # Jika callback adalah fungsi
            if callable(self._callback):
                self._callback(event='prediction_samples', samples=samples)
                return
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil prediction samples callback: {str(e)}")
            
    def _call_inference_time_callback(self, inference_time: float) -> None:
        """Panggil callback inference time dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah MetricsCallback interface
            if hasattr(self._callback, 'update_inference_time'):
                self._callback.update_inference_time(inference_time)
                return
                
            # Jika callback adalah dict dengan fungsi inference_time
            if isinstance(self._callback, dict) and 'inference_time' in self._callback:
                self._callback['inference_time'](inference_time)
                return
                
            # Jika callback adalah fungsi
            if callable(self._callback):
                self._callback(event='inference_time', time=inference_time)
                return
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil inference time callback: {str(e)}")
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        """Log metrics dengan format yang sesuai"""
        # Skip logging untuk batch metrics kecuali debug
        if phase == "train_batch":
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.debug(f"📊 Batch metrics [{phase}]: {metrics_str}")
            return
            
        # Log metrics untuk epoch
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Tambahkan improvement untuk metrics penting
        improvement_str = ""
        for k, v in metrics.items():
            if k in self._metrics_improvement[phase] and abs(self._metrics_improvement[phase][k]) > 0.1:
                sign = "+" if self._metrics_improvement[phase][k] > 0 else ""
                improvement_str += f" {k}: {sign}{self._metrics_improvement[phase][k]:.1f}%"
                
        if improvement_str:
            self.logger.info(f"📈 Metrics [{phase}]: {metrics_str} (Improvement:{improvement_str})")
        else:
            self.logger.info(f"📊 Metrics [{phase}]: {metrics_str}")
            
    # Properties untuk akses mudah
    @property
    def current_loss(self) -> float: return self._current_metrics.get("train", {}).get("loss", 0.0)
    @property
    def best_val_loss(self) -> float: return self._best_metrics.get("val", {}).get("loss", float('inf'))
    @property
    def best_val_map(self) -> float: return self._best_metrics.get("val", {}).get("mAP", 0.0)
    @property
    def total_time(self) -> float: return time.time() - self._start_time
