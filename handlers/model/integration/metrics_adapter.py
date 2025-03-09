# File: smartcash/handlers/model/integration/metrics_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan MetricsCalculator

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List

from smartcash.utils.logger import get_logger, SmartCashLogger

class MetricsAdapter:
    """
    Adapter untuk integrasi dengan MetricsCalculator dari utils.
    Menyediakan antarmuka yang konsisten untuk perhitungan metrik.
    """
    
    def __init__(
        self,
        logger: Optional[SmartCashLogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inisialisasi metrics adapter.
        
        Args:
            logger: Custom logger (opsional)
            config: Konfigurasi untuk metrics (opsional)
        """
        self.logger = logger or get_logger("metrics_adapter")
        self.config = config or {}
        
        # Metrik configuration
        self.metrics_config = self.config.get('evaluation', {}).get('metrics', {})
        
        # Lazy-initialized metrics calculator
        self._metrics_calculator = None
    
    @property
    def metrics_calculator(self):
        """Lazy initialization of metrics calculator."""
        if self._metrics_calculator is None:
            try:
                # Import MetricsCalculator
                from smartcash.utils.metrics import MetricsCalculator
                
                # Setup konfigurasi metrics calculator
                conf_threshold = self.metrics_config.get('conf_threshold', 0.25)
                iou_threshold = self.metrics_config.get('iou_threshold', 0.5)
                
                # Buat metrics calculator
                self._metrics_calculator = MetricsCalculator(
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                self.logger.info(
                    f"📊 MetricsCalculator diinisialisasi:\n"
                    f"   • Confidence threshold: {conf_threshold}\n"
                    f"   • IoU threshold: {iou_threshold}"
                )
                
            except ImportError:
                self.logger.error("❌ MetricsCalculator tidak ditemukan! Pastikan utils.metrics terimplementasi")
                # Fallback ke calculator dummy
                self._metrics_calculator = self._DummyMetricsCalculator()
                
        return self._metrics_calculator
    
    def reset(self) -> None:
        """Reset metrics calculator untuk perhitungan baru."""
        self.metrics_calculator.reset()
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray, List, Dict], 
        targets: Union[torch.Tensor, np.ndarray, List, Dict]
    ) -> None:
        """
        Update metrics dengan batch prediksi dan target.
        
        Args:
            predictions: Prediksi dari model
            targets: Target ground truth
        """
        self.metrics_calculator.update(predictions, targets)
    
    def compute(self) -> Dict[str, Any]:
        """
        Hitung metrik berdasarkan semua batch yang telah diupdate.
        
        Returns:
            Dictionary berisi metrik evaluasi
        """
        metrics = self.metrics_calculator.compute()
        
        # Log rangkuman metrik
        if metrics:
            self.logger.info(
                f"📊 Metrik evaluasi:\n"
                f"   • mAP: {metrics.get('mAP', 0):.4f}\n"
                f"   • Precision: {metrics.get('precision', 0):.4f}\n"
                f"   • Recall: {metrics.get('recall', 0):.4f}\n"
                f"   • F1 Score: {metrics.get('f1', 0):.4f}"
            )
        
        return metrics
    
    def compute_batch_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List, Dict],
        targets: Union[torch.Tensor, np.ndarray, List, Dict]
    ) -> Dict[str, Any]:
        """
        Hitung metrik untuk satu batch.
        
        Args:
            predictions: Prediksi dari model
            targets: Target ground truth
            
        Returns:
            Dictionary berisi metrik batch
        """
        # Reset, update dengan batch ini, lalu compute
        self.reset()
        self.update(predictions, targets)
        return self.compute()
    
    def measure_inference_time(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
        warm_up: int = 3
    ) -> Dict[str, float]:
        """
        Ukur waktu inferensi model.
        
        Args:
            model: Model PyTorch
            input_tensor: Tensor input
            num_runs: Jumlah pengukuran
            warm_up: Jumlah warm up runs
            
        Returns:
            Dictionary berisi statistik waktu inferensi
        """
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor)
            
        # Pastikan dalam mode eval
        model.eval()
        
        # Pindahkan ke device yang sama dengan model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(warm_up):
                _ = model(input_tensor)
        
        # Pengukuran
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    _ = model(input_tensor)
                    end.record()
                    
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1000)  # ms to seconds
                else:
                    import time
                    start = time.time()
                    _ = model(input_tensor)
                    times.append(time.time() - start)
        
        # Hitung statistik
        times = np.array(times)
        stats = {
            'inference_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times))
        }
        
        self.logger.info(
            f"⏱️ Pengukuran inferensi ({num_runs} runs):\n"
            f"   • Mean: {stats['inference_time']*1000:.2f} ms\n"
            f"   • FPS: {stats['fps']:.2f}\n"
            f"   • Min: {stats['min_time']*1000:.2f} ms\n"
            f"   • Max: {stats['max_time']*1000:.2f} ms"
        )
        
        return stats
    
    class _DummyMetricsCalculator:
        """Fallback calculator jika MetricsCalculator tidak tersedia."""
        
        def reset(self) -> None:
            """Reset metrics calculator."""
            pass
        
        def update(self, predictions, targets) -> None:
            """Update metrics calculator."""
            pass
        
        def compute(self) -> Dict[str, Any]:
            """Compute metrics."""
            return {
                'note': 'Metrics unavailable - using dummy calculator',
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }