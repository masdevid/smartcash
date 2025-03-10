# File: smartcash/handlers/evaluation/core/model_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen utama untuk evaluasi model yang disederhanakan

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.core.evaluation_component import EvaluationComponent
from smartcash.utils.metrics import MetricsCalculator
from smartcash.utils.observer.event_dispatcher import EventDispatcher

class ModelEvaluator(EvaluationComponent):
    """
    Komponen untuk evaluasi model dengan direct injection.
    Melakukan proses evaluasi pada model dengan dataset yang diberikan.
    """
    
    def __init__(
        self,
        config: Dict,
        metrics_calculator: Optional[MetricsCalculator] = None,
        logger: Optional[SmartCashLogger] = None,
        name: str = "ModelEvaluator"
    ):
        """
        Inisialisasi model evaluator.
        
        Args:
            config: Konfigurasi evaluasi
            metrics_calculator: Instance dari MetricsCalculator
            logger: Logger kustom (opsional)
            name: Nama komponen
        """
        super().__init__(config, logger, name)
        
        # Inisialisasi metrics_calculator
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

        # Parameter evaluasi dari config
        eval_config = self.config.get('evaluation', {})
        self.conf_threshold = eval_config.get('conf_threshold', 0.25)
        self.iou_threshold = eval_config.get('iou_threshold', 0.5)
        self.half_precision = eval_config.get('half_precision', False)
    
    def process(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses evaluasi model menggunakan direct injection.
        
        Args:
            model: Model PyTorch untuk evaluasi
            dataloader: DataLoader dengan dataset evaluasi
            device: Device untuk evaluasi ('cuda', 'cpu')
            **kwargs: Parameter tambahan
        
        Returns:
            Dictionary hasil evaluasi
        """
        # Default device jika tidak dispesifikasikan
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Pastikan model berada pada device yang benar dan dalam mode evaluasi
        model = model.to(device)
        model.eval()
        
        # Gunakan half precision jika diperlukan
        if self.half_precision and device == 'cuda':
            model = model.half()
            self.logger.info("🔄 Menggunakan half precision (FP16)")
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        # Notifikasi start
        self.notify('evaluation_start', {
            'num_batches': len(dataloader),
            'device': device
        })
        
        try:
            # Ukur waktu eksekusi
            start_time = time.time()
            
            # Evaluasi model
            results = self._evaluate(model, dataloader, device, **kwargs)
            
            # Hitung waktu total
            total_time = time.time() - start_time
            results['total_time'] = total_time
            
            # Notifikasi complete
            self.notify('evaluation_complete', results)
            
            self.logger.success(f"✅ Evaluasi selesai dalam {total_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            
            # Notifikasi error
            self.notify('evaluation_error', {
                'error': str(e)
            })
            
            raise
    
    def _evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model dengan dataloader.
        
        Args:
            model: Model PyTorch
            dataloader: DataLoader untuk evaluasi
            device: Device untuk evaluasi
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Inisialisasi tracking waktu
        inference_times = []
        
        # Progress bar dengan tqdm
        progress_bar = tqdm(dataloader, desc="Evaluasi", unit="batch")
        
        # Evaluasi batch-by-batch dengan no_grad untuk efisiensi memori
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Proses batch sesuai formatnya
                if isinstance(batch, dict) and 'image' in batch and 'targets' in batch:
                    # Format multilayer dataset
                    images = batch['image'].to(device)
                    targets = batch['targets']
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Format standar (images, targets)
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device) if isinstance(targets, torch.Tensor) else targets
                else:
                    self.logger.warning(f"⚠️ Format batch tidak didukung: {type(batch)}")
                    continue
                
                # Notifikasi batch start
                self.notify('batch_start', {
                    'batch_idx': batch_idx,
                    'batch_size': images.shape[0] if isinstance(images, torch.Tensor) else 0,
                })
                
                # Inferensi dengan pengukuran waktu
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Update metrics dengan hasil prediksi
                self.metrics_calculator.update(
                    predictions, 
                    targets,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold
                )
                
                # Update progress bar dengan informasi FPS
                if inference_times:
                    avg_time = np.mean(inference_times[-10:]) if len(inference_times) >= 10 else np.mean(inference_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    progress_bar.set_postfix({
                        'FPS': f'{fps:.1f}',
                        'Infer': f'{avg_time*1000:.1f}ms'
                    })
                
                # Notifikasi batch complete
                self.notify('batch_complete', {
                    'batch_idx': batch_idx,
                    'inference_time': inference_time
                })
        
        # Hitung metrik final
        metrics = self.metrics_calculator.compute()
        
        # Tambahkan informasi waktu inferensi
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            metrics['inference_time'] = avg_inference_time
            metrics['fps'] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            self.logger.info(f"⏱️ Waktu inferensi rata-rata: {avg_inference_time*1000:.2f}ms ({metrics['fps']:.1f} FPS)")
        
        return {
            'metrics': metrics,
            'inference_times': inference_times,
            'num_samples': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader) * dataloader.batch_size
        }
    
    def notify(self, event: str, data: Dict[str, Any] = None):
        """
        Notify observers menggunakan EventDispatcher.
        
        Args:
            event: Nama event
            data: Data event (opsional)
        """
        EventDispatcher.notify(f"evaluation.{event}", self, data or {})