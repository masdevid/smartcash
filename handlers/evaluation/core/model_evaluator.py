# File: smartcash/handlers/evaluation/core/model_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen utama untuk evaluasi model

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.core.evaluation_component import EvaluationComponent
from smartcash.handlers.evaluation.integration.metrics_adapter import MetricsAdapter
from smartcash.handlers.evaluation.integration.model_manager_adapter import ModelManagerAdapter
from smartcash.handlers.evaluation.integration.dataset_adapter import DatasetAdapter

class ModelEvaluator(EvaluationComponent):
    """
    Komponen untuk evaluasi model dengan berbagai strategi.
    Melakukan proses evaluasi pada model dengan dataset yang diberikan.
    """
    
    def __init__(
        self,
        config: Dict,
        metrics_adapter: Optional[MetricsAdapter] = None,
        model_adapter: Optional[ModelManagerAdapter] = None,
        dataset_adapter: Optional[DatasetAdapter] = None,
        logger: Optional[SmartCashLogger] = None,
        name: str = "ModelEvaluator"
    ):
        """
        Inisialisasi model evaluator.
        
        Args:
            config: Konfigurasi evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator
            model_adapter: Adapter untuk ModelManager
            dataset_adapter: Adapter untuk DatasetManager
            logger: Logger kustom (opsional)
            name: Nama komponen
        """
        super().__init__(config, logger, name)
        
        # Inisialisasi adapter
        self.metrics_adapter = metrics_adapter or MetricsAdapter(config, logger)
        self.model_adapter = model_adapter or ModelManagerAdapter(config, logger)
        self.dataset_adapter = dataset_adapter or DatasetAdapter(config, logger)
    
    def process(
        self,
        model_path: str,
        dataset_path: str,
        observers: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses evaluasi model.
        
        Args:
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            observers: List observer (opsional)
            **kwargs: Parameter tambahan
                - batch_size: Ukuran batch
                - num_workers: Jumlah worker
                - device: Device untuk evaluasi ('cuda', 'cpu')
                - half_precision: Gunakan half precision
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Validasi input
        self.validate_inputs(model_path=model_path, dataset_path=dataset_path)
        
        # Notifikasi observers
        self.notify_observers(observers, 'evaluation_start', {
            'model_path': model_path,
            'dataset_path': dataset_path
        })
        
        try:
            # Ambil parameter dari kwargs
            batch_size = kwargs.get('batch_size')
            num_workers = kwargs.get('num_workers')
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            half_precision = kwargs.get('half_precision')
            
            # Ukur waktu
            start_time = time.time()
            
            # Load model
            self.logger.info(f"🔄 Loading model: {model_path}")
            model = self.model_adapter.load_model(
                model_path=model_path,
                device=device,
                backbone=kwargs.get('backbone')
            )
            
            # Prepare model for evaluation
            model = self.model_adapter.prepare_model_for_evaluation(
                model=model,
                half_precision=half_precision
            )
            
            # Verifikasi dataset
            self.logger.info(f"🔄 Verifikasi dataset: {dataset_path}")
            self.dataset_adapter.verify_dataset(dataset_path)
            
            # Load data
            self.logger.info(f"🔄 Loading dataset untuk evaluasi")
            dataloader = self.dataset_adapter.get_eval_loader(
                dataset_path=dataset_path,
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # Reset metrics
            self.metrics_adapter.reset()
            
            # Notifikasi observers
            self.notify_observers(observers, 'evaluation_dataloader_ready', {
                'num_batches': len(dataloader),
                'num_samples': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
            })
            
            # Jalankan evaluasi
            self.logger.info(f"🚀 Menjalankan evaluasi pada {len(dataloader)} batch")
            results = self.evaluate(
                model=model,
                dataloader=dataloader,
                device=device,
                observers=observers,
                **kwargs
            )
            
            # Hitung waktu total
            total_time = time.time() - start_time
            results['total_time'] = total_time
            
            # Log hasil
            self.metrics_adapter.log_metrics(results)
            
            # Tambahkan info model dan dataset
            results['model_info'] = self.model_adapter.get_model_info(model_path)
            results['dataset_info'] = self.dataset_adapter.get_dataset_info(dataset_path)
            
            # Notifikasi observers
            self.notify_observers(observers, 'evaluation_complete', results)
            
            self.logger.success(f"✅ Evaluasi selesai dalam {total_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            
            # Notifikasi observers
            self.notify_observers(observers, 'evaluation_error', {
                'error': str(e),
                'model_path': model_path,
                'dataset_path': dataset_path
            })
            
            raise
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        observers: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model dengan dataloader.
        
        Args:
            model: Model PyTorch
            dataloader: DataLoader untuk evaluasi
            device: Device untuk evaluasi
            observers: List observer (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Pastikan model dalam mode evaluasi
        model.eval()
        
        # Inisialisasi variabel tracking
        inference_times = []
        
        # Evaluasi batch-by-batch
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluasi")):
                # Pindahkan ke device yang sesuai
                images = images.to(device)
                targets = targets.to(device)
                
                # Notifikasi observers
                self.notify_observers(observers, 'batch_start', {
                    'batch_idx': batch_idx,
                    'batch_size': images.shape[0],
                    'total_batches': len(dataloader)
                })
                
                # Inferensi
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Update metrics
                self.metrics_adapter.update(predictions, targets)
                
                # Notifikasi observers
                self.notify_observers(observers, 'batch_complete', {
                    'batch_idx': batch_idx,
                    'inference_time': inference_time,
                    'batch_size': images.shape[0]
                })
                
                # Log progress setiap 10 batch
                if (batch_idx + 1) % 10 == 0:
                    avg_time = np.mean(inference_times[-10:])
                    fps = 1.0 / avg_time
                    self.logger.debug(f"📊 Batch {batch_idx + 1}/{len(dataloader)}, FPS: {fps:.2f}")
        
        # Hitung metrik final
        metrics = self.metrics_adapter.compute()
        
        # Update inference time
        if inference_times:
            metrics['inference_time'] = np.mean(inference_times)
            metrics['fps'] = 1.0 / metrics['inference_time']
        
        return metrics
    
    def validate_inputs(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Validasi input sebelum evaluasi.
        
        Args:
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            **kwargs: Parameter tambahan
            
        Returns:
            True jika valid
            
        Raises:
            ValueError: Jika validasi gagal
        """
        # Validasi model_path
        if model_path and not model_path.endswith(('.pt', '.pth')):
            raise ValueError(f"Format checkpoint tidak valid: {model_path}")
        
        # Validasi dataset_path
        if dataset_path:
            if not self.dataset_adapter.verify_dataset(dataset_path):
                raise ValueError(f"Dataset tidak valid: {dataset_path}")
        
        return True