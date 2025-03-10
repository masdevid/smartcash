# File: smartcash/handlers/evaluation/pipeline/evaluation_pipeline.py
# Author: Alfrida Sabar
# Deskripsi: Pipeline evaluasi model yang disederhanakan dengan direct injection

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.metrics import MetricsCalculator
from smartcash.utils.observer.event_dispatcher import EventDispatcher
from smartcash.handlers.evaluation.pipeline.base_pipeline import BasePipeline
from smartcash.models.yolov5_model import YOLOv5Model

class EvaluationPipeline(BasePipeline):
    """Pipeline evaluasi standar untuk model tunggal dengan direct injection."""
    
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
        
        # Konfigurasi evaluasi
        eval_config = self.config.get('evaluation', {})
        self.batch_size = eval_config.get('batch_size', 16)
        self.num_workers = eval_config.get('num_workers', 4)
        self.device = eval_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.half_precision = eval_config.get('half_precision', False)
        self.conf_threshold = eval_config.get('conf_threshold', 0.25)
        self.iou_threshold = eval_config.get('iou_threshold', 0.5)
        
        # Load classes
        self.class_names = self.config.get('model', {}).get('class_names', [])
        
    def run(
        self,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi model.
        
        Args:
            model: Model PyTorch untuk evaluasi (opsional jika model_path disediakan)
            model_path: Path ke checkpoint model (opsional jika model disediakan)
            dataset_path: Path ke dataset evaluasi (opsional jika dataloader disediakan)
            dataloader: DataLoader untuk evaluasi (opsional jika dataset_path disediakan)
            device: Device untuk evaluasi (default dari config)
            conf_threshold: Threshold konfidiensi untuk deteksi (default dari config)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
            
        Raises:
            ValueError: Jika input tidak valid
        """
        # Notifikasi start
        self.notify('pipeline_start', {
            'model_path': model_path,
            'dataset_path': dataset_path,
            'device': device or self.device
        })
        
        self.logger.info(f"🚀 Menjalankan evaluasi model")
        
        try:
            # Validasi dan setup parameter
            device = device or self.device
            conf_threshold = conf_threshold or self.conf_threshold
            
            # Load model jika belum ada
            if model is None:
                model = self._load_model(model_path, device)
            
            # Siapkan model untuk evaluasi
            model = self._prepare_model(model, device)
            
            # Get dataloader jika belum ada
            if dataloader is None:
                dataloader = self._get_dataloader(dataset_path, batch_size=kwargs.get('batch_size', self.batch_size))
            
            # Buat metrics calculator
            metrics_calculator = MetricsCalculator(
                num_classes=len(self.class_names),
                class_names=self.class_names,
                conf_threshold=conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            # Notifikasi ready
            self.notify('evaluation_ready', {
                'model_info': str(model.__class__.__name__),
                'num_batches': len(dataloader)
            })
            
            # Evaluasi model
            result, execution_time = self.execute_with_timing(
                self._evaluate_model,
                model=model,
                dataloader=dataloader,
                metrics_calculator=metrics_calculator,
                device=device
            )
            
            # Tambahkan info
            result['model_path'] = model_path
            result['dataset_path'] = dataset_path
            result['total_execution_time'] = execution_time
            
            # Notifikasi complete
            self.notify('pipeline_complete', {
                'execution_time': execution_time,
                'metrics': result.get('metrics', {})
            })
            
            self.logger.success(f"✅ Evaluasi selesai dalam {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            self.notify('pipeline_error', {'error': str(e)})
            raise
    
    def _load_model(self, model_path: str, device: str) -> torch.nn.Module:
        """
        Load model dari checkpoint menggunakan direct loading.
        
        Args:
            model_path: Path ke checkpoint model
            device: Device untuk loading
            
        Returns:
            Model PyTorch
        """
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path tidak valid: {model_path}")
        
        self.logger.info(f"🔄 Loading model dari {Path(model_path).name}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Tentukan backbone
            backbone = None
            if 'config' in checkpoint and 'backbone' in checkpoint['config']:
                backbone = checkpoint['config']['backbone']
            elif 'efficientnet' in model_path.lower():
                backbone = 'efficientnet'
            else:
                backbone = 'cspdarknet'
            
            # Tentukan jumlah kelas
            num_classes = self.config.get('model', {}).get('num_classes', 17)
            
            # Buat model dengan backbone yang sesuai
            self.logger.info(f"🏗️ Menggunakan backbone {backbone}")
            model = YOLOv5Model(backbone_type=backbone, num_classes=num_classes)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            self.logger.info(f"✅ Model berhasil dimuat")
            return model
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise
    
    def _prepare_model(self, model: torch.nn.Module, device: str) -> torch.nn.Module:
        """
        Siapkan model untuk evaluasi.
        
        Args:
            model: Model PyTorch
            device: Device untuk evaluasi
            
        Returns:
            Model yang siap untuk evaluasi
        """
        # Pindahkan model ke device yang sesuai
        model = model.to(device)
        
        # Set model ke eval mode
        model.eval()
        
        # Gunakan half precision jika diperlukan dan didukung
        if self.half_precision and device == 'cuda':
            self.logger.info("🔄 Menggunakan half precision (FP16)")
            model = model.half()
        
        return model
    
    def _get_dataloader(self, dataset_path: str, batch_size: int = 16) -> torch.utils.data.DataLoader:
        """
        Buat dataloader untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset
            batch_size: Ukuran batch
            
        Returns:
            DataLoader
        """
        if not dataset_path or not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path tidak valid: {dataset_path}")
        
        self.logger.info(f"🔄 Loading dataset dari {Path(dataset_path).name}")
        
        try:
            # Import di sini untuk menghindari circular import
            from smartcash.handlers.dataset import DatasetManager
            
            # Dapatkan split dari path
            split = 'test'
            if 'train' in dataset_path:
                split = 'train'
            elif 'valid' in dataset_path or 'val' in dataset_path:
                split = 'valid'
            
            # Setup dataset manager dan dataloader
            dataset_manager = DatasetManager(config=self.config, data_dir=Path(dataset_path).parent)
            dataloader = dataset_manager.get_dataloader(
                split=split,
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
            
            self.logger.info(f"✅ Dataset berhasil dimuat: {len(dataloader)} batch")
            return dataloader
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat dataset: {str(e)}")
            raise
    
    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        metrics_calculator: MetricsCalculator,
        device: str
    ) -> Dict[str, Any]:
        """
        Evaluasi model dengan dataloader.
        
        Args:
            model: Model PyTorch
            dataloader: DataLoader
            metrics_calculator: MetricsCalculator instance
            device: Device untuk evaluasi
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Reset metrics
        metrics_calculator.reset()
        
        # Tracking waktu
        inference_times = []
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc="Evaluasi", unit="batch")
        
        # Evaluasi dalam no_grad context untuk efisiensi memori
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
                    'batch_size': images.shape[0] if isinstance(images, torch.Tensor) else 0
                })
                
                # Inferensi dengan pengukuran waktu
                batch_start = time.time()
                predictions = model(images)
                inference_time = time.time() - batch_start
                inference_times.append(inference_time)
                
                # Update metrics
                metrics_calculator.update(predictions, targets)
                
                # Update progress bar dengan FPS
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
        
        # Hitung metrik akhir
        metrics = metrics_calculator.compute()
        
        # Tambahkan informasi waktu inferensi
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            metrics['inference_time'] = avg_inference_time
            metrics['fps'] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            self.logger.info(f"⏱️ Waktu inferensi rata-rata: {avg_inference_time*1000:.2f}ms ({metrics['fps']:.1f} FPS)")
        
        return {
            'metrics': metrics,
            'num_samples': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader) * dataloader.batch_size
        }