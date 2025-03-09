# File: smartcash/handlers/model/core/model_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk evaluasi model yang diringkas dengan menghapus implementasi non-critical

import torch
import time
import numpy as np
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path
from tqdm import tqdm

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError, EvaluationError

class ModelEvaluator(ModelComponent):
    """
    Komponen untuk evaluasi model pada dataset test.
    Mendukung berbagai metrik evaluasi dan visualisasi hasil.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        model_factory = None,
        metrics_adapter = None
    ):
        """
        Inisialisasi model evaluator.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Custom logger (opsional)
            model_factory: Factory untuk membuat model (opsional, lazy-loaded)
            metrics_adapter: Adapter untuk metrics calculation (opsional, lazy-loaded)
        """
        super().__init__(config, logger, "model_evaluator")
        
        # Simpan factories dan adapters
        self._model_factory = model_factory
        self._metrics_adapter = metrics_adapter
    
    def _initialize(self) -> None:
        """Inisialisasi internal komponen."""
        self.evaluation_config = self.config.get('evaluation', {})
        
        # Default parameter evaluasi
        self.conf_threshold = self.evaluation_config.get('conf_threshold', 0.25)
        self.iou_threshold = self.evaluation_config.get('iou_threshold', 0.45)
        
        # Setup output direktori
        self.output_dir = Path(self.config.get('output_dir', 'runs/eval'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cek apakah di Google Colab
        self.in_colab = self._is_running_in_colab()
    
    def _is_running_in_colab(self) -> bool:
        """Deteksi apakah kode berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        if self._model_factory is None:
            from smartcash.handlers.model.core.model_factory import ModelFactory
            self._model_factory = ModelFactory(self.config, self.logger)
        return self._model_factory
    
    @property
    def metrics_adapter(self):
        """Lazy-loaded metrics adapter."""
        if self._metrics_adapter is None:
            from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter
            self._metrics_adapter = MetricsAdapter(self.logger, self.config, 
                                                  output_dir=str(self.output_dir / "metrics"))
        return self._metrics_adapter
    
    def process(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses evaluasi model. Alias untuk evaluate().
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi (opsional)
            **kwargs: Parameter tambahan untuk evaluasi
            
        Returns:
            Dict hasil evaluasi
        """
        return self.evaluate(test_loader, model, **kwargs)
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
        half_precision: Optional[bool] = None,
        per_class_metrics: bool = True,
        time_inference: bool = True,
        visualize: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi (opsional jika checkpoint_path diberikan)
            checkpoint_path: Path ke checkpoint (opsional jika model diberikan)
            conf_threshold: Threshold confidence untuk deteksi (opsional)
            iou_threshold: Threshold IoU untuk NMS (opsional)
            device: Device untuk evaluasi (opsional)
            half_precision: Gunakan half precision (FP16) (opsional)
            per_class_metrics: Hitung metrik per kelas (opsional)
            time_inference: Ukur waktu inferensi (opsional)
            visualize: Buat visualisasi hasil (opsional)
            **kwargs: Parameter tambahan untuk evaluasi
            
        Returns:
            Dict hasil evaluasi
        """
        start_time = time.time()
        
        try:
            # Pastikan ada model atau checkpoint
            if model is None and checkpoint_path is None:
                raise EvaluationError(
                    "Model atau checkpoint_path harus diberikan untuk evaluasi"
                )
            
            # Muat model dari checkpoint jika model tidak diberikan
            if model is None:
                self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
                model, checkpoint_meta = self.model_factory.load_model(checkpoint_path)
            
            # Tentukan device
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Pindahkan model ke device
            model = model.to(device)
            
            # Tentukan parameter evaluasi
            conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
            iou_threshold = iou_threshold if iou_threshold is not None else self.iou_threshold
            
            # Tentukan half precision
            if half_precision is None:
                # Default ke True jika CUDA tersedia
                half_precision = torch.cuda.is_available() and self.config.get('model', {}).get('half_precision', True)
            
            # Konversi ke half precision jika diminta
            if half_precision and device.type == 'cuda':
                model = model.half()
            
            # Reset metrics calculator
            self.metrics_adapter.reset()
            
            # Set model ke mode evaluasi
            model.eval()
            
            # Ukur waktu inferensi jika diminta
            inference_times = {}
            if time_inference:
                # Ukur waktu inferensi pada batch sample
                for batch in test_loader:
                    # Dapatkan input tensor
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs = batch[0]
                    else:
                        inputs = batch['image']
                    
                    if isinstance(inputs, list):
                        # Ambil satu item jika list
                        inputs = inputs[0] if len(inputs) > 0 else None
                    
                    if inputs is not None:
                        # Pastikan batch dimension
                        if len(inputs.shape) == 3:
                            inputs = inputs.unsqueeze(0)
                            
                        # Ukur waktu inferensi
                        inference_times = self.metrics_adapter.measure_inference_time(
                            model, inputs, num_runs=10, warm_up=2
                        )
                        break
            
            # Log informasi evaluasi
            self.logger.info(
                f"🔍 Mulai evaluasi model pada {len(test_loader)} batch:\n"
                f"   • Device: {device}\n"
                f"   • Confidence threshold: {conf_threshold}\n"
                f"   • IoU threshold: {iou_threshold}\n"
                f"   • Half precision: {half_precision}"
            )
            
            # Setup progress bar
            if self.in_colab:
                try:
                    from tqdm.notebook import tqdm as tqdm_notebook
                    pbar = tqdm_notebook(total=len(test_loader), desc="Evaluasi")
                except ImportError:
                    pbar = tqdm(total=len(test_loader), desc="Evaluasi")
            else:
                pbar = tqdm(total=len(test_loader), desc="Evaluasi")
            
            # Loop melalui batch
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # Dapatkan data dan target
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs, targets = batch['image'], batch['targets']
                    
                    # Pindahkan ke device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                    
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(device)
                    
                    # Inferensi
                    outputs = model(inputs)
                    
                    # Post-processing
                    if hasattr(model, 'post_process'):
                        # Model memiliki metode post-processing sendiri
                        predictions = model.post_process(
                            outputs, 
                            conf_threshold=conf_threshold, 
                            iou_threshold=iou_threshold
                        )
                    else:
                        # Gunakan output langsung sebagai prediksi
                        predictions = outputs
                    
                    # Update metrics pada setiap batch
                    self.metrics_adapter.update(predictions, targets)
                    
                    # Update progress
                    pbar.update(1)
            
            # Cleanup progress bar
            pbar.close()
            
            # Hitung metrik final
            metrics = self.metrics_adapter.compute()
            
            # Tambahkan waktu inferensi
            if time_inference:
                metrics.update(inference_times)
            
            # Tambahkan info evaluasi
            metrics['num_test_batches'] = len(test_loader)
            metrics['conf_threshold'] = conf_threshold
            metrics['iou_threshold'] = iou_threshold
            
            # Tambahkan metrik per layer jika model multilayer
            if hasattr(model, 'get_layer_metrics'):
                layer_metrics = model.get_layer_metrics(None, None)  # Simplified
                metrics['layer_metrics'] = layer_metrics
            
            # Hitung waktu eksekusi
            execution_time = time.time() - start_time
            metrics['execution_time'] = execution_time
            
            hours, remainder = divmod(execution_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.logger.success(
                f"✅ Evaluasi selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
                f"   • mAP: {metrics.get('mAP', 0):.4f}\n"
                f"   • Precision: {metrics.get('precision', 0):.4f}\n"
                f"   • Recall: {metrics.get('recall', 0):.4f}\n"
                f"   • F1 Score: {metrics.get('f1', 0):.4f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error saat evaluasi model: {str(e)}")
            raise EvaluationError(f"Gagal melakukan evaluasi: {str(e)}")
    
    def evaluate_on_layers(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        layers: List[str] = ['banknote', 'nominal', 'security'],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluasi model pada layer tertentu.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi
            layers: Daftar layer untuk dievaluasi
            **kwargs: Parameter tambahan untuk evaluate()
            
        Returns:
            Dict hasil evaluasi per layer
        """
        self.logger.info(f"🔍 Mulai evaluasi model pada {len(layers)} layer: {', '.join(layers)}")
        
        results = {}
        
        # Evaluasi pada semua layer sekaligus
        all_layers_results = self.evaluate(test_loader, model, **kwargs)
        results['all'] = all_layers_results
        
        # Evaluasi pada masing-masing layer jika model support
        if hasattr(model, 'set_active_layers'):
            for layer in layers:
                self.logger.info(f"🔍 Evaluasi layer: {layer}")
                
                # Set layer aktif
                model.set_active_layers([layer])
                
                # Evaluasi
                layer_results = self.evaluate(test_loader, model, **kwargs)
                
                # Simpan hasil
                results[layer] = layer_results
            
            # Reset ke semua layer
            model.set_active_layers(layers)
        
        # Log ringkasan hasil
        self.logger.info("📊 Ringkasan hasil evaluasi per layer:")
        for layer, metrics in results.items():
            self.logger.info(
                f"   • {layer}: mAP={metrics.get('mAP', 0):.4f}, "
                f"F1={metrics.get('f1', 0):.4f}"
            )
        
        return results