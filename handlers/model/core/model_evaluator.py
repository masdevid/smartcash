# File: smartcash/handlers/model/core/model_evaluator.py
# Deskripsi: Komponen untuk evaluasi model dengan dependency injection

import torch
import time
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from smartcash.exceptions.base import ModelError, EvaluationError
from smartcash.handlers.model.core.component_base import ComponentBase

class ModelEvaluator(ComponentBase):
    """Komponen untuk evaluasi model dengan dependency injection."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional = None,
        model_factory = None,
        metrics_calculator = None
    ):
        """
        Inisialisasi model evaluator.
        
        Args:
            config: Konfigurasi model dan evaluasi
            logger: Logger kustom (opsional)
            model_factory: ModelFactory instance (opsional)
            metrics_calculator: Kalkulator metrik (opsional)
        """
        super().__init__(config, logger, "model_evaluator")
        
        # Dependencies
        self.model_factory = model_factory
        self.metrics_calculator = metrics_calculator
    
    def _initialize(self):
        """Inisialisasi parameter evaluasi."""
        cfg = self.config.get('evaluation', {})
        self.params = {
            'conf_threshold': cfg.get('conf_threshold', 0.25),
            'iou_threshold': cfg.get('iou_threshold', 0.45)
        }
        self.output_dir = self.create_output_dir("eval")
    
    def evaluate(self, test_loader, model=None, checkpoint_path=None, **kwargs):
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil evaluasi
        """
        start_time = time.time()
        
        try:
            # Load model jika perlu
            model = self._prepare_model(model, checkpoint_path)
            
            # Setup parameter
            device = kwargs.get('device') or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device).eval()
            conf_threshold = kwargs.get('conf_threshold', self.params['conf_threshold'])
            iou_threshold = kwargs.get('iou_threshold', self.params['iou_threshold'])
            
            # Reset metrics calculator jika ada
            if self.metrics_calculator:
                self.metrics_calculator.reset()
            
            # Log info evaluasi
            self.logger.info(f"🔍 Evaluasi model ({len(test_loader)} batch)")
            
            # Setup progress bar
            pbar = self._create_progress_bar(test_loader)
            
            # Persiapan hasil
            all_predictions = []
            all_targets = []
            
            # Evaluasi
            with torch.no_grad():
                for batch in test_loader:
                    # Handle different batch formats
                    inputs, targets = self._extract_batch_data(batch, device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Post-processing detections
                    predictions = self._post_process(model, outputs, conf_threshold, iou_threshold)
                    
                    # Simpan prediksi dan target untuk evaluasi
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                    
                    # Update metrics langsung jika calculator tersedia
                    if self.metrics_calculator:
                        self.metrics_calculator.update(predictions, targets)
                    
                    # Update progress
                    if pbar:
                        pbar.update(1)
            
            # Cleanup progress bar
            if pbar:
                pbar.close()
            
            # Compute final metrics
            if self.metrics_calculator:
                metrics = self.metrics_calculator.compute()
            else:
                # Simple metrics jika tidak ada calculator
                metrics = self._compute_basic_metrics(all_predictions, all_targets)
            
            # Add timing dan config info
            metrics.update({
                'execution_time': time.time() - start_time,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            })
            
            # Log result
            self.logger.success(
                f"✅ Evaluasi selesai: "
                f"mAP={metrics.get('mAP', 0):.4f}, F1={metrics.get('f1', 0):.4f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluasi: {str(e)}")
            raise EvaluationError(f"Gagal evaluasi: {str(e)}")
    
    def _prepare_model(self, model, checkpoint_path):
        """Persiapkan model untuk evaluasi."""
        if model is None and checkpoint_path is None:
            raise EvaluationError("Model atau checkpoint_path harus diberikan")
            
        if model is None:
            if not self.model_factory:
                raise EvaluationError("Model factory diperlukan untuk load checkpoint")
                
            self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
            model, _ = self.model_factory.load_model(checkpoint_path)
            
        return model
    
    def _create_progress_bar(self, test_loader):
        """Buat progress bar sesuai environment."""
        if self.in_colab:
            try:
                from tqdm.notebook import tqdm as tqdm_notebook
                return tqdm_notebook(total=len(test_loader), desc="Evaluasi")
            except ImportError:
                return tqdm(total=len(test_loader), desc="Evaluasi")
        else:
            return tqdm(total=len(test_loader), desc="Evaluasi")
    
    def _extract_batch_data(self, batch, device):
        """Ekstrak data dari batch dengan format berbeda."""
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch['image'], batch['targets']
        
        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
            
        return inputs, targets
    
    def _post_process(self, model, outputs, conf_threshold, iou_threshold):
        """Post-process output model."""
        if hasattr(model, 'post_process'):
            return model.post_process(outputs, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        else:
            return outputs
            
    def _compute_basic_metrics(self, predictions, targets):
        """Hitung metrik dasar jika tidak ada metrics calculator."""
        # Implementasi sederhana, idealnya menggunakan metrics calculator yang sesuai
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }