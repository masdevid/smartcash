# File: smartcash/handlers/model/core/model_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk evaluasi model dengan implementasi minimal

import torch
import time
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm

from smartcash.utils.logger import get_logger
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError, EvaluationError

class ModelEvaluator(ModelComponent):
    """Komponen untuk evaluasi model pada dataset test."""
    
    def _initialize(self):
        """Inisialisasi parameter evaluasi."""
        cfg = self.config.get('evaluation', {})
        self.params = {
            'conf_threshold': cfg.get('conf_threshold', 0.25),
            'iou_threshold': cfg.get('iou_threshold', 0.45)
        }
        self.output_dir = Path(self.config.get('output_dir', 'runs/eval'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, test_loader, model=None, **kwargs):
        """Alias untuk evaluate()."""
        return self.evaluate(test_loader, model, **kwargs)
    
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
            
            # Reset metrics calculator
            self.metrics_adapter.reset()
            
            # Log info evaluasi
            self.logger.info(f"🔍 Evaluasi model ({len(test_loader)} batch): device={device}, conf={conf_threshold:.2f}")
            
            # Setup progress bar
            pbar = self._create_progress_bar(test_loader)
            
            # Evaluasi
            with torch.no_grad():
                for batch in test_loader:
                    # Handle different batch formats
                    inputs, targets = self._extract_batch_data(batch, device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Post-processing detections
                    predictions = self._post_process(model, outputs, conf_threshold, iou_threshold)
                    
                    # Update metrics
                    self.metrics_adapter.update(predictions, targets)
                    
                    # Update progress
                    if pbar:
                        pbar.update(1)
            
            # Cleanup progress bar
            if pbar:
                pbar.close()
            
            # Compute final metrics
            metrics = self.metrics_adapter.compute()
            
            # Add timing and config info
            metrics.update({
                'execution_time': time.time() - start_time,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            })
            
            # Log result
            self.logger.success(f"✅ Evaluasi selesai: mAP={metrics.get('mAP', 0):.4f}, F1={metrics.get('f1', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluasi: {str(e)}")
            raise EvaluationError(f"Gagal evaluasi: {str(e)}")
    
    def _prepare_model(self, model, checkpoint_path):
        """Persiapkan model untuk evaluasi."""
        if model is None and checkpoint_path is None:
            raise EvaluationError("Model atau checkpoint_path harus diberikan")
            
        if model is None:
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