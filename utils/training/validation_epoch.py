"""
File: smartcash/utils/training/validation_epoch.py
Author: Alfrida Sabar
Deskripsi: Handler untuk proses validasi pada satu epoch dengan dukungan berbagai metrik
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

from smartcash.utils.training.training_callbacks import TrainingCallbacks

class ValidationEpoch:
    """
    Handler untuk proses validasi pada satu epoch dengan dukungan
    untuk berbagai format data, model, dan perhitungan metrik.
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi handler epoch validasi
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger
    
    def run(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        callbacks: Optional[TrainingCallbacks] = None
    ) -> Tuple[float, Dict]:
        """
        Jalankan satu epoch validasi.
        
        Args:
            epoch: Nomor epoch
            model: Model yang akan divalidasi
            val_loader: Dataloader untuk validasi
            device: Device (cuda/cpu)
            callbacks: Handler callback
            
        Returns:
            Tuple (rata-rata validation loss, dict metrik)
        """
        total_loss = 0
        batch_count = 0
        val_metrics = {}
        
        # Kumpulkan semua prediksi dan target untuk menghitung metrik
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # Handle berbagai format data
                images, targets = self._process_batch_data(data, device)
                
                # Skip jika data tidak valid
                if images.numel() == 0:
                    continue
                
                # Forward pass
                predictions = model(images)
                
                # Compute loss
                loss, component_losses = self._compute_loss(model, predictions, targets)
                
                # Update metrik komponen jika ada
                for name, value in component_losses.items():
                    if name not in val_metrics:
                        val_metrics[name] = 0
                    val_metrics[name] += value
                
                # Update loss tracking
                total_loss += loss.item()
                batch_count += 1
                
                # Collect predictions dan targets untuk metrics
                self._collect_predictions(predictions, targets, all_predictions, all_targets)
        
        # Calculate average loss
        avg_loss = total_loss / max(1, batch_count)
        
        # Average component metrics
        for k in val_metrics:
            val_metrics[k] /= max(1, batch_count)
        
        # Calculate additional metrics if we have predictions and targets
        if len(all_targets) > 0 and len(all_predictions) > 0:
            additional_metrics = self._calculate_metrics(all_predictions, all_targets)
            val_metrics.update(additional_metrics)
        
        # Trigger validation end callback
        if callbacks:
            callbacks.trigger(
                event='validation_end',
                epoch=epoch,
                val_loss=avg_loss,
                metrics=val_metrics
            )
        
        return avg_loss, val_metrics
    
    def _process_batch_data(self, data: Any, device: torch.device) -> tuple:
        """Proses data batch ke format yang konsisten"""
        # Handle berbagai format data
        if isinstance(data, dict):
            # Format multilayer dataset
            images = data['image'].to(device)
            targets = {k: v.to(device) for k, v in data['targets'].items()}
        elif isinstance(data, tuple) and len(data) == 2:
            # Format (images, targets)
            images, targets = data
            images = images.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, dict):
                targets = {k: v.to(device) for k, v in targets.items()}
        else:
            if self.logger:
                self.logger.warning(f"⚠️ Format data tidak didukung: {type(data)}")
            # Set default values to prevent failure
            images = torch.tensor([], device=device)
            targets = torch.tensor([], device=device)
            
        return images, targets
    
    def _compute_loss(
        self, 
        model: torch.nn.Module, 
        predictions: Any, 
        targets: Any
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Hitung loss berdasarkan format prediction dan target
        
        Returns:
            Tuple (loss, dict component losses)
        """
        component_losses = {}
        
        # Jika model memiliki metode compute_loss sendiri
        if hasattr(model, 'compute_loss'):
            loss_dict = model.compute_loss(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Tambahkan component losses ke metrics
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    component_losses[k] = v.item()
                    
            return loss, component_losses
        
        # Jika model multi-layer dan targets adalah dict
        if isinstance(predictions, dict) and isinstance(targets, dict):
            total_loss = 0
            
            for layer_name in predictions:
                if layer_name in targets:
                    layer_pred = predictions[layer_name]
                    layer_target = targets[layer_name]
                    layer_loss = torch.nn.functional.mse_loss(layer_pred, layer_target)
                    total_loss += layer_loss
                    component_losses[f"{layer_name}_loss"] = layer_loss.item()
                    
            return total_loss, component_losses
        
        # Format standar
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, targets)
        return loss, component_losses
    
    def _collect_predictions(
        self, 
        predictions: Any, 
        targets: Any,
        all_predictions: list,
        all_targets: list
    ) -> None:
        """Kumpulkan prediksi dan target untuk perhitungan metrik"""
        if not isinstance(targets, dict):
            try:
                # Handle predictions berupa tensor scores/logits
                if isinstance(predictions, torch.Tensor) and predictions.dim() > 1:
                    pred_classes = predictions.argmax(dim=1).cpu().numpy()
                else:
                    pred_classes = predictions.cpu().numpy()
                
                # Handle targets berupa one-hot
                if isinstance(targets, torch.Tensor):
                    if targets.dim() > 1 and targets.size(1) > 1:
                        true_classes = targets.argmax(dim=1).cpu().numpy()
                    else:
                        true_classes = targets.cpu().numpy()
                
                all_predictions.extend(pred_classes)
                all_targets.extend(true_classes)
            except:
                # Lewati jika format tidak kompatibel
                pass
    
    def _calculate_metrics(self, predictions: list, targets: list) -> Dict:
        """Hitung metrik evaluasi dari prediksi dan target"""
        metrics = {}
        
        # Try to calculate precision, recall, f1, etc.
        try:
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions, average='weighted', zero_division=0
            )
            
            accuracy = accuracy_score(targets, predictions)
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })
            
            # Tambahkan mAP jika bisa
            try:
                from sklearn.metrics import average_precision_score
                # Coba hitung mAP jika jumlah kelas > 2
                unique_classes = np.unique(np.concatenate([targets, predictions]))
                if len(unique_classes) > 2:
                    # Convert ke format one-hot untuk mAP
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(targets, classes=unique_classes)
                    y_pred_bin = label_binarize(predictions, classes=unique_classes)
                    
                    # Hitung mAP
                    mAP = average_precision_score(y_true_bin, y_pred_bin, average='macro')
                    metrics['mAP'] = mAP
            except:
                # Lewati jika error saat menghitung mAP
                pass
                
        except ImportError:
            if self.logger:
                self.logger.warning("⚠️ scikit-learn tidak ditemukan, metrics lanjutan tidak dihitung")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Error saat menghitung metrics: {str(e)}")
        
        return metrics