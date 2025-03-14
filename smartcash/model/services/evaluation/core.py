"""
File: smartcash/model/services/evaluation/core.py
Deskripsi: Layanan inti untuk evaluasi model deteksi mata uang
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import time
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.model.services.evaluation.metrics import MetricsComputation
from smartcash.model.services.evaluation.visualization import EvaluationVisualizer
from smartcash.model.components.losses import compute_loss

class EvaluationService:
    """
    Layanan untuk evaluasi model deteksi mata uang dengan dukungan lengkap.
    
    Fitur:
    - Evaluasi performa model pada dataset validasi
    - Perhitungan berbagai metrik evaluasi
    - Visualisasi hasil prediksi dan metrik
    - Integrasi dengan eksperimen tracking
    """
    
    def __init__(
        self, 
        config: Dict,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi layanan evaluasi.
        
        Args:
            config: Konfigurasi model dan evaluasi
            output_dir: Direktori output untuk hasil evaluasi
            logger: Logger instance
        """
        # Inisialisasi dasar
        self.config = config
        self.logger = logger or get_logger("model.evaluation")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup direktori output
        self.output_dir = Path(output_dir) if output_dir else Path("runs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi komponen evaluasi
        self.metrics = MetricsComputation(config, logger=self.logger)
        self.visualizer = EvaluationVisualizer(config, output_dir=str(self.output_dir), logger=self.logger)
        
        # Konfigurasi layer
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config.get_layer_names())
        
        self.logger.info(f"🔍 Layanan evaluasi diinisialisasi dengan {len(self.active_layers)} layer aktif")
    
    def evaluate(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        visualize: bool = False,
        batch_size: Optional[int] = None,
        return_samples: bool = False,
        experiment_tracker = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model pada dataset.
        
        * old: handlers.model.evaluation.evaluate_model()
        * migrated: Flexible service-based evaluation
        
        Args:
            model: Model yang akan dievaluasi
            dataloader: DataLoader untuk dataset validasi
            conf_thres: Threshold confidence untuk deteksi
            iou_thres: Threshold IoU untuk NMS
            max_det: Jumlah maksimum deteksi per gambar
            visualize: Flag untuk mengaktifkan visualisasi hasil
            batch_size: Batch size (jika tidak menggunakan dataloader.batch_size)
            return_samples: Flag untuk mengembalikan sampel hasil deteksi
            experiment_tracker: Objek experiment tracker
            
        Returns:
            Dict berisi hasil evaluasi
        """
        # Siapkan model
        model.eval()
        if not hasattr(model, 'warmup'):
            # Buat warmup jika belum ada
            def warmup():
                # Dummy forward pass untuk warmup CUDA
                if self.device.type != 'cpu':
                    img = torch.zeros((1, 3, *model.img_size), device=self.device)
                    for _ in range(3):
                        model(img)
            model.warmup = warmup
        
        # Warmup jika menggunakan CUDA
        if self.device.type != 'cpu':
            model.warmup()
        
        # Mulai evaluasi
        self.logger.info(f"🔄 Mulai evaluasi model pada {len(dataloader)} batch")
        start_time = time.time()
        
        # Reset metrics
        self.metrics.reset()
        
        # Struktur untuk menyimpan hasil
        validation_loss = 0
        samples = [] if return_samples else None
        
        # Config evaluasi
        hyp = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'max_det': max_det
        }
        
        # Evaluasi model
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validasi", bar_format='{l_bar}{bar:10}{r_bar}')
            for batch_idx, batch in enumerate(pbar):
                # Pindahkan data ke device
                images = batch['images'].to(self.device)
                targets = batch['targets']
                
                # Inferensi
                batch_start = time.time()
                predictions = model(images)
                batch_time = time.time() - batch_start
                
                # Hitung loss
                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(predictions, targets)[0]
                else:
                    loss = compute_loss(predictions, targets, model, self.active_layers)
                validation_loss += loss.item()
                
                # Update metrik
                self.metrics.update(predictions, targets, batch_time)
                
                # Simpan sampel jika diminta
                if return_samples and batch_idx < 5:  # Batasi 5 batch saja
                    for i, img in enumerate(images):
                        # Hanya simpan 5 sampel per batch
                        if i >= 5:
                            break
                        # Simpan sampel gambar, prediksi, dan target
                        samples.append({
                            'image': img.cpu().numpy(),
                            'prediction': predictions[i].detach().cpu() if isinstance(predictions, list) else predictions.detach().cpu()[i],
                            'target': {layer: targets[layer][i].cpu() for layer in targets}
                        })
                
                # Update progress bar
                batch_metrics = self.metrics.get_last_batch_metrics()
                desc = f"Val: {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}"
                if 'precision' in batch_metrics:
                    desc += f", Prec: {batch_metrics['precision']:.4f}"
                if 'recall' in batch_metrics:
                    desc += f", Rcl: {batch_metrics['recall']:.4f}"
                pbar.set_description(desc)
        
        # Hitung metrik final
        metrics = self.metrics.compute()
        metrics['val_loss'] = validation_loss / len(dataloader)
        
        # Buat visualisasi jika diminta
        if visualize:
            # Visualisasi confusion matrix
            cm_path = self.visualizer.plot_confusion_matrix(
                metrics.get('confusion_matrix', {}),
                title="Confusion Matrix",
                normalized=True
            )
            metrics['confusion_matrix_path'] = cm_path
            
            # Visualisasi precision-recall curve
            pr_path = self.visualizer.plot_pr_curve(
                metrics.get('precision_curve', {}),
                metrics.get('recall_curve', {}),
                metrics.get('f1_curve', {}),
                title="Precision-Recall Curve"
            )
            metrics['pr_curve_path'] = pr_path
            
            # Visualisasi contoh prediksi jika ada sampel
            if samples:
                samples_path = self.visualizer.visualize_predictions(
                    samples[:10],  # Batasi 10 sampel saja
                    conf_thres=conf_thres,
                    title="Sample Predictions"
                )
                metrics['samples_path'] = samples_path
        
        # Log hasil evaluasi
        duration = time.time() - start_time
        self.logger.info(
            f"✅ Evaluasi selesai dalam {duration:.2f} detik:\n"
            f"   • Loss: {metrics['val_loss']:.4f}\n"
            f"   • mAP@.5: {metrics.get('mAP50', 0):.4f}\n"
            f"   • mAP@.5:.95: {metrics.get('mAP', 0):.4f}\n"
            f"   • Presisi: {metrics.get('precision', 0):.4f}\n"
            f"   • Recall: {metrics.get('recall', 0):.4f}\n"
            f"   • Inference: {metrics.get('inference_time', 0):.2f}ms/img"
        )
        
        # Log ke experiment tracker jika ada
        if experiment_tracker:
            log_dict = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            experiment_tracker.log_metrics(log_dict)
        
        return {
            'metrics': metrics,
            'samples': samples
        }
    
    def evaluate_by_layer(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluasi model per layer.
        
        * new: Layer-specific evaluation metrics
        
        Args:
            model: Model yang akan dievaluasi
            dataloader: DataLoader untuk dataset validasi
            **kwargs: Parameter tambahan untuk evaluate()
            
        Returns:
            Dict berisi metrik per layer
        """
        self.logger.info(f"🔍 Mengevaluasi model per layer ({len(self.active_layers)} layer)")
        
        # Evaluasi model secara keseluruhan
        eval_results = self.evaluate(model, dataloader, **kwargs)
        metrics = eval_results['metrics']
        
        # Ekstrak metrik per layer
        layer_metrics = {}
        
        for layer in self.active_layers:
            # Ekstrak metrik spesifik layer
            layer_prefix = f"{layer}_"
            layer_metrics[layer] = {
                k.replace(layer_prefix, ''): v
                for k, v in metrics.items()
                if k.startswith(layer_prefix)
            }
            
            # Tambahkan metrik umum
            layer_metrics[layer].update({
                'val_loss': metrics.get(f'{layer}_loss', metrics.get('val_loss', 0)),
                'precision': metrics.get(f'{layer}_precision', 0),
                'recall': metrics.get(f'{layer}_recall', 0),
                'f1': metrics.get(f'{layer}_f1', 0),
                'mAP50': metrics.get(f'{layer}_mAP50', 0),
                'mAP': metrics.get(f'{layer}_mAP', 0)
            })
            
            # Log metrik layer
            self.logger.info(
                f"📊 Metrik layer {layer}:\n"
                f"   • mAP@.5: {layer_metrics[layer]['mAP50']:.4f}\n"
                f"   • Presisi: {layer_metrics[layer]['precision']:.4f}\n"
                f"   • Recall: {layer_metrics[layer]['recall']:.4f}"
            )
        
        return {
            'metrics': metrics,
            'layer_metrics': layer_metrics
        }
    
    def evaluate_by_class(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluasi model per kelas.
        
        * new: Class-specific evaluation metrics
        
        Args:
            model: Model yang akan dievaluasi
            dataloader: DataLoader untuk dataset validasi
            **kwargs: Parameter tambahan untuk evaluate()
            
        Returns:
            Dict berisi metrik per kelas
        """
        self.logger.info("🔍 Mengevaluasi model per kelas")
        
        # Evaluasi model
        eval_results = self.evaluate(model, dataloader, **kwargs)
        metrics = eval_results['metrics']
        
        # Ekstrak metrik per kelas
        class_metrics = {}
        class_map = {}
        
        # Bangun mapping class_id ke nama
        for layer in self.active_layers:
            layer_config = self.layer_config.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                if i < len(layer_config['classes']):
                    class_map[cls_id] = layer_config['classes'][i]
        
        # Ekstrak metrik kelas dari per-class metrics
        for cls_id, cls_name in class_map.items():
            # Ekstrak metrik spesifik kelas
            cls_prefix = f"cls_{cls_id}_"
            class_metrics[cls_name] = {
                k.replace(cls_prefix, ''): v
                for k, v in metrics.items()
                if k.startswith(cls_prefix)
            }
            
            # Log metrik kelas
            if 'precision' in class_metrics[cls_name]:
                self.logger.info(
                    f"📊 Metrik kelas {cls_name} (ID: {cls_id}):\n"
                    f"   • Presisi: {class_metrics[cls_name].get('precision', 0):.4f}\n"
                    f"   • Recall: {class_metrics[cls_name].get('recall', 0):.4f}\n"
                    f"   • F1: {class_metrics[cls_name].get('f1', 0):.4f}"
                )
        
        return {
            'metrics': metrics,
            'class_metrics': class_metrics
        }