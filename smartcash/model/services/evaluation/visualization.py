"""
File: smartcash/model/services/evaluation/visualization.py
Deskripsi: Layanan visualisasi hasil evaluasi model deteksi mata uang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
import cv2

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.model.services.evaluation.visualization import setup_visualization
from smartcash.model.visualization.metrics_visualizer import MetricsVisualizer
from smartcash.model.visualization.detection_visualizer import DetectionVisualizer

class EvaluationVisualizer:
    """
    Visualisasi hasil evaluasi model deteksi mata uang.
    
    Fitur:
    - Visualisasi confusion matrix
    - Visualisasi kurva precision-recall
    - Visualisasi contoh deteksi dengan bounding box
    - Plot metrik evaluasi per epoch
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi EvaluationVisualizer.
        
        Args:
            config: Konfigurasi model dan evaluasi
            output_dir: Direktori output untuk hasil visualisasi
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or get_logger("model.evaluation.viz")
        
        # Setup direktori output
        self.output_dir = Path(output_dir) if output_dir else Path("runs/evaluation/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan konfigurasi layer
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config.get_layer_names())
        
        # Setup visualisasi
        setup_visualization()
        
        # Inisialisasi visualizers
        self.metrics_viz = MetricsVisualizer(output_dir=str(self.output_dir), logger=self.logger)
        self.detection_viz = DetectionVisualizer(output_dir=str(self.output_dir), logger=self.logger)
        
        self.logger.info(f"🎨 EvaluationVisualizer diinisialisasi di {self.output_dir}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: Dict[int, Dict[int, int]],
        title: str = "Confusion Matrix",
        normalized: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues',
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualisasi confusion matrix.
        
        * old: handlers.model.visualization.plot_confusion_matrix()
        * migrated: Enhanced visualization with better color mapping
        
        Args:
            confusion_matrix: Dictionary {pred_class: {true_class: count}}
            title: Judul plot
            normalized: Flag untuk normalisasi confusion matrix
            figsize: Ukuran figure (width, height)
            cmap: Colormap untuk plot
            save_path: Path untuk menyimpan hasil (optional)
            
        Returns:
            Path hasil visualisasi
        """
        # Konversi confusion matrix ke format numpy
        classes = set()
        for pred_cls, true_dict in confusion_matrix.items():
            classes.add(pred_cls)
            classes.update(true_dict.keys())
        
        classes = sorted(list(classes))
        n_classes = len(classes)
        
        # Jika tidak ada kelas atau confusion matrix kosong
        if n_classes == 0:
            self.logger.warning("⚠️ Confusion matrix kosong, tidak ada kelas untuk divisualisasikan")
            # Buat confusion matrix dummy 1x1
            cm_array = np.zeros((1, 1))
            classes = ['No Detection']
        else:
            # Buat array confusion matrix
            cm_array = np.zeros((n_classes, n_classes))
            
            for i, pred_cls in enumerate(classes):
                if pred_cls in confusion_matrix:
                    for j, true_cls in enumerate(classes):
                        if true_cls in confusion_matrix[pred_cls]:
                            cm_array[i, j] = confusion_matrix[pred_cls][true_cls]
        
        # Convert class IDs to class names
        class_names = []
        for cls_id in classes:
            found = False
            for layer in self.active_layers:
                layer_config = self.layer_config.get_layer_config(layer)
                if cls_id in layer_config['class_ids']:
                    idx = layer_config['class_ids'].index(cls_id)
                    if idx < len(layer_config['classes']):
                        class_names.append(f"{layer_config['classes'][idx]} ({cls_id})")
                        found = True
                        break
            
            if not found:
                class_names.append(f"Class {cls_id}")
        
        # Normalisasi jika diminta
        if normalized and cm_array.sum() > 0:
            cm_array = cm_array.astype('float') / cm_array.sum(axis=1, keepdims=True)
            cm_array = np.nan_to_num(cm_array, nan=0)  # Replace NaN with 0
        
        # Generate save path jika tidak diberikan
        if save_path is None:
            save_path = str(self.output_dir / f"confusion_matrix.png")
            
        # Gunakan metrics_viz untuk plot confusion matrix
        result_path = self.metrics_viz.plot_confusion_matrix(
            confusion_matrix=cm_array,
            classes=class_names,
            title=title,
            normalize=False,  # Sudah dinormalisasi di atas jika perlu
            filepath=save_path,
            figsize=figsize,
            cmap=cmap
        )
        
        self.logger.info(f"📊 Confusion matrix disimpan di {result_path}")
        return result_path
    
    def plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        f1: Optional[np.ndarray] = None,
        title: str = "Precision-Recall Curve",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualisasi kurva precision-recall.
        
        * old: handlers.model.visualization.plot_pr_curve()
        * migrated: Enhanced visualization with F1 curve
        
        Args:
            precision: Array nilai precision
            recall: Array nilai recall
            f1: Array nilai F1 (optional)
            title: Judul plot
            figsize: Ukuran figure (width, height)
            save_path: Path untuk menyimpan hasil (optional)
            
        Returns:
            Path hasil visualisasi
        """
        # Generate save path jika tidak diberikan
        if save_path is None:
            save_path = str(self.output_dir / f"pr_curve.png")
            
        # Convert arrays ke list jika diperlukan
        precision_list = precision.tolist() if isinstance(precision, np.ndarray) else precision
        recall_list = recall.tolist() if isinstance(recall, np.ndarray) else recall
        f1_list = f1.tolist() if f1 is not None and isinstance(f1, np.ndarray) else f1
        
        # Gunakan metrics_viz untuk plot kurva PR
        curves_data = {
            'precision': precision_list,
            'recall': recall_list
        }
        
        if f1_list is not None:
            curves_data['f1'] = f1_list
            
        result_path = self.metrics_viz.plot_curves(
            data=curves_data,
            title=title,
            xlabel='Recall',
            ylabel='Precision / F1',
            labels={
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1 Score'
            } if f1_list is not None else {
                'precision': 'Precision', 
                'recall': 'Recall'
            },
            filepath=save_path,
            figsize=figsize
        )
        
        self.logger.info(f"📊 Kurva PR disimpan di {result_path}")
        return result_path
    
    def visualize_predictions(
        self,
        samples: List[Dict],
        conf_thres: float = 0.25,
        title: str = "Sample Predictions",
        figsize: Tuple[int, int] = (16, 16),
        max_samples: int = 4,
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualisasi hasil prediksi model.
        
        * old: handlers.model.visualization.visualize_detections()
        * migrated: Enhanced visualization with multi-sample support
        
        Args:
            samples: List sampel {image, prediction, target}
            conf_thres: Threshold confidence untuk deteksi
            title: Judul plot
            figsize: Ukuran figure (width, height)
            max_samples: Jumlah maksimum sampel yang ditampilkan
            save_path: Path untuk menyimpan hasil (optional)
            
        Returns:
            Path hasil visualisasi
        """
        # Validasi samples
        if not samples:
            self.logger.warning("⚠️ Tidak ada sampel untuk divisualisasikan")
            return ""
            
        # Batasi jumlah sampel
        samples = samples[:max_samples]
        
        # Generate save path jika tidak diberikan
        if save_path is None:
            save_path = str(self.output_dir / f"predictions.png")
            
        # Preprocess samples
        processed_images = []
        
        for sample in samples:
            # Ekstrak image dari sampel (convert ke uint8 jika perlu)
            image = sample['image']
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWH
                
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Ekstrak prediksi dan targets
            pred = sample['prediction']
            targets = sample.get('target', {})
            
            # Konversi prediksi ke format yang didukung
            if isinstance(pred, torch.Tensor):
                # Filter berdasarkan confidence
                if pred.shape[0] > 0:
                    mask = pred[:, 4] > conf_thres
                    pred = pred[mask].cpu().numpy()
                else:
                    pred = np.array([])
                    
            # Konversi target ke format yang didukung
            target_boxes = []
            for layer, layer_target in targets.items():
                if isinstance(layer_target, torch.Tensor):
                    # Filter berdasarkan confidence (jika ada)
                    if layer_target.shape[0] > 0 and layer_target.shape[1] > 4:
                        mask = layer_target[:, 4] > 0
                        layer_target = layer_target[mask].cpu().numpy()
                    else:
                        layer_target = layer_target.cpu().numpy()
                    
                    # Tambahkan ke target_boxes
                    target_boxes.append(layer_target)
            
            target_boxes = np.concatenate(target_boxes, axis=0) if target_boxes else np.array([])
            
            # Visualisasi gambar dengan prediksi dan target
            vis_image = self.detection_viz.draw_detections(
                image, pred, target_boxes, 
                conf_threshold=conf_thres,
                detection_format='xywh'
            )
            
            processed_images.append(vis_image)
        
        # Gabungkan semua gambar dalam satu grid
        grid_image = self.detection_viz.create_image_grid(
            processed_images,
            title=title,
            cols=2,  # Gunakan 2 kolom untuk layout yang baik
            filepath=save_path
        )
        
        self.logger.info(f"📊 Visualisasi prediksi disimpan di {grid_image}")
        return grid_image
    
    def plot_metrics_history(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Training Metrics",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualisasi history metrik training/validasi.
        
        * new: Metrics history visualization for experiment tracking
        
        Args:
            metrics_history: Dict history metrik {metric_name: [values]}
            title: Judul plot
            figsize: Ukuran figure (width, height)
            save_path: Path untuk menyimpan hasil (optional)
            
        Returns:
            Path hasil visualisasi
        """
        # Generate save path jika tidak diberikan
        if save_path is None:
            save_path = str(self.output_dir / f"metrics_history.png")
            
        # Gunakan metrics_viz untuk plot metrics history
        result_path = self.metrics_viz.plot_curves(
            data=metrics_history,
            title=title,
            xlabel='Epoch',
            ylabel='Value',
            labels={k: k for k in metrics_history.keys()},
            filepath=save_path,
            figsize=figsize
        )
        
        self.logger.info(f"📊 Metrics history disimpan di {result_path}")
        return result_path