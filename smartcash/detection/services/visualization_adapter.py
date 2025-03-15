"""
File: smartcash/detection/services/visualization_adapter.py
Deskripsi: Adapter untuk mengintegrasikan visualisasi dari domain model ke domain detection
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np

from smartcash.model.visualization.detection_visualizer import DetectionVisualizer
from smartcash.model.visualization.metrics_visualizer import MetricsVisualizer
from smartcash.common.logger import get_logger

class DetectionVisualizationAdapter:
    """
    Adapter untuk mengintegrasikan visualisasi dari domain model ke domain detection.
    Mengimplementasikan prinsip DRY dengan memanfaatkan kelas visualisasi yang sudah ada.
    """
    
    def __init__(
        self,
        detection_output_dir: str = "results/detections",
        metrics_output_dir: str = "results/metrics",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi adapter visualisasi.
        
        Args:
            detection_output_dir: Direktori output untuk visualisasi deteksi
            metrics_output_dir: Direktori output untuk visualisasi metrik
            logger: Logger untuk logging
        """
        self.logger = logger or get_logger("detection_visualization_adapter")
        
        # Inisialisasi visualizer dari domain model
        self.detection_visualizer = DetectionVisualizer(detection_output_dir)
        self.metrics_visualizer = MetricsVisualizer(metrics_output_dir)
        
        self.logger.debug(f"🔄 Adapter visualisasi deteksi diinisialisasi")
    
    def visualize_detection(
        self,
        image: np.ndarray,
        detections: List[Dict],
        filename: Optional[str] = None,
        conf_threshold: float = 0.25,
        show_labels: bool = True,
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Visualisasikan deteksi pada gambar menggunakan DetectionVisualizer.
        
        Args:
            image: Gambar input dalam format numpy array
            detections: List deteksi dalam format dict
            filename: Nama file untuk menyimpan output
            conf_threshold: Threshold confidence
            show_labels: Tampilkan label
            show_conf: Tampilkan nilai confidence
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Delegasi ke DetectionVisualizer
        return self.detection_visualizer.visualize_detection(
            image=image,
            detections=detections,
            filename=filename,
            conf_threshold=conf_threshold,
            show_labels=show_labels,
            show_conf=show_conf
        )
    
    def visualize_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        filename: Optional[str] = None,
        normalize: bool = True
    ) -> Any:
        """
        Visualisasikan confusion matrix menggunakan MetricsVisualizer.
        
        Args:
            cm: Confusion matrix
            class_names: List nama kelas
            title: Judul plot
            filename: Nama file untuk menyimpan output
            normalize: Normalisasi confusion matrix
            
        Returns:
            Figure matplotlib
        """
        # Delegasi ke MetricsVisualizer
        return self.metrics_visualizer.plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=title,
            filename=filename,
            normalize=normalize
        )
    
    def visualize_model_comparison(
        self,
        comparison_data: Any,
        metric_cols: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP'],
        title: str = "Model Comparison",
        filename: Optional[str] = None
    ) -> Any:
        """
        Visualisasikan perbandingan model menggunakan MetricsVisualizer.
        
        Args:
            comparison_data: DataFrame dengan data perbandingan
            metric_cols: Kolom metrik yang akan divisualisasikan
            title: Judul plot
            filename: Nama file untuk menyimpan output
            
        Returns:
            Figure matplotlib
        """
        # Delegasi ke MetricsVisualizer
        return self.metrics_visualizer.plot_model_comparison(
            comparison_data=comparison_data,
            metric_cols=metric_cols,
            title=title,
            filename=filename
        )