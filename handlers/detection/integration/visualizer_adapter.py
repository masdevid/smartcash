# File: smartcash/handlers/detection/integration/visualizer_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan ResultVisualizer

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import cv2

from smartcash.utils.logger import get_logger

class VisualizerAdapter:
    """
    Adapter untuk integrasi dengan ResultVisualizer dari utils.visualization.
    Mengelola visualisasi hasil deteksi dengan dukungan untuk custom styling.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger = None
    ):
        """
        Inisialisasi visualizer adapter.
        
        Args:
            config: Konfigurasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("visualizer_adapter")
        
        # Parameter visualisasi dari konfigurasi
        inference_config = config.get('inference', {})
        self.show_labels = inference_config.get('show_labels', True)
        self.show_conf = inference_config.get('show_conf', True)
        self.show_value = inference_config.get('show_value', True)
        
        # Untuk load ResultVisualizer lazily
        self._visualizer = None
        
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        show_labels: Optional[bool] = None,
        show_conf: Optional[bool] = None,
        **kwargs
    ) -> Union[np.ndarray, str]:
        """
        Visualisasikan deteksi pada gambar.
        
        Args:
            image: Gambar sebagai array numpy (RGB)
            detections: List deteksi
            output_path: Path untuk menyimpan hasil (opsional)
            show_labels: Flag untuk menampilkan labels (opsional)
            show_conf: Flag untuk menampilkan confidence (opsional)
            **kwargs: Parameter tambahan untuk visualizer
            
        Returns:
            Gambar hasil visualisasi atau path file output jika disimpan
        """
        # Gunakan parameter dari init jika tidak diberikan
        show_labels = show_labels if show_labels is not None else self.show_labels
        show_conf = show_conf if show_conf is not None else self.show_conf
        
        # Load ResultVisualizer jika belum
        visualizer = self._get_visualizer()
        
        # Convert detections ke format yang diharapkan oleh visualizer
        # Format asli:
        # [{'bbox_xyxy': [x1, y1, x2, y2], 'class_name': 'name', 'confidence': conf, ...}, ...]
        # Format untuk visualizer:
        # [{'box': [x1, y1, x2, y2], 'label': 'name', 'score': conf}, ...]
        
        viz_detections = []
        for det in detections:
            # Gunakan bbox_xyxy jika tersedia, jika tidak konversi dari format lain
            if 'bbox_xyxy' in det:
                box = det['bbox_xyxy']
            elif 'bbox_pixels' in det:
                # Konversi [x_center, y_center, width, height] ke [x1, y1, x2, y2]
                x_center, y_center, width, height = det['bbox_pixels']
                box = [
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2
                ]
            elif 'bbox' in det:
                # Asumsikan normalisasi, tidak bisa divisualisasikan tanpa ukuran gambar
                self.logger.warning("⚠️ Visualisasi dengan bbox normalisasi tidak didukung")
                continue
            else:
                self.logger.warning(f"⚠️ Format deteksi tidak didukung: {det}")
                continue
                
            # Format label
            label = det['class_name']
            if 'layer' in det and self.show_value and det['layer'] == 'banknote':
                # Untuk layer banknote, tambahkan nilai mata uang
                label = f"{label} ({self._get_currency_value(det['class_name'])})"
                
            # Tambahkan confidence jika diminta
            if show_conf:
                conf = det.get('confidence', 0)
                label = f"{label} {conf:.2f}"
                
            # Buat entry untuk visualizer
            viz_det = {
                'box': box,
                'label': label if show_labels else None,
                'score': det.get('confidence', 0),
                'class_id': det.get('class_id', 0),
                'layer': det.get('layer', 'unknown')
            }
            
            viz_detections.append(viz_det)
            
        # Visualisasikan
        try:
            result_img = visualizer.visualize_detections(
                image,
                viz_detections,
                output_path=output_path,
                **kwargs
            )
            
            return result_img
            
        except Exception as e:
            self.logger.error(f"❌ Error visualisasi: {str(e)}")
            
            # Fallback jika visualizer gagal
            if output_path:
                # Simpan gambar original saja
                if isinstance(output_path, str):
                    output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert RGB ke BGR untuk OpenCV
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_bgr)
                return str(output_path)
            
            return image
    
    def _get_visualizer(self):
        """Lazy-load ResultVisualizer dari utils."""
        if self._visualizer is None:
            try:
                # Import dari utils, handle jika tidak tersedia
                from smartcash.utils.visualization import DetectionVisualizer
                
                # Setup visualizer dengan konfigurasi
                self._visualizer = DetectionVisualizer(
                    output_dir=self.config.get('inference', {}).get('output_dir', 'results/detections')
                )
                
            except ImportError:
                self.logger.warning("⚠️ DetectionVisualizer tidak tersedia, menggunakan fallback visualization")
                # Disini kita bisa implementasikan visualizer fallback jika dibutuhkan
                # Untuk sekarang, biarkan None dan gunakan metode fallback di visualize_detections
        
        return self._visualizer
    
    def _get_currency_value(self, class_name: str) -> str:
        """
        Convert class_name mata uang ke nilai satuan mata uang.
        
        Args:
            class_name: Nama kelas ('001', '002', '005', dll)
            
        Returns:
            Nilai mata uang (Rp 1.000, Rp 2.000, dll)
        """
        currency_map = {
            '001': 'Rp 1.000',
            '002': 'Rp 2.000',
            '005': 'Rp 5.000',
            '010': 'Rp 10.000',
            '020': 'Rp 20.000',
            '050': 'Rp 50.000',
            '100': 'Rp 100.000'
        }
        
        return currency_map.get(class_name, class_name)