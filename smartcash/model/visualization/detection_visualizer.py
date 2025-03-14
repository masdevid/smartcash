"""
File: smartcash/utils/visualization/detection.py
Author: Alfrida Sabar
Deskripsi: Visualisasi hasil deteksi objek dengan bounding box, label, dan informasi tambahan
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from smartcash.model.visualization.base import VisualizationHelper
from smartcash.common.logger import get_logger

class DetectionVisualizer:
    """
    Visualisasi hasil deteksi objek dengan bounding box, label dan informasi tambahan.
    """
    
    # Konfigurasi warna untuk visualisasi kelas
    DEFAULT_COLORS = {
        # Denominasi Layer 1
        '100': (255, 0, 0),     # 100rb - Merah
        '050': (0, 0, 255),     # 50rb - Biru
        '020': (0, 255, 0),     # 20rb - Hijau
        '010': (128, 0, 128),   # 10rb - Ungu
        '005': (0, 128, 128),   # 5rb - Teal
        '002': (128, 128, 0),   # 2rb - Olive
        '001': (0, 0, 128),     # 1rb - Navy
        
        # Layer 2 (nominal)
        'l2_100': (255, 50, 50),    # 100rb
        'l2_050': (50, 50, 255),    # 50rb
        'l2_020': (50, 255, 50),    # 20rb
        'l2_010': (178, 50, 178),   # 10rb
        'l2_005': (50, 178, 178),   # 5rb
        'l2_002': (178, 178, 50),   # 2rb
        'l2_001': (50, 50, 178),    # 1rb
        
        # Layer 3 (security)
        'l3_sign': (255, 150, 0),   # Tanda tangan
        'l3_text': (150, 255, 0),   # Teks
        'l3_thread': (0, 150, 255)  # Benang
    }
    
    # Pemetaan denominasi ke nilai nominal
    DENOMINATION_MAP = {
        '001': 1000,
        '002': 2000,
        '005': 5000,
        '010': 10000,
        '020': 20000,
        '050': 50000,
        '100': 100000
    }
    
    def __init__(
        self, 
        output_dir: str = "results/detections",
        class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi visualizer deteksi.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            class_colors: Dictionary warna kelas kustom
            logger: Logger untuk logging
        """
        self.output_dir = VisualizationHelper.create_output_directory(output_dir)
        self.logger = logger or get_logger("detection_visualizer")
        
        # Gunakan warna default jika tidak ada kustom
        self.class_colors = class_colors or self.DEFAULT_COLORS
    
    def visualize_detection(
        self,
        image: np.ndarray,
        detections: List[Dict],
        filename: Optional[str] = None,
        conf_threshold: float = 0.25,
        show_labels: bool = True,
        show_conf: bool = True,
        show_total: bool = True,
        show_value: bool = True
    ) -> np.ndarray:
        """
        Visualisasikan deteksi pada gambar.
        
        Args:
            image: Gambar input (BGR/OpenCV format)
            detections: List deteksi dalam format [{'bbox': [x1,y1,x2,y2], 'class_name': str, 'confidence': float}]
            filename: Nama file untuk menyimpan output
            conf_threshold: Threshold confidence untuk menampilkan deteksi
            show_labels: Tampilkan label kelas
            show_conf: Tampilkan nilai confidence
            show_total: Tampilkan total deteksi
            show_value: Tampilkan total nilai mata uang
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Buat salinan gambar
        vis_img = image.copy()
        
        # Filter deteksi berdasarkan confidence
        filtered_detections = [det for det in detections if det.get('confidence', 1.0) >= conf_threshold]
        
        # Hitung total nilai yang terdeteksi
        total_value = 0
        denominations = []
        
        # Gambar setiap deteksi
        for det in filtered_detections:
            # Extract info
            bbox = det['bbox']
            class_name = det.get('class_name', str(det.get('class_id', 'unknown')))
            confidence = det.get('confidence', 1.0)
            
            # Tambahkan ke total nilai jika ini adalah uang kertas utuh
            if class_name in self.DENOMINATION_MAP:
                value = self.DENOMINATION_MAP[class_name]
                total_value += value
                denominations.append(class_name)
            
            # Get warna untuk kelas
            color = self.class_colors.get(class_name, (255, 255, 255))  # Default putih
            
            # Gambar bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Tambahkan label jika diminta
            if show_labels:
                # Konversi nama kelas ke format yang lebih pendek
                display_name = class_name
                if class_name in self.DENOMINATION_MAP:
                    value = self.DENOMINATION_MAP[class_name]
                    display_name = f"{value//1000}rb"
                
                # Buat teks label
                label = display_name
                if show_conf:
                    label += f" {confidence:.2f}"
                
                # Hitung ukuran teks
                text_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Gambar background label
                cv2.rectangle(
                    vis_img,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1
                )
                
                # Gambar teks label
                cv2.putText(
                    vis_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # Putih
                    1
                )
        
        # Tampilkan info jumlah deteksi
        if show_total:
            cv2.putText(
                vis_img,
                f"Deteksi: {len(filtered_detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Hijau
                2
            )
        
        # Tampilkan total nilai di bagian bawah
        if show_value and total_value > 0:
            h = vis_img.shape[0]
            total_text = f"Total: Rp {total_value:,}"
            denom_text = f"Denominasi: {', '.join(f'{d}' for d in denominations)}"
            
            # Background untuk teks
            cv2.rectangle(
                vis_img,
                (10, h - 60),
                (10 + max(cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0],
                        cv2.getTextSize(denom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]) + 10,
                h - 10),
                (0, 0, 0),  # Hitam
                -1
            )
            
            # Tampilkan teks total dan denominasi
            cv2.putText(
                vis_img,
                total_text,
                (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # Putih
                2
            )
            
            cv2.putText(
                vis_img,
                denom_text,
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # Putih
                1
            )
        
        # Simpan hasil jika diminta
        if filename:
            output_path = self.output_dir / filename
            try:
                cv2.imwrite(str(output_path), vis_img)
                self.logger.info(f"✅ Hasil deteksi disimpan ke {output_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyimpan gambar deteksi: {str(e)}")
        
        return vis_img
    
    def visualize_detections_grid(
        self,
        images: List[np.ndarray],
        detections_list: List[List[Dict]],
        title: str = "Hasil Deteksi",
        filename: Optional[str] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        conf_threshold: float = 0.25
    ) -> np.ndarray:
        """
        Visualisasikan multiple deteksi dalam grid.
        
        Args:
            images: List gambar input
            detections_list: List deteksi untuk setiap gambar
            title: Judul grid
            filename: Nama file untuk menyimpan output
            grid_size: Ukuran grid (rows, cols)
            conf_threshold: Threshold confidence
            
        Returns:
            Gambar grid dengan deteksi
        """
        if not images or len(images) == 0:
            self.logger.warning("⚠️ Tidak ada gambar untuk divisualisasikan")
            return None
            
        if len(images) != len(detections_list):
            self.logger.warning("⚠️ Jumlah gambar dan deteksi tidak sama")
            return None
        
        # Tentukan ukuran grid
        n_images = len(images)
        
        if grid_size is None:
            # Hitung grid optimal
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            grid_size = (rows, cols)
        else:
            rows, cols = grid_size
            
        # Validasi grid size
        if rows * cols < n_images:
            self.logger.warning(f"⚠️ Grid size {grid_size} terlalu kecil untuk {n_images} gambar")
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            grid_size = (rows, cols)
        
        # Visualisasikan setiap gambar
        vis_images = []
        
        for img, detections in zip(images, detections_list):
            if img is None:
                continue
                
            # Visualisasikan deteksi
            vis_img = self.visualize_detection(
                img, 
                detections, 
                conf_threshold=conf_threshold
            )
            
            vis_images.append(vis_img)
        
        # Resize semua gambar ke ukuran yang sama
        if vis_images:
            # Tentukan ukuran target
            target_height = max(img.shape[0] for img in vis_images)
            target_width = max(img.shape[1] for img in vis_images)
            
            # Resize semua gambar
            vis_images = [
                cv2.resize(img, (target_width, target_height))
                for img in vis_images
            ]
            
            # Buat grid
            grid_img = self._create_grid(vis_images, grid_size, title)
            
            # Simpan hasil
            if filename:
                output_path = self.output_dir / filename
                try:
                    cv2.imwrite(str(output_path), grid_img)
                    self.logger.info(f"✅ Grid deteksi disimpan ke {output_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menyimpan grid deteksi: {str(e)}")
            
            return grid_img
        
        return None
    
    def _create_grid(
        self,
        images: List[np.ndarray],
        grid_size: Tuple[int, int],
        title: str = ""
    ) -> np.ndarray:
        """
        Buat grid dari gambar.
        
        Args:
            images: List gambar
            grid_size: Ukuran grid (rows, cols)
            title: Judul grid
            
        Returns:
            Gambar grid
        """
        rows, cols = grid_size
        
        # Pastikan semua gambar memiliki ukuran yang sama
        h, w = images[0].shape[:2]
        
        # Buat grid kosong (termasuk ruang untuk judul)
        title_height = 50  # Pixel untuk judul
        grid = np.zeros((h * rows + title_height, w * cols, 3), dtype=np.uint8)
        
        # Tambahkan judul
        if title:
            cv2.putText(
                grid,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        
        # Isi grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
                
            r = i // cols
            c = i % cols
            
            y_start = r * h + title_height
            y_end = (r + 1) * h + title_height
            x_start = c * w
            x_end = (c + 1) * w
            
            grid[y_start:y_end, x_start:x_end] = img
        
        return grid
    
    def calculate_denomination_total(self, detections: List[Dict]) -> int:
        """
        Hitung total nilai mata uang dari deteksi.
        
        Args:
            detections: List deteksi
            
        Returns:
            Total nilai dalam Rupiah
        """
        total = 0
        for det in detections:
            class_name = det.get('class_name', '')
            if class_name in self.DENOMINATION_MAP:
                total += self.DENOMINATION_MAP[class_name]
        
        return total


# Fungsi helper untuk visualisasi cepat tanpa membuat instance penuh
def visualize_detection(
    image: np.ndarray,
    detections: List[Dict],
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25
) -> np.ndarray:
    """
    Fungsi helper untuk visualisasi cepat.
    
    Args:
        image: Gambar input
        detections: List deteksi
        output_path: Path untuk menyimpan output
        conf_threshold: Threshold confidence
        
    Returns:
        Gambar dengan deteksi
    """
    visualizer = DetectionVisualizer()
    result = visualizer.visualize_detection(
        image, 
        detections, 
        filename=Path(output_path).name if output_path else None,
        conf_threshold=conf_threshold
    )
    
    return result