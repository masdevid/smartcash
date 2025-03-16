"""
File: smartcash/detection/adapters/torchscript_adapter.py
Deskripsi: Adapter untuk model TorchScript yang dioptimasi.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import cv2
from PIL import Image

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection, ImageType
from smartcash.common.layer_config import get_layer_config
from smartcash.model.services.postprocessing.nms_processor import NMSProcessor


class TorchScriptAdapter:
    """Adapter untuk model TorchScript yang dioptimasi"""
    
    def __init__(self, 
                model_path: str,
                input_shape: tuple = (1, 3, 640, 640),
                class_map: Optional[Dict[int, str]] = None,
                device: str = "auto",
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi TorchScript Adapter
        
        Args:
            model_path: Path ke file model TorchScript
            input_shape: Bentuk input model
            class_map: Map ID kelas ke nama kelas (opsional)
            device: Device untuk inferensi ('cpu', 'cuda', 'mps', atau 'auto')
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.class_map = class_map
        self.device_str = device
        self.logger = logger or get_logger("TorchScriptAdapter")
        self.model = None
        self.device = None
        
        # Load class map dari layer config jika tidak disediakan
        if self.class_map is None:
            layer_config = get_layer_config()
            self.class_map = layer_config.get_class_map()
        
        # Inisialisasi NMS processor dari model domain
        self.nms_processor = NMSProcessor(logger=self.logger)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model TorchScript"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ File model TorchScript tidak ditemukan: {self.model_path}")
                raise FileNotFoundError(f"File model TorchScript tidak ditemukan: {self.model_path}")
            
            # Deteksi device yang tersedia
            if self.device_str == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    self.logger.info(f"✓ Menggunakan CUDA untuk inferensi")
                elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    self.logger.info(f"✓ Menggunakan MPS (Apple Silicon) untuk inferensi")
                else:
                    self.device = torch.device("cpu")
                    self.logger.info(f"✓ Menggunakan CPU untuk inferensi")
            else:
                self.device = torch.device(self.device_str)
            
            # Load model ke device
            self.logger.info(f"🔄 Memuat model TorchScript dari {self.model_path}")
            
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # Aktifkan cudnn benchmark jika menggunakan CUDA
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            
            self.logger.info(f"✅ Model TorchScript berhasil dimuat ke {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saat memuat model TorchScript: {str(e)}")
            raise
    
    def preprocess(self, image: ImageType) -> torch.Tensor:
        """
        Preprocess gambar untuk inferensi
        
        Args:
            image: Gambar input (path, numpy array, atau PIL Image)
            
        Returns:
            Tensor yang sudah dipreproses
        """
        try:
            # Load gambar jika berupa path
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert PIL Image ke numpy array
            elif isinstance(image, Image.Image):
                img = np.array(image)
            # Gunakan langsung jika sudah berupa numpy array
            elif isinstance(image, np.ndarray):
                img = image.copy()
                # Convert BGR ke RGB jika perlu
                if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.logger.error(f"❌ Format gambar tidak didukung: {type(image)}")
                raise ValueError(f"Format gambar tidak didukung: {type(image)}")
            
            # Resize gambar ke ukuran input model
            height, width = self.input_shape[2], self.input_shape[3]
            img_resized = cv2.resize(img, (width, height))
            
            # Normalisasi (0-255 -> 0-1)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Transpose ke format NCHW
            img_transposed = img_normalized.transpose(2, 0, 1)  # HWC -> CHW
            
            # Convert ke tensor torch
            tensor = torch.from_numpy(img_transposed).unsqueeze(0)  # CHW -> NCHW
            
            # Pindahkan ke device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"❌ Error saat preprocessing gambar: {str(e)}")
            raise
    
    def postprocess(self, 
                   output: torch.Tensor, 
                   original_image: ImageType,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45) -> List[Detection]:
        """
        Postprocess hasil model untuk mendapatkan deteksi
        
        Args:
            output: Output model TorchScript
            original_image: Gambar original
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi
        """
        try:
            # Convert output ke numpy jika perlu
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = output
            
            # Dapatkan dimensi gambar original
            if isinstance(original_image, str):
                img = cv2.imread(original_image)
                orig_height, orig_width = img.shape[:2]
            elif isinstance(original_image, Image.Image):
                orig_width, orig_height = original_image.size
            elif isinstance(original_image, np.ndarray):
                orig_height, orig_width = original_image.shape[:2]
            else:
                self.logger.error(f"❌ Format gambar tidak didukung: {type(original_image)}")
                raise ValueError(f"Format gambar tidak didukung: {type(original_image)}")
            
            # Format output model (tergantung pada arsitektur model)
            # Kita asumsikan output YOLOv5: [batch, num_detections, 6]
            # dimana 6 = [x1, y1, x2, y2, confidence, class]
            
            # Ekstrak hasil deteksi
            detections = []
            
            # Check format output
            if len(output_np.shape) == 3 and output_np.shape[2] >= 6:
                # Format YOLOv5: [batch, num_detections, 6]
                detections_data = output_np[0]  # Ambil batch pertama
            elif len(output_np.shape) == 2 and output_np.shape[1] >= 6:
                # Format alternatif: [num_detections, 6]
                detections_data = output_np
            else:
                self.logger.error(f"❌ Format output model tidak didukung: {output_np.shape}")
                return []
            
            # Filter berdasarkan confidence
            mask = detections_data[:, 4] >= conf_threshold
            filtered_data = detections_data[mask]
            
            if len(filtered_data) == 0:
                return []
            
            # Convert ke format Detection
            for det in filtered_data:
                x1, y1, x2, y2, confidence, class_id = det[:6]
                
                # Convert ke koordinat relatif (0-1)
                x1_rel = max(0, min(1, x1 / self.input_shape[3]))
                y1_rel = max(0, min(1, y1 / self.input_shape[2]))
                x2_rel = max(0, min(1, x2 / self.input_shape[3]))
                y2_rel = max(0, min(1, y2 / self.input_shape[2]))
                
                # Convert class_id ke int
                class_id = int(class_id)
                
                # Dapatkan nama kelas
                class_name = self.class_map.get(class_id, f"class_{class_id}")
                
                # Buat objek Detection
                detection = Detection(
                    bbox=np.array([x1_rel, y1_rel, x2_rel, y2_rel]),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=class_name
                )
                
                detections.append(detection)
            
            # Terapkan NMS menggunakan processor dari domain model
            filtered_detections = self.nms_processor.process(
                detections=detections,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold
            )
            
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"❌ Error saat postprocessing hasil: {str(e)}")
            raise
    
    def predict(self, 
               image: ImageType,
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45) -> List[Detection]:
        """
        Prediksi objek pada gambar
        
        Args:
            image: Gambar input (path, numpy array, atau PIL Image)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi
        """
        if self.model is None:
            self.logger.error(f"❌ Model belum dimuat")
            return []
            
        try:
            # Preprocess gambar
            input_tensor = self.preprocess(image)
            
            # Lakukan inferensi
            with torch.no_grad():
                start_time = time.time()
                output = self.model(input_tensor)
                inference_time = time.time() - start_time
            
            # Postprocess hasil
            detections = self.postprocess(output, image, conf_threshold, iou_threshold)
            
            self.logger.debug(f"✅ Inferensi selesai dalam {inference_time:.4f}s: {len(detections)} objek terdeteksi")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ Error saat prediksi: {str(e)}")
            return []
    
    def visualize(self, 
                 image: ImageType,
                 detections: List[Detection],
                 conf_threshold: float = 0.0) -> np.ndarray:
        """
        Visualisasikan hasil deteksi pada gambar
        
        Args:
            image: Gambar input (path, numpy array, atau PIL Image)
            detections: List hasil deteksi
            conf_threshold: Threshold confidence minimum
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        try:
            # Load gambar jika berupa path
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert PIL Image ke numpy array
            elif isinstance(image, Image.Image):
                img = np.array(image)
            # Gunakan langsung jika sudah berupa numpy array
            elif isinstance(image, np.ndarray):
                img = image.copy()
                # Convert BGR ke RGB jika perlu
                if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.logger.error(f"❌ Format gambar tidak didukung: {type(image)}")
                raise ValueError(f"Format gambar tidak didukung: {type(image)}")
            
            # Filter deteksi berdasarkan conf_threshold
            filtered_detections = [d for d in detections if d.confidence >= conf_threshold]
            
            # Dapatkan dimensi gambar
            height, width = img.shape[:2]
            
            # Generate warna unik untuk setiap kelas
            unique_classes = set(d.class_id for d in filtered_detections)
            colors = {}
            
            for class_id in unique_classes:
                # Generate warna berbeda untuk setiap kelas
                hue = (class_id * 60) % 360
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                colors[class_id] = rgb[0, 0].tolist()
            
            # Visualisasikan deteksi
            for detection in filtered_detections:
                # Dapatkan koordinat bbox dalam piksel
                x1, y1, x2, y2 = detection.bbox
                x1_px = int(x1 * width)
                y1_px = int(y1 * height)
                x2_px = int(x2 * width)
                y2_px = int(y2 * height)
                
                # Dapatkan warna untuk kelas ini
                color = colors.get(detection.class_id, [0, 255, 0])
                
                # Gambar bounding box
                cv2.rectangle(img, (x1_px, y1_px), (x2_px, y2_px), color, 2)
                
                # Siapkan teks label
                label_text = f"{detection.class_name}: {detection.confidence:.2f}"
                
                # Gambar background untuk teks
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1_px, y1_px - text_size[1] - 5), (x1_px + text_size[0], y1_px), color, -1)
                
                # Gambar teks
                cv2.putText(img, label_text, (x1_px, y1_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
            
            # Tambahkan informasi jumlah deteksi
            info_text = f"Deteksi: {len(filtered_detections)}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
            
            return img
            
        except Exception as e:
            self.logger.error(f"❌ Error saat visualisasi: {str(e)}")
            # Kembalikan gambar kosong
            return np.zeros((100, 100, 3), dtype=np.uint8)