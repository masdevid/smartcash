"""
File: smartcash/detection/adapters/onnx_adapter.py
Deskripsi: Adapter untuk model ONNX yang dioptimasi.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
from PIL import Image

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection, ImageType
from smartcash.common.layer_config import get_layer_config
from smartcash.model.services.postprocessing.nms_processor import NMSProcessor


class ONNXModelAdapter:
    """Adapter untuk model ONNX yang dioptimasi"""
    
    def __init__(self, 
                onnx_path: str,
                input_shape: tuple = (1, 3, 640, 640),
                class_map: Optional[Dict[int, str]] = None,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi ONNX Model Adapter
        
        Args:
            onnx_path: Path ke file model ONNX
            input_shape: Bentuk input model
            class_map: Map ID kelas ke nama kelas (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.class_map = class_map
        self.logger = logger or get_logger("ONNXModelAdapter")
        self.session = None
        
        # Load class map dari layer config jika tidak disediakan
        if self.class_map is None:
            layer_config = get_layer_config()
            self.class_map = layer_config.get_class_map()
        
        # Inisialisasi NMS processor dari model domain
        self.nms_processor = NMSProcessor(logger=self.logger)
        
        # Load model ONNX
        self._load_model()
    
    def _load_model(self):
        """Load model ONNX"""
        try:
            import onnxruntime as ort
            
            if not os.path.exists(self.onnx_path):
                self.logger.error(f"❌ File model ONNX tidak ditemukan: {self.onnx_path}")
                raise FileNotFoundError(f"File model ONNX tidak ditemukan: {self.onnx_path}")
            
            # Buat session untuk inferensi
            self.logger.info(f"🔄 Memuat model ONNX dari {self.onnx_path}")
            
            # Dapatkan provider terbaik yang tersedia
            providers = ort.get_available_providers()
            
            # Prioritaskan GPU provider jika tersedia
            preferred_providers = []
            if 'CUDAExecutionProvider' in providers:
                preferred_providers.append('CUDAExecutionProvider')
                self.logger.info(f"✓ Menggunakan CUDA untuk inferensi ONNX")
            elif 'TensorrtExecutionProvider' in providers:
                preferred_providers.append('TensorrtExecutionProvider')
                self.logger.info(f"✓ Menggunakan TensorRT untuk inferensi ONNX")
            elif 'ROCMExecutionProvider' in providers:
                preferred_providers.append('ROCMExecutionProvider')
                self.logger.info(f"✓ Menggunakan ROCm untuk inferensi ONNX")
            
            # Selalu tambahkan CPU provider sebagai fallback
            preferred_providers.append('CPUExecutionProvider')
            
            # Buat session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 0  # Gunakan semua thread yang tersedia
            
            # Load model dengan provider yang tersedia
            self.session = ort.InferenceSession(
                self.onnx_path,
                sess_options=sess_options,
                providers=preferred_providers
            )
            
            # Dapatkan input dan output model
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Log informasi model
            actual_input_shape = self.session.get_inputs()[0].shape
            self.logger.info(f"✅ Model ONNX berhasil dimuat: input shape {actual_input_shape}")
            
        except ImportError:
            self.logger.error(f"❌ Package ONNX Runtime tidak terinstal")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error saat memuat model ONNX: {str(e)}")
            raise
    
    def preprocess(self, image: ImageType) -> np.ndarray:
        """
        Preprocess gambar untuk inferensi
        
        Args:
            image: Gambar input (path, numpy array, atau PIL Image)
            
        Returns:
            Array numpy yang sudah dipreproses
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
            
            # Transpose ke format NCHW jika perlu (input ONNX biasanya NCHW)
            if len(self.input_shape) == 4 and self.input_shape[1] == 3:
                # Format yang diharapkan: NCHW
                img_transposed = img_normalized.transpose(2, 0, 1)  # HWC -> CHW
                img_batched = np.expand_dims(img_transposed, axis=0)  # CHW -> NCHW
            else:
                # Format yang diharapkan: NHWC
                img_batched = np.expand_dims(img_normalized, axis=0)  # HWC -> NHWC
            
            return img_batched
            
        except Exception as e:
            self.logger.error(f"❌ Error saat preprocessing gambar: {str(e)}")
            raise
    
    def postprocess(self, 
                   outputs: Dict[str, np.ndarray], 
                   original_image: ImageType,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45) -> List[Detection]:
        """
        Postprocess hasil model untuk mendapatkan deteksi
        
        Args:
            outputs: Output model ONNX
            original_image: Gambar original
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi
        """
        try:
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
            
            # Format output model
            # Catatan: format output tergantung pada model ONNX yang digunakan
            # Kita asumsikan output model YOLOv5 (dets, proto)
            
            # Ekstrak hasil deteksi
            # Format output YOLOv5: [batch, num_detections, 6] (x1, y1, x2, y2, confidence, class)
            output_data = None
            
            # Cari output yang mengandung hasil deteksi
            for name in self.output_names:
                if name in outputs:
                    output_data = outputs[name]
                    break
            
            if output_data is None:
                self.logger.error(f"❌ Tidak dapat menemukan hasil deteksi dalam output model")
                return []
            
            # Ekstrak deteksi dari output
            if len(output_data.shape) == 3 and output_data.shape[2] >= 6:
                # Format YOLOv5: [batch, num_detections, 6]
                detections_data = output_data[0]  # Ambil batch pertama
            elif len(output_data.shape) == 2 and output_data.shape[1] >= 6:
                # Format alternatif: [num_detections, 6]
                detections_data = output_data
            else:
                self.logger.error(f"❌ Format output model tidak didukung: {output_data.shape}")
                return []
            
            # Filter berdasarkan confidence
            mask = detections_data[:, 4] >= conf_threshold
            filtered_data = detections_data[mask]
            
            if len(filtered_data) == 0:
                return []
            
            # Convert ke format Detection
            detections = []
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
        if self.session is None:
            self.logger.error(f"❌ Model belum dimuat")
            return []
            
        try:
            # Preprocess gambar
            input_data = self.preprocess(image)
            
            # Lakukan inferensi
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            inference_time = time.time() - start_time
            
            # Convert output ke dictionary
            outputs_dict = {name: outputs[i] for i, name in enumerate(self.output_names)}
            
            # Postprocess hasil
            detections = self.postprocess(outputs_dict, image, conf_threshold, iou_threshold)
            
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
                # Generate warna yang berbeda untuk setiap kelas
                hue = (class_id * 60) % 360  # Hue dalam HSV (0-360)
                # Convert ke RGB
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)  # HSV: (hue, saturation, value)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                colors[class_id] = rgb[0, 0].tolist()  # Extract RGB
            
            # Visualisasikan deteksi
            for detection in filtered_detections:
                # Dapatkan koordinat bbox dalam piksel
                x1, y1, x2, y2 = detection.bbox
                x1_px = int(x1 * width)
                y1_px = int(y1 * height)
                x2_px = int(x2 * width)
                y2_px = int(y2 * height)
                
                # Dapatkan warna untuk kelas ini
                color = colors.get(detection.class_id, [0, 255, 0])  # Default: hijau
                
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