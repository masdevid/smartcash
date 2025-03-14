"""
File: smartcash/model/services/prediction/core.py
Deskripsi: Layanan inti prediksi untuk model deteksi mata uang SmartCash
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.model.exceptions import ModelInferenceError


class PredictionService:
    """
    Layanan prediksi terintegrasi untuk model SmartCash.
    
    Menyediakan:
    - Inferensi model dengan batching untuk optimal throughput
    - Praproses input (gambar)
    - Pascaproses output (deteksi)
    - Caching hasil untuk endpoint API
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        img_size: Tuple[int, int] = (640, 640),
        batch_size: int = 4,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        enable_cache: bool = True,
        logger = None
    ):
        """
        Inisialisasi layanan prediksi.
        
        Args:
            model: Model PyTorch yang digunakan untuk prediksi
            img_size: Ukuran gambar input untuk model (width, height)
            batch_size: Ukuran batch untuk inferensi
            confidence_threshold: Threshold confidence untuk deteksi
            iou_threshold: Threshold IoU untuk NMS
            device: Device untuk inferensi ('cuda', 'cpu', atau None untuk auto)
            enable_cache: Flag untuk mengaktifkan caching hasil
            logger: Logger untuk pencatatan
        """
        self.logger = logger or get_logger("prediction_service")
        self.layer_config = get_layer_config()
        
        # Setup model dan parameter
        self.model = model
        self.img_size = tuple(img_size)
        self.batch_size = batch_size
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Setup device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cache hasil
        self.enable_cache = enable_cache
        self._prediction_cache = {}  # {image_hash: prediction_result}
        
        self.logger.info(
            f"🔮 Layanan prediksi diinisialisasi:\n"
            f"   • Device: {self.device}\n"
            f"   • Ukuran gambar: {self.img_size}\n"
            f"   • Batch size: {self.batch_size}\n"
            f"   • Threshold confidence: {self.conf_threshold}\n"
            f"   • Threshold IoU: {self.iou_threshold}"
        )
    
    def predict(
        self, 
        images: Union[np.ndarray, List[np.ndarray]],
        return_annotated: bool = False
    ) -> Dict[str, Any]:
        """
        Membuat prediksi untuk satu gambar atau batch gambar.
        
        Args:
            images: Numpy array gambar atau list gambar
            return_annotated: Jika True, kembalikan gambar dengan anotasi
            
        Returns:
            Dict hasil prediksi dengan format:
            {
                'predictions': List deteksi untuk setiap gambar,
                'annotated_images': (opsional) Gambar dengan anotasi,
                'inference_time': Waktu inferensi (ms)
            }
        """
        # Handle kasus satu gambar
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
            
        if not images:
            return {'predictions': [], 'inference_time': 0}
        
        # Praproses gambar
        t_start = time.time()
        preprocessed = self._preprocess_images(images)
        
        # Periksa cache jika diaktifkan
        if self.enable_cache:
            cached_results = self._check_prediction_cache(preprocessed)
            if cached_results:
                return cached_results
        
        # Jalankan inferensi dalam mode evaluasi dan tanpa gradient
        with torch.no_grad():
            try:
                # Batching untuk memproses banyak gambar lebih efisien
                predictions = []
                
                # Proses gambar dalam batch
                for i in range(0, len(preprocessed), self.batch_size):
                    batch = preprocessed[i:i+self.batch_size]
                    batch_tensor = torch.stack(batch).to(self.device)
                    
                    # Inferensi
                    batch_predictions = self.model(batch_tensor)
                    
                    # Tambahkan hasil ke list
                    if isinstance(batch_predictions, torch.Tensor):
                        predictions.extend(batch_predictions.detach().cpu())
                    elif isinstance(batch_predictions, (tuple, list)):
                        # Jika model mengembalikan multiple output, ambil yang pertama
                        predictions.extend(batch_predictions[0].detach().cpu())
                    else:
                        # Handle kasus output dict (model multilayer)
                        batch_preds = self._handle_multilayer_output(batch_predictions)
                        predictions.extend(batch_preds)
                
                # Pascaproses deteksi
                results = self._postprocess_predictions(predictions, images)
                
                # Hitung waktu inferensi
                inference_time = (time.time() - t_start) * 1000  # Konversi ke ms
                results['inference_time'] = inference_time
                
                # Tambahkan gambar dengan anotasi jika diminta
                if return_annotated:
                    results['annotated_images'] = self._annotate_images(images, results['predictions'])
                
                # Cache hasil
                if self.enable_cache:
                    self._update_prediction_cache(preprocessed, results)
                
                return results
                
            except Exception as e:
                error_msg = f"❌ Error saat inferensi: {str(e)}"
                self.logger.error(error_msg)
                raise ModelInferenceError(error_msg) from e
    
    def predict_from_files(
        self, 
        image_paths: Union[str, List[str]],
        return_annotated: bool = False
    ) -> Dict[str, Any]:
        """
        Membuat prediksi dari file gambar.
        
        Args:
            image_paths: Path file gambar atau list path
            return_annotated: Jika True, kembalikan gambar dengan anotasi
            
        Returns:
            Dict hasil prediksi
        """
        import cv2
        
        # Handle satu path
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # Load gambar
        images = []
        loaded_paths = []
        
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    self.logger.warning(f"⚠️ Gagal load gambar: {path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                loaded_paths.append(path)
            except Exception as e:
                self.logger.warning(f"⚠️ Error membaca gambar {path}: {str(e)}")
        
        if not images:
            return {'predictions': [], 'inference_time': 0, 'paths': []}
        
        # Buat prediksi
        results = self.predict(images, return_annotated)
        
        # Tambahkan informasi path
        results['paths'] = loaded_paths
        
        return results
    
    def _preprocess_images(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Praproses gambar untuk inferensi.
        
        Args:
            images: List numpy array gambar (RGB)
            
        Returns:
            List PyTorch tensor yang siap untuk inferensi
        """
        import cv2
        preprocessed = []
        
        for img in images:
            # Resize ke ukuran model
            resized = cv2.resize(img, self.img_size)
            
            # Normalisasi
            normalized = resized / 255.0
            
            # Konversi ke tensor dan atur channel
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
            
            preprocessed.append(tensor)
            
        return preprocessed
    
    def _postprocess_predictions(
        self, 
        predictions: List[torch.Tensor], 
        original_images: List[np.ndarray]
    ) -> Dict[str, List]:
        """
        Pascaproses hasil raw prediksi.
        
        Args:
            predictions: Hasil prediksi dari model
            original_images: List gambar asli
            
        Returns:
            Dictionary berisi deteksi yang telah diproses
        """
        from smartcash.model.services.prediction.postprocessing import process_detections
        
        processed_results = []
        
        for i, pred in enumerate(predictions):
            # Dapatkan ukuran gambar asli
            orig_h, orig_w = original_images[i].shape[:2]
            
            # Proses deteksi (NMS, scale ke original size, dll)
            detections = process_detections(
                pred, 
                orig_size=(orig_w, orig_h),
                model_size=self.img_size,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            processed_results.append(detections)
            
        return {'predictions': processed_results}
    
    def _handle_multilayer_output(self, output: Any) -> List[torch.Tensor]:
        """
        Handle berbagai format output model.
        
        Args:
            output: Output dari model
            
        Returns:
            List tensor prediksi yang distandarisasi
        """
        # Cek output dict (model multilayer)
        if isinstance(output, dict) and 'banknote' in output:
            # Kasus SmartCash multilayer model
            # Combine semua deteksi dari berbagai layer
            
            batch_size = next(iter(output.values())).shape[0]
            result = []
            
            for i in range(batch_size):
                detections = []
                
                for layer_name, layer_output in output.items():
                    # Extract prediksi untuk satu item dalam batch
                    layer_pred = layer_output[i]
                    if torch.numel(layer_pred) > 0:
                        detections.append(layer_pred)
                
                # Gabungkan semua deteksi jika ada
                if detections:
                    result.append(torch.cat(detections))
                else:
                    # Tidak ada deteksi
                    result.append(torch.zeros((0, 6), device=self.device))
            
            return result
        
        # Format output YOLO standar
        if isinstance(output, (tuple, list)) and len(output) > 0:
            return output[0].detach().cpu()
        
        # Sudah dalam format yang benar
        return output.detach().cpu() if isinstance(output, torch.Tensor) else output
    
    def _check_prediction_cache(self, preprocessed_images: List[torch.Tensor]) -> Optional[Dict]:
        """
        Periksa cache untuk hasil prediksi.
        
        Args:
            preprocessed_images: List gambar yang telah diproses
            
        Returns:
            Dict hasil prediksi dari cache, atau None jika tidak ditemukan
        """
        if not self.enable_cache:
            return None
            
        # Gunakan hash tensor sebagai key
        image_hashes = []
        for img_tensor in preprocessed_images:
            # Buat hash sederhana dari content tensor
            img_hash = hash(img_tensor.numpy().tobytes())
            image_hashes.append(img_hash)
            
        # Jika hash dari semua gambar ada di cache, return hasil
        cache_key = tuple(image_hashes)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
            
        return None
    
    def _update_prediction_cache(self, preprocessed_images: List[torch.Tensor], results: Dict) -> None:
        """
        Update cache dengan hasil prediksi baru.
        
        Args:
            preprocessed_images: List gambar yang telah diproses
            results: Hasil prediksi untuk dicache
        """
        if not self.enable_cache:
            return
            
        # Gunakan hash tensor sebagai key
        image_hashes = []
        for img_tensor in preprocessed_images:
            # Buat hash sederhana dari content tensor
            img_hash = hash(img_tensor.numpy().tobytes())
            image_hashes.append(img_hash)
            
        # Simpan di cache
        cache_key = tuple(image_hashes)
        self._prediction_cache[cache_key] = results
        
        # Batasi ukuran cache (max 100 items)
        if len(self._prediction_cache) > 100:
            # Hapus entry tertua
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
    
    def _annotate_images(self, images: List[np.ndarray], predictions: List[List[Dict]]) -> List[np.ndarray]:
        """
        Buat gambar dengan anotasi prediksi.
        
        Args:
            images: List gambar asli
            predictions: List deteksi untuk setiap gambar
            
        Returns:
            List gambar dengan bounding box dan label
        """
        import cv2
        annotated_images = []
        
        for i, img in enumerate(images):
            img_copy = img.copy()
            
            # Get deteksi untuk gambar ini
            if i < len(predictions):
                detections = predictions[i]
                
                # Draw bounding box untuk setiap deteksi
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    class_id = det['class_id']
                    conf = det['confidence']
                    label = det['class_name']
                    
                    # Convert koordinat ke integer
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Gunakan warna berbeda untuk tiap kelas
                    color = self._get_color_for_class(class_id)
                    
                    # Draw bounding box
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label dengan confidence
                    label_text = f"{label}: {conf:.2f}"
                    cv2.putText(
                        img_copy, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            
            annotated_images.append(img_copy)
            
        return annotated_images
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """
        Dapatkan warna RGB untuk kelas.
        
        Args:
            class_id: ID kelas
            
        Returns:
            Tuple RGB color (untuk cv2, format BGR)
        """
        # Warna dasar untuk beberapa kelas
        colors = [
            (255, 0, 0),    # Merah
            (0, 255, 0),    # Hijau
            (0, 0, 255),    # Biru
            (255, 255, 0),  # Kuning
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Hijau tua
            (0, 0, 128),    # Biru tua
            (128, 128, 0),  # Olive
        ]
        
        # Gunakan modulo untuk support banyak kelas
        color_idx = class_id % len(colors)
        
        # Return format BGR untuk cv2
        r, g, b = colors[color_idx]
        return (b, g, r)
    
    def clear_cache(self) -> None:
        """Reset prediction cache."""
        self._prediction_cache = {}
        
    def enable_gpu(self) -> None:
        """Aktifkan GPU untuk inferensi (jika tersedia)."""
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.to(self.device)
            self.logger.info("✅ GPU diaktifkan untuk inferensi")
        else:
            self.logger.warning("⚠️ GPU tidak tersedia, tetap menggunakan CPU")
    
    def enable_cpu(self) -> None:
        """Aktifkan CPU untuk inferensi."""
        self.device = 'cpu'
        self.model = self.model.to(self.device)
        self.logger.info("✅ CPU diaktifkan untuk inferensi")