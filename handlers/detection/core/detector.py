# File: smartcash/handlers/detection/core/detector.py
# Author: Alfrida Sabar
# Deskripsi: Komponen inti untuk deteksi objek mata uang

import time
from typing import Dict, Any, List, Union, Optional
import torch
import numpy as np

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import ModelError

class DefaultDetector:
    """
    Komponen inti untuk deteksi objek mata uang.
    Menggunakan model yang dimuat oleh model_adapter untuk
    mendeteksi objek dalam gambar yang telah dipreprocess.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        logger = None
    ):
        """
        Inisialisasi detector.
        
        Args:
            model: Model deteksi yang sudah dimuat
            config: Konfigurasi
            logger: Logger kustom (opsional)
        """
        self.model = model
        self.config = config
        self.logger = logger or get_logger("detector")
        
        # Parameter deteksi dari konfigurasi
        inference_config = config.get('inference', {})
        self.conf_threshold = inference_config.get('conf_threshold', 0.25)
        self.iou_threshold = inference_config.get('iou_threshold', 0.45)
        
        # Deteksi device
        self.device = next(model.parameters()).device
        self.logger.info(f"🔍 Detector diinisialisasi (device: {self.device})")
        
    def detect(
        self, 
        image: torch.Tensor, 
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        measure_time: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dalam gambar.
        
        Args:
            image: Tensor gambar yang telah dipreprocess
            conf_thres: Confidence threshold (opsional)
            iou_thres: IOU threshold untuk NMS (opsional)
            measure_time: Flag untuk mengukur waktu inferensi
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi
        """
        # Gunakan nilai default jika tidak diberikan
        conf_thres = conf_thres if conf_thres is not None else self.conf_threshold
        iou_thres = iou_thres if iou_thres is not None else self.iou_threshold
        
        try:
            # Ubah model ke mode evaluasi jika belum
            self.model.eval()
            
            # Jalankan inferensi dalam no_grad context untuk efisiensi memori
            start_time = time.time()
            
            with torch.no_grad():
                # Pastikan gambar ada di device yang sama dengan model
                if image.device != self.device:
                    image = image.to(self.device)
                    
                # Jalankan model untuk deteksi
                predictions = self.model(image)
            
            # Jika model mengembalikan tuple/list, ambil elemen pertama
            if isinstance(predictions, (tuple, list)):
                predictions = predictions[0]
                
            # Hitung waktu inferensi jika diminta
            inference_time = None
            if measure_time:
                inference_time = time.time() - start_time
                self.logger.debug(f"⏱️ Waktu inferensi: {inference_time:.4f} detik")
            
            # Siapkan hasil
            result = {
                'predictions': predictions,
                'inference_time': inference_time,
                'conf_threshold': conf_thres,
                'iou_threshold': iou_thres,
            }
            
            return result
            
        except RuntimeError as e:
            # Deteksi error memori
            if "out of memory" in str(e).lower():
                self.logger.error(f"🚫 GPU out of memory: {e}")
                torch.cuda.empty_cache()
                raise ModelError(f"GPU out of memory saat deteksi. Coba kurangi batch size atau ukuran gambar.")
            else:
                self.logger.error(f"🚫 Runtime error saat deteksi: {e}")
                raise ModelError(f"Error saat menjalankan model deteksi: {e}")
        except Exception as e:
            self.logger.error(f"🚫 Error tidak terduga saat deteksi: {e}")
            raise ModelError(f"Error saat deteksi: {e}")