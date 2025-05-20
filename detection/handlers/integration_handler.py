"""
File: smartcash/detection/handlers/integration_handler.py
Deskripsi: Handler untuk integrasi deteksi dengan UI/API.
"""

import os
import json
import base64
import numpy as np
import io
from PIL import Image
from typing import Dict, List, Optional, Union, Any, Callable
import threading
import queue
import time

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection, ImageType
from smartcash.common.utils import format_time


class IntegrationHandler:
    """Handler untuk integrasi deteksi dengan UI dan API"""
    
    def __init__(self, 
                 detection_handler, 
                 logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Integration Handler
        
        Args:
            detection_handler: Handler untuk deteksi gambar tunggal
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.detection_handler = detection_handler
        self.logger = logger or get_logger()
        
        # Queue untuk proses asinkron
        self._task_queue = queue.Queue()
        self._result_cache = {}
        self._worker_thread = None
        self._stop_worker = False
        
    def detect_from_base64(self, 
                          image_base64: str, 
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45,
                          return_visualization: bool = False,
                          visualization_format: str = 'base64') -> Dict:
        """
        Deteksi objek dari gambar base64
        
        Args:
            image_base64: String base64 gambar
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            return_visualization: Flag untuk mengembalikan visualisasi hasil
            visualization_format: Format visualisasi ('base64', 'numpy')
            
        Returns:
            Dictionary berisi hasil deteksi dan visualisasi (opsional)
        """
        try:
            # Decode base64 ke gambar
            if ',' in image_base64:
                # Handle data URL scheme (data:image/jpeg;base64,...)
                image_base64 = image_base64.split(',', 1)[1]
                
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            start_time = time.time()
            
            # Deteksi objek
            if return_visualization:
                detections, visualization = self.detection_handler.detect(
                    image=image_np,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    return_visualization=True
                )
            else:
                detections = self.detection_handler.detect(
                    image=image_np,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    return_visualization=False
                )
                visualization = None
            
            processing_time = time.time() - start_time
            
            # Format hasil deteksi
            result = {
                "status": "success",
                "processing_time": processing_time,
                "detections": [d.__dict__ for d in detections],
                "count": len(detections)
            }
            
            # Format visualisasi jika diminta
            if return_visualization and visualization is not None:
                if visualization_format == 'base64':
                    # Konversi numpy array ke base64
                    pil_img = Image.fromarray(visualization)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    result["visualization"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
                elif visualization_format == 'numpy':
                    result["visualization"] = visualization.tolist()
            
            self.logger.debug(f"✅ Deteksi base64 selesai: {len(detections)} objek, {processing_time:.4f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error pada deteksi base64: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def start_async_worker(self, num_workers: int = 1):
        """
        Memulai worker thread untuk pemrosesan asinkron
        
        Args:
            num_workers: Jumlah worker thread
        """
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.logger.warning("⚠️ Worker thread sudah berjalan")
            return
            
        self._stop_worker = False
        self._worker_thread = threading.Thread(target=self._async_worker_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        
        self.logger.info(f"🔄 Memulai worker thread asinkron")
    
    def stop_async_worker(self):
        """Menghentikan worker thread untuk pemrosesan asinkron"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
            
        self._stop_worker = True
        self._worker_thread.join(timeout=5.0)
        
        # Bersihkan queue
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                break
                
        self.logger.info(f"🛑 Menghentikan worker thread asinkron")
    
    def detect_async(self, 
                    image: ImageType, 
                    task_id: Optional[str] = None,
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45,
                    callback: Optional[Callable] = None,
                    return_visualization: bool = False) -> str:
        """
        Menambahkan task deteksi asinkron ke queue
        
        Args:
            image: Gambar yang akan dideteksi
            task_id: ID unik untuk task (opsional, akan dibuat secara otomatis)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            callback: Callback function saat task selesai
            return_visualization: Flag untuk mengembalikan visualisasi
            
        Returns:
            task_id untuk polling hasil
        """
        # Buat task_id jika tidak disediakan
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        # Tambahkan task ke queue
        task = {
            "id": task_id,
            "image": image,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "return_visualization": return_visualization,
            "callback": callback,
            "status": "queued",
            "timestamp": time.time()
        }
        
        self._result_cache[task_id] = {
            "status": "queued",
            "timestamp": time.time()
        }
        
        self._task_queue.put(task)
        
        # Mulai worker jika belum berjalan
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self.start_async_worker()
            
        self.logger.debug(f"➕ Menambahkan task asinkron {task_id} ke queue")
        return task_id
    
    def get_task_result(self, task_id: str, remove_after_get: bool = True) -> Dict:
        """
        Dapatkan hasil task asinkron
        
        Args:
            task_id: ID task
            remove_after_get: Flag untuk menghapus hasil setelah diambil
            
        Returns:
            Dictionary berisi hasil task atau status task jika belum selesai
        """
        if task_id not in self._result_cache:
            return {"status": "not_found"}
            
        result = self._result_cache[task_id]
        
        # Hapus hasil jika diminta dan task sudah selesai
        if remove_after_get and result["status"] in ["success", "error"]:
            if task_id in self._result_cache:
                del self._result_cache[task_id]
                
        return result
    
    def get_queue_status(self) -> Dict:
        """
        Dapatkan status queue dan task
        
        Returns:
            Dictionary berisi statistik queue dan task
        """
        # Hitung status task
        task_counts = {"queued": 0, "processing": 0, "success": 0, "error": 0}
        
        for task_id, result in self._result_cache.items():
            status = result.get("status", "unknown")
            if status in task_counts:
                task_counts[status] += 1
                
        # Dapatkan info queue
        queue_size = self._task_queue.qsize()
        worker_running = self._worker_thread is not None and self._worker_thread.is_alive()
        
        return {
            "queue_size": queue_size,
            "worker_running": worker_running,
            "task_counts": task_counts,
            "total_tasks": len(self._result_cache)
        }
    
    def clean_old_results(self, max_age_seconds: float = 3600):
        """
        Bersihkan hasil task lama
        
        Args:
            max_age_seconds: Usia maksimum hasil dalam detik
        """
        current_time = time.time()
        task_ids_to_remove = []
        
        for task_id, result in self._result_cache.items():
            # Hanya hapus task yang sudah selesai
            if result["status"] in ["success", "error"]:
                age = current_time - result.get("timestamp", current_time)
                if age > max_age_seconds:
                    task_ids_to_remove.append(task_id)
        
        # Hapus task lama
        for task_id in task_ids_to_remove:
            if task_id in self._result_cache:
                del self._result_cache[task_id]
                
        self.logger.debug(f"🧹 Membersihkan {len(task_ids_to_remove)} hasil task lama")
    
    def _async_worker_loop(self):
        """Loop worker untuk pemrosesan asinkron"""
        self.logger.info(f"🔄 Worker thread asinkron dimulai")
        
        while not self._stop_worker:
            try:
                # Ambil task dari queue dengan timeout
                try:
                    task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update status
                task_id = task["id"]
                self._result_cache[task_id]["status"] = "processing"
                
                # Proses task
                start_time = time.time()
                self.logger.debug(f"▶️ Memproses task asinkron {task_id}")
                
                try:
                    # Deteksi objek
                    if task["return_visualization"]:
                        detections, visualization = self.detection_handler.detect(
                            image=task["image"],
                            conf_threshold=task["conf_threshold"],
                            iou_threshold=task["iou_threshold"],
                            return_visualization=True
                        )
                    else:
                        detections = self.detection_handler.detect(
                            image=task["image"],
                            conf_threshold=task["conf_threshold"],
                            iou_threshold=task["iou_threshold"],
                            return_visualization=False
                        )
                        visualization = None
                    
                    processing_time = time.time() - start_time
                    
                    # Format hasil
                    result = {
                        "status": "success",
                        "processing_time": processing_time,
                        "timestamp": time.time(),
                        "detections": [d.__dict__ for d in detections],
                        "count": len(detections),
                        "queue_time": start_time - task["timestamp"]
                    }
                    
                    # Tambahkan visualisasi jika diminta
                    if task["return_visualization"] and visualization is not None:
                        pil_img = Image.fromarray(visualization)
                        buffered = io.BytesIO()
                        pil_img.save(buffered, format="JPEG")
                        result["visualization"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Simpan hasil
                    self._result_cache[task_id] = result
                    
                    self.logger.debug(f"✅ Task {task_id} selesai dalam {processing_time:.2f}s: {len(detections)} objek")
                    
                except Exception as e:
                    # Simpan error
                    self._result_cache[task_id] = {
                        "status": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    }
                    
                    self.logger.error(f"❌ Error pada task {task_id}: {str(e)}")
                
                # Panggil callback jika ada
                if task["callback"] is not None:
                    try:
                        task["callback"](task_id, self._result_cache[task_id])
                    except Exception as e:
                        self.logger.error(f"❌ Error pada callback task {task_id}: {str(e)}")
                
                # Tandai task selesai
                self._task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"❌ Error pada worker thread: {str(e)}")
                
        self.logger.info(f"🛑 Worker thread asinkron berhenti")
    
    def to_json_response(self, result: Dict) -> str:
        """
        Konversi hasil ke respons JSON untuk API
        
        Args:
            result: Dictionary hasil deteksi
            
        Returns:
            String JSON
        """
        return json.dumps(result, indent=2, default=str)