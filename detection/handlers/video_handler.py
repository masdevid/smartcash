"""
File: smartcash/detection/handlers/video_handler.py
Deskripsi: Handler untuk deteksi objek pada video dan webcam.
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm

from smartcash.common.utils import ensure_dir, get_timestamp
from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class VideoHandler:
    """Handler untuk mengelola proses deteksi pada video dan webcam"""
    
    def __init__(self, 
                 detection_handler, 
                 logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Video Handler
        
        Args:
            detection_handler: Handler untuk deteksi gambar tunggal
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.detection_handler = detection_handler
        self.logger = logger or get_logger("VideoHandler")
        self._stop_flag = False
    
    def detect_video(self, 
                    video_path: str, 
                    output_path: Optional[str] = None,
                    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                    start_frame: int = 0,
                    end_frame: Optional[int] = None,
                    step: int = 1,
                    show_progress: bool = True,
                    show_preview: bool = False,
                    overlay_info: bool = True,
                    callback: Optional[Callable] = None) -> Dict:
        """
        Deteksi objek pada file video
        
        Args:
            video_path: Path ke file video
            output_path: Path ke file video output (opsional)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            start_frame: Frame pertama yang diproses
            end_frame: Frame terakhir yang diproses (None=sampai akhir)
            step: Langkah frame (1=setiap frame, 2=setiap 2 frame, dll)
            show_progress: Flag untuk menampilkan progress bar
            show_preview: Flag untuk menampilkan preview saat proses
            overlay_info: Flag untuk menampilkan info overlay pada output
            callback: Callback function untuk setiap frame
            
        Returns:
            Dictionary berisi hasil deteksi dan statistik
        """
        # Buka video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"❌ Tidak dapat membuka video: {video_path}")
            return {"status": "error", "message": "Tidak dapat membuka video"}
        
        # Dapatkan properti video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tentukan end_frame jika tidak ditentukan
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        # Jumlah frame yang akan diproses
        frames_to_process = (end_frame - start_frame) // step
        
        # Setup video writer jika output_path ditentukan
        video_writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk MP4
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup progress bar jika diperlukan
        progress = None
        if show_progress:
            progress = tqdm(total=frames_to_process, desc="🎬 Memproses video", unit="frame")
        
        # Statistik
        stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "processing_time": 0,
            "start_time": time.time()
        }
        
        # Pindah ke frame awal
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Proses frame video
        frame_idx = start_frame
        self._stop_flag = False
        
        while frame_idx < end_frame and not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Proses frame jika sesuai step
            if (frame_idx - start_frame) % step == 0:
                # Hitung waktu deteksi
                start_time = time.time()
                
                # Deteksi objek pada frame
                detections = self.detection_handler.detect(
                    image=frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Update statistik
                processing_time = time.time() - start_time
                stats["frames_processed"] += 1
                stats["total_detections"] += len(detections)
                stats["processing_time"] += processing_time
                
                # Visualisasi hasil deteksi
                if output_path or show_preview:
                    # Dapatkan frame dengan visualisasi deteksi
                    visualized_frame = self.detection_handler.inference_service.visualize(frame, detections)
                    
                    # Tambahkan overlay info jika diminta
                    if overlay_info:
                        visualized_frame = self._add_overlay_text(
                            visualized_frame,
                            [
                                f"FPS: {1/processing_time:.1f}" if processing_time > 0 else "FPS: N/A",
                                f"Frame: {frame_idx}/{total_frames}",
                                f"Deteksi: {len(detections)}"
                            ]
                        )
                    
                    # Tulis ke file output jika ada
                    if video_writer:
                        video_writer.write(visualized_frame)
                    
                    # Tampilkan preview jika diminta
                    if show_preview:
                        cv2.imshow('Video Detection Preview', visualized_frame)
                        # Tekan 'q' untuk keluar
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self._stop_flag = True
                            break
                
                # Panggil callback jika ada
                if callback:
                    stop_requested = callback(frame_idx, frame, detections, processing_time)
                    if stop_requested:
                        self._stop_flag = True
                        break
                
                # Update progress bar
                if progress:
                    progress.update(1)
            
            # Pindah ke frame berikutnya
            frame_idx += 1
        
        # Hitung statistik akhir
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        stats["avg_processing_time"] = stats["processing_time"] / max(stats["frames_processed"], 1)
        stats["avg_fps"] = stats["frames_processed"] / max(stats["total_time"], 0.001)
        stats["avg_detections_per_frame"] = stats["total_detections"] / max(stats["frames_processed"], 1)
        
        # Tutup resources
        cap.release()
        if video_writer:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        if progress:
            progress.close()
        
        if self._stop_flag:
            self.logger.info(f"🛑 Proses deteksi dihentikan (frame {frame_idx}/{end_frame})")
        else:
            self.logger.info(f"✅ Proses video selesai: {stats['frames_processed']} frame, "
                           f"{stats['total_detections']} objek terdeteksi, "
                           f"{stats['avg_fps']:.2f} FPS")
                           
        return {
            "status": "success" if not self._stop_flag else "stopped",
            "stats": stats,
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": stats["frames_processed"],
                "input_path": video_path,
                "output_path": output_path
            }
        }
    
    def detect_webcam(self, 
                     camera_id: int = 0,
                     output_path: Optional[str] = None,
                     conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                     iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                     display_width: int = 1280,
                     display_height: int = 720,
                     overlay_info: bool = True,
                     max_time: Optional[float] = None,
                     callback: Optional[Callable] = None) -> Dict:
        """
        Deteksi objek pada webcam
        
        Args:
            camera_id: ID kamera (default 0)
            output_path: Path ke file video output (opsional)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            display_width: Lebar display
            display_height: Tinggi display
            overlay_info: Flag untuk menampilkan info overlay
            max_time: Waktu maksimum rekaman (None=tanpa batas)
            callback: Callback function untuk setiap frame
            
        Returns:
            Dictionary berisi hasil deteksi dan statistik
        """
        # Buka webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"❌ Tidak dapat membuka kamera ID {camera_id}")
            return {"status": "error", "message": "Tidak dapat membuka kamera"}
        
        # Set resolusi webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        
        # Dapatkan properti webcam aktual
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer jika output_path ditentukan
        video_writer = None
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            # Jika path adalah direktori, buat nama file otomatis
            if os.path.isdir(output_path):
                timestamp = get_timestamp()
                output_path = os.path.join(output_path, f"webcam_{timestamp}.mp4")
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk MP4
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"📹 Merekam ke {output_path}")
        
        # Statistik
        stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "processing_time": 0,
            "start_time": time.time()
        }
        
        # Variabel untuk penghitungan FPS
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        # Proses frame webcam
        self._stop_flag = False
        
        self.logger.info(f"🎥 Memulai deteksi dari webcam (ID: {camera_id})")
        
        # Hitung waktu selesai jika max_time ditentukan
        end_time = None
        if max_time is not None:
            end_time = time.time() + max_time
            
        while not self._stop_flag:
            # Cek waktu maksimum
            if end_time and time.time() > end_time:
                self.logger.info(f"⏱️ Waktu maksimum ({max_time}s) tercapai")
                break
                
            # Baca frame
            ret, frame = cap.read()
            if not ret:
                self.logger.error("❌ Tidak dapat membaca frame dari kamera")
                break
            
            # Hitung waktu deteksi
            start_time = time.time()
            
            # Deteksi objek pada frame
            detections = self.detection_handler.detect(
                image=frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Update statistik
            processing_time = time.time() - start_time
            stats["frames_processed"] += 1
            stats["total_detections"] += len(detections)
            stats["processing_time"] += processing_time
            
            # Hitung FPS setiap detik
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            
            # Visualisasi hasil deteksi
            visualized_frame = self.detection_handler.inference_service.visualize(frame, detections)
            
            # Tambahkan overlay info jika diminta
            if overlay_info:
                visualized_frame = self._add_overlay_text(
                    visualized_frame,
                    [
                        f"FPS: {current_fps:.1f}",
                        f"Deteksi: {len(detections)}",
                        f"Waktu: {(time.time() - stats['start_time']):.1f}s"
                    ]
                )
            
            # Tulis ke file output jika ada
            if video_writer:
                video_writer.write(visualized_frame)
            
            # Tampilkan preview
            cv2.imshow('Webcam Detection', visualized_frame)
            
            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop_flag = True
                break
            
            # Panggil callback jika ada
            if callback:
                stop_requested = callback(stats["frames_processed"], frame, detections, processing_time)
                if stop_requested:
                    self._stop_flag = True
                    break
        
        # Hitung statistik akhir
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        stats["avg_processing_time"] = stats["processing_time"] / max(stats["frames_processed"], 1)
        stats["avg_fps"] = stats["frames_processed"] / max(stats["total_time"], 0.001)
        stats["avg_detections_per_frame"] = stats["total_detections"] / max(stats["frames_processed"], 1)
        
        # Tutup resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        self.logger.info(f"✅ Selesai deteksi webcam: {stats['frames_processed']} frame, "
                       f"{stats['total_detections']} objek terdeteksi, "
                       f"{stats['avg_fps']:.2f} FPS")
                       
        return {
            "status": "success" if not self._stop_flag else "stopped",
            "stats": stats,
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "input": f"Webcam ID {camera_id}",
                "output_path": output_path
            }
        }
    
    def stop(self):
        """Menghentikan proses deteksi yang sedang berjalan"""
        self._stop_flag = True
        self.logger.info("🛑 Menghentikan proses deteksi video/webcam")
    
    def _add_overlay_text(self, frame, text_lines, start_y=30, line_height=30, color=(0, 255, 0), thickness=2, font_scale=0.7):
        """
        Tambahkan teks overlay ke frame
        
        Args:
            frame: Frame untuk ditambahkan teks
            text_lines: List teks yang akan ditampilkan
            start_y: Posisi y awal
            line_height: Tinggi baris
            color: Warna teks (B,G,R)
            thickness: Ketebalan teks
            font_scale: Skala font
            
        Returns:
            Frame dengan teks overlay
        """
        for i, text in enumerate(text_lines):
            y_pos = start_y + (i * line_height)
            cv2.putText(frame, text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return frame