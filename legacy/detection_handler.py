# File: smartcash/handlers/detection_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk deteksi mata uang Rupiah dari gambar atau video

import os
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.utils.visualization import ResultVisualizer
from smartcash.exceptions.base import DataError

class DetectionHandler:
    """Handler untuk deteksi mata uang Rupiah menggunakan model YOLOv5."""
    
    # Konfigurasi kelas mata uang
    CURRENCY_CLASSES = {
        0: {'name': '100', 'color': (255, 0, 0)},     # 100rb - Biru
        1: {'name': '050', 'color': (0, 0, 255)},     # 50rb - Merah
        2: {'name': '020', 'color': (0, 255, 0)},     # 20rb - Hijau
        3: {'name': '010', 'color': (128, 0, 128)},   # 10rb - Ungu
        4: {'name': '005', 'color': (0, 128, 128)},   # 5rb - Coklat
        5: {'name': '002', 'color': (128, 128, 0)},   # 2rb - Abu-Abu
        6: {'name': '001', 'color': (0, 0, 128)},     # 1rb - Merah Tua
        7: {'name': 'l2_100', 'color': (255, 50, 50)},    # Layer 2 (nominal) 100rb
        8: {'name': 'l2_050', 'color': (50, 50, 255)},    # Layer 2 (nominal) 50rb
        9: {'name': 'l2_020', 'color': (50, 255, 50)},    # Layer 2 (nominal) 20rb
        10: {'name': 'l2_010', 'color': (178, 50, 178)},  # Layer 2 (nominal) 10rb
        11: {'name': 'l2_005', 'color': (50, 178, 178)},  # Layer 2 (nominal) 5rb
        12: {'name': 'l2_002', 'color': (178, 178, 50)},  # Layer 2 (nominal) 2rb
        13: {'name': 'l2_001', 'color': (50, 50, 178)},   # Layer 2 (nominal) 1rb
        14: {'name': 'l3_sign', 'color': (255, 150, 0)},  # Layer 3 (security) tanda tangan
        15: {'name': 'l3_text', 'color': (150, 255, 0)},  # Layer 3 (security) teks
        16: {'name': 'l3_thread', 'color': (0, 150, 255)} # Layer 3 (security) benang
    }
    
    def __init__(
        self, 
        config: Dict,
        weights_path: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi detection handler.
        
        Args:
            config: Konfigurasi aplikasi
            weights_path: Path ke model weights (opsional)
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Siapkan output dir jika belum ada
        self.output_dir = Path(config.get('output_dir', 'results/detection'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Siapkan model
        self.model = self._load_model(weights_path)
        
        # Siapkan visualizer
        self.visualizer = ResultVisualizer(output_dir=str(self.output_dir))
        
    def _load_model(self, weights_path: Optional[str] = None) -> YOLOv5Model:
        """
        Load model untuk deteksi.
        
        Args:
            weights_path: Path ke model weights
            
        Returns:
            Model yang sudah dimuat
            
        Raises:
            FileNotFoundError: Jika weights tidak ditemukan
        """
        # Gunakan weights dari config jika tidak disediakan
        if weights_path is None:
            weights_path = self.config.get('model', {}).get('weights_path')
            
        # Cek ketersediaan weights
        if weights_path is None or not Path(weights_path).exists():
            # Coba cari weights terbaru di direktori checkpoints
            checkpoints_dir = Path(self.config.get('checkpoints_dir', 'runs/train/weights'))
            
            if checkpoints_dir.exists():
                # Cari weights terbaik
                best_weights = list(checkpoints_dir.glob('*_best.pt'))
                if best_weights:
                    weights_path = str(max(best_weights, key=os.path.getmtime))
                    self.logger.info(f"🔍 Menggunakan weights terbaru: {weights_path}")
                else:
                    # Cari weights apapun
                    all_weights = list(checkpoints_dir.glob('*.pt'))
                    if all_weights:
                        weights_path = str(max(all_weights, key=os.path.getmtime))
                        self.logger.info(f"🔍 Menggunakan weights yang tersedia: {weights_path}")
                    else:
                        raise FileNotFoundError("❌ Tidak ditemukan model weights")
            else:
                raise FileNotFoundError(f"❌ Direktori weights tidak ditemukan: {checkpoints_dir}")
        
        # Load YOLOv5 model
        backbone = self.config.get('backbone', 'cspdarknet')
        num_classes = len(self.CURRENCY_CLASSES)
        
        self.logger.info(f"🔄 Memuat model YOLOv5 dengan backbone {backbone}...")
        
        # Inisialisasi model
        model = YOLOv5Model(
            backbone_type=backbone,
            num_classes=num_classes
        )
        
        # Load weights
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.success(f"✅ Model berhasil dimuat dari {weights_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise
        
        # Pindahkan ke GPU jika tersedia
        if torch.cuda.is_available():
            model = model.cuda()
            self.logger.info("🚀 Model berjalan di GPU")
        else:
            self.logger.info("💻 Model berjalan di CPU")
        
        # Set ke mode evaluasi
        model.eval()
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess gambar untuk inferensi.
        
        Args:
            image: Gambar input (BGR)
            
        Returns:
            Tensor yang siap untuk inferensi
        """
        # Resize
        img_size = self.config.get('model', {}).get('img_size', [640, 640])
        image = cv2.resize(image, tuple(img_size))
        
        # BGR ke RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalisasi
        image = image.astype(np.float32) / 255.0
        
        # HWC ke CHW
        image = image.transpose(2, 0, 1)
        
        # Tambahkan batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Konversi ke tensor
        image_tensor = torch.from_numpy(image)
        
        # Pindahkan ke device yang sama dengan model
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            
        return image_tensor
    
    def detect(
        self, 
        source: str,
        conf_thres: float = 0.25,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deteksi mata uang dari gambar atau video.
        
        Args:
            source: Path ke gambar/video atau "0" untuk webcam
            conf_thres: Threshold confidence
            output_dir: Direktori untuk menyimpan hasil (opsional)
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        # Update output dir jika ada
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Periksa source
        if source == "0":
            return self._detect_from_webcam(conf_thres)
        
        source_path = Path(source)
        if not source_path.exists():
            raise DataError(f"❌ Source tidak ditemukan: {source}")
            
        if source_path.is_file():
            # Cek ekstensi
            suffix = source_path.suffix.lower()
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                return self._detect_from_image(source_path, conf_thres)
            elif suffix in ['.mp4', '.avi', '.mov']:
                return self._detect_from_video(source_path, conf_thres)
            else:
                raise DataError(f"❌ Format file tidak didukung: {suffix}")
        elif source_path.is_dir():
            return self._detect_from_directory(source_path, conf_thres)
        else:
            raise DataError(f"❌ Tipe source tidak valid: {source}")
    
    def _detect_from_image(
        self, 
        image_path: Path, 
        conf_thres: float
    ) -> Dict[str, Any]:
        """
        Deteksi dari file gambar.
        
        Args:
            image_path: Path ke file gambar
            conf_thres: Threshold confidence
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        self.logger.info(f"🔍 Mendeteksi dari gambar: {image_path}")
        
        try:
            # Baca gambar
            image = cv2.imread(str(image_path))
            if image is None:
                raise DataError(f"❌ Gagal membaca gambar: {image_path}")
                
            # Simpan ukuran asli
            original_size = image.shape[:2]  # (height, width)
            
            # Preprocess
            image_tensor = self.preprocess_image(image)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                predictions = self.model(image_tensor)
            inference_time = time.time() - start_time
            
            # Postprocess
            detections = self._postprocess_predictions(
                predictions,
                conf_thres,
                original_size
            )
            
            # Visualisasi hasil
            result_path = self._visualize_result(
                image, 
                detections,
                str(image_path.stem)
            )
            
            self.logger.success(
                f"✅ Deteksi berhasil: {len(detections)} objek terdeteksi\n"
                f"   Waktu inferensi: {inference_time*1000:.2f}ms\n"
                f"   Hasil tersimpan di: {result_path}"
            )
            
            return {
                'source': str(image_path),
                'detections': detections,
                'inference_time': inference_time,
                'result_path': result_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendeteksi dari gambar: {str(e)}")
            raise
    
    def _detect_from_video(
        self, 
        video_path: Path, 
        conf_thres: float
    ) -> Dict[str, Any]:
        """
        Deteksi dari file video.
        
        Args:
            video_path: Path ke file video
            conf_thres: Threshold confidence
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        self.logger.info(f"🔍 Mendeteksi dari video: {video_path}")
        
        try:
            # Buka video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise DataError(f"❌ Gagal membuka video: {video_path}")
                
            # Dapatkan info video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Siapkan output video
            output_path = self.output_dir / f"{video_path.stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Process video
            frame_detections = []
            total_inference_time = 0
            
            with tqdm(total=total_frames, desc="⏳ Memproses video") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Preprocess
                    image_tensor = self.preprocess_image(frame)
                    
                    # Inference
                    start_time = time.time()
                    with torch.no_grad():
                        predictions = self.model(image_tensor)
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time
                    
                    # Postprocess
                    detections = self._postprocess_predictions(
                        predictions,
                        conf_thres,
                        (height, width)
                    )
                    
                    # Tambahkan ke hasil
                    frame_detections.append(detections)
                    
                    # Visualisasi pada frame
                    annotated_frame = self._draw_detections(frame, detections)
                    
                    # Tambahkan info FPS
                    fps_text = f"FPS: {1/inference_time:.1f}"
                    cv2.putText(
                        annotated_frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Tulis ke output
                    out.write(annotated_frame)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({'FPS': f"{1/inference_time:.1f}"})
            
            # Tutup resources
            cap.release()
            out.release()
            
            # Hitung rata-rata inferensi
            avg_inference_time = total_inference_time / total_frames
            
            self.logger.success(
                f"✅ Deteksi video berhasil\n"
                f"   Total frame: {total_frames}\n"
                f"   Rata-rata FPS: {1/avg_inference_time:.1f}\n"
                f"   Hasil tersimpan di: {output_path}"
            )
            
            return {
                'source': str(video_path),
                'detections': frame_detections,
                'inference_time': avg_inference_time,
                'result_path': str(output_path),
                'total_frames': total_frames
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendeteksi dari video: {str(e)}")
            raise
            
    def _detect_from_directory(
        self, 
        dir_path: Path, 
        conf_thres: float
    ) -> Dict[str, Any]:
        """
        Deteksi dari direktori berisi gambar.
        
        Args:
            dir_path: Path ke direktori
            conf_thres: Threshold confidence
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        self.logger.info(f"🔍 Mendeteksi dari direktori: {dir_path}")
        
        # Cari semua file gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise DataError(f"❌ Tidak ada file gambar di direktori: {dir_path}")
            
        # Proses setiap gambar
        results = []
        total_detections = 0
        total_time = 0
        
        with tqdm(image_files, desc="🖼️ Memproses gambar") as pbar:
            for img_path in pbar:
                try:
                    result = self._detect_from_image(img_path, conf_thres)
                    results.append(result)
                    
                    total_detections += len(result['detections'])
                    total_time += result['inference_time']
                    
                    # Update progress
                    pbar.set_postfix({
                        'deteksi': len(result['detections']),
                        'waktu': f"{result['inference_time']*1000:.1f}ms"
                    })
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memproses {img_path.name}: {str(e)}")
        
        # Hitung rata-rata
        avg_inference_time = total_time / len(image_files)
        
        self.logger.success(
            f"✅ Deteksi batch berhasil\n"
            f"   Total gambar: {len(image_files)}\n"
            f"   Total deteksi: {total_detections}\n"
            f"   Rata-rata waktu: {avg_inference_time*1000:.1f}ms\n"
            f"   Hasil tersimpan di: {self.output_dir}"
        )
        
        return {
            'source': str(dir_path),
            'results': results,
            'total_images': len(image_files),
            'total_detections': total_detections,
            'average_inference_time': avg_inference_time
        }
    
    def _detect_from_webcam(self, conf_thres: float) -> Dict[str, Any]:
        """
        Deteksi dari webcam.
        
        Args:
            conf_thres: Threshold confidence
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        self.logger.info("🎥 Memulai deteksi dari webcam")
        
        try:
            # Buka webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise DataError("❌ Gagal membuka webcam")
                
            # Dapatkan info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Siapkan output video
            output_path = self.output_dir / f"webcam_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                20,  # FPS
                (width, height)
            )
            
            # Informasi runtime
            self.logger.info(
                f"ℹ️ Deteksi webcam aktif\n"
                f"   Tekan 'q' untuk keluar\n"
                f"   Output: {output_path}"
            )
            
            # Process webcam
            frame_count = 0
            total_inference_time = 0
            fps_history = []
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Preprocess
                    image_tensor = self.preprocess_image(frame)
                    
                    # Inference
                    start_time = time.time()
                    with torch.no_grad():
                        predictions = self.model(image_tensor)
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time
                    
                    # Update FPS history (last 10 frames)
                    fps_history.append(1/inference_time)
                    if len(fps_history) > 10:
                        fps_history.pop(0)
                    current_fps = sum(fps_history) / len(fps_history)
                    
                    # Postprocess
                    detections = self._postprocess_predictions(
                        predictions,
                        conf_thres,
                        (height, width)
                    )
                    
                    # Visualisasi pada frame
                    annotated_frame = self._draw_detections(frame, detections)
                    
                    # Tambahkan info FPS
                    fps_text = f"FPS: {current_fps:.1f}"
                    cv2.putText(
                        annotated_frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Tambahkan jumlah deteksi
                    det_text = f"Deteksi: {len(detections)}"
                    cv2.putText(
                        annotated_frame,
                        det_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Tulis ke output
                    out.write(annotated_frame)
                    
                    # Tampilkan
                    cv2.imshow("SmartCash Detection", annotated_frame)
                    
                    # Update counter
                    frame_count += 1
                    
                    # Cek keyboard
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                self.logger.info("👋 Deteksi dihentikan oleh pengguna")
            
            # Tutup resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Hitung rata-rata inferensi
            avg_inference_time = total_inference_time / max(1, frame_count)
            
            self.logger.success(
                f"✅ Deteksi webcam selesai\n"
                f"   Total frame: {frame_count}\n"
                f"   Rata-rata FPS: {1/avg_inference_time:.1f}\n"
                f"   Hasil tersimpan di: {output_path}"
            )
            
            return {
                'source': 'webcam',
                'result_path': str(output_path),
                'total_frames': frame_count,
                'average_fps': 1/avg_inference_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendeteksi dari webcam: {str(e)}")
            raise
    
    def _postprocess_predictions(
        self,
        predictions: Union[torch.Tensor, List[torch.Tensor]],
        conf_thres: float,
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess prediksi model.
        
        Args:
            predictions: Output dari model
            conf_thres: Threshold confidence
            original_size: Ukuran asli gambar (height, width)
            
        Returns:
            List dictionary berisi bounding box, kelas, dan confidence yang terdeteksi
        """
        # Handle multi-scale predictions
        if isinstance(predictions, list):
            # Ambil prediksi dari skala terbaik (medium scale)
            if len(predictions) >= 2:
                predictions = predictions[1]  # Medium scale
            else:
                predictions = predictions[0]
        
        # Konversi ke CPU jika diperlukan
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Inisialisasi list deteksi
        detections = []
        
        # Ekstrak ukuran asli
        orig_h, orig_w = original_size
        
        # Loop untuk setiap batch
        for pred in predictions:
            # Loop untuk setiap anchor
            for anchor_pred in pred:
                # Loop untuk setiap grid cell
                for y in range(anchor_pred.shape[0]):
                    for x in range(anchor_pred.shape[1]):
                        # Ekstrak nilai prediksi
                        box = anchor_pred[y, x, :4]
                        conf = anchor_pred[y, x, 4]
                        cls_scores = anchor_pred[y, x, 5:]
                        
                        # Filter berdasarkan confidence
                        if conf < conf_thres:
                            continue
                            
                        # Temukan kelas dengan skor tertinggi
                        cls_id = np.argmax(cls_scores)
                        cls_score = cls_scores[cls_id]
                        
                        # Filter berdasarkan skor kelas
                        if cls_score < conf_thres:
                            continue
                            
                        # Kalkulasi total confidence
                        total_conf = conf * cls_score
                        
                        # Konversi koordinat relatif ke absolut
                        cx, cy, w, h = box
                        x1 = (cx - w/2) * orig_w
                        y1 = (cy - h/2) * orig_h
                        x2 = (cx + w/2) * orig_w
                        y2 = (cy + h/2) * orig_h
                        
                        # Simpan deteksi
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class_id': int(cls_id),
                            'class_name': self.CURRENCY_CLASSES[cls_id]['name'],
                            'confidence': float(total_conf)
                        })
        
        # Sort berdasarkan confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Gambar hasil deteksi pada gambar.
        
        Args:
            image: Gambar asli
            detections: List deteksi
            
        Returns:
            Gambar dengan anotasi
        """
        # Copy gambar agar tidak memodifikasi aslinya
        annotated_img = image.copy()
        
        # Gambar setiap deteksi
        for det in detections:
            # Ekstrak info
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            cls_name = det['class_name']
            conf = det['confidence']
            
            # Ambil warna
            color = self.CURRENCY_CLASSES[cls_id]['color']
            
            # Gambar bbox
            cv2.rectangle(
                annotated_img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            
            # Siapkan label
            label = f"{cls_name} {conf:.2f}"
            
            # Gambar background label
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )[0]
            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0], y1),
                color,
                -1
            )
            
            # Gambar label
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
        return annotated_img
    
    def _visualize_result(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        name_prefix: str
    ) -> str:
        """
        Visualisasi dan simpan hasil deteksi.
        
        Args:
            image: Gambar asli
            detections: List deteksi
            name_prefix: Prefix untuk nama file
            
        Returns:
            Path ke gambar hasil
        """
        # Gambar deteksi
        annotated_img = self._draw_detections(image, detections)
        
        # Tambahkan label total
        cv2.putText(
            annotated_img,
            f"Deteksi: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Generate nama file
        result_path = self.output_dir / f"{name_prefix}_detected.jpg"
        
        # Simpan gambar
        cv2.imwrite(str(result_path), annotated_img)
        
        return str(result_path)