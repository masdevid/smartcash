# File: smartcash/handlers/model/core/model_predictor.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk prediksi menggunakan model yang sudah dilatih

import torch
import time
import numpy as np
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path
import cv2

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError
from smartcash.utils.visualization.detection import DetectionVisualizer

class ModelPredictor(ModelComponent):
    """
    Komponen untuk prediksi menggunakan model yang sudah dilatih.
    Mendukung prediksi pada gambar tunggal atau batch, dengan visualisasi hasil.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        model_factory = None
    ):
        """
        Inisialisasi model predictor.
        
        Args:
            config: Konfigurasi model dan prediksi
            logger: Custom logger (opsional)
            model_factory: Factory untuk membuat model (opsional, lazy-loaded)
        """
        super().__init__(config, logger, "model_predictor")
        
        # Simpan factory
        self._model_factory = model_factory
    
    def _initialize(self) -> None:
        """Inisialisasi internal komponen."""
        self.inference_config = self.config.get('inference', {})
        
        # Default parameter inference
        self.conf_threshold = self.inference_config.get('conf_threshold', 0.25)
        self.iou_threshold = self.inference_config.get('iou_threshold', 0.45)
        self.img_size = self.config.get('model', {}).get('img_size', [640, 640])
        
        # Parameter visualisasi
        self.visualize = self.inference_config.get('visualize', True)
        self.show_labels = self.inference_config.get('show_labels', True)
        self.show_conf = self.inference_config.get('show_conf', True)
        self.show_value = self.inference_config.get('show_value', True)
        
        # Direktori output
        self.output_dir = self.inference_config.get(
            'output_dir', 
            str(Path(self.config.get('output_dir', 'runs/predict')) / "results")
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi visualizer
        self.detection_visualizer = DetectionVisualizer(output_dir=self.output_dir, logger=self.logger)
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        if self._model_factory is None:
            from smartcash.handlers.model.core.model_factory import ModelFactory
            self._model_factory = ModelFactory(self.config, self.logger)
        return self._model_factory
    
    def process(
        self,
        images: Union[torch.Tensor, np.ndarray, List, str, Path],
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses prediksi. Alias untuk predict().
        
        Args:
            images: Input gambar untuk prediksi
            model: Model untuk prediksi (opsional)
            **kwargs: Parameter tambahan untuk prediksi
            
        Returns:
            Dict hasil prediksi
        """
        return self.predict(images, model, **kwargs)
    
    def predict(
        self,
        images: Union[torch.Tensor, np.ndarray, List, str, Path],
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
        half_precision: Optional[bool] = None,
        visualize: Optional[bool] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prediksi dengan model.
        
        Args:
            images: Input gambar untuk prediksi
            model: Model untuk prediksi (opsional jika checkpoint_path diberikan)
            checkpoint_path: Path ke checkpoint (opsional jika model diberikan)
            conf_threshold: Threshold confidence untuk deteksi (opsional)
            iou_threshold: Threshold IoU untuk NMS (opsional)
            device: Device untuk prediksi (opsional)
            half_precision: Gunakan half precision (FP16) (opsional)
            visualize: Buat visualisasi hasil (opsional)
            output_dir: Direktori output untuk visualisasi (opsional)
            **kwargs: Parameter tambahan untuk prediksi
            
        Returns:
            Dict hasil prediksi
        """
        start_time = time.time()
        
        try:
            # Pastikan ada model atau checkpoint
            if model is None and checkpoint_path is None:
                raise ModelError(
                    "Model atau checkpoint_path harus diberikan untuk prediksi"
                )
            
            # Muat model dari checkpoint jika model tidak diberikan
            if model is None:
                self.logger.info(f"🔄 Loading model dari checkpoint: {checkpoint_path}")
                model, _ = self.model_factory.load_model(checkpoint_path)
            
            # Tentukan device
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Pindahkan model ke device
            model = model.to(device)
            
            # Tentukan parameter prediksi
            conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
            iou_threshold = iou_threshold if iou_threshold is not None else self.iou_threshold
            visualize = visualize if visualize is not None else self.visualize
            
            # Tentukan half precision
            if half_precision is None:
                # Default ke True jika CUDA tersedia
                half_precision = torch.cuda.is_available() and self.config.get('model', {}).get('half_precision', True)
            
            # Konversi ke half precision jika diminta
            if half_precision and device.type == 'cuda':
                model.half()
            
            # Siapkan output directory untuk visualisasi
            if visualize:
                if output_dir is None:
                    output_dir = self.output_dir
                
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Update visualizer dengan output_dir baru jika berbeda
                if output_dir != self.output_dir:
                    self.detection_visualizer = DetectionVisualizer(output_dir=output_dir, logger=self.logger)
            
            # Set model ke mode evaluasi
            model.eval()
            
            # Preprocess gambar
            input_tensors, original_images, image_sizes = self._preprocess_images(images, device)
            
            # Log informasi prediksi
            self.logger.info(
                f"🔎 Prediksi pada {len(input_tensors)} gambar:\n"
                f"   • Device: {device}\n"
                f"   • Confidence threshold: {conf_threshold}\n"
                f"   • IoU threshold: {iou_threshold}\n"
                f"   • Half precision: {half_precision}"
            )
            
            # Prediksi
            predictions = []
            
            with torch.no_grad():
                for i, inputs in enumerate(input_tensors):
                    # Inferensi
                    outputs = model(inputs)
                    
                    # Post-processing
                    if hasattr(model, 'post_process'):
                        # Model memiliki metode post-processing sendiri
                        batch_predictions = model.post_process(
                            outputs, 
                            conf_threshold=conf_threshold, 
                            iou_threshold=iou_threshold
                        )
                    else:
                        # Gunakan output langsung sebagai prediksi
                        batch_predictions = outputs
                    
                    # Tambahkan ke daftar prediksi
                    predictions.append(batch_predictions)
            
            # Proses hasil prediksi
            processed_results = self._process_predictions(
                predictions, 
                original_images, 
                image_sizes
            )
            
            # Visualisasi jika diminta
            if visualize:
                visualization_paths = self._visualize_predictions(
                    original_images,
                    processed_results,
                    output_dir=output_dir
                )
                processed_results['visualization_paths'] = visualization_paths
            
            # Hitung waktu eksekusi
            execution_time = time.time() - start_time
            processed_results['execution_time'] = execution_time
            
            # Hitung FPS
            fps = len(input_tensors) / execution_time
            processed_results['fps'] = fps
            
            self.logger.success(
                f"✅ Prediksi selesai dalam {execution_time:.2f}s ({fps:.2f} FPS)\n"
                f"   • Terdeteksi {sum(len(det['detections']) for det in processed_results['detections'])} objek\n"
                f"   • Visualisasi: {visualize}"
            )
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"❌ Error saat prediksi: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi: {str(e)}")
    
    def _preprocess_images(
        self,
        images: Union[torch.Tensor, np.ndarray, List, str, Path],
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[np.ndarray], List[Tuple[int, int]]]:
        """
        Preprocess gambar untuk prediksi.
        
        Args:
            images: Input gambar untuk prediksi
            device: Device untuk tensor
            
        Returns:
            Tuple (List tensor input, List gambar asli, List ukuran gambar asli)
        """
        # Buat list dari input jika bukan list
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        input_tensors = []
        original_images = []
        image_sizes = []
        
        for img in images:
            # Konversi ke numpy array jika path
            if isinstance(img, (str, Path)):
                # Load gambar dari path
                img_path = str(img)
                img = cv2.imread(img_path)
                if img is None:
                    raise ModelError(f"Tidak dapat membaca gambar: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Konversi torch tensor ke numpy jika tensor
            if isinstance(img, torch.Tensor):
                # Jika tensor sudah dalam format [B, C, H, W], gunakan langsung
                if len(img.shape) == 4:
                    input_tensor = img.to(device)
                    # Simpan original shape untuk resize hasil nanti
                    img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0]
                    orig_h, orig_w = img_np.shape[:2]
                else:
                    # Jika tensor dalam format [C, H, W], convert ke [B, C, H, W]
                    input_tensor = img.unsqueeze(0).to(device)
                    # Simpan original shape untuk resize hasil nanti
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                    orig_h, orig_w = img_np.shape[:2]
            else:
                # Untuk numpy array atau PIL Image
                if not isinstance(img, np.ndarray):
                    # Konversi PIL Image ke numpy array
                    img = np.array(img)
                
                # Simpan gambar asli
                img_np = img.copy()
                orig_h, orig_w = img.shape[:2]
                
                # Resize ke img_size jika perlu
                input_img = cv2.resize(img, tuple(self.img_size))
                
                # Normalisasi (0-1)
                input_img = input_img.astype(np.float32) / 255.0
                
                # Channel pertama
                input_img = input_img.transpose(2, 0, 1)
                
                # Buat tensor
                input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)
            
            # Tambahkan ke list
            input_tensors.append(input_tensor)
            original_images.append(img_np)
            image_sizes.append((orig_h, orig_w))
        
        return input_tensors, original_images, image_sizes
    
    def _process_predictions(
        self,
        predictions: List,
        original_images: List[np.ndarray],
        image_sizes: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Proses hasil prediksi menjadi format yang lebih mudah digunakan.
        
        Args:
            predictions: Hasil prediksi model
            original_images: Gambar asli
            image_sizes: Ukuran gambar asli (H, W)
            
        Returns:
            Dict hasil prediksi yang terstruktur
        """
        processed_results = {
            'num_images': len(original_images),
            'detections': []
        }
        
        # Dapatkan pemetaan class_id ke class_name dari config
        class_names = {}
        for layer_name, layer_config in self.config.get('layers', {}).items():
            if 'classes' in layer_config and 'class_ids' in layer_config:
                for cls_name, cls_id in zip(layer_config['classes'], layer_config['class_ids']):
                    class_names[cls_id] = cls_name
        
        for i, pred in enumerate(predictions):
            img_h, img_w = image_sizes[i]
            
            # Konversi format prediksi ke format DetectionVisualizer
            detections_list = []
            
            # Format deteksi tergantung pada format output model
            if isinstance(pred, torch.Tensor):
                # Format output: [x1, y1, x2, y2, conf, class_id]
                pred_np = pred.cpu().numpy()
                
                if len(pred_np.shape) > 1 and pred_np.shape[0] > 0:
                    for det in pred_np:
                        # Rescale box ke ukuran asli
                        x1, y1, x2, y2 = det[:4]
                        x1 = x1 * img_w / self.img_size[0]
                        y1 = y1 * img_h / self.img_size[1]
                        x2 = x2 * img_w / self.img_size[0]
                        y2 = y2 * img_h / self.img_size[1]
                        
                        # Format prediksi untuk DetectionVisualizer
                        class_id = int(det[5])
                        detections_list.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, str(class_id)),
                            'confidence': float(det[4])
                        })
            
            elif isinstance(pred, dict):
                # Format output: {'boxes': [...], 'scores': [...], 'labels': [...]}
                if 'boxes' in pred and isinstance(pred['boxes'], torch.Tensor):
                    boxes_tensor = pred['boxes'].cpu()
                    scores = pred['scores'].cpu() if 'scores' in pred else []
                    classes = pred['labels'].cpu() if 'labels' in pred else []
                    
                    # Rescale boxes ke ukuran asli
                    for j, box in enumerate(boxes_tensor):
                        x1, y1, x2, y2 = box.tolist()
                        x1 = x1 * img_w / self.img_size[0]
                        y1 = y1 * img_h / self.img_size[1]
                        x2 = x2 * img_w / self.img_size[0]
                        y2 = y2 * img_h / self.img_size[1]
                        
                        class_id = int(classes[j]) if j < len(classes) else 0
                        confidence = float(scores[j]) if j < len(scores) else 1.0
                        
                        # Format prediksi untuk DetectionVisualizer
                        detections_list.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, str(class_id)),
                            'confidence': confidence
                        })
            
            # Masukkan ke hasil
            processed_results['detections'].append({
                'image_size': (img_h, img_w),
                'detections': detections_list
            })
        
        return processed_results
    
    def _visualize_predictions(
        self,
        original_images: List[np.ndarray],
        processed_results: Dict[str, Any],
        output_dir: str
    ) -> List[str]:
        """
        Visualisasikan hasil prediksi menggunakan DetectionVisualizer.
        
        Args:
            original_images: Gambar asli
            processed_results: Hasil prediksi yang sudah diproses
            output_dir: Direktori output
            
        Returns:
            List path hasil visualisasi
        """
        visualization_paths = []
        
        # Visualisasi setiap gambar
        for i, (img, det_result) in enumerate(zip(original_images, processed_results['detections'])):
            # Buat filename output
            output_filename = f"detection_{i}.jpg"
            
            # Gunakan DetectionVisualizer untuk visualisasi
            vis_img = self.detection_visualizer.visualize_detection(
                image=img,
                detections=det_result['detections'],
                filename=output_filename,
                conf_threshold=self.conf_threshold,
                show_labels=self.show_labels,
                show_conf=self.show_conf,
                show_value=self.show_value
            )
            
            # Tambahkan path ke list
            output_path = str(Path(output_dir) / output_filename)
            visualization_paths.append(output_path)
        
        return visualization_paths
    
    def predict_on_video(
        self,
        video_path: Union[str, Path],
        model: torch.nn.Module,
        output_path: Optional[str] = None,
        frame_skip: int = 0,
        **kwargs
    ) -> str:
        """
        Prediksi pada video dengan visualisasi hasil.
        
        Args:
            video_path: Path ke file video
            model: Model untuk prediksi
            output_path: Path output (opsional)
            frame_skip: Jumlah frame yang dilewati (opsional)
            **kwargs: Parameter tambahan untuk predict()
            
        Returns:
            Path ke video hasil
        """
        self.logger.info(f"🎬 Prediksi pada video: {video_path}")
        
        try:
            # Buka video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ModelError(f"Tidak dapat membuka video: {video_path}")
                
            # Dapatkan informasi video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output video
            if output_path is None:
                output_path = str(Path(self.output_dir) / f"output_{Path(video_path).stem}.mp4")
                
            # Codec untuk output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Setup progress bar
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_frames, desc="Proses Video")
            except ImportError:
                progress_bar = None
                
            # Tentukan device untuk model
            device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Hitung FPS
            frame_count = 0
            start_time = time.time()
            
            # Loop melalui frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frame jika diperlukan
                frame_count += 1
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue
                    
                # Predict pada frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prediksi dengan model
                results = self.predict(
                    frame_rgb,
                    model=model,
                    device=device,
                    visualize=False,  # Jangan simpan visualisasi terpisah
                    **kwargs
                )
                
                # Dapatkan deteksi untuk frame ini
                detection_result = results['detections'][0]
                
                # Visualisasi deteksi pada frame menggunakan DetectionVisualizer
                vis_frame = self.detection_visualizer.visualize_detection(
                    image=frame,
                    detections=detection_result['detections'],
                    conf_threshold=self.conf_threshold,
                    show_total=True,
                    show_value=self.show_value
                )
                
                # Tampilkan FPS
                processing_fps = frame_count / (time.time() - start_time)
                cv2.putText(
                    vis_frame,
                    f"FPS: {processing_fps:.1f}",
                    (10, 70),  # Diposisikan di bawah info deteksi
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                # Tulis frame ke output
                out.write(vis_frame)
                
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.update(1)
            
            # Bersihkan
            cap.release()
            out.release()
            
            if progress_bar is not None:
                progress_bar.close()
                
            # Hitung waktu total
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            
            self.logger.success(
                f"✅ Prediksi video selesai:\n"
                f"   • Output: {output_path}\n"
                f"   • Frames: {frame_count}/{total_frames}\n"
                f"   • Waktu: {total_time:.2f}s\n"
                f"   • FPS: {avg_fps:.2f}"
            )
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Error saat prediksi video: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi video: {str(e)}")
    
    def predict_on_directory(
        self,
        directory: Union[str, Path],
        model: torch.nn.Module,
        output_dir: Optional[str] = None,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prediksi pada semua gambar dalam direktori.
        
        Args:
            directory: Path ke direktori gambar
            model: Model untuk prediksi
            output_dir: Direktori output (opsional)
            image_extensions: Ekstensi file gambar yang diproses
            **kwargs: Parameter tambahan untuk predict()
            
        Returns:
            Dict hasil prediksi
        """
        self.logger.info(f"📁 Prediksi pada direktori: {directory}")
        
        try:
            # Cari semua gambar dalam direktori
            directory = Path(directory)
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(list(directory.glob(f"*{ext}")))
                image_paths.extend(list(directory.glob(f"*{ext.upper()}")))
            
            if not image_paths:
                raise ModelError(f"Tidak ada gambar ditemukan di {directory}")
                
            self.logger.info(f"🔍 Ditemukan {len(image_paths)} gambar")
            
            # Setup output direktori
            if output_dir is None:
                output_dir = str(Path(self.output_dir) / directory.name)
                
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Setup visualizer dengan output_dir baru
            self.detection_visualizer = DetectionVisualizer(output_dir=output_dir, logger=self.logger)
            
            # Setup progress bar
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(image_paths), desc="Prediksi Gambar")
            except ImportError:
                progress_bar = None
            
            # Predict pada setiap gambar
            results = {}
            
            for image_path in image_paths:
                # Predict
                result = self.predict(
                    str(image_path),
                    model=model,
                    output_dir=output_dir,
                    **kwargs
                )
                
                # Simpan hasil
                results[str(image_path)] = result
                
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.update(1)
            
            if progress_bar is not None:
                progress_bar.close()
                
            # Buat grid visualisasi jika jumlah gambar > 1
            if len(image_paths) > 1:
                try:
                    # Siapkan data untuk grid visualisasi
                    images = []
                    detection_lists = []
                    
                    for i, (img_path, result) in enumerate(results.items()):
                        # Load gambar
                        img = cv2.imread(str(img_path))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Dapatkan deteksi
                        detections = result['detections'][0]['detections']
                        
                        images.append(img)
                        detection_lists.append(detections)
                    
                    # Buat grid visualisasi
                    grid_img = self.detection_visualizer.visualize_detections_grid(
                        images=images,
                        detections_list=detection_lists,
                        title=f"Deteksi pada {directory.name}",
                        filename="grid_detections.jpg",
                        conf_threshold=self.conf_threshold
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membuat grid visualisasi: {str(e)}")
                
            # Buat ringkasan hasil
            total_detections = sum(
                len(det_result['detections']) 
                for result in results.values() 
                for det_result in result['detections']
            )
            
            # Hitung total nilai yang terdeteksi
            total_value = 0
            for result in results.values():
                for det_result in result['detections']:
                    total_value += self.detection_visualizer.calculate_denomination_total(
                        det_result['detections']
                    )
            
            summary = {
                'total_images': len(image_paths),
                'total_detections': total_detections,
                'total_value': total_value,
                'output_directory': output_dir
            }
            
            self.logger.success(
                f"✅ Prediksi direktori selesai:\n"
                f"   • Total gambar: {summary['total_images']}\n"
                f"   • Total deteksi: {summary['total_detections']}\n"
                f"   • Total nilai: Rp {summary['total_value']:,}\n"
                f"   • Output: {summary['output_directory']}"
            )
            
            # Gabungkan summary dengan hasil
            return {
                'summary': summary,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error saat prediksi direktori: {str(e)}")
            raise ModelError(f"Gagal melakukan prediksi direktori: {str(e)}")
    
    def get_class_info(self, class_id: int) -> Dict[str, Any]:
        """
        Dapatkan informasi kelas dari config.
        
        Args:
            class_id: ID kelas
            
        Returns:
            Dict informasi kelas
        """
        # Cari kelas dari config
        class_info = {
            'id': class_id,
            'name': f"Class {class_id}",
            'layer': None
        }
        
        for layer_name, layer_config in self.config.get('layers', {}).items():
            if 'classes' in layer_config and 'class_ids' in layer_config:
                if class_id in layer_config['class_ids']:
                    idx = layer_config['class_ids'].index(class_id)
                    class_info['name'] = layer_config['classes'][idx]
                    class_info['layer'] = layer_name
                    break
        
        return class_info