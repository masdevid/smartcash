# File: smartcash/handlers/model/core/model_predictor.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk prediksi menggunakan model yang sudah dilatih

import torch
import time
import numpy as np
import cv2
from typing import Dict, Optional, Any, List, Union
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.handlers.model.core.model_component import ModelComponent
from smartcash.exceptions.base import ModelError
from smartcash.utils.visualization.detection import DetectionVisualizer

class ModelPredictor(ModelComponent):
    """Komponen untuk prediksi menggunakan model yang sudah dilatih."""
    
    def _initialize(self):
        """Inisialisasi parameter prediksi."""
        inf_cfg = self.config.get('inference', {})
        self.params = {
            'conf_threshold': inf_cfg.get('conf_threshold', 0.25),
            'iou_threshold': inf_cfg.get('iou_threshold', 0.45),
            'visualize': inf_cfg.get('visualize', True),
            'img_size': self.config.get('model', {}).get('img_size', [640, 640])
        }
        
        # Setup visualizer
        output_dir = inf_cfg.get('output_dir', str(Path(self.config.get('output_dir', 'runs/predict')) / "results"))
        self.detection_visualizer = DetectionVisualizer(output_dir=output_dir, logger=self.logger)
    
    def process(self, images, model=None, **kwargs):
        """Alias untuk predict()."""
        return self.predict(images, model, **kwargs)
    
    def predict(
        self,
        images, 
        model=None,
        checkpoint_path=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prediksi dengan model.
        
        Args:
            images: Input gambar untuk prediksi
            model: Model untuk prediksi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil prediksi
        """
        start_time = time.time()
        
        try:
            # Load model jika perlu
            if model is None and checkpoint_path is not None:
                model, _ = self.model_factory.load_model(checkpoint_path)
                
            if model is None:
                raise ModelError("Model atau checkpoint_path harus diberikan")
                
            # Setup parameter
            device = kwargs.get('device') or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device).eval()
            conf_threshold = kwargs.get('conf_threshold', self.params['conf_threshold'])
            iou_threshold = kwargs.get('iou_threshold', self.params['iou_threshold'])
            visualize = kwargs.get('visualize', self.params['visualize'])
            
            # Setup output directory untuk visualisasi
            output_dir = kwargs.get('output_dir')
            if visualize and output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                self.detection_visualizer = DetectionVisualizer(output_dir=output_dir, logger=self.logger)
            
            # Preprocess gambar
            input_tensors, original_images, image_sizes = self._preprocess_images(images, device)
            
            # Log info prediksi
            self.logger.info(
                f"🔎 Prediksi pada {len(input_tensors)} gambar (conf: {conf_threshold:.2f}, iou: {iou_threshold:.2f})"
            )
            
            # Prediksi
            predictions = []
            with torch.no_grad():
                for inputs in input_tensors:
                    outputs = model(inputs)
                    
                    # Post-processing
                    if hasattr(model, 'post_process'):
                        batch_predictions = model.post_process(
                            outputs, conf_threshold=conf_threshold, iou_threshold=iou_threshold
                        )
                    else:
                        batch_predictions = outputs
                        
                    predictions.append(batch_predictions)
            
            # Proses hasil prediksi
            results = self._process_predictions(predictions, original_images, image_sizes)
            
            # Visualisasi jika diminta
            if visualize:
                results['visualization_paths'] = self._visualize_predictions(
                    original_images, results, output_dir or self.detection_visualizer.output_dir
                )
            
            # Hitung waktu dan FPS
            duration = time.time() - start_time
            fps = len(input_tensors) / duration
            results.update({'execution_time': duration, 'fps': fps})
            
            # Log hasil
            total_detections = sum(len(det['detections']) for det in results['detections'])
            self.logger.success(
                f"✅ Prediksi selesai: {total_detections} objek terdeteksi ({fps:.2f} FPS)"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error prediksi: {str(e)}")
            raise ModelError(f"Gagal prediksi: {str(e)}")
    
    def predict_on_video(
        self,
        video_path,
        model,
        output_path=None,
        frame_skip=0,
        **kwargs
    ) -> str:
        """
        Prediksi pada video dengan visualisasi hasil.
        
        Args:
            video_path: Path ke file video
            model: Model untuk prediksi
            output_path: Path output (opsional)
            frame_skip: Jumlah frame yang dilewati
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke video hasil
        """
        self.logger.info(f"🎬 Prediksi video: {video_path}")
        
        try:
            # Buka video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ModelError(f"Video tidak dapat dibuka: {video_path}")
                
            # Info video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output video
            if output_path is None:
                output_path = str(Path(self.detection_visualizer.output_dir) / f"output_{Path(video_path).stem}.mp4")
                
            # Codec untuk output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Setup progress bar
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_frames, desc="Video")
            except ImportError:
                pbar = None
                
            # Parameter prediksi
            conf_threshold = kwargs.get('conf_threshold', self.params['conf_threshold'])
            
            # Proses video
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frame jika diperlukan
                frame_count += 1
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    if pbar:
                        pbar.update(1)
                    continue
                    
                # Predict pada frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.predict(frame_rgb, model=model, visualize=False, **kwargs)
                
                # Visualisasi deteksi
                detection = results['detections'][0]
                vis_frame = self.detection_visualizer.visualize_detection(
                    image=frame,
                    detections=detection['detections'],
                    conf_threshold=conf_threshold,
                    show_total=True
                )
                
                # Tampilkan FPS
                fps_text = f"FPS: {frame_count / (time.time() - start_time):.1f}"
                cv2.putText(vis_frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Tulis frame ke output
                out.write(vis_frame)
                
                # Update progress
                if pbar:
                    pbar.update(1)
            
            # Cleanup
            cap.release()
            out.release()
            if pbar:
                pbar.close()
                
            # Log hasil
            duration = time.time() - start_time
            self.logger.success(
                f"✅ Video selesai: {frame_count} frames diproses ({frame_count/duration:.2f} FPS)"
            )
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Error video: {str(e)}")
            raise ModelError(f"Gagal proses video: {str(e)}")
    
    def _preprocess_images(self, images, device):
        """
        Preprocess gambar untuk prediksi.
        
        Returns:
            (input_tensors, original_images, image_sizes)
        """
        # Pastikan format list
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        input_tensors = []
        original_images = []
        image_sizes = []
        
        for img in images:
            # Jika path, load gambar
            if isinstance(img, (str, Path)):
                img = cv2.imread(str(img))
                if img is None:
                    raise ModelError(f"Tidak dapat membaca gambar: {img}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Simpan gambar asli
            img_np = img.copy() if isinstance(img, np.ndarray) else img
            original_images.append(img_np)
            
            # Dapatkan ukuran original
            if isinstance(img, torch.Tensor):
                orig_h, orig_w = img.shape[-2:] if len(img.shape) == 4 else img.shape[1:3]
            else:
                orig_h, orig_w = img.shape[:2]
                
            image_sizes.append((orig_h, orig_w))
            
            # Preprocessing untuk tensor input
            if isinstance(img, torch.Tensor):
                if len(img.shape) == 3:  # [C,H,W] -> [1,C,H,W]
                    img = img.unsqueeze(0)
                input_tensor = img.to(device)
            else:
                # Resize dan normalisasi
                input_img = cv2.resize(img, tuple(self.params['img_size']))
                input_img = input_img.astype(np.float32) / 255.0
                input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
                input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)
            
            input_tensors.append(input_tensor)
            
        return input_tensors, original_images, image_sizes
    
    def _process_predictions(self, predictions, original_images, image_sizes):
        """Proses hasil prediksi menjadi format yang lebih mudah digunakan."""
        # Dapatkan mapping class_id ke nama
        class_names = self._get_class_names()
        
        # Kumpulkan deteksi
        results = {'num_images': len(original_images), 'detections': []}
        
        for i, pred in enumerate(predictions):
            img_h, img_w = image_sizes[i]
            detections_list = []
            
            # Format output [x1, y1, x2, y2, conf, class_id]
            if isinstance(pred, torch.Tensor):
                pred_np = pred.cpu().numpy()
                
                if len(pred_np.shape) > 1 and pred_np.shape[0] > 0:
                    for det in pred_np:
                        # Rescale box ke ukuran asli
                        x1, y1, x2, y2 = det[:4]
                        x1 = x1 * img_w / self.params['img_size'][0]
                        y1 = y1 * img_h / self.params['img_size'][1]
                        x2 = x2 * img_w / self.params['img_size'][0]
                        y2 = y2 * img_h / self.params['img_size'][1]
                        
                        # Format deteksi
                        class_id = int(det[5])
                        detections_list.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, str(class_id)),
                            'confidence': float(det[4])
                        })
            
            # Format output dictionary
            elif isinstance(pred, dict) and 'boxes' in pred:
                boxes = pred['boxes'].cpu() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes']
                scores = pred['scores'].cpu() if 'scores' in pred else []
                classes = pred['labels'].cpu() if 'labels' in pred else []
                
                for j, box in enumerate(boxes):
                    if j >= len(classes) or j >= len(scores):
                        break
                        
                    # Rescale box
                    x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
                    x1 = x1 * img_w / self.params['img_size'][0]
                    y1 = y1 * img_h / self.params['img_size'][1]
                    x2 = x2 * img_w / self.params['img_size'][0]
                    y2 = y2 * img_h / self.params['img_size'][1]
                    
                    # Format deteksi
                    class_id = int(classes[j])
                    detections_list.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'class_name': class_names.get(class_id, str(class_id)),
                        'confidence': float(scores[j])
                    })
            
            # Tambahkan ke hasil
            results['detections'].append({
                'image_size': (img_h, img_w),
                'detections': detections_list
            })
            
        return results
    
    def _visualize_predictions(self, original_images, processed_results, output_dir):
        """Visualisasikan hasil prediksi."""
        visualization_paths = []
        
        # Visualisasi setiap gambar
        for i, (img, det_result) in enumerate(zip(original_images, processed_results['detections'])):
            # Buat filename
            output_filename = f"detection_{i}.jpg"
            
            # Visualisasi
            vis_img = self.detection_visualizer.visualize_detection(
                image=img,
                detections=det_result['detections'],
                filename=output_filename,
                conf_threshold=self.params['conf_threshold']
            )
            
            # Tambahkan path
            output_path = str(Path(output_dir) / output_filename)
            visualization_paths.append(output_path)
            
        return visualization_paths
    
    def _get_class_names(self):
        """Dapatkan mapping class_id ke class_name dari config."""
        class_names = {}
        
        # Ambil dari konfigurasi layer
        for layer_name, layer_config in self.config.get('layers', {}).items():
            if 'classes' in layer_config and 'class_ids' in layer_config:
                for name, id_ in zip(layer_config['classes'], layer_config['class_ids']):
                    class_names[id_] = name
                    
        return class_names