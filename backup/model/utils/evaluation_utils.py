"""
File: smartcash/model/utils/evaluation_utils.py
Deskripsi: Utilitas untuk evaluasi model YOLOv5 dengan berbagai backbone dan skenario
"""

import torch
import numpy as np
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union

def run_model_inference(model, dataloader, conf_thresh: float = 0.25, iou_thresh: float = 0.45, 
                       device: str = 'cuda', progress_callback=None) -> Dict[str, Any]:
    """
    Menjalankan inferensi model pada dataloader dengan perhitungan waktu dan NMS
    
    Args:
        model: Model YOLOv5 yang akan digunakan
        dataloader: Dataloader berisi gambar test
        conf_thresh: Confidence threshold untuk deteksi
        iou_thresh: IoU threshold untuk NMS
        device: Device untuk inferensi ('cuda' atau 'cpu')
        progress_callback: Callback untuk melaporkan progress
        
    Returns:
        Dict berisi hasil prediksi dan metrik inferensi
    """
    # Pastikan model dalam mode evaluasi
    model.eval()
    
    # Inisialisasi hasil
    all_predictions = []
    inference_times = []
    batch_times = []
    
    # Tentukan device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Pindahkan model ke device yang sesuai
    model = model.to(device)
    
    # Jalankan inferensi
    total_batches = len(dataloader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Update progress
        if progress_callback:
            progress = int((batch_idx / total_batches) * 100)
            progress_callback(progress, f"⚙️ Inferensi batch {batch_idx+1}/{total_batches}")
        
        # Ekstrak data batch
        images = batch['image'].to(device)
        image_paths = batch.get('image_path', [None] * len(images))
        original_shapes = batch.get('original_shape', [(None, None, None)] * len(images))
        
        # Jalankan inferensi dengan pengukuran waktu
        with torch.no_grad():
            inference_start = time.time()
            outputs = model(images)
            inference_end = time.time()
            
            # Hitung waktu inferensi per gambar
            batch_inference_time = inference_end - inference_start
            per_image_time = batch_inference_time / len(images)
            inference_times.append(per_image_time)
        
        # Terapkan NMS
        try:
            processed_outputs = apply_nms(outputs, conf_thresh, iou_thresh)
        except Exception as e:
            # Fallback ke simple NMS jika gagal
            processed_outputs = simple_nms(outputs, conf_thresh, iou_thresh)
        
        # Proses hasil untuk setiap gambar
        for i, (preds, img_path, orig_shape) in enumerate(zip(processed_outputs, image_paths, original_shapes)):
            # Konversi ke format yang lebih mudah digunakan
            if preds is not None and len(preds) > 0:
                # Ekstrak boxes, scores, dan class_ids
                boxes = preds[:, :4].cpu().numpy()
                scores = preds[:, 4].cpu().numpy()
                class_ids = preds[:, 5].cpu().numpy().astype(int)
                
                # Simpan prediksi
                prediction = {
                    'image_path': img_path,
                    'original_shape': orig_shape,
                    'predictions': [
                        {
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class_id': int(class_id)
                        } for box, score, class_id in zip(boxes, scores, class_ids)
                    ]
                }
            else:
                # Tidak ada prediksi
                prediction = {
                    'image_path': img_path,
                    'original_shape': orig_shape,
                    'predictions': []
                }
            
            all_predictions.append(prediction)
        
        # Hitung waktu batch total
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
    
    # Hitung metrik inferensi
    total_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    # Kompilasi hasil
    result = {
        'predictions': all_predictions,
        'inference_metrics': {
            'total_time': total_time,
            'average_inference_time': avg_inference_time,
            'average_batch_time': avg_batch_time,
            'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
            'total_images': len(all_predictions),
            'device': device
        }
    }
    
    return result

def apply_nms(outputs, conf_thresh: float, iou_thresh: float) -> List[torch.Tensor]:
    """
    Terapkan Non-Maximum Suppression pada output model
    
    Args:
        outputs: Output dari model YOLOv5
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold untuk NMS
        
    Returns:
        List of tensors dengan prediksi setelah NMS
    """
    # Ekstrak prediksi dari output model
    if isinstance(outputs, list):
        # Output dari model.forward()
        predictions = outputs
    elif isinstance(outputs, torch.Tensor):
        # Output langsung berupa tensor
        predictions = [outputs]
    elif isinstance(outputs, dict) and 'pred' in outputs:
        # Output dari model dengan format dict
        predictions = outputs['pred']
    else:
        raise ValueError(f"Format output tidak didukung: {type(outputs)}")
    
    # Terapkan NMS
    processed = []
    
    for pred in predictions:
        if pred is None or len(pred) == 0:
            processed.append(torch.empty((0, 6), device=pred.device))
            continue
        
        # Filter berdasarkan confidence
        mask = pred[:, 4] > conf_thresh
        filtered = pred[mask]
        
        if len(filtered) == 0:
            processed.append(torch.empty((0, 6), device=pred.device))
            continue
        
        # Terapkan NMS
        # Format: [x1, y1, x2, y2, confidence, class_id]
        processed_pred = non_max_suppression(
            filtered, 
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            classes=None,
            agnostic=False,
            multi_label=True
        )
        
        processed.extend(processed_pred)
    
    return processed

def simple_nms(outputs, conf_thresh: float, iou_thresh: float) -> List[torch.Tensor]:
    """
    Implementasi sederhana Non-Maximum Suppression sebagai fallback
    
    Args:
        outputs: Output dari model YOLOv5
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold untuk NMS
        
    Returns:
        List of tensors dengan prediksi setelah NMS
    """
    import torchvision.ops as ops
    
    # Ekstrak prediksi dari output model
    if isinstance(outputs, list):
        # Output dari model.forward()
        predictions = outputs
    elif isinstance(outputs, torch.Tensor):
        # Output langsung berupa tensor
        predictions = [outputs]
    elif isinstance(outputs, dict) and 'pred' in outputs:
        # Output dari model dengan format dict
        predictions = outputs['pred']
    else:
        raise ValueError(f"Format output tidak didukung: {type(outputs)}")
    
    # Terapkan NMS
    processed = []
    
    for pred in predictions:
        if pred is None or len(pred) == 0:
            processed.append(torch.empty((0, 6), device=pred.device))
            continue
        
        # Filter berdasarkan confidence
        mask = pred[:, 4] > conf_thresh
        filtered = pred[mask]
        
        if len(filtered) == 0:
            processed.append(torch.empty((0, 6), device=pred.device))
            continue
        
        # Terapkan NMS dengan torchvision
        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        keep = ops.nms(boxes, scores, iou_thresh)
        
        processed.append(filtered[keep])
    
    return processed

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False):
    """
    Implementasi Non-Maximum Suppression dari YOLOv5
    
    Args:
        prediction: Output dari model YOLOv5
        conf_thres: Confidence threshold
        iou_thres: IoU threshold untuk NMS
        classes: Filter berdasarkan class
        agnostic: Abaikan class saat menghitung IoU
        multi_label: Allow multiple labels per box
        
    Returns:
        List of tensors dengan prediksi setelah NMS
    """
    import torchvision.ops as ops
    
    # Settings
    min_wh, max_wh = 2, 4096  # min dan max box width dan height
    max_det = 300  # maximum detections per image
    max_nms = 30000  # maximum boxes ke dalam torchvision.ops.nms()
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    
    # Compute conf = obj_conf * cls_conf
    prediction[..., 5:] *= prediction[..., 4:5]  # conf = obj_conf * cls_conf
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[..., 4] > conf_thres]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
            
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        
    return output

def xywh2xyxy(x):
    """
    Convert bounding box format dari [x, y, w, h] ke [x1, y1, x2, y2]
    
    Args:
        x: Tensor dengan format [x, y, w, h]
        
    Returns:
        Tensor dengan format [x1, y1, x2, y2]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def create_inference_dataloader(images: List[np.ndarray], image_paths: List[str], 
                               augmentation_pipeline=None, batch_size: int = 8, 
                               img_size: int = 416) -> torch.utils.data.DataLoader:
    """
    Buat dataloader untuk inferensi dengan augmentasi opsional
    
    Args:
        images: List gambar untuk inferensi
        image_paths: List path gambar
        augmentation_pipeline: Pipeline augmentasi opsional
        batch_size: Ukuran batch
        img_size: Target image size
        
    Returns:
        DataLoader untuk inferensi
    """
    from torch.utils.data import Dataset, DataLoader
    
    class InferenceDataset(Dataset):
        def __init__(self, images, image_paths, augmentation_pipeline, img_size):
            self.images = images
            self.image_paths = image_paths
            self.augmentation_pipeline = augmentation_pipeline
            self.img_size = img_size
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            image_path = self.image_paths[idx]
            original_shape = image.shape
            
            # Terapkan augmentasi jika ada
            if self.augmentation_pipeline:
                try:
                    # Augmentasi tanpa bounding box
                    transformed = self.augmentation_pipeline(image=image)
                    image = transformed['image']
                except Exception:
                    # Jika gagal, gunakan preprocessing default
                    image = self._default_preprocess(image)
            else:
                # Preprocessing default
                image = self._default_preprocess(image)
            
            # Konversi ke tensor
            if isinstance(image, np.ndarray):
                # Jika masih numpy array, konversi ke tensor
                if image.ndim == 2:
                    # Grayscale ke RGB
                    image = np.stack([image] * 3, axis=2)
                
                # Normalisasi jika belum
                if image.max() > 1.0:
                    image = image / 255.0
                
                # HWC ke CHW
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
            
            return {
                'image': image,
                'image_path': image_path,
                'original_shape': original_shape
            }
        
        def _default_preprocess(self, image):
            """Preprocessing default jika tidak ada augmentasi atau augmentasi gagal"""
            # Resize
            h, w = image.shape[:2]
            scale = min(self.img_size / h, self.img_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize dengan maintain aspect ratio
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad ke target size
            padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            padded[:new_h, :new_w] = resized
            
            # Normalisasi
            padded = padded.astype(np.float32) / 255.0
            
            return padded
    
    # Buat dataset dan dataloader
    dataset = InferenceDataset(images, image_paths, augmentation_pipeline, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return dataloader

def load_ground_truth_labels(image_paths: List[str], class_names: List[str]) -> Dict[str, Any]:
    """
    Muat ground truth labels dari file YOLO format
    
    Args:
        image_paths: List path gambar
        class_names: List nama kelas
        
    Returns:
        Dict berisi ground truth labels dan metadata
    """
    import os
    
    # Inisialisasi hasil
    labels_info = {
        'available': False,
        'labels': {},
        'class_counts': {},
        'total_labels': 0
    }
    
    # Hitung jumlah kelas
    num_classes = len(class_names)
    
    # Inisialisasi class counts
    for cls_name in class_names:
        labels_info['class_counts'][cls_name] = 0
    
    # Cek apakah ada label
    valid_labels = 0
    
    for img_path in image_paths:
        # Ganti ekstensi gambar dengan .txt untuk file label
        label_path = os.path.splitext(img_path)[0] + '.txt'
        
        if os.path.exists(label_path):
            try:
                # Baca label
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse label
                img_labels = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < num_classes:
                            # Format YOLO: class_id, x_center, y_center, width, height
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Validasi koordinat
                            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                                img_labels.append({
                                    'class_id': class_id,
                                    'class_name': class_names[class_id],
                                    'bbox': [x_center, y_center, width, height]
                                })
                                
                                # Update class counts
                                labels_info['class_counts'][class_names[class_id]] += 1
                                labels_info['total_labels'] += 1
                
                # Simpan label
                if img_labels:
                    labels_info['labels'][img_path] = img_labels
                    valid_labels += 1
                
            except Exception:
                # Abaikan file label yang tidak valid
                pass
    
    # Update status ketersediaan label
    labels_info['available'] = valid_labels > 0
    labels_info['valid_label_count'] = valid_labels
    labels_info['total_images'] = len(image_paths)
    
    return labels_info

# Alias untuk kompatibilitas dengan test
run_inference_core = run_model_inference
