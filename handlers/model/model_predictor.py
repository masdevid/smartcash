# File: smartcash/handlers/model/model_predictor.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk melakukan prediksi dengan model

import torch
import numpy as np
from typing import Dict, Optional, List, Union, Any, Tuple

from smartcash.utils.logger import SmartCashLogger
from smartcash.models.yolov5_model import YOLOv5Model
from smartcash.models.baseline import BaselineModel
from smartcash.handlers.checkpoint_handler import CheckpointHandler
from smartcash.handlers.model.model_factory import ModelFactory

class ModelPredictor:
    """
    Handler untuk melakukan prediksi dengan model terlatih.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        checkpoint_handler: Optional[CheckpointHandler] = None
    ):
        """
        Inisialisasi model predictor.
        
        Args:
            config: Konfigurasi model dan prediksi
            logger: Custom logger (opsional)
            checkpoint_handler: Handler untuk manajemen checkpoint (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup factory
        self.model_factory = ModelFactory(config, logger)
        
        # Setup checkpoint handler
        if checkpoint_handler is None:
            checkpoints_dir = config.get('output_dir', 'runs/train') + '/weights'
            self.checkpoint_handler = CheckpointHandler(
                output_dir=checkpoints_dir,
                logger=self.logger
            )
        else:
            self.checkpoint_handler = checkpoint_handler
            
        self.logger.info(f"🔧 ModelPredictor diinisialisasi")
    
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mencari checkpoint terbaik)
            device: Device untuk menempatkan model
            
        Returns:
            Tuple (Model yang dimuat dari checkpoint, metadata checkpoint)
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cari checkpoint terbaik jika tidak ada path yang diberikan
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_handler.find_best_checkpoint()
            if checkpoint_path is None:
                self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan, membuat model baru")
                model = self.model_factory.create_model()
                return model, {'epoch': 0, 'metrics': {}}
                
        try:
            # Muat checkpoint
            checkpoint = self.checkpoint_handler.load_checkpoint(checkpoint_path)
            checkpoint_config = checkpoint.get('config', {})
            
            # Dapatkan informasi backbone dari checkpoint
            backbone = checkpoint_config.get('model', {}).get('backbone', 
                    self.config.get('model', {}).get('backbone', 'efficientnet'))
            
            # Buat model baru dengan konfigurasi yang sama dengan checkpoint
            model = self.model_factory.create_model(backbone_type=backbone)
            
            # Muat state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Pindahkan model ke device
            model = model.to(device)
            
            # Log informasi
            self.logger.success(
                f"✅ Model berhasil dimuat dari checkpoint:\n"
                f"   • Path: {checkpoint_path}\n"
                f"   • Epoch: {checkpoint.get('epoch', 'unknown')}\n"
                f"   • Backbone: {backbone}"
            )
            
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise e
    
    def predict(
        self,
        images: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        Lakukan prediksi pada satu atau beberapa gambar.
        
        Args:
            images: Tensor gambar input [B, C, H, W]
            model: Model yang akan digunakan (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            device: Device untuk prediksi
            conf_threshold: Confidence threshold untuk prediksi
            
        Returns:
            Dict berisi hasil prediksi
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Muat model jika belum ada
        if model is None:
            if checkpoint_path:
                model, _ = self.load_model(checkpoint_path, device)
            else:
                self.logger.error("❌ Tidak ada model yang diberikan, dan tidak ada checkpoint path")
                raise ValueError("Model atau checkpoint_path harus diberikan")
        else:
            model = model.to(device)
        
        # Pastikan model dalam mode evaluasi
        model.eval()
        
        # Pindahkan gambar ke device
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        # Tambahkan dimensi batch jika hanya 1 gambar
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        images = images.to(device)
        
        # Lakukan prediksi
        with torch.no_grad():
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            predictions = model(images)
            end_time.record()
            
            # Synchronize CUDA
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # ms to s
            
            # Format yang berbeda untuk YOLOv5Model dan BaselineModel
            if isinstance(model, YOLOv5Model):
                # Gunakan predict dengan threshold
                detections = model.predict(images, conf_threshold=conf_threshold)
            else:
                # Format output untuk model baseline
                detections = []
                for pred in predictions:
                    scores, indices = torch.max(torch.softmax(pred, dim=0), dim=0)
                    mask = scores > conf_threshold
                    
                    detection = {
                        'boxes': torch.tensor([]),  # Empty tensor for boxes
                        'scores': scores[mask],
                        'labels': indices[mask]
                    }
                    detections.append(detection)
        
        # Format hasil
        result = {
            'detections': detections,
            'inference_time': inference_time,
            'conf_threshold': conf_threshold
        }
        
        self.logger.info(
            f"🔍 Prediksi selesai dalam {inference_time*1000:.2f} ms\n"
            f"   • Batch size: {images.shape[0]}\n"
            f"   • Confidence threshold: {conf_threshold}"
        )
        
        return result
        
    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        conf_threshold: float = 0.5,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Lakukan prediksi pada seluruh batch data.
        
        Args:
            dataloader: DataLoader yang berisi data untuk prediksi
            model: Model yang akan digunakan (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            device: Device untuk prediksi
            conf_threshold: Confidence threshold untuk prediksi
            max_samples: Maksimum jumlah sampel yang akan diproses (opsional)
            
        Returns:
            List berisi hasil prediksi untuk setiap batch
        """
        from tqdm.auto import tqdm
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Muat model jika belum ada
        if model is None:
            if checkpoint_path:
                model, _ = self.load_model(checkpoint_path, device)
            else:
                self.logger.error("❌ Tidak ada model yang diberikan, dan tidak ada checkpoint path")
                raise ValueError("Model atau checkpoint_path harus diberikan")
        else:
            model = model.to(device)
        
        # Pastikan model dalam mode evaluasi
        model.eval()
        
        # Lakukan prediksi pada semua batch
        results = []
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(dataloader, desc="Prediksi batch")):
                # Handle different data formats
                if isinstance(data, dict):
                    # Multilayer dataset format
                    images = data['images'].to(device)
                    metadata = data.get('metadata', [{}] * images.shape[0])
                else:
                    # Standard format
                    if len(data) == 2:
                        images, _ = data
                    else:
                        images = data
                    
                    metadata = [{}] * images.shape[0]
                    
                    if not isinstance(images, torch.Tensor):
                        images = torch.tensor(images)
                    
                    images = images.to(device)
                
                # Lakukan prediksi
                batch_result = self.predict(
                    images=images,
                    model=model,
                    device=device,
                    conf_threshold=conf_threshold
                )
                
                # Tambahkan metadata ke hasil
                batch_result['metadata'] = metadata
                results.append(batch_result)
                
                # Update total samples
                total_samples += images.shape[0]
                
                # Cek apakah sudah mencapai max_samples
                if max_samples is not None and total_samples >= max_samples:
                    break
        
        self.logger.success(f"✅ Prediksi batch selesai: {total_samples} sampel diproses")
        return results