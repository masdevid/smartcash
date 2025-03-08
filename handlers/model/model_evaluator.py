# File: smartcash/handlers/model/model_evaluator.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk evaluasi model dengan metrics yang telah direfaktor

import torch
import numpy as np
from typing import Dict, Optional, List, Union, Any, Tuple
from tqdm.auto import tqdm
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.evaluation_metrics import MetricsCalculator
from smartcash.handlers.checkpoint import CheckpointManager
from smartcash.handlers.model.model_factory import ModelFactory

class ModelEvaluator:
    """
    Handler untuk evaluasi model dengan metrics yang telah direfaktor.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Inisialisasi model evaluator.
        
        Args:
            config: Konfigurasi model dan evaluasi
            logger: Custom logger (opsional)
            checkpoint_manager: Handler untuk manajemen checkpoint (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup factory
        self.model_factory = ModelFactory(config, logger)
        
        # Setup checkpoint handler
        if checkpoint_manager is None:
            checkpoints_dir = config.get('output_dir', 'runs/train') + '/weights'
            self.checkpoint_manager = CheckpointManager(
                output_dir=checkpoints_dir,
                logger=self.logger
            )
        else:
            # Backward compatibility - jika masih menerima CheckpointManager
            self.checkpoint_manager = checkpoint_manager
            
        # Output direktori
        self.output_dir = Path(config.get('output_dir', 'runs/eval'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔧 ModelEvaluator diinisialisasi")
    
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
            checkpoint_path = self.checkpoint_manager.find_best_checkpoint()
            if checkpoint_path is None:
                self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan, membuat model baru")
                model = self.model_factory.create_model()
                return model, {'epoch': 0, 'metrics': {}}
                
        try:
            # Muat checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
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
                f"   • Loss: {checkpoint.get('metrics', {}).get('loss', 'unknown')}\n"
                f"   • Backbone: {backbone}"
            )
            
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat model: {str(e)}")
            raise e
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model yang akan dievaluasi (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            device: Device untuk evaluasi
            
        Returns:
            Dict berisi metrik evaluasi
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
        
        model.eval()
        
        # Gunakan MetricsCalculator dari utils/evaluation_metrics
        metrics_calc = MetricsCalculator()
        
        # Evaluasi model
        self.logger.info("🔍 Mengevaluasi model pada test dataset...")
        
        # Track total loss dan prediksi
        total_loss = 0
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluasi")):
                # Handle different data formats
                if isinstance(data, dict):
                    # Multilayer dataset format
                    images = data['images'].to(device)
                    targets = data['targets']
                    
                    # Transfer targets to device
                    for layer_name in targets:
                        targets[layer_name] = targets[layer_name].to(device)
                else:
                    # Standard format
                    images, targets = data
                    images = images.to(device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(device)
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                predictions = model(images)
                end_time.record()
                
                # Synchronize CUDA
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # ms to s
                inference_times.append(inference_time)
                
                # Compute loss
                loss_dict = model.compute_loss(predictions, targets)
                loss = loss_dict['total_loss']
                total_loss += loss.item()
                
                # Update metrics calculator
                metrics_calc.update(predictions, targets)
        
        # Compute final metrics
        metrics = metrics_calc.compute()
        
        # Add loss and inference time
        metrics['loss'] = total_loss / len(test_loader)
        metrics['inference_time'] = np.mean(inference_times)
        
        # Log hasil evaluasi
        self.logger.success(
            f"✅ Evaluasi selesai:\n"
            f"   • Loss: {metrics.get('loss', 'N/A'):.4f}\n"
            f"   • Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n"
            f"   • Precision: {metrics.get('precision', 'N/A'):.4f}\n"
            f"   • Recall: {metrics.get('recall', 'N/A'):.4f}\n"
            f"   • F1 Score: {metrics.get('f1', 'N/A'):.4f}\n"
            f"   • mAP: {metrics.get('mAP', 'N/A'):.4f}\n"
            f"   • Inference Time: {metrics.get('inference_time', 'N/A')*1000:.2f} ms/batch"
        )
        
        # Simpan metrics ke file
        self._save_metrics(metrics)
        
        return metrics
        
    def _save_metrics(self, metrics: Dict) -> None:
        """
        Simpan metrics evaluasi ke file.
        
        Args:
            metrics: Dictionary berisi metrics evaluasi
        """
        try:
            import json
            from datetime import datetime
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Simpan ke file JSON
            metrics_file = self.output_dir / f"eval_metrics_{timestamp}.json"
            
            # Konversi numpy types ke Python native types
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                elif isinstance(v, np.float32) or isinstance(v, np.float64):
                    clean_metrics[k] = float(v)
                elif isinstance(v, np.int32) or isinstance(v, np.int64):
                    clean_metrics[k] = int(v)
                else:
                    clean_metrics[k] = v
            
            with open(metrics_file, 'w') as f:
                json.dump(clean_metrics, f, indent=2)
                
            self.logger.info(f"📊 Metrics tersimpan di {metrics_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyimpan metrics: {str(e)}")
    
    def evaluate_multiple_runs(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        num_runs: int = 3,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Evaluasi model beberapa kali dan ambil rata-rata metrics.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model yang akan dievaluasi (jika None, muat dari checkpoint)
            checkpoint_path: Path ke checkpoint (jika model None)
            num_runs: Jumlah run evaluasi
            device: Device untuk evaluasi
            
        Returns:
            Dict berisi rata-rata metrics
        """
        all_metrics = []
        
        for run in range(num_runs):
            self.logger.info(f"🔄 Evaluasi run {run+1}/{num_runs}")
            
            # Evaluasi model
            metrics = self.evaluate(
                test_loader=test_loader,
                model=model,
                checkpoint_path=checkpoint_path,
                device=device
            )
            
            all_metrics.append(metrics)
        
        # Hitung rata-rata
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Log hasil rata-rata
        self.logger.success(
            f"✅ Evaluasi multi-run selesai ({num_runs} runs):\n"
            f"   • Avg Loss: {avg_metrics.get('loss', 'N/A'):.4f}\n"
            f"   • Avg Accuracy: {avg_metrics.get('accuracy', 'N/A'):.4f}\n"
            f"   • Avg Precision: {avg_metrics.get('precision', 'N/A'):.4f}\n"
            f"   • Avg Recall: {avg_metrics.get('recall', 'N/A'):.4f}\n"
            f"   • Avg F1 Score: {avg_metrics.get('f1', 'N/A'):.4f}\n"
            f"   • Avg Inference Time: {avg_metrics.get('inference_time', 'N/A')*1000:.2f} ms/batch"
        )
        
        return avg_metrics