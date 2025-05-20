"""
File: smartcash/model/services/experiment/experiment_service.py
Deskripsi: Layanan inti untuk mengelola eksperimen model individual
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

from smartcash.common.logger import get_logger
from smartcash.model.config.experiment_config import ExperimentConfig
from smartcash.model.manager import ModelManager
from smartcash.model.services.training import TrainingService
from smartcash.model.services.evaluation import EvaluationService


class ExperimentService:
    """
    Layanan untuk mengelola satu eksperimen model individual.
    
    Bertanggung jawab untuk:
    - Setup dan konfigurasi eksperimen
    - Pelatihan dan evaluasi model
    - Manajemen checkpoint dan hasil
    - Tracking metrik dan performa
    """
    
    def __init__(
        self,
        experiment_dir: str = "runs/experiment",
        training_service: Optional[TrainingService] = None,
        evaluation_service: Optional[EvaluationService] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment service.
        
        Args:
            experiment_dir: Direktori untuk menyimpan hasil eksperimen
            training_service: Layanan training (opsional)
            evaluation_service: Layanan evaluasi (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or get_logger()
        self.training_service = training_service or TrainingService(logger=self.logger)
        self.evaluation_service = evaluation_service or EvaluationService(logger=self.logger)
        
        # Status eksperimen
        self.experiment_config = None
        self.model = None
        self.results = {}
        
        self.logger.info(f"🧪 ExperimentService diinisialisasi (dir={self.experiment_dir})")
    
    def setup_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> ExperimentConfig:
        """
        Setup eksperimen baru.
        
        Args:
            name: Nama eksperimen
            config: Konfigurasi eksperimen (opsional)
            description: Deskripsi eksperimen
            
        Returns:
            ExperimentConfig untuk eksperimen
        """
        # Buat konfigurasi eksperimen
        self.experiment_config = ExperimentConfig(
            name=name,
            experiment_dir=str(self.experiment_dir),
            description=description
        )
        
        # Update konfigurasi jika ada
        if config:
            for key, value in config.items():
                self.experiment_config.parameters[key] = value
                
        self.logger.info(
            f"✅ Eksperimen {name} disetup:\n"
            f"   • ID: {self.experiment_config.experiment_id}\n"
            f"   • Dir: {self.experiment_dir}"
        )
        
        return self.experiment_config
    
    def setup_model(
        self,
        model_type: str = "efficient_basic",
        batch_size: int = 16,
        learning_rate: float = 0.001,
        **kwargs
    ) -> torch.nn.Module:
        """
        Setup model untuk eksperimen.
        
        Args:
            model_type: Tipe model
            batch_size: Ukuran batch
            learning_rate: Learning rate
            **kwargs: Parameter tambahan
            
        Returns:
            Instance model yang sudah dibangun
        """
        # Validasi eksperimen
        if not self.experiment_config:
            self.setup_experiment(f"experiment_{model_type}")
        
        # Setup model config
        model_config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            **kwargs
        }
        
        # Inisialisasi model manager
        model_manager = ModelManager(
            config=model_config,
            model_type=model_type,
            logger=self.logger
        )
        
        # Build model
        self.model = model_manager.build_model()
        self.model_type = model_type
        
        self.logger.info(
            f"🔄 Model {model_type} disetup untuk eksperimen"
        )
        
        return self.model
    
    def run_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        callbacks: List[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan training untuk model.
        
        Args:
            train_loader: DataLoader untuk training data
            val_loader: DataLoader untuk validation data
            epochs: Jumlah epoch
            callbacks: List callback function
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil training
        """
        # Validasi model
        if not self.model:
            raise ValueError("Model belum disetup. Panggil setup_model terlebih dahulu.")
        
        # Jalankan training
        start_time = time.time()
        
        training_result = self.training_service.train(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            callbacks=callbacks,
            experiment_config=self.experiment_config,
            **kwargs
        )
        
        # Catat hasil
        training_time = time.time() - start_time
        training_result["training_time"] = training_time
        
        self.results["training"] = training_result
        
        self.logger.success(
            f"✅ Training selesai dalam {training_time:.2f} detik\n"
            f"   • Loss akhir: {training_result.get('final_loss', 'N/A')}"
        )
        
        return training_result
    
    def run_evaluation(
        self,
        test_loader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi untuk model.
        
        Args:
            test_loader: DataLoader untuk test data
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
        # Validasi model
        if not self.model:
            raise ValueError("Model belum disetup. Panggil setup_model terlebih dahulu.")
        
        # Jalankan evaluasi
        start_time = time.time()
        
        evaluation_result = self.evaluation_service.evaluate(
            model=self.model,
            test_loader=test_loader,
            **kwargs
        )
        
        # Catat hasil
        evaluation_time = time.time() - start_time
        evaluation_result["evaluation_time"] = evaluation_time
        
        self.results["evaluation"] = evaluation_result
        
        self.logger.success(
            f"✅ Evaluasi selesai dalam {evaluation_time:.2f} detik\n"
            f"   • mAP: {evaluation_result.get('mAP', 'N/A'):.4f}"
        )
        
        return evaluation_result
    
    def run_complete_experiment(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_type: str = "efficient_basic",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan eksperimen lengkap dari setup hingga evaluasi.
        
        Args:
            train_loader: DataLoader untuk training data
            val_loader: DataLoader untuk validation data
            test_loader: DataLoader untuk test data
            model_type: Tipe model
            epochs: Jumlah epoch
            batch_size: Ukuran batch
            learning_rate: Learning rate
            name: Nama eksperimen (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil eksperimen
        """
        # Setup eksperimen
        if not name:
            name = f"experiment_{model_type}_{epochs}ep"
            
        self.setup_experiment(name)
        
        # Setup model
        self.setup_model(
            model_type=model_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
        
        start_time = time.time()
        
        # Training
        training_result = self.run_training(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            **kwargs
        )
        
        # Evaluasi
        evaluation_result = self.run_evaluation(
            test_loader=test_loader,
            **kwargs
        )
        
        # Proses hasil
        elapsed_time = time.time() - start_time
        
        result = {
            "status": "success",
            "experiment_id": self.experiment_config.experiment_id,
            "model_type": model_type,
            "training": training_result,
            "evaluation": evaluation_result,
            "elapsed_time": elapsed_time,
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                **kwargs
            }
        }
        
        # Simpan hasil
        self.save_results(result)
        
        self.logger.success(
            f"✅ Eksperimen {name} selesai dalam {elapsed_time:.2f} detik\n"
            f"   • Model: {model_type}\n"
            f"   • mAP: {evaluation_result.get('mAP', 'N/A'):.4f}"
        )
        
        return result
    
    def save_checkpoint(self, filename: str = "model.pt") -> str:
        """
        Simpan checkpoint model.
        
        Args:
            filename: Nama file checkpoint
            
        Returns:
            Path file checkpoint
        """
        if not self.model:
            raise ValueError("Model belum disetup")
            
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        # Simpan model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': getattr(self, 'model_type', 'unknown'),
            'experiment_id': self.experiment_config.experiment_id if self.experiment_config else None
        }, checkpoint_path)
        
        self.logger.info(f"💾 Model checkpoint disimpan: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Muat checkpoint model.
        
        Args:
            checkpoint_path: Path file checkpoint
            
        Returns:
            Model yang dimuat
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Muat state dict ke model
        if self.model:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Jika model belum disetup, coba setup berdasarkan informasi dari checkpoint
            model_type = checkpoint.get('model_type', 'efficient_basic')
            self.setup_model(model_type=model_type)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"📂 Model dimuat dari checkpoint: {checkpoint_path}")
        
        return self.model
    
    def save_results(self, results: Dict[str, Any] = None) -> str:
        """
        Simpan hasil eksperimen.
        
        Args:
            results: Hasil eksperimen (opsional, default: self.results)
            
        Returns:
            Path file hasil
        """
        if not results:
            # Gabungkan semua hasil yang ada
            results = {
                "status": "success",
                "experiment_id": self.experiment_config.experiment_id if self.experiment_config else "unknown",
                "model_type": getattr(self, 'model_type', 'unknown'),
                **self.results
            }
        
        # Simpan hasil dalam format JSON
        results_path = self.experiment_dir / "results.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Hasil eksperimen disimpan: {results_path}")
        
        # Update metrics dalam experiment_config jika ada
        if self.experiment_config and "evaluation" in results and "training" in results:
            self.experiment_config.log_metrics(
                epoch=results.get("parameters", {}).get("epochs", 0),
                train_loss=results["training"].get("final_loss", 0),
                val_loss=results["training"].get("final_val_loss", 0),
                mAP=results["evaluation"].get("mAP", 0),
                precision=results["evaluation"].get("precision", 0),
                recall=results["evaluation"].get("recall", 0),
                f1=results["evaluation"].get("f1", 0)
            )
        
        return str(results_path)
    
    def load_results(self) -> Dict[str, Any]:
        """
        Muat hasil eksperimen.
        
        Returns:
            Dictionary hasil eksperimen
        """
        results_path = self.experiment_dir / "results.json"
        
        if not results_path.exists():
            return {}
        
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            
            # Update results internal
            self.results = results
            
            self.logger.info(f"📂 Hasil eksperimen dimuat: {results_path}")
            
            return results
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat hasil: {str(e)}")
            return {}
    
    def predict(
        self,
        inputs: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        Lakukan prediksi dengan model.
        
        Args:
            inputs: Input tensor
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil prediksi
        """
        if not self.model:
            raise ValueError("Model belum disetup")
        
        # Set model ke mode evaluasi
        self.model.eval()
        
        # Lakukan prediksi
        with torch.no_grad():
            predictions = self.model(inputs, **kwargs)
        
        return predictions