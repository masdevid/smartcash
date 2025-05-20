"""
File: smartcash/model/services/research/experiment_runner.py
Deskripsi: Komponen untuk menjalankan eksperimen model
"""

import os
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


class ExperimentRunner:
    """
    Komponen untuk menjalankan eksperimen model.
    
    Bertanggung jawab untuk:
    - Jalankan eksperimen pelatihan dan evaluasi
    - Proses dan simpan hasil eksperimen
    - Mengelola eksekusi model
    """
    
    def __init__(
        self,
        base_dir: str,
        training_service: TrainingService,
        evaluation_service: EvaluationService,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment runner.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan hasil eksperimen
            training_service: Layanan training
            evaluation_service: Layanan evaluasi
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger()
        self.training_service = training_service
        self.evaluation_service = evaluation_service
        
        self.logger.debug(f"🏃 ExperimentRunner diinisialisasi")
    
    def run_experiment(
        self,
        experiment: Union[str, ExperimentConfig],
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        model_type: str = "efficient_basic",
        callbacks: List[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan eksperimen pelatihan dan evaluasi.
        
        Args:
            experiment: ID eksperimen atau objek ExperimentConfig
            dataset_path: Path ke dataset
            epochs: Jumlah epoch
            batch_size: Ukuran batch
            learning_rate: Learning rate
            model_type: Tipe model ('basic', 'efficient_basic', dll)
            callbacks: List callback yang akan dipanggil selama eksperimen
            **kwargs: Parameter tambahan untuk eksperimen
            
        Returns:
            Dictionary hasil eksperimen
        """
        # Dapatkan konfigurasi eksperimen
        experiment_config = self._get_experiment_config(experiment)
        if not experiment_config:
            return {"status": "error", "message": "Eksperimen tidak ditemukan"}
        
        experiment_id = experiment_config.experiment_id
        self.logger.info(f"🚀 Menjalankan eksperimen: {experiment_id}")
        
        start_time = time.time()
        
        try:
            # Setup model
            model = self._setup_model(model_type, batch_size, learning_rate, **kwargs)
            
            # Setup data
            data_loaders = self._prepare_dataloaders(dataset_path, batch_size)
            train_loader, val_loader, test_loader = data_loaders
            
            # Jalankan training
            training_result = self._execute_training(
                model, train_loader, val_loader, epochs, 
                learning_rate, experiment_config, callbacks, **kwargs
            )
            
            # Evaluasi model
            evaluation_result = self._execute_evaluation(model, test_loader, **kwargs)
            
            # Proses hasil
            result = self._process_experiment_results(
                experiment_id, model_type, training_result, evaluation_result,
                start_time, epochs, batch_size, learning_rate, **kwargs
            )
            
            # Simpan hasil
            self._save_experiment_result(experiment_config, result)
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"❌ Eksperimen {experiment_id} gagal: {str(e)}")
            return {
                "status": "error",
                "experiment_id": experiment_id,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
    
    def _setup_model(
        self, 
        model_type: str, 
        batch_size: int, 
        learning_rate: float,
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
        return model_manager.build_model()
    
    def _execute_training(
        self, 
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        learning_rate: float,
        experiment_config: ExperimentConfig,
        callbacks: List[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan proses training model.
        
        Args:
            model: Model yang akan dilatih
            train_loader: DataLoader untuk training data
            val_loader: DataLoader untuk validation data
            epochs: Jumlah epoch
            learning_rate: Learning rate
            experiment_config: Konfigurasi eksperimen
            callbacks: List callback function
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil training
        """
        return self.training_service.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            callbacks=callbacks,
            experiment_config=experiment_config,
            **kwargs
        )
    
    def _execute_evaluation(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi model.
        
        Args:
            model: Model yang akan dievaluasi
            test_loader: DataLoader untuk test data
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
        return self.evaluation_service.evaluate(
            model=model,
            test_loader=test_loader,
            **kwargs
        )
    
    def _process_experiment_results(
        self,
        experiment_id: str,
        model_type: str,
        training_result: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        start_time: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses hasil eksperimen menjadi format yang konsisten.
        
        Args:
            experiment_id: ID eksperimen
            model_type: Tipe model
            training_result: Hasil training
            evaluation_result: Hasil evaluasi
            start_time: Waktu mulai eksperimen
            epochs: Jumlah epoch
            batch_size: Ukuran batch
            learning_rate: Learning rate
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil eksperimen terformat
        """
        elapsed_time = time.time() - start_time
        
        # Log hasil
        self.logger.success(
            f"✅ Eksperimen {experiment_id} selesai dalam {elapsed_time:.2f} detik\n"
            f"   • Model: {model_type}\n"
            f"   • Loss akhir: {training_result.get('final_loss', 'N/A')}\n"
            f"   • mAP: {evaluation_result.get('mAP', 'N/A'):.4f}"
        )
        
        # Format hasil
        return {
            "status": "success",
            "experiment_id": experiment_id,
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
    
    def _save_experiment_result(
        self,
        experiment_config: ExperimentConfig,
        result: Dict[str, Any]
    ) -> None:
        """
        Simpan hasil eksperimen.
        
        Args:
            experiment_config: Konfigurasi eksperimen
            result: Hasil eksperimen
        """
        # Simpan hasil dalam format JSON
        results_path = Path(experiment_config.experiment_dir) / "results.json"
        
        try:
            with open(results_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            # Update metrics dalam experiment_config
            if "evaluation" in result:
                experiment_config.log_metrics(
                    epoch=result.get("parameters", {}).get("epochs", 0),
                    train_loss=result.get("training", {}).get("final_loss", 0),
                    val_loss=result.get("training", {}).get("final_val_loss", 0),
                    mAP=result.get("evaluation", {}).get("mAP", 0),
                    precision=result.get("evaluation", {}).get("precision", 0),
                    recall=result.get("evaluation", {}).get("recall", 0),
                    f1=result.get("evaluation", {}).get("f1", 0)
                )
            
            self.logger.info(f"💾 Hasil eksperimen disimpan: {results_path}")
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil eksperimen: {str(e)}")
    
    def _prepare_dataloaders(
        self,
        dataset_path: str,
        batch_size: int
    ) -> Tuple:
        """
        Persiapkan data loaders untuk training dan evaluasi.
        
        Args:
            dataset_path: Path ke dataset
            batch_size: Ukuran batch
            
        Returns:
            Tuple (train_loader, val_loader, test_loader)
        """
        # Implementasi sederhana
        # Seharusnya menggunakan DatasetService atau DataLoader yang sudah ada
        
        # Placeholder - implementasi sebenarnya harus menggunakan dataset yang ada
        self.logger.info(f"🔄 Mempersiapkan data loaders dari {dataset_path}")
        
        # Dataset dummy
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        
        # Dummy datasets
        train_dataset = TensorDataset(
            torch.from_numpy(np.random.rand(100, 3, 640, 640).astype(np.float32)),
            torch.from_numpy(np.random.rand(100, 10, 5).astype(np.float32))
        )
        
        val_dataset = TensorDataset(
            torch.from_numpy(np.random.rand(20, 3, 640, 640).astype(np.float32)),
            torch.from_numpy(np.random.rand(20, 10, 5).astype(np.float32))
        )
        
        test_dataset = TensorDataset(
            torch.from_numpy(np.random.rand(20, 3, 640, 640).astype(np.float32)),
            torch.from_numpy(np.random.rand(20, 10, 5).astype(np.float32))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def _get_experiment_config(
        self,
        experiment: Union[str, ExperimentConfig]
    ) -> Optional[ExperimentConfig]:
        """
        Dapatkan konfigurasi eksperimen.
        
        Args:
            experiment: ID eksperimen atau objek ExperimentConfig
            
        Returns:
            ExperimentConfig untuk eksperimen
        """
        if isinstance(experiment, ExperimentConfig):
            return experiment
            
        # Jika ID, coba load dari file
        experiment_dir = self.base_dir / experiment
        if experiment_dir.exists():
            try:
                return ExperimentConfig(name="loaded", experiment_dir=str(experiment_dir))
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat konfigurasi eksperimen: {str(e)}")
                
        return None