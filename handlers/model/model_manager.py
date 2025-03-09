# File: smartcash/handlers/model/model_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama model sebagai facade untuk semua komponen model (versi yang diringkas)

import torch
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError

def is_running_in_colab() -> bool:
    """Deteksi apakah kode berjalan di Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

class ModelManager:
    """
    Manager utama model sebagai facade.
    Menyembunyikan kompleksitas implementasi dan meningkatkan usability.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None
    ):
        """
        Inisialisasi model manager.
        
        Args:
            config: Konfigurasi model dan training
            logger: Custom logger (opsional)
            colab_mode: Flag untuk mode Colab (opsional, deteksi otomatis jika None)
        """
        self.config = config
        self.logger = logger or get_logger("model_manager")
        
        # Deteksi otomatis colab_mode jika tidak diberikan
        self.colab_mode = is_running_in_colab() if colab_mode is None else colab_mode
        
        # Setup lazy loading untuk factories dan adapters
        self._model_factory = None
        self._optimizer_factory = None
        self._checkpoint_adapter = None
        self._metrics_adapter = None
        self._trainer = None
        self._evaluator = None 
        self._predictor = None
        self._experiment_manager = None
        
        self.logger.info(
            f"🔧 ModelManager diinisialisasi:\n"
            f"   • Colab mode: {self.colab_mode}\n"
            f"   • Output dir: {self.config.get('output_dir', 'runs/train')}"
        )
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        if self._model_factory is None:
            from smartcash.handlers.model.core.model_factory import ModelFactory
            self._model_factory = ModelFactory(self.config, self.logger)
        return self._model_factory
    
    @property
    def optimizer_factory(self):
        """Lazy-loaded optimizer factory."""
        if self._optimizer_factory is None:
            from smartcash.handlers.model.core.optimizer_factory import OptimizerFactory
            self._optimizer_factory = OptimizerFactory(self.config, self.logger)
        return self._optimizer_factory
    
    @property
    def checkpoint_adapter(self):
        """Lazy-loaded checkpoint adapter."""
        if self._checkpoint_adapter is None:
            from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
            self._checkpoint_adapter = CheckpointAdapter(self.config, self.logger)
        return self._checkpoint_adapter
    
    @property
    def metrics_adapter(self):
        """Lazy-loaded metrics adapter."""
        if self._metrics_adapter is None:
            from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter
            self._metrics_adapter = MetricsAdapter(self.logger, self.config)
        return self._metrics_adapter
    
    @property
    def trainer(self):
        """Lazy-loaded model trainer."""
        if self._trainer is None:
            from smartcash.handlers.model.core.model_trainer import ModelTrainer
            self._trainer = ModelTrainer(
                self.config,
                self.logger,
                self.model_factory,
                self.optimizer_factory,
                self.checkpoint_adapter,
                self.metrics_adapter
            )
        return self._trainer
    
    @property
    def evaluator(self):
        """Lazy-loaded model evaluator."""
        if self._evaluator is None:
            from smartcash.handlers.model.core.model_evaluator import ModelEvaluator
            self._evaluator = ModelEvaluator(
                self.config,
                self.logger,
                self.model_factory,
                self.metrics_adapter
            )
        return self._evaluator
    
    @property
    def predictor(self):
        """Lazy-loaded model predictor."""
        if self._predictor is None:
            from smartcash.handlers.model.core.model_predictor import ModelPredictor
            self._predictor = ModelPredictor(
                self.config,
                self.logger,
                self.model_factory
            )
        return self._predictor
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        if self._experiment_manager is None:
            from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager
            self._experiment_manager = ExperimentManager(
                self.config,
                self.logger,
                self.model_factory,
                self.optimizer_factory,
                self.checkpoint_adapter,
                self.metrics_adapter
            )
        return self._experiment_manager
    
    def create_model(
        self, 
        backbone_type: Optional[str] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        Buat model baru dengan konfigurasi tertentu.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            **kwargs: Parameter tambahan untuk ModelFactory
            
        Returns:
            Model yang diinisialisasi
        """
        return self.model_factory.create_model(backbone_type=backbone_type, **kwargs)
    
    def load_model(
        self, 
        checkpoint_path: str,
        **kwargs
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke file checkpoint
            **kwargs: Parameter tambahan untuk load_model
            
        Returns:
            Tuple (Model, Metadata checkpoint)
        """
        return self.model_factory.load_model(checkpoint_path, **kwargs)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> Dict:
        """
        Train model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model yang akan dilatih (opsional, buat baru jika None)
            **kwargs: Parameter tambahan untuk training
            
        Returns:
            Dict hasil training
        """
        # Setup observers untuk Colab jika perlu
        if self.colab_mode and 'observers' not in kwargs:
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            kwargs['observers'] = [ColabObserver(self.logger)]
        elif self.colab_mode and isinstance(kwargs.get('observers', []), list):
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            
            # Tambahkan ColabObserver jika belum ada
            if not any(isinstance(obs, ColabObserver) for obs in kwargs['observers']):
                kwargs['observers'].append(ColabObserver(self.logger))
        
        # Train model dengan trainer
        return self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            **kwargs
        )
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model yang akan dievaluasi (opsional)
            checkpoint_path: Path ke checkpoint (opsional, jika model None)
            **kwargs: Parameter tambahan untuk evaluasi
            
        Returns:
            Dict hasil evaluasi
        """
        # Pastikan ada model atau checkpoint
        if model is None and checkpoint_path is None:
            checkpoint_path = self.checkpoint_adapter.find_best_checkpoint()
            if checkpoint_path is None:
                raise ModelError(
                    "Tidak ada model yang diberikan, dan tidak ada checkpoint yang ditemukan"
                )
        
        # Evaluasi model
        return self.evaluator.evaluate(
            test_loader=test_loader,
            model=model,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
    
    def predict(
        self,
        images: Union[torch.Tensor, np.ndarray, List, str, Path],
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Prediksi dengan model.
        
        Args:
            images: Input images
            model: Model yang akan digunakan (opsional)
            checkpoint_path: Path ke checkpoint (opsional, jika model None)
            **kwargs: Parameter tambahan untuk prediksi
            
        Returns:
            Dict hasil prediksi
        """
        # Pastikan ada model atau checkpoint
        if model is None and checkpoint_path is None:
            checkpoint_path = self.checkpoint_adapter.find_best_checkpoint()
            if checkpoint_path is None:
                raise ModelError(
                    "Tidak ada model yang diberikan, dan tidak ada checkpoint yang ditemukan"
                )
        
        # Prediksi
        return self.predictor.predict(
            images=images,
            model=model,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
    
    def predict_on_video(
        self,
        video_path: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Prediksi pada video dengan visualisasi hasil.
        
        Args:
            video_path: Path ke file video
            model: Model untuk prediksi (opsional)
            checkpoint_path: Path ke checkpoint (opsional, jika model None)
            **kwargs: Parameter tambahan untuk predict()
            
        Returns:
            Path ke video hasil
        """
        # Load model jika perlu
        if model is None:
            if checkpoint_path is not None:
                model, _ = self.load_model(checkpoint_path)
            else:
                checkpoint_path = self.checkpoint_adapter.find_best_checkpoint()
                if checkpoint_path is None:
                    raise ModelError("Tidak ada model atau checkpoint yang tersedia")
                model, _ = self.load_model(checkpoint_path)
        
        # Prediksi pada video
        return self.predictor.predict_on_video(
            video_path=video_path,
            model=model,
            **kwargs
        )
    
    def compare_backbones(
        self,
        backbones: List[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs
    ) -> Dict:
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone yang akan dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            **kwargs: Parameter tambahan untuk eksperimen
            
        Returns:
            Dict hasil perbandingan eksperimen
        """
        # Setup Colab observer jika perlu
        if self.colab_mode and 'observers' not in kwargs:
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            kwargs['observers'] = [ColabObserver(self.logger)]
        
        # Gunakan experiment manager untuk perbandingan
        return self.experiment_manager.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
    
    def export_model(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        format: str = 'torchscript',
        **kwargs
    ) -> Optional[str]:
        """
        Export model ke format untuk deployment.
        
        Args:
            model: Model yang akan diexport (opsional)
            checkpoint_path: Path ke checkpoint (opsional, jika model None)
            format: Format export ('torchscript', 'onnx')
            **kwargs: Parameter tambahan untuk export
            
        Returns:
            Path ke file hasil export, atau None jika gagal
        """
        # Export model terbaik jika tidak ada model atau checkpoint
        if model is None and checkpoint_path is None:
            return self.checkpoint_adapter.export_best_model(
                format=format,
                **kwargs
            )
        
        # Load model dari checkpoint jika perlu
        if model is None and checkpoint_path is not None:
            model, _ = self.load_model(checkpoint_path)
        
        # Implementasi export berdasarkan format
        try:
            # Persiapkan parameter export
            input_shape = kwargs.get('input_shape', [1, 3, 640, 640])
            output_dir = Path(kwargs.get('output_dir', Path(self.config.get('output_dir', 'runs/export'))))
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = kwargs.get('filename', f"model_{format}.{format.lower()}")
            output_path = str(output_dir / filename)
            
            # Export model sesuai format yang diminta
            if format.lower() == 'torchscript':
                model.eval()
                dummy_input = torch.rand(*input_shape)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(output_path)
                self.logger.success(f"✅ Model berhasil diexport ke TorchScript: {output_path}")
            elif format.lower() == 'onnx':
                # Import onnx jika tersedia
                try:
                    import onnx
                    import onnxruntime
                    
                    model.eval()
                    dummy_input = torch.rand(*input_shape)
                    
                    torch.onnx.export(
                        model, dummy_input, output_path, export_params=True,
                        opset_version=12, do_constant_folding=True,
                        input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    
                    # Verifikasi model ONNX
                    onnx_model = onnx.load(output_path)
                    onnx.checker.check_model(onnx_model)
                    
                    self.logger.success(f"✅ Model berhasil diexport ke ONNX: {output_path}")
                except ImportError:
                    self.logger.error("❌ ONNX export membutuhkan package 'onnx' dan 'onnxruntime'")
                    return None
            else:
                self.logger.error(f"❌ Format export '{format}' tidak didukung")
                return None
                
            return output_path
        except Exception as e:
            self.logger.error(f"❌ Gagal mengexport model: {str(e)}")
            return None