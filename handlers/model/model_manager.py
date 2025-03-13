# File: smartcash/handlers/model/model_manager.py
# Deskripsi: Manager utama model sebagai facade dengan dependency injection

import torch
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.core.component_base import ComponentBase

class ModelManager(ComponentBase):
    """
    Manager utama model sebagai facade.
    Menggunakan dependency injection langsung untuk komponen-komponen.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        model_factory = None,
        trainer = None,
        evaluator = None,
        predictor = None,
        experiment_manager = None
    ):
        """
        Inisialisasi model manager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            model_factory: ModelFactory instance (opsional)
            trainer: ModelTrainer instance (opsional)
            evaluator: ModelEvaluator instance (opsional)
            predictor: ModelPredictor instance (opsional)
            experiment_manager: ExperimentManager instance (opsional)
        """
        super().__init__(config, logger, "model_manager")
        
        # Dependency components
        self._model_factory = model_factory
        self._trainer = trainer
        self._evaluator = evaluator
        self._predictor = predictor
        self._experiment_manager = experiment_manager
        
    def _initialize(self) -> None:
        """Inisialisasi parameter manager."""
        # Deteksi environment
        self.in_colab = self._is_colab()
        
        # Output directory
        self.output_dir = self.create_output_dir()
    
    # ===== Lazy-loaded Properties =====
    
    @property
    def model_factory(self):
        """Lazy-loaded model factory."""
        if self._model_factory is None:
            from smartcash.handlers.model.core.model_factory import ModelFactory
            self._model_factory = ModelFactory(self.config, self.logger)
        return self._model_factory
    
    @property
    def trainer(self):
        """Lazy-loaded model trainer."""
        if self._trainer is None:
            from smartcash.handlers.model.core.model_trainer import ModelTrainer
            self._trainer = ModelTrainer(
                self.config, 
                self.logger,
                self.model_factory
            )
        return self._trainer
    
    @property
    def evaluator(self):
        """Lazy-loaded model evaluator."""
        if self._evaluator is None:
            from smartcash.handlers.model.core.model_evaluator import ModelEvaluator
            
            # Create metrics calculator if needed
            metrics_calculator = None
            try:
                from smartcash.utils.metrics import MetricsCalculator
                metrics_calculator = MetricsCalculator()
            except ImportError:
                self.logger.warning("⚠️ MetricsCalculator tidak tersedia")
                
            self._evaluator = ModelEvaluator(
                self.config, 
                self.logger,
                self.model_factory,
                metrics_calculator
            )
        return self._evaluator
    
    @property
    def predictor(self):
        """Lazy-loaded model predictor."""
        if self._predictor is None:
            from smartcash.handlers.model.core.model_predictor import ModelPredictor
            
            # Create visualizer if needed
            visualizer = None
            try:
                from smartcash.utils.visualization.detection import DetectionVisualizer
                visualizer = DetectionVisualizer(
                    output_dir=str(self.output_dir / "results"),
                    logger=self.logger
                )
            except ImportError:
                self.logger.warning("⚠️ DetectionVisualizer tidak tersedia")
                
            self._predictor = ModelPredictor(
                self.config, 
                self.logger,
                self.model_factory,
                visualizer
            )
        return self._predictor
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        if self._experiment_manager is None:
            from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager
            
            # Buat visualizer jika diperlukan
            visualizer = None
            try:
                from smartcash.utils.visualization import ExperimentVisualizer
                visualizer = ExperimentVisualizer(
                    output_dir=str(self.output_dir / "experiments" / "visualizations")
                )
            except ImportError:
                self.logger.warning("⚠️ ExperimentVisualizer tidak tersedia")
                
            self._experiment_manager = ExperimentManager(
                self.config, 
                self.logger, 
                self.model_factory,
                visualizer
            )
        return self._experiment_manager
    
    # ===== Core Functionality =====
    
    def create_model(self, backbone_type=None, **kwargs) -> torch.nn.Module:
        """
        Buat model baru dengan konfigurasi tertentu.
        
        Args:
            backbone_type: Tipe backbone (optional)
            **kwargs: Parameter tambahan untuk model
            
        Returns:
            Model yang siap digunakan
        """
        return self.model_factory.create_model(backbone_type=backbone_type, **kwargs)
    
    def load_model(self, checkpoint_path, **kwargs) -> Tuple[torch.nn.Module, Dict]:
        """
        Load model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            **kwargs: Parameter tambahan
            
        Returns:
            Tuple (Model yang dimuat, Metadata checkpoint)
        """
        return self.model_factory.load_model(checkpoint_path, **kwargs)
    
    def train(self, train_loader, val_loader, model=None, **kwargs) -> Dict:
        """
        Train model dengan dataset yang diberikan.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            model: Model untuk dilatih (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil training
        """
        return self.trainer.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            model=model, 
            **kwargs
        )
    
    def evaluate(self, test_loader, model=None, checkpoint_path=None, **kwargs) -> Dict:
        """
        Evaluasi model pada test dataset.
        
        Args:
            test_loader: DataLoader untuk testing
            model: Model untuk evaluasi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil evaluasi
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.evaluator.evaluate(
            test_loader=test_loader, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict(self, images, model=None, checkpoint_path=None, **kwargs) -> Dict:
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
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict(
            images=images, 
            model=model, 
            checkpoint_path=checkpoint_path, 
            **kwargs
        )
    
    def predict_on_video(self, video_path, model=None, checkpoint_path=None, **kwargs) -> str:
        """
        Prediksi pada video dengan visualisasi hasil.
        
        Args:
            video_path: Path ke file video
            model: Model untuk prediksi (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke video hasil
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        return self.predictor.predict_on_video(
            video_path=video_path, 
            model=model,
            **kwargs
        )
    
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs) -> Dict:
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        return self.experiment_manager.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
    
    def export_model(self, model=None, checkpoint_path=None, format='torchscript', **kwargs) -> Optional[str]:
        """
        Export model ke format untuk deployment.
        
        Args:
            model: Model untuk export (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            format: Format export ('torchscript', 'onnx')
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke model yang diexport
        """
        # Pastikan ada model atau checkpoint
        model, checkpoint_path = self._ensure_model_or_checkpoint(model, checkpoint_path)
        
        # Export berdasarkan format
        if format.lower() == 'torchscript':
            return self._export_to_torchscript(model, **kwargs)
        elif format.lower() == 'onnx':
            return self._export_to_onnx(model, **kwargs)
        else:
            self.logger.error(f"❌ Format export '{format}' tidak didukung")
            return None
            
    def _export_to_torchscript(self, model, **kwargs):
        """Export model ke TorchScript format."""
        try:
            output_dir = Path(kwargs.get('output_dir', self.output_dir / 'exports'))
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Simpan ke TorchScript
            dummy_input = torch.zeros(1, 3, 640, 640)
            script_model = torch.jit.trace(model, dummy_input)
            
            output_path = output_dir / f"model_torchscript.pt"
            script_model.save(str(output_path))
            
            self.logger.success(f"✅ Model berhasil diekspor ke: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor ke TorchScript: {e}")
            return None
            
    def _export_to_onnx(self, model, **kwargs):
        """Export model ke ONNX format."""
        try:
            import onnx
            import onnxruntime
            
            output_dir = Path(kwargs.get('output_dir', self.output_dir / 'exports'))
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Siapkan input
            dummy_input = torch.zeros(1, 3, 640, 640)
            output_path = output_dir / "model.onnx"
            
            # Export ke ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                input_names=['input'],
                output_names=['output']
            )
            
            self.logger.success(f"✅ Model berhasil diekspor ke: {output_path}")
            return str(output_path)
        except ImportError:
            self.logger.error("❌ ONNX atau ONNXRuntime tidak tersedia")
            return None
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor ke ONNX: {e}")
            return None
    
    def _ensure_model_or_checkpoint(self, model, checkpoint_path):
        """
        Pastikan ada model atau checkpoint path.
        
        Args:
            model: Model yang sudah ada (opsional)
            checkpoint_path: Path ke checkpoint (opsional)
            
        Returns:
            Tuple (model, checkpoint_path)
            
        Raises:
            ModelError: Jika tidak ada model dan tidak ada checkpoint
        """
        if model is None and checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
            if checkpoint_path is None:
                raise ModelError("Tidak ada model yang diberikan, dan tidak ada checkpoint yang ditemukan")
        
        if model is None and checkpoint_path is not None:
            model, _ = self.load_model(checkpoint_path)
            checkpoint_path = None  # Sudah di-load, tidak perlu lagi
        
        return model, checkpoint_path
        
    def _find_best_checkpoint(self):
        """Cari checkpoint terbaik di direktori output."""
        try:
            weights_dir = self.output_dir / "weights"
            if not weights_dir.exists():
                return None
                
            # Cari semua file checkpoint
            checkpoints = list(weights_dir.glob("*.pt"))
            if not checkpoints:
                return None
                
            # Coba cari yang memiliki 'best' di namanya
            best_checkpoints = [cp for cp in checkpoints if "best" in cp.name]
            if best_checkpoints:
                return str(sorted(best_checkpoints)[-1])  # Ambil yang terbaru
                
            # Jika tidak ada, ambil checkpoint terakhir
            return str(sorted(checkpoints)[-1])
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mencari checkpoint: {e}")
            return None