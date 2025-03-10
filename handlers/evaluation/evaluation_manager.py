# File: smartcash/handlers/evaluation/evaluation_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager untuk evaluasi model dengan direct injection

import os
import time
import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.metrics import MetricsCalculator
from smartcash.utils.observer.event_dispatcher import EventDispatcher
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.handlers.evaluation.core.model_evaluator import ModelEvaluator
from smartcash.handlers.evaluation.core.report_generator import ReportGenerator
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from smartcash.handlers.evaluation.pipeline.batch_evaluation_pipeline import BatchEvaluationPipeline
from smartcash.handlers.evaluation.pipeline.research_pipeline import ResearchPipeline

class EvaluationManager:
    """
    Manager utama untuk evaluasi model dengan direct injection.
    Menyediakan antarmuka sederhana untuk berbagai operasi evaluasi.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None
    ):
        """
        Inisialisasi evaluation manager.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            colab_mode: Mode Google Colab (auto-detect jika None)
        """
        self.config = config
        self.logger = logger or get_logger("evaluation_manager")
        
        # Coba auto-detect colab mode jika tidak dispesifikasikan
        if colab_mode is None:
            try:
                import google.colab
                colab_mode = True
            except ImportError:
                colab_mode = False
        
        self.colab_mode = colab_mode
        
        # Setup observer manager
        self.observer_manager = ObserverManager()
        
        # Cache untuk komponen
        self._model_evaluator = None
        self._report_generator = None
        self._evaluation_pipeline = None
        self._batch_pipeline = None
        self._research_pipeline = None
        
        # Parameter evaluasi dari config
        eval_config = self.config.get('evaluation', {})
        self.output_dir = Path(eval_config.get('output_dir', 'results/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔧 EvaluationManager diinisialisasi (colab_mode={colab_mode})")
        
        # Setup observers
        self._setup_default_observers()
    
    def evaluate_model(
        self,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        generate_report: bool = True,
        report_format: str = 'json',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model dengan dataset.
        
        Args:
            model: Model PyTorch (opsional)
            model_path: Path ke checkpoint model (opsional)
            dataset_path: Path ke dataset evaluasi (opsional)
            dataloader: DataLoader untuk evaluasi (opsional)
            device: Device untuk evaluasi ('cuda', 'cpu')
            conf_threshold: Threshold konfidiensi untuk deteksi
            visualize: Buat visualisasi hasil
            generate_report: Buat laporan hasil
            report_format: Format laporan ('json', 'csv', 'markdown', 'html')
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
            
        Raises:
            ValueError: Jika tidak cukup parameter untuk evaluasi
        """
        self.logger.info("🚀 Mulai evaluasi model")
        
        # Validasi input
        if model is None and model_path is None:
            raise ValueError("Harus menyediakan model atau model_path")
        
        if dataloader is None and dataset_path is None:
            raise ValueError("Harus menyediakan dataloader atau dataset_path")
        
        # Notifikasi evaluasi dimulai
        EventDispatcher.notify("evaluation.manager.start", self, {
            'model_path': model_path,
            'dataset_path': dataset_path
        })
        
        try:
            # Ukur waktu eksekusi
            start_time = time.time()
            
            # Gunakan evaluation pipeline untuk evaluasi model
            eval_pipeline = self._get_evaluation_pipeline()
            
            # Set parameter yang belum ditentukan dari config
            if device is None:
                device = self.config.get('evaluation', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            if conf_threshold is None:
                conf_threshold = self.config.get('evaluation', {}).get('conf_threshold', 0.25)
            
            # Jalankan pipeline
            results = eval_pipeline.run(
                model=model,
                model_path=model_path,
                dataset_path=dataset_path,
                dataloader=dataloader,
                device=device,
                conf_threshold=conf_threshold,
                **kwargs
            )
            
            # Tambahkan waktu eksekusi total
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Buat visualisasi jika diminta
            if visualize:
                visualization_paths = self._create_visualizations(results)
                results['visualization_paths'] = visualization_paths
            
            # Buat laporan jika diminta
            if generate_report:
                report_path = self._generate_report(results, report_format)
                results['report_path'] = report_path
            
            # Notifikasi evaluasi selesai
            EventDispatcher.notify("evaluation.manager.complete", self, {
                'execution_time': execution_time,
                'metrics': results.get('metrics', {})
            })
            
            self.logger.success(f"✅ Evaluasi model selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi model gagal: {str(e)}")
            
            # Notifikasi error
            EventDispatcher.notify("evaluation.manager.error", self, {
                'error': str(e)
            })
            
            raise
    
    def evaluate_batch(
        self,
        models: Optional[List[torch.nn.Module]] = None,
        model_paths: Optional[List[str]] = None,
        dataset_path: str = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        parallel: bool = True,
        visualize: bool = True,
        generate_report: bool = True,
        report_format: str = 'markdown',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi beberapa model dengan dataset yang sama.
        
        Args:
            models: List model PyTorch (opsional)
            model_paths: List path ke checkpoint model (opsional)
            dataset_path: Path ke dataset evaluasi (opsional)
            dataloader: DataLoader untuk evaluasi (opsional)
            parallel: Evaluasi secara paralel
            visualize: Buat visualisasi hasil
            generate_report: Buat laporan hasil
            report_format: Format laporan ('json', 'csv', 'markdown', 'html')
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
            
        Raises:
            ValueError: Jika tidak cukup parameter untuk evaluasi
        """
        self.logger.info("🚀 Mulai evaluasi batch model")
        
        # Validasi input
        if (models is None or len(models) == 0) and (model_paths is None or len(model_paths) == 0):
            raise ValueError("Harus menyediakan models atau model_paths")
        
        if dataloader is None and dataset_path is None:
            raise ValueError("Harus menyediakan dataloader atau dataset_path")
        
        # Ukur waktu eksekusi
        start_time = time.time()
        
        # Notifikasi evaluasi batch dimulai
        EventDispatcher.notify("evaluation.batch.start", self, {
            'num_models': len(models) if models is not None else len(model_paths),
            'dataset_path': dataset_path
        })
        
        try:
            # Gunakan batch evaluation pipeline
            batch_pipeline = self._get_batch_pipeline()
            
            # Convert models ke model_paths jika perlu
            if models is not None and model_paths is None:
                model_paths = []
                for i, model in enumerate(models):
                    # Simpan model ke temporary checkpoint
                    temp_path = str(self.output_dir / f"temp_model_{i}.pt")
                    torch.save(model.state_dict(), temp_path)
                    model_paths.append(temp_path)
            
            # Jalankan batch evaluation
            results = batch_pipeline.run(
                model_paths=model_paths,
                dataset_path=dataset_path,
                dataloader=dataloader,
                parallel=parallel,
                **kwargs
            )
            
            # Tambahkan waktu eksekusi total
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Buat visualisasi jika diminta
            if visualize:
                visualization_paths = self._create_batch_visualizations(results)
                results['plots'] = visualization_paths
            
            # Buat laporan jika diminta
            if generate_report:
                report_path = self._generate_report(results, report_format)
                results['report_path'] = report_path
            
            # Notifikasi evaluasi batch selesai
            EventDispatcher.notify("evaluation.batch.complete", self, {
                'execution_time': execution_time,
                'num_models': len(model_paths)
            })
            
            self.logger.success(f"✅ Evaluasi batch selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi batch gagal: {str(e)}")
            
            # Notifikasi error
            EventDispatcher.notify("evaluation.batch.error", self, {
                'error': str(e)
            })
            
            raise
        finally:
            # Hapus temporary checkpoint jika ada
            if models is not None and model_paths is not None:
                for path in model_paths:
                    if path.startswith(str(self.output_dir / "temp_model_")):
                        try:
                            os.remove(path)
                        except:
                            pass
    
    def evaluate_research_scenarios(
        self,
        scenarios: Dict[str, Dict],
        visualize: bool = True,
        generate_report: bool = True,
        report_format: str = 'markdown',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi skenario penelitian.
        
        Args:
            scenarios: Dictionary skenario penelitian
            visualize: Buat visualisasi hasil
            generate_report: Buat laporan hasil
            report_format: Format laporan ('json', 'csv', 'markdown', 'html')
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
            
        Raises:
            ValueError: Jika tidak ada skenario yang valid
        """
        self.logger.info(f"🔬 Mulai evaluasi {len(scenarios)} skenario penelitian")
        
        # Validasi input
        if not scenarios:
            raise ValueError("Harus menyediakan minimal satu skenario")
        
        # Notifikasi evaluasi skenario dimulai
        EventDispatcher.notify("evaluation.research.start", self, {
            'num_scenarios': len(scenarios)
        })
        
        try:
            # Ukur waktu eksekusi
            start_time = time.time()
            
            # Gunakan research pipeline
            research_pipeline = self._get_research_pipeline()
            
            # Jalankan evaluasi skenario
            results = research_pipeline.run(
                scenarios=scenarios,
                **kwargs
            )
            
            # Tambahkan waktu eksekusi total
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Buat visualisasi jika diminta
            if visualize:
                visualization_paths = self._create_research_visualizations(results)
                results['plots'] = visualization_paths
            
            # Buat laporan jika diminta
            if generate_report:
                report_path = self._generate_report(results, report_format)
                results['report_path'] = report_path
            
            # Notifikasi evaluasi skenario selesai
            EventDispatcher.notify("evaluation.research.complete", self, {
                'execution_time': execution_time,
                'num_scenarios': len(scenarios)
            })
            
            self.logger.success(f"✅ Evaluasi skenario penelitian selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi skenario penelitian gagal: {str(e)}")
            
            # Notifikasi error
            EventDispatcher.notify("evaluation.research.error", self, {
                'error': str(e)
            })
            
            raise
    
    def generate_report(
        self,
        results: Dict[str, Any],
        format: str = 'json',
        output_path: Optional[str] = None,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Buat laporan hasil evaluasi.
        
        Args:
            results: Hasil evaluasi
            format: Format laporan ('json', 'csv', 'markdown', 'html')
            output_path: Path output laporan (opsional)
            include_plots: Sertakan visualisasi
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan
        """
        report_generator = self._get_report_generator()
        
        # Buat path output jika belum ada
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / "reports" / f"report_{timestamp}.{format}")
        
        # Generate report
        report_path = report_generator.generate(
            results=results,
            format=format,
            output_path=output_path,
            include_plots=include_plots,
            **kwargs
        )
        
        return report_path
    
    def _setup_default_observers(self):
        """Setup observer default untuk logging dan progress tracking."""
        # Observer untuk logging evaluasi
        self.observer_manager.create_logging_observer(
            event_types=["evaluation.manager.start", "evaluation.manager.complete", "evaluation.manager.error"],
            log_level="info",
            name="EvaluationLogger",
            group="default_observers"
        )
        
        # Observer untuk progress tracking dalam penelitian
        if self.colab_mode:
            # Tambahan observer khusus untuk Colab
            pass
    
    def _get_model_evaluator(self) -> ModelEvaluator:
        """
        Lazy-load model evaluator.
        
        Returns:
            ModelEvaluator instance
        """
        if self._model_evaluator is None:
            metrics_calculator = MetricsCalculator()
            self._model_evaluator = ModelEvaluator(
                config=self.config,
                metrics_calculator=metrics_calculator,
                logger=self.logger.get_child("model_evaluator") if hasattr(self.logger, 'get_child') else self.logger
            )
            
        return self._model_evaluator
    
    def _get_report_generator(self) -> ReportGenerator:
        """
        Lazy-load report generator.
        
        Returns:
            ReportGenerator instance
        """
        if self._report_generator is None:
            self._report_generator = ReportGenerator(
                config=self.config,
                logger=self.logger.get_child("report_generator") if hasattr(self.logger, 'get_child') else self.logger
            )
            
        return self._report_generator
    
    def _get_evaluation_pipeline(self) -> EvaluationPipeline:
        """
        Lazy-load evaluation pipeline.
        
        Returns:
            EvaluationPipeline instance
        """
        if self._evaluation_pipeline is None:
            self._evaluation_pipeline = EvaluationPipeline(
                config=self.config,
                logger=self.logger.get_child("evaluation_pipeline") if hasattr(self.logger, 'get_child') else self.logger
            )
            
        return self._evaluation_pipeline
    
    def _get_batch_pipeline(self) -> BatchEvaluationPipeline:
        """
        Lazy-load batch evaluation pipeline.
        
        Returns:
            BatchEvaluationPipeline instance
        """
        if self._batch_pipeline is None:
            self._batch_pipeline = BatchEvaluationPipeline(
                config=self.config,
                logger=self.logger.get_child("batch_pipeline") if hasattr(self.logger, 'get_child') else self.logger
            )
            
        return self._batch_pipeline
    
    def _get_research_pipeline(self) -> ResearchPipeline:
        """
        Lazy-load research pipeline.
        
        Returns:
            ResearchPipeline instance
        """
        if self._research_pipeline is None:
            self._research_pipeline = ResearchPipeline(
                config=self.config,
                logger=self.logger.get_child("research_pipeline") if hasattr(self.logger, 'get_child') else self.logger
            )
            
        return self._research_pipeline
    
    def _create_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Buat visualisasi untuk hasil evaluasi model tunggal.
        
        Args:
            results: Hasil evaluasi
            
        Returns:
            Dictionary path visualisasi
        """
        # Di sini kita akan menggunakan visualizer untuk membuat plot
        # Karena implementasi visualization_adapter sudah baik, kita bisa tetap menggunakannya
        # atau beralih ke utils.visualization langsung
        
        # Untuk sekarang, kita akan mengembalikan dummy paths
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        return {
            'confusion_matrix': str(self.output_dir / "plots" / f"confusion_matrix_{timestamp}.png"),
            'pr_curve': str(self.output_dir / "plots" / f"pr_curve_{timestamp}.png")
        }
    
    def _create_batch_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Buat visualisasi untuk hasil evaluasi batch.
        
        Args:
            results: Hasil evaluasi batch
            
        Returns:
            Dictionary path visualisasi
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        return {
            'map_comparison': str(self.output_dir / "plots" / f"map_comparison_{timestamp}.png"),
            'inference_time': str(self.output_dir / "plots" / f"inference_time_{timestamp}.png")
        }
    
    def _create_research_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Buat visualisasi untuk hasil evaluasi skenario penelitian.
        
        Args:
            results: Hasil evaluasi skenario
            
        Returns:
            Dictionary path visualisasi
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        return {
            'backbone_comparison': str(self.output_dir / "plots" / f"backbone_comparison_{timestamp}.png"),
            'condition_comparison': str(self.output_dir / "plots" / f"condition_comparison_{timestamp}.png")
        }
    
    def _generate_report(self, results: Dict[str, Any], format: str) -> str:
        """
        Buat laporan hasil evaluasi.
        
        Args:
            results: Hasil evaluasi
            format: Format laporan
            
        Returns:
            Path ke laporan
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(self.output_dir / "reports" / f"report_{timestamp}.{format}")
        
        report_generator = self._get_report_generator()
        report_path = report_generator.generate(
            results=results,
            format=format,
            output_path=output_path
        )
        
        return report_path