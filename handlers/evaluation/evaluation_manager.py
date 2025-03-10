# File: smartcash/handlers/evaluation/evaluation_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager untuk evaluasi model dengan pendekatan ringkas dan direct injection

import os
import time
import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.metrics import MetricsCalculator
from smartcash.utils.observer import EventDispatcher, ObserverManager
from smartcash.handlers.evaluation.pipeline import EvaluationPipeline, BatchEvaluationPipeline, ResearchPipeline

class EvaluationManager:
    """
    Manager utama untuk evaluasi model dengan direct injection.
    Menyediakan antarmuka sederhana untuk berbagai operasi evaluasi.
    """
    
    def __init__(
        self,
        config: Dict,
        logger = None,
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
        
        # Auto-detect colab mode jika tidak dispesifikasikan
        if colab_mode is None:
            try:
                import google.colab
                colab_mode = True
            except ImportError:
                colab_mode = False
        
        self.colab_mode = colab_mode
        
        # Setup observer manager
        self.observer_manager = ObserverManager()
        
        # Cache untuk komponen lazy-load
        self._components = {}
        
        # Parameter dari config
        eval_config = self.config.get('evaluation', {})
        self.output_dir = Path(eval_config.get('output_dir', 'results/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔧 EvaluationManager diinisialisasi (colab_mode={colab_mode})")
        
        # Setup observers
        self._setup_default_observers()
    
    def evaluate_model(
        self,
        model = None,
        model_path = None,
        dataset_path = None,
        dataloader = None,
        device = None,
        conf_threshold = None,
        visualize = True,
        generate_report = True,
        report_format = 'json',
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
        """
        self.logger.info("🚀 Mulai evaluasi model")
        
        # Validasi input dasar
        if model is None and model_path is None:
            raise ValueError("Harus menyediakan model atau model_path")
        
        if dataloader is None and dataset_path is None:
            raise ValueError("Harus menyediakan dataloader atau dataset_path")
        
        # Notifikasi event
        EventDispatcher.notify("evaluation.manager.start", self, {
            'model_path': model_path,
            'dataset_path': dataset_path
        })
        
        try:
            # Ukur waktu eksekusi
            start_time = time.time()
            
            # Dapatkan evaluation pipeline
            pipeline = self._get_component('evaluation_pipeline', EvaluationPipeline)
            
            # Set parameter dari config jika belum ditentukan
            if device is None:
                device = self.config.get('evaluation', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
            if conf_threshold is None:
                conf_threshold = self.config.get('evaluation', {}).get('conf_threshold', 0.25)
            
            # Jalankan pipeline
            results = pipeline.run(
                model=model,
                model_path=model_path,
                dataset_path=dataset_path,
                dataloader=dataloader,
                device=device,
                conf_threshold=conf_threshold,
                **kwargs
            )
            
            # Tambahkan waktu eksekusi
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Buat visualisasi
            if visualize:
                visualization_paths = self._create_visualizations(results)
                results['visualization_paths'] = visualization_paths
            
            # Buat laporan
            if generate_report:
                report_path = self._generate_report(results, report_format)
                results['report_path'] = report_path
            
            # Notifikasi selesai
            EventDispatcher.notify("evaluation.manager.complete", self, {
                'execution_time': execution_time,
                'metrics': results.get('metrics', {})
            })
            
            self.logger.success(f"✅ Evaluasi model selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi model gagal: {str(e)}")
            EventDispatcher.notify("evaluation.manager.error", self, {'error': str(e)})
            raise
    
    def evaluate_batch(
        self,
        models = None,
        model_paths = None,
        dataset_path = None,
        dataloader = None,
        parallel = True,
        visualize = True,
        generate_report = True,
        report_format = 'markdown',
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
        """
        self.logger.info("🚀 Mulai evaluasi batch model")
        
        # Validasi input dasar
        if (models is None or len(models) == 0) and (model_paths is None or len(model_paths) == 0):
            raise ValueError("Harus menyediakan models atau model_paths")
        
        if dataloader is None and dataset_path is None:
            raise ValueError("Harus menyediakan dataloader atau dataset_path")
        
        # Notifikasi dan timing
        start_time = time.time()
        EventDispatcher.notify("evaluation.batch.start", self, {
            'num_models': len(models or model_paths),
            'dataset_path': dataset_path
        })
        
        try:
            # Dapatkan batch pipeline
            batch_pipeline = self._get_component('batch_pipeline', BatchEvaluationPipeline)
            
            # Convert models ke model_paths jika perlu
            if models is not None and model_paths is None:
                model_paths = []
                for i, model in enumerate(models):
                    temp_path = str(self.output_dir / f"temp_model_{i}.pt")
                    torch.save(model.state_dict(), temp_path)
                    model_paths.append(temp_path)
            
            # Jalankan evaluasi batch
            results = batch_pipeline.run(
                model_paths=model_paths,
                dataset_path=dataset_path,
                dataloader=dataloader,
                parallel=parallel,
                **kwargs
            )
            
            # Tambahkan waktu eksekusi
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Visualisasi dan laporan
            if visualize:
                results['plots'] = self._create_batch_visualizations(results)
            
            if generate_report:
                results['report_path'] = self._generate_report(results, report_format)
            
            # Notifikasi selesai
            EventDispatcher.notify("evaluation.batch.complete", self, {
                'execution_time': execution_time,
                'num_models': len(model_paths)
            })
            
            self.logger.success(f"✅ Evaluasi batch selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi batch gagal: {str(e)}")
            EventDispatcher.notify("evaluation.batch.error", self, {'error': str(e)})
            raise
        finally:
            # Cleanup temporary files
            if models is not None and model_paths is not None:
                for path in model_paths:
                    if str(path).startswith(str(self.output_dir / "temp_model_")):
                        try:
                            os.remove(path)
                        except:
                            pass
    
    def evaluate_research_scenarios(
        self,
        scenarios: Dict[str, Dict],
        visualize = True,
        generate_report = True,
        report_format = 'markdown',
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
        """
        self.logger.info(f"🔬 Mulai evaluasi {len(scenarios)} skenario penelitian")
        
        # Validasi input dasar
        if not scenarios:
            raise ValueError("Harus menyediakan minimal satu skenario")
        
        # Notifikasi dan timing
        start_time = time.time()
        EventDispatcher.notify("evaluation.research.start", self, {
            'num_scenarios': len(scenarios)
        })
        
        try:
            # Dapatkan research pipeline
            research_pipeline = self._get_component('research_pipeline', ResearchPipeline)
            
            # Jalankan evaluasi skenario
            results = research_pipeline.run(scenarios=scenarios, **kwargs)
            
            # Tambahkan waktu eksekusi
            execution_time = time.time() - start_time
            results['total_execution_time'] = execution_time
            
            # Visualisasi dan laporan
            if visualize:
                results['plots'] = self._create_research_visualizations(results)
            
            if generate_report:
                results['report_path'] = self._generate_report(results, report_format)
            
            # Notifikasi selesai
            EventDispatcher.notify("evaluation.research.complete", self, {
                'execution_time': execution_time,
                'num_scenarios': len(scenarios)
            })
            
            self.logger.success(f"✅ Evaluasi skenario penelitian selesai dalam {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluasi skenario penelitian gagal: {str(e)}")
            EventDispatcher.notify("evaluation.research.error", self, {'error': str(e)})
            raise
    
    def generate_report(
        self,
        results: Dict[str, Any],
        format = 'json',
        output_path = None,
        include_plots = True,
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
        # Dapatkan report generator
        from smartcash.handlers.evaluation.core.report_generator import ReportGenerator
        report_generator = self._get_component('report_generator', ReportGenerator)
        
        # Buat output path jika belum ada
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / "reports" / f"report_{timestamp}.{format}")
        
        # Buat report
        return report_generator.generate(
            results=results,
            format=format,
            output_path=output_path,
            include_plots=include_plots,
            **kwargs
        )
    
    def _setup_default_observers(self):
        """Setup observer default untuk logging dan progress tracking."""
        # Observer untuk logging evaluasi
        self.observer_manager.create_logging_observer(
            event_types=["evaluation.manager.start", "evaluation.manager.complete", "evaluation.manager.error"],
            log_level="info",
            name="EvaluationLogger",
            group="default_observers"
        )
    
    def _get_component(self, name, component_class, **kwargs):
        """
        Lazy-load komponen.
        
        Args:
            name: Nama komponen untuk cache
            component_class: Kelas komponen yang akan dibuat
            **kwargs: Parameter tambahan untuk constructor
            
        Returns:
            Instance komponen
        """
        if name not in self._components:
            self._components[name] = component_class(
                config=self.config,
                logger=self.logger.get_child(name) if hasattr(self.logger, 'get_child') else self.logger,
                **kwargs
            )
        return self._components[name]
    
    def _create_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Buat visualisasi untuk hasil evaluasi model tunggal."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'confusion_matrix': str(plots_dir / f"confusion_matrix_{timestamp}.png"),
            'pr_curve': str(plots_dir / f"pr_curve_{timestamp}.png")
        }
    
    def _create_batch_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Buat visualisasi untuk hasil evaluasi batch."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'map_comparison': str(plots_dir / f"map_comparison_{timestamp}.png"),
            'inference_time': str(plots_dir / f"inference_time_{timestamp}.png")
        }
    
    def _create_research_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Buat visualisasi untuk hasil evaluasi skenario penelitian."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'backbone_comparison': str(plots_dir / f"backbone_comparison_{timestamp}.png"),
            'condition_comparison': str(plots_dir / f"condition_comparison_{timestamp}.png")
        }
    
    def _generate_report(self, results: Dict[str, Any], format: str) -> str:
        """Generate laporan hasil evaluasi."""
        return self.generate_report(
            results=results,
            format=format
        )