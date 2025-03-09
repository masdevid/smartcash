# File: smartcash/handlers/evaluation/evaluation_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama untuk evaluasi model dengan pola desain Facade yang telah dioptimasi

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import time

from smartcash.utils.logger import get_logger
from smartcash.handlers.evaluation.integration.adapters_factory import AdaptersFactory
from smartcash.handlers.evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from smartcash.handlers.evaluation.pipeline.batch_evaluation_pipeline import BatchEvaluationPipeline
from smartcash.handlers.evaluation.pipeline.research_pipeline import ResearchPipeline
from smartcash.handlers.evaluation.core.report_generator import ReportGenerator

class EvaluationManager:
    """
    Manager utama evaluasi sebagai facade.
    Menyederhanakan antarmuka untuk evaluasi model dengan menggunakan berbagai adapter dan pipeline.
    """
    
    def __init__(
        self,
        config: Dict,
        logger = None,
        colab_mode: bool = False
    ):
        """
        Inisialisasi evaluation manager dengan berbagai adapter dan pipeline.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            colab_mode: Mode Google Colab
        """
        self.config = config
        self.logger = logger or get_logger("evaluation_manager")
        self.colab_mode = colab_mode
        
        # Setup direktori output
        self.output_dir = Path(config.get('output_dir', 'results/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Factory untuk adapter
        self.adapters_factory = AdaptersFactory(config, self.logger)
        
        # Inisialisasi pipelines
        self.standard_pipeline = EvaluationPipeline(config, self.logger)
        self.research_pipeline = ResearchPipeline(config, self.logger)
        self.batch_pipeline = BatchEvaluationPipeline(config, self.logger)
        
        self.logger.success(f"✅ EvaluationManager berhasil diinisialisasi")
    
    def evaluate_model(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluasi model tunggal menggunakan pipeline standar.
        
        Args:
            model_path: Path ke file model (opsional, gunakan terbaik jika None)
            dataset_path: Path ke dataset evaluasi (opsional, gunakan test_dir dari config)
            **kwargs: Parameter tambahan untuk pipeline
            
        Returns:
            Dictionary berisi hasil evaluasi
        """
        self.logger.info("🚀 Memulai evaluasi model tunggal...")
        
        # Gunakan default jika parameter tidak disediakan
        checkpoint_adapter = self.adapters_factory.get_checkpoint_adapter()
        model_path = model_path or checkpoint_adapter.get_best_checkpoint()
        dataset_path = dataset_path or self.config.get('data', {}).get('test_dir', 'data/test')
        
        # Dapatkan adapter yang diperlukan
        metrics_adapter = self.adapters_factory.get_metrics_adapter()
        model_adapter = self.adapters_factory.get_model_adapter()
        dataset_adapter = self.adapters_factory.get_dataset_adapter()
        
        # Jalankan pipeline standar
        return self.standard_pipeline.run(
            model_path=model_path,
            dataset_path=dataset_path,
            metrics_adapter=metrics_adapter,
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            **kwargs
        )
    
    def evaluate_batch(
        self,
        model_paths: Optional[List[str]] = None,
        dataset_path: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluasi batch model menggunakan pipeline batch.
        
        Args:
            model_paths: List path ke file model (opsional, gunakan semua checkpoint jika None)
            dataset_path: Path ke dataset evaluasi (opsional, gunakan test_dir dari config)
            **kwargs: Parameter tambahan untuk pipeline
            
        Returns:
            Dictionary berisi hasil evaluasi untuk setiap model
        """
        self.logger.info("🚀 Memulai evaluasi batch model...")
        
        # Gunakan default jika parameter tidak disediakan
        checkpoint_adapter = self.adapters_factory.get_checkpoint_adapter()
        model_paths = model_paths or checkpoint_adapter.list_checkpoints()
        dataset_path = dataset_path or self.config.get('data', {}).get('test_dir', 'data/test')
        
        # Dapatkan adapter yang diperlukan
        metrics_adapter = self.adapters_factory.get_metrics_adapter()
        model_adapter = self.adapters_factory.get_model_adapter()
        dataset_adapter = self.adapters_factory.get_dataset_adapter()
        
        # Jalankan pipeline batch
        return self.batch_pipeline.run(
            model_paths=model_paths,
            dataset_path=dataset_path,
            metrics_adapter=metrics_adapter,
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            **kwargs
        )
    
    def evaluate_research_scenarios(
        self,
        scenarios: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluasi skenario penelitian menggunakan pipeline penelitian.
        
        Args:
            scenarios: Dictionary skenario penelitian (opsional, gunakan default jika None)
            **kwargs: Parameter tambahan untuk pipeline
            
        Returns:
            Dictionary berisi hasil evaluasi untuk setiap skenario
        """
        self.logger.info("🚀 Memulai evaluasi skenario penelitian...")
        
        # Gunakan default jika parameter tidak disediakan
        scenarios = scenarios or self._get_default_scenarios()
        
        # Dapatkan adapter yang diperlukan
        metrics_adapter = self.adapters_factory.get_metrics_adapter()
        model_adapter = self.adapters_factory.get_model_adapter()
        dataset_adapter = self.adapters_factory.get_dataset_adapter()
        
        # Jalankan pipeline penelitian
        return self.research_pipeline.run(
            scenarios=scenarios,
            metrics_adapter=metrics_adapter,
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            **kwargs
        )
    
    def generate_report(
        self,
        results: Dict,
        format: str = 'json',
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Buat laporan hasil evaluasi.
        
        Args:
            results: Hasil evaluasi 
            format: Format laporan ('json', 'csv', 'markdown', 'html')
            output_path: Path output laporan (opsional)
            **kwargs: Parameter tambahan untuk generator laporan
            
        Returns:
            Path ke file laporan
        """
        self.logger.info(f"📊 Membuat laporan evaluasi ({format})...")
        
        # Generate output path jika tidak disediakan
        output_path = output_path or str(self.output_dir / f"evaluation_report.{format}")
        
        # Buat generator laporan dan generate laporan
        report_generator = ReportGenerator(self.config, self.logger)
        report_path = report_generator.generate(
            results=results,
            format=format,
            output_path=output_path,
            **kwargs
        )
        
        self.logger.success(f"✅ Laporan evaluasi berhasil dibuat: {report_path}")
        return report_path
    
    def visualize_results(
        self,
        results: Dict,
        prefix: str = "",
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Visualisasikan hasil evaluasi menggunakan VisualizationAdapter.
        
        Args:
            results: Hasil evaluasi (dari evaluate_model, evaluate_batch, atau evaluate_research_scenarios)
            prefix: Prefix untuk nama file output (opsional)
            output_dir: Direktori output untuk visualisasi (opsional)
            **kwargs: Parameter tambahan untuk visualisasi
            
        Returns:
            Dictionary berisi path ke file visualisasi yang dihasilkan
        """
        self.logger.info("🎨 Membuat visualisasi hasil evaluasi...")
        visualization_adapter = self.adapters_factory.get_visualization_adapter(
            output_dir=output_dir or str(self.plot_dir)
        )
        
        # Tentukan tipe hasil untuk visualisasi yang sesuai
        result_type = self._determine_result_type(results)
        
        if result_type == 'batch':
            return visualization_adapter.generate_batch_plots(results, prefix=prefix or "batch", **kwargs)
        elif result_type == 'research':
            return visualization_adapter.generate_research_plots(results, prefix=prefix or "research", **kwargs)
        elif result_type == 'single':
            # Format data untuk visualisasi
            metrics_data = self._prepare_single_model_metrics(results)
            return visualization_adapter.generate_custom_plots(
                data=metrics_data,
                prefix=prefix or "single",
                **kwargs
            )
        else:
            self.logger.warning("⚠️ Format hasil tidak dikenali untuk visualisasi")
            return {}
    
    def _determine_result_type(self, results: Dict) -> str:
        """
        Tentukan tipe hasil evaluasi.
        
        Args:
            results: Hasil evaluasi
            
        Returns:
            Tipe hasil: 'batch', 'research', 'single' atau 'unknown'
        """
        if 'model_results' in results:
            return 'batch'
        elif 'scenario_results' in results:
            return 'research'
        elif 'metrics' in results:
            return 'single'
        else:
            return 'unknown'
    
    def _prepare_single_model_metrics(self, results: Dict) -> List[Dict]:
        """
        Siapkan metrik model tunggal untuk visualisasi.
        
        Args:
            results: Hasil evaluasi model tunggal
            
        Returns:
            List metrics untuk visualisasi
        """
        model_name = Path(results.get('model_path', '')).stem
        return [{
            'model': model_name,
            'mAP': results['metrics'].get('mAP', 0),
            'F1': results['metrics'].get('f1', 0),
            'precision': results['metrics'].get('precision', 0),
            'recall': results['metrics'].get('recall', 0),
            'inference_time': results['metrics'].get('inference_time', 0) * 1000  # ms
        }]
    
    def _get_default_scenarios(self) -> Dict:
        """
        Dapatkan skenario penelitian default.
        
        Returns:
            Dictionary skenario penelitian default
        """
        return {
            'Skenario-1': {
                'desc': 'YOLOv5 Default (CSPDarknet) - Posisi Bervariasi',
                'model': 'cspdarknet_position_varied.pt',
                'data': 'test_position_varied'
            },
            'Skenario-2': {
                'desc': 'YOLOv5 Default (CSPDarknet) - Pencahayaan Bervariasi',
                'model': 'cspdarknet_lighting_varied.pt',
                'data': 'test_lighting_varied'
            },
            'Skenario-3': {
                'desc': 'YOLOv5 EfficientNet-B4 - Posisi Bervariasi',
                'model': 'efficientnet_position_varied.pt',
                'data': 'test_position_varied'
            },
            'Skenario-4': {
                'desc': 'YOLOv5 EfficientNet-B4 - Pencahayaan Bervariasi',
                'model': 'efficientnet_lighting_varied.pt',
                'data': 'test_lighting_varied'
            }
        }