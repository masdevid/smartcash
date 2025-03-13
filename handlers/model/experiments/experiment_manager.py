# File: smartcash/handlers/model/experiments/experiment_manager.py
# Deskripsi: Manager untuk eksperimen model dengan dependency injection langsung

import torch
import time
import json
import numpy as np
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from smartcash.exceptions.base import ModelError, TrainingError
from smartcash.utils.early_stopping import EarlyStopping
from smartcash.handlers.model.core.component_base import ComponentBase

class ExperimentManager(ComponentBase):
    """Manager untuk eksperimen model dengan dependency injection langsung."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional = None,
        model_factory = None,
        visualizer = None
    ):
        """
        Inisialisasi experiment manager.
        
        Args:
            config: Konfigurasi model dan eksperimen
            logger: Logger kustom (opsional)
            model_factory: ModelFactory instance (opsional)
            visualizer: ExperimentVisualizer instance (opsional)
        """
        super().__init__(config, logger, "experiment_manager")
        
        # Dependencies
        self.model_factory = model_factory
        self.visualizer = visualizer
    
    def _initialize(self) -> None:
        """Inisialisasi parameter eksperimen."""
        # Setup output dir dan konfigurasi
        self.output_dir = self.create_output_dir("experiments")
        self.experiment_config = self.config.get('experiment', {})
    
    def compare_backbones(
        self,
        backbones: List[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = 'val_loss',
        parallel: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            test_loader: DataLoader untuk testing (opsional)
            epochs: Jumlah epoch training
            batch_size: Ukuran batch
            early_stopping: Gunakan early stopping
            early_stopping_patience: Jumlah epoch tunggu untuk early stopping
            early_stopping_metric: Metrik untuk early stopping
            parallel: Jalankan perbandingan secara paralel
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        return self.safe_execute(
            self._compare_backbones_internal,
            "Gagal membandingkan backbone",
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            parallel=parallel,
            **kwargs
        )
    
    def _compare_backbones_internal(
        self,
        backbones: List[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        early_stopping: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = 'val_loss',
        parallel: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Implementasi internal untuk perbandingan backbone."""
        # Setup parameter eksperimen
        epochs = epochs or self.config.get('training', {}).get('epochs', 30)
        batch_size = batch_size or self.config.get('training', {}).get('batch_size', 16)
        early_stopping_patience = early_stopping_patience or self.config.get('training', {}).get('early_stopping_patience', 10)
        
        self.logger.info(
            f"🔬 Membandingkan {len(backbones)} backbone: {', '.join(backbones)}\n"
            f"   • Epochs: {epochs}\n"
            f"   • Batch size: {batch_size}\n"
            f"   • Early stopping: {early_stopping} "
            f"(metric: {early_stopping_metric}, patience: {early_stopping_patience})"
        )
        
        # Setup output directory dan waktu mulai
        experiment_name = kwargs.get('experiment_name', f"backbone_comparison_{int(time.time())}")
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        
        # Parameter eksperimen untuk dilewatkan ke metode pelatihan
        experiment_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_metric': early_stopping_metric,
            **kwargs
        }
        
        # Jalankan eksperimen (paralel atau serial)
        results = self._run_experiments(
            backbones, train_loader, val_loader, test_loader,
            experiment_dir, parallel, experiment_params
        )
        
        # Hitung waktu eksekusi total
        execution_time = time.time() - start_time
        
        # Visualisasi hasil jika semua eksperimen berhasil
        visualization_paths = {}
        if all('error' not in result for result in results.values()) and self.visualizer:
            visualization_paths = self.visualizer.visualize_backbone_comparison(
                results=results,
                title=f"Perbandingan Backbone - {experiment_name}",
                output_filename=f"{experiment_name}_comparison"
            )
        
        # Buat ringkasan dan gabungkan hasil
        summary = self._create_summary(results)
        final_results = {
            'experiment_name': experiment_name,
            'experiment_dir': str(experiment_dir),
            'num_backbones': len(backbones),
            'execution_time': execution_time,
            'backbones': backbones,
            'results': results,
            'summary': summary,
            'visualization_paths': visualization_paths
        }
        
        # Simpan hasil dan log sukses
        self._save_results(final_results, experiment_dir)
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.success(
            f"✅ Perbandingan backbone selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        
        return final_results
    
    def _run_experiments(
        self,
        backbones: List[str],
        train_loader,
        val_loader,
        test_loader,
        experiment_dir: Path,
        parallel: bool,
        params: Dict
    ) -> Dict[str, Any]:
        """
        Menjalankan eksperimen untuk semua backbone (paralel atau serial).
        
        Args:
            backbones: List backbone untuk diuji
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing
            experiment_dir: Direktori eksperimen
            parallel: Flag untuk eksekusi paralel
            params: Parameter eksperimen
            
        Returns:
            Dict hasil untuk semua backbone
        """
        results = {}
        
        # Jalankan paralel jika diminta dan ada lebih dari 1 backbone
        if parallel and len(backbones) > 1:
            max_workers = min(len(backbones), self.config.get('model', {}).get('workers', 2))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Setup futures untuk semua backbone
                futures = {
                    executor.submit(
                        self._train_and_evaluate_backbone,
                        backbone_type=backbone,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        experiment_dir=experiment_dir / backbone,
                        **params
                    ): backbone for backbone in backbones
                }
                
                # Progress bar
                pbar = tqdm(total=len(futures), desc="Eksperimen") if not self.in_colab else None
                
                # Process completed futures
                for future in as_completed(futures):
                    backbone = futures[future]
                    try:
                        results[backbone] = future.result()
                        self.logger.success(f"✅ Eksperimen {backbone} selesai")
                    except Exception as e:
                        self.logger.error(f"❌ Eksperimen {backbone} gagal: {str(e)}")
                        results[backbone] = {'error': str(e)}
                    
                    if pbar:
                        pbar.update(1)
                
                if pbar:
                    pbar.close()
        else:
            # Jalankan secara serial
            for backbone in backbones:
                try:
                    self.logger.info(f"🚀 Memulai eksperimen backbone: {backbone}")
                    results[backbone] = self._train_and_evaluate_backbone(
                        backbone_type=backbone,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        experiment_dir=experiment_dir / backbone,
                        **params
                    )
                    self.logger.success(f"✅ Eksperimen {backbone} selesai")
                except Exception as e:
                    self.logger.error(f"❌ Eksperimen {backbone} gagal: {str(e)}")
                    results[backbone] = {'error': str(e)}
                    
        return results
    
    def _train_and_evaluate_backbone(
        self,
        backbone_type: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 30,
        batch_size: int = 16,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = 'val_loss',
        experiment_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train dan evaluasi model dengan backbone tertentu.
        
        Args:
            backbone_type: Tipe backbone
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            epochs: Jumlah epoch training
            batch_size: Ukuran batch
            early_stopping: Gunakan early stopping
            early_stopping_patience: Jumlah epoch tunggu untuk early stopping
            early_stopping_metric: Metrik untuk early stopping
            experiment_dir: Direktori output eksperimen
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil training dan evaluasi
        """
        if experiment_dir:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
        # Setup metrics tracking
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rate': []
        }
        
        # Init early stopper
        early_stopper = None
        if early_stopping:
            # Deteksi mode berdasarkan metrik (min untuk loss, max untuk metrik seperti mAP)
            mode = 'min' if 'loss' in early_stopping_metric else 'max'
            early_stopper = EarlyStopping(
                monitor=early_stopping_metric,
                patience=early_stopping_patience,
                mode=mode,
                logger=self.logger
            )
            
        # Siapkan model
        if not self.model_factory:
            raise ModelError("Model factory diperlukan untuk eksperimen")
            
        model = self.model_factory.create_model(backbone_type=backbone_type)
        
        # Siapkan trainer
        from smartcash.handlers.model.core.model_trainer import ModelTrainer
        trainer = ModelTrainer(
            self.config, 
            self.logger,
            self.model_factory
        )
        
        # Train model
        self.logger.info(f"🏋️ Training model dengan backbone {backbone_type}")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            save_dir=str(experiment_dir / "weights") if experiment_dir else None,
            **kwargs
        )
        
        # Simpan metrics history
        if 'metrics_history' in training_results:
            metrics_history = training_results['metrics_history']
        
        # Evaluasi jika ada test_loader
        eval_results = {}
        if test_loader is not None:
            self.logger.info(f"🔍 Evaluasi model dengan backbone {backbone_type}")
            
            # Siapkan evaluator
            from smartcash.handlers.model.core.model_evaluator import ModelEvaluator
            
            # Buat metrics calculator jika diperlukan
            metrics_calculator = None
            try:
                from smartcash.utils.metrics import MetricsCalculator
                metrics_calculator = MetricsCalculator()
            except ImportError:
                self.logger.warning("⚠️ MetricsCalculator tidak tersedia")
                
            evaluator = ModelEvaluator(
                self.config, 
                self.logger,
                self.model_factory,
                metrics_calculator
            )
            
            # Load model terbaik atau gunakan model terakhir
            if 'best_checkpoint_path' in training_results and training_results['best_checkpoint_path']:
                best_model, _ = self.model_factory.load_model(training_results['best_checkpoint_path'])
                eval_results = evaluator.evaluate(
                    test_loader=test_loader,
                    model=best_model,
                    **kwargs
                )
            else:
                eval_results = evaluator.evaluate(
                    test_loader=test_loader,
                    model=model,
                    **kwargs
                )
                
        # Gabungkan dan return hasil
        return {
            'backbone': backbone_type,
            'training': training_results,
            'evaluation': eval_results,
            'epochs_completed': training_results.get('epoch', 0),
            'early_stopped': training_results.get('early_stopped', False),
            'best_val_loss': training_results.get('best_val_loss', float('inf')),
            'metrics_history': metrics_history,
        }
    
    def _create_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Buat ringkasan hasil perbandingan backbone."""
        summary = {
            'num_backbones': len(results),
            'successful_backbones': sum(1 for r in results.values() if 'error' not in r),
            'failed_backbones': sum(1 for r in results.values() if 'error' in r),
            'metrics_comparison': {},
            'training_summary': {}
        }
        
        # Metrics to compare and data collection
        metrics_to_compare = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
        metrics_data = {metric: {} for metric in metrics_to_compare}
        
        # Collect data for each backbone and metric
        for backbone, result in results.items():
            if 'error' in result:
                continue
                
            # Add evaluation metrics
            if 'evaluation' in result:
                eval_metrics = result['evaluation']
                for metric in metrics_to_compare:
                    if metric in eval_metrics:
                        metrics_data[metric][backbone] = eval_metrics[metric]
            
            # Add training info
            if 'training' in result:
                summary['training_summary'][backbone] = {
                    'epochs_completed': result.get('epochs_completed', 0),
                    'early_stopped': result.get('early_stopped', False),
                    'best_val_loss': result.get('best_val_loss', float('inf')),
                    'execution_time': result['training'].get('execution_time', 0)
                }
        
        # Find best for each metric
        for metric, values in metrics_data.items():
            if not values:
                continue
                
            # For inference time, lower is better
            is_minimize = metric == 'inference_time'
            compare_func = min if is_minimize else max
            best_backbone = compare_func(values.items(), key=lambda x: x[1])[0]
            
            summary['metrics_comparison'][metric] = {
                'values': values,
                'best_backbone': best_backbone,
                'best_value': values[best_backbone],
                'average': sum(values.values()) / len(values)
            }
        
        # Determine overall best backbone
        for preferred_metric in ['mAP', 'f1', *metrics_data.keys()]:
            if preferred_metric in summary['metrics_comparison']:
                summary['best_metric'] = preferred_metric
                summary['best_backbone'] = summary['metrics_comparison'][preferred_metric]['best_backbone']
                summary['best_value'] = summary['metrics_comparison'][preferred_metric]['best_value']
                break
                
        return summary
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Simpan hasil eksperimen ke file."""
        try:
            # Simpan hasil ke JSON
            results_path = output_dir / "experiment_results.json"
            
            # Konversi nilai untuk JSON
            def convert_to_serializable(obj):
                if isinstance(obj, (np.number, np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (Path, set)):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj
            
            # Simpan ke JSON dan Markdown
            serializable_results = convert_to_serializable(results)
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Buat ringkasan Markdown
            self._create_markdown_summary(results, output_dir)
                
            self.logger.info(f"💾 Hasil eksperimen disimpan: {results_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil: {str(e)}")
            
    def _create_markdown_summary(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Buat ringkasan eksperimen dalam format Markdown."""
        summary_path = output_dir / "experiment_summary.md"
        
        try:
            with open(summary_path, 'w') as f:
                f.write(f"# Ringkasan Eksperimen Perbandingan Backbone\n\n")
                
                # Informasi umum
                hours, remainder = divmod(results['execution_time'], 3600)
                minutes, seconds = divmod(remainder, 60)
                f.write("## Informasi Umum\n\n")
                f.write(f"- **Waktu Eksekusi**: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
                f.write(f"- **Backbone yang Diuji**: {', '.join(results['backbones'])}\n\n")
                
                # Backbone terbaik
                if 'summary' in results and 'best_backbone' in results['summary']:
                    summary = results['summary']
                    f.write(f"- **Backbone Terbaik**: {summary['best_backbone']} ({summary['best_metric']})\n")
                    f.write(f"- **Nilai Terbaik**: {summary['best_value']:.4f}\n\n")
                
                # Referensi visualisasi
                if 'visualization_paths' in results and results['visualization_paths']:
                    f.write("## Visualisasi\n\n")
                    for name, path in results['visualization_paths'].items():
                        rel_path = Path(path).relative_to(output_dir) if Path(path).is_absolute() else Path(path)
                        f.write(f"- [{name.replace('_', ' ').title()}]({rel_path})\n")
                        
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat ringkasan Markdown: {str(e)}")