"""
File: smartcash/model/services/research/experiment_analyzer.py
Deskripsi: Komponen untuk menganalisis hasil eksperimen
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from smartcash.common.logger import get_logger
from smartcash.model.config.experiment_config import ExperimentConfig
from smartcash.model.utils.research_model_utils import format_metric_name


class ExperimentAnalyzer:
    """
    Komponen untuk menganalisis hasil eksperimen.
    
    Bertanggung jawab untuk:
    - Daftar dan filter eksperimen
    - Mendapatkan dan memformat hasil eksperimen
    - Membandingkan beberapa eksperimen
    - Menghasilkan laporan dan visualisasi
    """
    
    def __init__(
        self,
        base_dir: str,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment analyzer.
        
        Args:
            base_dir: Direktori dasar untuk eksperimen
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger()
        
        self.logger.debug(f"🔍 ExperimentAnalyzer diinisialisasi")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Dapatkan hasil eksperimen berdasarkan ID.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            Dictionary hasil eksperimen
        """
        experiment_dir = self.base_dir / experiment_id
        results_path = experiment_dir / "results.json"
        
        if not results_path.exists():
            self.logger.warning(f"⚠️ Hasil eksperimen tidak ditemukan: {experiment_id}")
            return {"status": "error", "message": "Hasil eksperimen tidak ditemukan"}
        
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            
            self.logger.info(f"📊 Loaded results for experiment: {experiment_id}")
            return results
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat hasil eksperimen: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def list_experiments(self, filter_tags: List[str] = None) -> pd.DataFrame:
        """
        Dapatkan daftar semua eksperimen.
        
        Args:
            filter_tags: Filter berdasarkan tag (opsional)
            
        Returns:
            DataFrame berisi informasi semua eksperimen
        """
        experiments = []
        
        # Cari semua direktori eksperimen
        for exp_dir in self.base_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            config_path = exp_dir / "config.yaml"
            results_path = exp_dir / "results.json"
            
            if not config_path.exists():
                continue
                
            # Baca config
            try:
                exp_config = ExperimentConfig(name="temp", experiment_dir=str(exp_dir))
                
                # Filter berdasarkan tag jika diperlukan
                if filter_tags and not any(tag in exp_config.parameters.get("tags", []) for tag in filter_tags):
                    continue
                
                # Periksa apakah ada hasil
                has_results = results_path.exists()
                
                # Tambahkan ke daftar
                experiments.append({
                    "experiment_id": exp_config.experiment_id,
                    "name": exp_config.parameters.get("name", "Unknown"),
                    "description": exp_config.parameters.get("description", ""),
                    "tags": ", ".join(exp_config.parameters.get("tags", [])),
                    "timestamp": exp_config.timestamp,
                    "has_results": has_results
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat konfigurasi eksperimen di {exp_dir}: {str(e)}")
        
        # Konversi ke DataFrame
        return pd.DataFrame(experiments)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Bandingkan beberapa eksperimen.
        
        Args:
            experiment_ids: List ID eksperimen yang akan dibandingkan
            metrics: List metrik yang akan dibandingkan (opsional)
            
        Returns:
            DataFrame perbandingan eksperimen
        """
        if metrics is None:
            metrics = ["mAP", "precision", "recall", "f1"]
            
        comparison_data = []
        
        for exp_id in experiment_ids:
            results = self.get_experiment_results(exp_id)
            
            if results.get("status") == "error":
                self.logger.warning(f"⚠️ Gagal memuat hasil untuk {exp_id}")
                continue
                
            # Dapatkan nama eksperimen
            exp_config = self._get_experiment_config(exp_id)
            name = exp_config.parameters.get("name", exp_id) if exp_config else exp_id
            
            # Ekstrak metrik yang diminta
            row = {"Experiment": name, "ID": exp_id}
            
            for metric in metrics:
                if metric.lower() in ["final_loss", "loss"]:
                    value = results.get("training", {}).get("final_loss", "N/A")
                else:
                    value = results.get("evaluation", {}).get(metric.lower(), "N/A")
                
                row[format_metric_name(metric)] = value
                
            # Tambahkan data parameter
            row["Model Type"] = results.get("model_type", "Unknown")
            row["Batch Size"] = results.get("parameters", {}).get("batch_size", "N/A")
            row["Learning Rate"] = results.get("parameters", {}).get("learning_rate", "N/A")
            row["Epochs"] = results.get("parameters", {}).get("epochs", "N/A")
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_experiment_report(
        self,
        experiment_id: str,
        include_plots: bool = True
    ) -> str:
        """
        Generate report untuk eksperimen.
        
        Args:
            experiment_id: ID eksperimen
            include_plots: Sertakan plot dalam report
            
        Returns:
            Path ke file report
        """
        # Dapatkan konfigurasi eksperimen
        experiment_config = self._get_experiment_config(experiment_id)
        if not experiment_config:
            self.logger.warning(f"⚠️ Eksperimen tidak ditemukan: {experiment_id}")
            return ""
        
        # Generate report
        report_path = experiment_config.generate_report(include_plots=include_plots)
        
        self.logger.info(f"📝 Report generated: {report_path}")
        return report_path
    
    def analyze_experiment_performance(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Analisis performa eksperimen secara mendalam.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            Dictionary hasil analisis
        """
        results = self.get_experiment_results(experiment_id)
        if results.get("status") == "error":
            return {"status": "error", "message": "Tidak dapat menganalisis eksperimen"}
        
        # Dapatkan metrics dan informasi performa
        evaluation = results.get("evaluation", {})
        training = results.get("training", {})
        
        analysis = {
            "status": "success",
            "experiment_id": experiment_id,
            "model_type": results.get("model_type", "Unknown"),
            "summary": {
                "mAP": evaluation.get("mAP", 0),
                "precision": evaluation.get("precision", 0),
                "recall": evaluation.get("recall", 0),
                "f1": evaluation.get("f1", 0),
                "final_loss": training.get("final_loss", 0),
                "inference_time": evaluation.get("inference_time", 0),
            },
            "training_convergence": self._analyze_training_convergence(training),
            "class_performance": self._analyze_class_performance(evaluation),
            "recommended_improvements": self._generate_improvement_recommendations(results)
        }
        
        return analysis
    
    def _analyze_training_convergence(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis konvergensi training.
        
        Args:
            training_results: Hasil training
            
        Returns:
            Dictionary hasil analisis konvergensi
        """
        losses = training_results.get("losses", [])
        val_losses = training_results.get("val_losses", [])
        
        if not losses or not val_losses:
            return {"status": "unknown", "message": "Data loss tidak tersedia"}
        
        # Analisis sederhana
        early_stopping = training_results.get("early_stopping", False)
        final_loss = training_results.get("final_loss", 0)
        final_val_loss = training_results.get("final_val_loss", 0)
        
        # Apakah model overfit?
        overfit = final_val_loss > final_loss * 1.1
        
        # Apakah model konvergen?
        converged = len(losses) > 1 and abs(losses[-1] - losses[-2]) < 0.001
        
        return {
            "converged": converged,
            "overfit": overfit,
            "early_stopping_triggered": early_stopping,
            "final_loss_ratio": final_val_loss / final_loss if final_loss > 0 else None
        }
    
    def _analyze_class_performance(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis performa per kelas.
        
        Args:
            evaluation_results: Hasil evaluasi
            
        Returns:
            Dictionary hasil analisis per kelas
        """
        class_metrics = evaluation_results.get("class_metrics", {})
        
        if not class_metrics:
            return {"status": "unknown", "message": "Data metrik per kelas tidak tersedia"}
        
        # Cari kelas dengan performa terbaik dan terburuk
        best_class = None
        worst_class = None
        best_f1 = -1
        worst_f1 = float('inf')
        
        for class_id, metrics in class_metrics.items():
            f1 = metrics.get("f1", 0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_class = class_id
                
            if f1 < worst_f1:
                worst_f1 = f1
                worst_class = class_id
        
        return {
            "best_class": {
                "id": best_class,
                "f1": best_f1,
                "metrics": class_metrics.get(best_class, {})
            },
            "worst_class": {
                "id": worst_class,
                "f1": worst_f1,
                "metrics": class_metrics.get(worst_class, {})
            },
            "class_variance": self._calculate_class_variance(class_metrics)
        }
    
    def _calculate_class_variance(self, class_metrics: Dict[str, Dict]) -> float:
        """
        Hitung variance performa antar kelas.
        
        Args:
            class_metrics: Metrik per kelas
            
        Returns:
            Nilai variance
        """
        if not class_metrics:
            return 0.0
            
        f1_scores = [metrics.get("f1", 0) for metrics in class_metrics.values()]
        
        if not f1_scores:
            return 0.0
            
        import numpy as np
        return float(np.var(f1_scores)) if len(f1_scores) > 1 else 0.0
    
    def _generate_improvement_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate rekomendasi perbaikan berdasarkan hasil.
        
        Args:
            results: Hasil eksperimen
            
        Returns:
            List rekomendasi
        """
        recommendations = []
        
        # Dapatkan evaluasi dan training
        evaluation = results.get("evaluation", {})
        training = results.get("training", {})
        parameters = results.get("parameters", {})
        
        # 1. Periksa overfitting
        if training.get("final_val_loss", 0) > training.get("final_loss", 0) * 1.1:
            recommendations.append(
                "Model menunjukkan tanda-tanda overfitting. Pertimbangkan untuk menambahkan "
                "regularisasi atau augmentasi data."
            )
        
        # 2. Periksa convergence
        losses = training.get("losses", [])
        if losses and losses[-1] > 0.1:
            recommendations.append(
                "Model belum sepenuhnya konvergen. Pertimbangkan training dengan "
                "epoch lebih banyak atau learning rate yang berbeda."
            )
        
        # 3. Periksa class imbalance
        class_metrics = evaluation.get("class_metrics", {})
        if class_metrics:
            f1_scores = [metrics.get("f1", 0) for metrics in class_metrics.values()]
            if max(f1_scores) - min(f1_scores) > 0.3:
                recommendations.append(
                    "Terdapat perbedaan besar dalam performa antar kelas. "
                    "Periksa keseimbangan dataset dan pertimbangkan teknik sampling."
                )
        
        # 4. Model size vs. performance
        if evaluation.get("inference_time", 0) > 100:  # 100ms threshold
            recommendations.append(
                "Waktu inferensi cukup tinggi. Pertimbangkan penggunaan model yang lebih ringan "
                "atau teknik optimasi seperti pruning atau quantization."
            )
        
        # 5. Fallback jika tidak ada rekomendasi
        if not recommendations:
            metrics = evaluation.get("mAP", 0)
            if metrics > 0.9:
                recommendations.append(
                    "Model menunjukkan performa yang sangat baik. Pertimbangkan untuk menguji "
                    "dengan dataset lebih beragam untuk mengkonfirmasi ketahanan model."
                )
            else:
                recommendations.append(
                    "Pertimbangkan untuk eksperimen dengan arsitektur backbone yang berbeda "
                    "atau parameter training yang dioptimasi."
                )
        
        return recommendations
    
    def _get_experiment_config(
        self,
        experiment_id: str
    ) -> Optional[ExperimentConfig]:
        """
        Dapatkan konfigurasi eksperimen berdasarkan ID.
        
        Args:
            experiment_id: ID eksperimen
            
        Returns:
            ExperimentConfig untuk eksperimen atau None jika tidak ditemukan
        """
        experiment_dir = self.base_dir / experiment_id
        if experiment_dir.exists() and (experiment_dir / "config.yaml").exists():
            try:
                config = ExperimentConfig(name="loaded", experiment_dir=str(experiment_dir))
                return config
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
        
        return None