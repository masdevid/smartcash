"""
File: smartcash/model/services/research/comparison_runner.py
Deskripsi: Komponen untuk menjalankan eksperimen perbandingan model
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger


class ComparisonRunner:
    """
    Komponen untuk menjalankan eksperimen perbandingan model.
    
    Bertanggung jawab untuk:
    - Menjalankan perbandingan beberapa model
    - Mengumpulkan dan menganalisis hasil perbandingan
    - Identifikasi model terbaik
    """
    
    def __init__(
        self,
        base_dir: str,
        experiment_runner: Any,
        experiment_creator: Any,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi comparison runner.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan hasil eksperimen
            experiment_runner: Komponen experiment runner
            experiment_creator: Komponen experiment creator
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger()
        self.experiment_runner = experiment_runner
        self.experiment_creator = experiment_creator
        
        self.logger.debug(f"🔄 ComparisonRunner diinisialisasi")
    
    def run_comparison_experiment(
        self,
        name: str,
        dataset_path: str,
        models_to_compare: List[str],
        epochs: int = 10,
        batch_size: int = 16,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan eksperimen perbandingan beberapa model.
        
        Args:
            name: Nama eksperimen perbandingan
            dataset_path: Path ke dataset
            models_to_compare: List tipe model yang akan dibandingkan
            epochs: Jumlah epoch
            batch_size: Ukuran batch
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil perbandingan
        """
        self.logger.info(f"🔄 Memulai eksperimen perbandingan: {name}")
        self.logger.info(f"📋 Model yang akan dibandingkan: {models_to_compare}")
        
        # Setup grup eksperimen
        group_info = self.experiment_creator.create_experiment_group(name, "comparison")
        group_id, group_dir = group_info["id"], group_info["dir"]
        
        # Jalankan eksperimen untuk setiap model
        results, comparison_data = self._run_model_comparison_experiments(
            name, dataset_path, models_to_compare, epochs, batch_size, **kwargs
        )
        
        # Proses dan simpan hasil
        comparison_df = self._process_comparison_results(results, comparison_data, group_dir)
        
        # Log hasil
        best_model = self._get_best_model(comparison_df)
        self.logger.success(
            f"✅ Eksperimen perbandingan selesai\n"
            f"   • Hasil disimpan di: {group_dir}\n"
            f"   • Model terbaik: {best_model}"
        )
        
        return {
            "status": "success",
            "comparison_id": group_id,
            "results": results,
            "comparison_df": comparison_df,
            "best_model": best_model
        }
    
    def _run_model_comparison_experiments(
        self,
        name: str,
        dataset_path: str,
        models_to_compare: List[str],
        epochs: int,
        batch_size: int,
        **kwargs
    ) -> Tuple[Dict, List]:
        """
        Jalankan eksperimen untuk setiap model dalam perbandingan.
        
        Args:
            name: Nama eksperimen
            dataset_path: Path ke dataset
            models_to_compare: List model yang akan dibandingkan
            epochs: Jumlah epoch
            batch_size: Ukuran batch
            **kwargs: Parameter tambahan
            
        Returns:
            Tuple (results, comparison_data)
        """
        # Dictionary untuk menyimpan hasil
        results = {}
        comparison_data = []
        
        # Progress bar
        with tqdm(total=len(models_to_compare), desc="Membandingkan model") as pbar:
            for model_type in models_to_compare:
                # Buat sub-eksperimen
                exp_name = f"{name}_{model_type}"
                experiment = self.experiment_creator.create_experiment(
                    name=exp_name,
                    description=f"Bagian dari perbandingan {name}",
                    tags=["comparison", model_type]
                )
                
                # Jalankan eksperimen
                result = self.experiment_runner.run_experiment(
                    experiment=experiment,
                    dataset_path=dataset_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    model_type=model_type,
                    **kwargs
                )
                
                # Simpan hasil
                results[model_type] = result
                
                # Tambahkan ke comparison data
                comparison_row = self._extract_comparison_data(result, model_type)
                comparison_data.append(comparison_row)
                
                # Update progress
                pbar.update(1)
                pbar.set_description(f"Model {model_type} selesai")
        
        return results, comparison_data
    
    def _extract_comparison_data(
        self,
        result: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """
        Ekstrak data untuk perbandingan dari hasil eksperimen.
        
        Args:
            result: Hasil eksperimen
            model_type: Tipe model
            
        Returns:
            Dictionary data perbandingan
        """
        if result["status"] == "success":
            return {
                "Model": model_type,
                "Final Loss": result["training"].get("final_loss", 0),
                "mAP": result["evaluation"].get("mAP", 0),
                "Precision": result["evaluation"].get("precision", 0),
                "Recall": result["evaluation"].get("recall", 0),
                "F1-Score": result["evaluation"].get("f1", 0),
                "Inference Time (ms)": result["evaluation"].get("inference_time", 0)
            }
        else:
            return {
                "Model": model_type,
                "Status": "Error",
                "Error": result.get("error", "Unknown error")
            }
    
    def _process_comparison_results(
        self,
        results: Dict[str, Dict],
        comparison_data: List[Dict],
        group_dir: Path
    ) -> pd.DataFrame:
        """
        Proses dan simpan hasil perbandingan.
        
        Args:
            results: Dictionary hasil eksperimen
            comparison_data: List data perbandingan
            group_dir: Direktori grup
            
        Returns:
            DataFrame hasil perbandingan
        """
        # Buat DataFrame hasil perbandingan
        comparison_df = pd.DataFrame(comparison_data)
        
        # Simpan hasil perbandingan
        comparison_path = group_dir / "comparison_results.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Simpan dalam format JSON
        results_path = group_dir / "comparison_details.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return comparison_df
    
    def _get_best_model(self, comparison_df: pd.DataFrame) -> str:
        """
        Dapatkan model terbaik dari DataFrame perbandingan.
        
        Args:
            comparison_df: DataFrame hasil perbandingan
            
        Returns:
            Nama model terbaik
        """
        if comparison_df.empty:
            return "Unknown"
            
        # Cek jika ada kolom mAP
        if "mAP" in comparison_df.columns:
            best_idx = comparison_df["mAP"].idxmax()
            best_model = comparison_df.loc[best_idx, "Model"]
            best_map = comparison_df.loc[best_idx, "mAP"]
            return f"{best_model} (mAP={best_map:.4f})"
        
        # Fallback ke F1-Score
        elif "F1-Score" in comparison_df.columns:
            best_idx = comparison_df["F1-Score"].idxmax()
            best_model = comparison_df.loc[best_idx, "Model"]
            best_f1 = comparison_df.loc[best_idx, "F1-Score"]
            return f"{best_model} (F1={best_f1:.4f})"
            
        return "Unknown"