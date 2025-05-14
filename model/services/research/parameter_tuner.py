"""
File: smartcash/model/services/research/parameter_tuner.py
Deskripsi: Komponen untuk melakukan tuning parameter model
"""

import itertools
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger


class ParameterTuner:
    """
    Komponen untuk melakukan tuning parameter model.
    
    Bertanggung jawab untuk:
    - Melakukan grid search atau random search untuk parameter
    - Menjalankan eksperimen dengan berbagai kombinasi parameter
    - Menganalisis hasil tuning dan menemukan parameter optimal
    """
    
    def __init__(
        self,
        base_dir: str,
        experiment_runner: Any,
        experiment_creator: Any,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi parameter tuner.
        
        Args:
            base_dir: Direktori dasar untuk menyimpan hasil eksperimen
            experiment_runner: Komponen experiment runner
            experiment_creator: Komponen experiment creator
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.base_dir = Path(base_dir)
        self.logger = logger or get_logger("parameter_tuner")
        self.experiment_runner = experiment_runner
        self.experiment_creator = experiment_creator
        
        self.logger.debug(f"🔧 ParameterTuner diinisialisasi")
    
    def run_parameter_tuning(
        self,
        name: str,
        dataset_path: str,
        model_type: str,
        param_grid: Dict[str, List],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan eksperimen tuning parameter.
        
        Args:
            name: Nama eksperimen
            dataset_path: Path ke dataset
            model_type: Tipe model yang akan dituning
            param_grid: Grid parameter untuk tuning
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil tuning
        """
        self.logger.info(f"🔧 Memulai parameter tuning: {name}")
        self.logger.info(f"📋 Parameter yang akan dituning: {param_grid}")
        
        # Setup grup eksperimen
        group_info = self.experiment_creator.create_experiment_group(name, "tuning")
        group_id, group_dir = group_info["id"], group_info["dir"]
        
        # Generate kombinasi parameter
        param_combinations = self._generate_param_combinations(param_grid)
        self.logger.info(f"🧮 Total kombinasi parameter: {len(param_combinations)}")
        
        # Jalankan eksperimen untuk setiap kombinasi parameter
        results, tuning_data = self._run_parameter_tuning_experiments(
            name, dataset_path, model_type, param_combinations, **kwargs
        )
        
        # Proses dan simpan hasil
        tuning_df = self._process_tuning_results(results, tuning_data, group_dir)
        
        # Identifikasi parameter terbaik
        best_params = self._get_best_params(tuning_df)
        
        self.logger.success(
            f"✅ Parameter tuning selesai\n"
            f"   • Hasil disimpan di: {group_dir}\n"
            f"   • Parameter terbaik: {best_params}"
        )
        
        return {
            "status": "success",
            "tuning_id": group_id,
            "results": results,
            "tuning_df": tuning_df,
            "best_params": best_params
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Generate semua kombinasi parameter dari grid.
        
        Args:
            param_grid: Grid parameter
            
        Returns:
            List kombinasi parameter
        """
        # Dapatkan semua kombinasi
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        # Konversi ke list dictionary
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _run_parameter_tuning_experiments(
        self,
        name: str,
        dataset_path: str,
        model_type: str,
        param_combinations: List[Dict],
        **kwargs
    ) -> Tuple[Dict, List]:
        """
        Jalankan eksperimen untuk setiap kombinasi parameter.
        
        Args:
            name: Nama eksperimen
            dataset_path: Path ke dataset
            model_type: Tipe model
            param_combinations: List kombinasi parameter
            **kwargs: Parameter tambahan
            
        Returns:
            Tuple (results, tuning_data)
        """
        # Dictionary untuk menyimpan hasil
        results = {}
        tuning_data = []
        
        # Progress bar
        with tqdm(total=len(param_combinations), desc="Tuning parameter") as pbar:
            # Jalankan eksperimen untuk setiap kombinasi parameter
            for i, params in enumerate(param_combinations):
                # Buat nama yang sesuai dengan parameter
                param_str = "_".join([f"{k}={v}" for k, v in params.items()])
                exp_name = f"{name}_params_{i}"
                
                # Buat sub-eksperimen
                experiment = self.experiment_creator.create_experiment(
                    name=exp_name,
                    description=f"Parameter tuning: {param_str}",
                    tags=["tuning", model_type]
                )
                
                # Jalankan eksperimen dengan parameter ini
                result = self.experiment_runner.run_experiment(
                    experiment=experiment,
                    dataset_path=dataset_path,
                    model_type=model_type,
                    **params,
                    **kwargs
                )
                
                # Simpan hasil
                param_key = str(params)
                results[param_key] = result
                
                # Tambahkan ke tuning data
                tuning_row = self._extract_tuning_data(result, params)
                tuning_data.append(tuning_row)
                
                # Update progress
                pbar.update(1)
                pbar.set_description(f"Kombinasi {i+1}/{len(param_combinations)}")
        
        return results, tuning_data
    
    def _extract_tuning_data(
        self,
        result: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ekstrak data untuk tuning dari hasil eksperimen.
        
        Args:
            result: Hasil eksperimen
            params: Parameter eksperimen
            
        Returns:
            Dictionary data tuning
        """
        if result["status"] == "success":
            return {
                **params,
                "Final Loss": result["training"].get("final_loss", 0),
                "mAP": result["evaluation"].get("mAP", 0),
                "Precision": result["evaluation"].get("precision", 0),
                "Recall": result["evaluation"].get("recall", 0),
                "F1-Score": result["evaluation"].get("f1", 0),
                "Inference Time (ms)": result["evaluation"].get("inference_time", 0)
            }
        else:
            return {
                **params,
                "Status": "Error",
                "Error": result.get("error", "Unknown error")
            }
    
    def _process_tuning_results(
        self,
        results: Dict[str, Dict],
        tuning_data: List[Dict],
        group_dir: Path
    ) -> pd.DataFrame:
        """
        Proses dan simpan hasil tuning.
        
        Args:
            results: Dictionary hasil eksperimen
            tuning_data: List data tuning
            group_dir: Direktori grup
            
        Returns:
            DataFrame hasil tuning
        """
        # Buat DataFrame hasil tuning
        tuning_df = pd.DataFrame(tuning_data)
        
        # Simpan hasil tuning
        tuning_path = group_dir / "tuning_results.csv"
        tuning_df.to_csv(tuning_path, index=False)
        
        # Simpan dalam format JSON
        results_path = group_dir / "tuning_details.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return tuning_df
    
    def _get_best_params(self, tuning_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dapatkan parameter terbaik dari DataFrame hasil tuning.
        
        Args:
            tuning_df: DataFrame hasil tuning
            
        Returns:
            Dictionary parameter terbaik
        """
        if tuning_df.empty:
            return {}
            
        # Cek jika ada kolom mAP
        if "mAP" in tuning_df.columns:
            best_idx = tuning_df["mAP"].idxmax()
        # Fallback ke F1-Score
        elif "F1-Score" in tuning_df.columns:
            best_idx = tuning_df["F1-Score"].idxmax()
        else:
            return {}
        
        # Ekstrak parameter
        param_columns = [col for col in tuning_df.columns if col not in [
            "Final Loss", "mAP", "Precision", "Recall", "F1-Score",
            "Inference Time (ms)", "Status", "Error"
        ]]
        
        return {param: tuning_df.loc[best_idx, param] for param in param_columns}