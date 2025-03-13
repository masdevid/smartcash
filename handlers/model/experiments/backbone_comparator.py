# File: smartcash/handlers/model/experiments/backbone_comparator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk perbandingan backbone dengan berbagai parameter

import torch
import time
from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path
import numpy as np

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager

class BackboneComparator:
    """
    Komponen khusus untuk perbandingan backbone dengan parameter yang berbeda.
    Memperluas fungsionalitas ExperimentManager dengan lebih banyak opsi perbandingan.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        experiment_manager: Optional[ExperimentManager] = None
    ):
        """
        Inisialisasi backbone comparator.
        
        Args:
            config: Konfigurasi model dan training
            logger: Custom logger (opsional)
            experiment_manager: ExperimentManager (opsional, dibuat baru jika None)
        """
        self.config = config
        self.logger = logger or SmartCashLogger("backbone_comparator")
        
        # Gunakan experiment_manager yang diberikan atau buat baru
        self._experiment_manager = experiment_manager
        
        # Setup konfigurasi eksperimen
        self.experiment_config = config.get('experiment', {})
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'runs/train')) / "experiments" / "backbone_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        if self._experiment_manager is None:
            self._experiment_manager = ExperimentManager(self.config, self.logger)
        return self._experiment_manager
    
    def compare_with_image_sizes(
        self,
        backbones: List[str],
        image_sizes: List[List[int]],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan backbone dengan ukuran gambar yang berbeda.
        
        Args:
            backbones: List backbone untuk dibandingkan
            image_sizes: List ukuran gambar (H, W) untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            test_loader: DataLoader untuk testing (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        self.logger.info(
            f"🔬 Membandingkan {len(backbones)} backbone dengan {len(image_sizes)} ukuran gambar"
        )
        
        experiment_name = kwargs.get('experiment_name', f"image_size_comparison_{int(time.time())}")
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Loop untuk setiap backbone
        for backbone in backbones:
            backbone_results = {}
            
            # Loop untuk setiap ukuran gambar
            for img_size in image_sizes:
                # Update config untuk ukuran gambar
                config_copy = self.config.copy()
                config_copy['training'] = config_copy.get('training', {}).copy()
                config_copy['training']['img_size'] = img_size
                
                # Setup nama eksperimen
                size_name = f"{img_size[0]}x{img_size[1]}"
                exp_name = f"{backbone}_{size_name}"
                
                # Jalankan eksperimen dengan backbone dan ukuran gambar ini
                try:
                    # Buat experiment manager dengan config yang diupdate
                    experiment_manager = ExperimentManager(config_copy, self.logger)
                    
                    # Jalankan eksperimen
                    result = experiment_manager._train_and_evaluate_backbone(
                        backbone_type=backbone,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        experiment_dir=experiment_dir / exp_name,
                        **kwargs
                    )
                    
                    # Simpan hasil
                    backbone_results[size_name] = result
                    self.logger.success(f"✅ Eksperimen {exp_name} selesai")
                except Exception as e:
                    self.logger.error(f"❌ Eksperimen {exp_name} gagal: {str(e)}")
                    backbone_results[size_name] = {'error': str(e)}
            
            # Simpan hasil backbone
            results[backbone] = backbone_results
        
        # Buat ringkasan
        summary = self._create_image_size_summary(results)
        
        # Gabungkan hasil
        final_results = {
            'experiment_name': experiment_name,
            'experiment_dir': str(experiment_dir),
            'backbones': backbones,
            'image_sizes': image_sizes,
            'results': results,
            'summary': summary
        }
        
        # Simpan hasil ke file
        self._save_results(final_results, experiment_dir, "image_size_comparison")
        
        return final_results
    
    def compare_with_augmentations(
        self,
        backbones: List[str],
        augmentation_types: List[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan backbone dengan strategi augmentasi yang berbeda.
        
        Args:
            backbones: List backbone untuk dibandingkan
            augmentation_types: List tipe augmentasi untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            test_loader: DataLoader untuk testing (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        self.logger.info(
            f"🔬 Membandingkan {len(backbones)} backbone dengan {len(augmentation_types)} tipe augmentasi"
        )
        
        experiment_name = kwargs.get('experiment_name', f"augmentation_comparison_{int(time.time())}")
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Loop untuk setiap backbone
        for backbone in backbones:
            backbone_results = {}
            
            # Loop untuk setiap tipe augmentasi
            for aug_type in augmentation_types:
                # Update config untuk tipe augmentasi
                config_copy = self.config.copy()
                config_copy['augmentation'] = config_copy.get('augmentation', {}).copy()
                config_copy['augmentation']['types'] = [aug_type]
                
                # Setup nama eksperimen
                exp_name = f"{backbone}_{aug_type}"
                
                # Jalankan eksperimen dengan backbone dan tipe augmentasi ini
                try:
                    # Buat experiment manager dengan config yang diupdate
                    experiment_manager = ExperimentManager(config_copy, self.logger)
                    
                    # Jalankan eksperimen
                    result = experiment_manager._train_and_evaluate_backbone(
                        backbone_type=backbone,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        experiment_dir=experiment_dir / exp_name,
                        **kwargs
                    )
                    
                    # Simpan hasil
                    backbone_results[aug_type] = result
                    self.logger.success(f"✅ Eksperimen {exp_name} selesai")
                except Exception as e:
                    self.logger.error(f"❌ Eksperimen {exp_name} gagal: {str(e)}")
                    backbone_results[aug_type] = {'error': str(e)}
            
            # Simpan hasil backbone
            results[backbone] = backbone_results
        
        # Buat ringkasan
        summary = self._create_augmentation_summary(results)
        
        # Gabungkan hasil
        final_results = {
            'experiment_name': experiment_name,
            'experiment_dir': str(experiment_dir),
            'backbones': backbones,
            'augmentation_types': augmentation_types,
            'results': results,
            'summary': summary
        }
        
        # Simpan hasil ke file
        self._save_results(final_results, experiment_dir, "augmentation_comparison")
        
        return final_results
    
    def _create_image_size_summary(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Buat ringkasan dasar hasil perbandingan ukuran gambar."""
        summary = {
            'best_combinations': {},
            'size_impact': {}
        }
        
        # Identifikasi metrik-metrik yang ada
        metrics_to_analyze = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
        metrics_data = {metric: {} for metric in metrics_to_analyze}
        
        # Ekstrak metrik untuk analisis
        for backbone, sizes in results.items():
            for metric in metrics_to_analyze:
                if backbone not in metrics_data[metric]:
                    metrics_data[metric][backbone] = {}
                
                for size_name, result in sizes.items():
                    if 'error' not in result and 'evaluation' in result:
                        if metric in result['evaluation']:
                            metrics_data[metric][backbone][size_name] = result['evaluation'][metric]
        
        # Temukan kombinasi terbaik
        for metric, backbone_data in metrics_data.items():
            best_combination = None
            best_value = 0 if metric != 'inference_time' else float('inf')
            
            for backbone, sizes in backbone_data.items():
                for size_name, value in sizes.items():
                    if metric == 'inference_time':
                        if value < best_value:
                            best_value = value
                            best_combination = (backbone, size_name)
                    else:
                        if value > best_value:
                            best_value = value
                            best_combination = (backbone, size_name)
            
            if best_combination:
                summary['best_combinations'][metric] = {
                    'backbone': best_combination[0],
                    'image_size': best_combination[1],
                    'value': best_value
                }
        
        return summary
    
    def _create_augmentation_summary(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Buat ringkasan dasar hasil perbandingan tipe augmentasi."""
        summary = {
            'best_combinations': {},
            'augmentation_impact': {}
        }
        
        # Identifikasi metrik-metrik yang ada
        metrics_to_analyze = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
        metrics_data = {metric: {} for metric in metrics_to_analyze}
        
        # Ekstrak metrik untuk analisis
        for backbone, augs in results.items():
            for metric in metrics_to_analyze:
                if backbone not in metrics_data[metric]:
                    metrics_data[metric][backbone] = {}
                
                for aug_type, result in augs.items():
                    if 'error' not in result and 'evaluation' in result:
                        if metric in result['evaluation']:
                            metrics_data[metric][backbone][aug_type] = result['evaluation'][metric]
        
        # Temukan kombinasi terbaik
        for metric, backbone_data in metrics_data.items():
            best_combination = None
            best_value = 0 if metric != 'inference_time' else float('inf')
            
            for backbone, augs in backbone_data.items():
                for aug_type, value in augs.items():
                    if metric == 'inference_time':
                        if value < best_value:
                            best_value = value
                            best_combination = (backbone, aug_type)
                    else:
                        if value > best_value:
                            best_value = value
                            best_combination = (backbone, aug_type)
            
            if best_combination:
                summary['best_combinations'][metric] = {
                    'backbone': best_combination[0],
                    'augmentation': best_combination[1],
                    'value': best_value
                }
                
        return summary
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path, result_type: str) -> None:
        """Simpan hasil eksperimen ke file."""
        try:
            # Simpan hasil ke JSON
            results_path = output_dir / f"{result_type}_results.json"
            
            # Konversi nilai yang tidak JSON-serializable
            import json
            
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
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
            
            # Konversi semua nilai
            serializable_results = convert_to_serializable(results)
            
            # Simpan ke file
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            self.logger.info(f"💾 Hasil eksperimen disimpan ke: {results_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil: {str(e)}")