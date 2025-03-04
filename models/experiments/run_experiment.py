# File: scripts/run_experiments.py
# Author: Alfrida Sabar
# Deskripsi: Script untuk menjalankan eksperimen perbandingan backbone

import argparse
from typing import List

from smartcash.models.experiments.backbone_experiment import BackboneExperiment
from smartcash.utils.logger import SmartCashLogger

def run_all_experiments(
    config_path: str,
    scenarios: List[str] = ['position', 'lighting'],
    logger: SmartCashLogger = None
) -> None:
    """
    Jalankan semua eksperimen sesuai dokumen evaluasi
    
    Args:
        config_path: Path ke file konfigurasi
        scenarios: List skenario yang akan dijalankan
        logger: Logger untuk tracking progress
    """
    logger = logger or SmartCashLogger(__name__)
    logger.start("🚀 Memulai eksperimen SmartCash")
    
    # Setup eksperimen
    experiment = BackboneExperiment(
        config_path=config_path,
        experiment_name="smartcash_backbone_comparison",
        logger=logger
    )
    
    try:
        # 1. Perbandingan awal backbone
        logger.info("📊 Membandingkan karakteristik backbone...")
        experiment.compare_backbones()
        
        # 2. Skenario dengan CSPDarknet (Baseline)
        logger.info("🔬 Menjalankan skenario baseline (CSPDarknet)...")
        for scenario in scenarios:
            experiment.run_experiment_scenario(
                scenario=scenario,
                backbone_name='cspdarknet'
            )
            
        # 3. Skenario dengan EfficientNet
        logger.info("🔬 Menjalankan skenario EfficientNet...")
        for scenario in scenarios:
            experiment.run_experiment_scenario(
                scenario=scenario,
                backbone_name='efficientnet'
            )
            
        logger.success("✨ Semua eksperimen selesai!")
        
    except Exception as e:
        logger.error(f"❌ Eksperimen gagal: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jalankan eksperimen backbone SmartCash"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path ke file konfigurasi"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=['position', 'lighting'],
        help="Skenario yang akan dijalankan"
    )
    
    args = parser.parse_args()
    run_all_experiments(args.config, args.scenarios)