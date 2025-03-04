# File: handlers/evaluation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk evaluasi model dengan pendekatan modular

import os
from typing import Dict, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.base_evaluation_handler import BaseEvaluationHandler
from smartcash.handlers.research_scenario_handler import ResearchScenarioHandler
from smartcash.cli.configuration_manager import ConfigurationManager

class EvaluationHandler:
    """
    Handler utama untuk evaluasi model dengan dukungan berbagai skenario.
    
    Fitur:
    - Evaluasi model reguler
    - Evaluasi skenario penelitian
    - Manajemen konfigurasi fleksibel
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi EvaluationHandler.
        
        Args:
            config: Konfigurasi kustom (opsional)
            logger: Logger kustom (opsional)
        """
        # Setup logger
        self.logger = logger or SmartCashLogger(__name__)
        
        # Gunakan ConfigurationManager untuk mengelola konfigurasi
        self.config_manager = ConfigurationManager(
            base_config_path='configs/base_config.yaml'
        )
        
        # Update konfigurasi jika ada
        if config:
            for key, value in config.items():
                self.config_manager.update(key, value)
        
        # Ambil konfigurasi yang sudah diproses
        self.config = self.config_manager.current_config
        
        # Inisialisasi handler evaluasi
        try:
            self.base_evaluator = BaseEvaluationHandler(
                config=self.config, 
                logger=self.logger
            )
            
            self.research_evaluator = ResearchScenarioHandler(
                config=self.config, 
                logger=self.logger
            )
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi evaluator: {str(e)}")
            raise
    
    def evaluate(
        self, 
        eval_type: str = 'regular', 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluasi model berdasarkan tipe evaluasi.
        
        Args:
            eval_type: Tipe evaluasi ('regular' atau 'research')
            **kwargs: Argumen tambahan untuk kustomisasi evaluasi
        
        Returns:
            Dict berisi hasil evaluasi
        """
        try:
            # Update konfigurasi dengan argumen tambahan
            if kwargs:
                for key, value in kwargs.items():
                    self.config_manager.update(key, value)
            
            # Pilih evaluator berdasarkan tipe
            if eval_type == 'regular':
                return self._evaluate_regular()
            elif eval_type == 'research':
                return self._evaluate_research()
            else:
                raise ValueError(f"Tipe evaluasi tidak valid: {eval_type}")
        
        except Exception as e:
            self.logger.error(f"❌ Evaluasi gagal: {str(e)}")
            raise
    
    def _evaluate_regular(self) -> Dict:
        """
        Evaluasi model regular dengan checkpoint terbaru.
        
        Returns:
            Dict berisi hasil evaluasi
        """
        self.logger.info("🔍 Memulai evaluasi reguler...")
        
        # Cari checkpoint model terbaru
        checkpoints_dir = Path(self.config.get('checkpoints_dir', 'checkpoints'))
        
        # Pola pencarian checkpoint
        checkpoint_patterns = [
            '*_best.pth',    # Checkpoint terbaik
            '*_latest.pth',  # Checkpoint terakhir
            '*_epoch_*.pth'  # Checkpoint epoch tertentu
        ]
        
        # Cari checkpoint yang valid
        latest_checkpoint = None
        for pattern in checkpoint_patterns:
            matches = list(checkpoints_dir.glob(pattern))
            if matches:
                latest_checkpoint = max(matches, key=os.path.getmtime)
                break
        
        if not latest_checkpoint:
            raise FileNotFoundError("❌ Tidak ada checkpoint model yang ditemukan")
        
        # Path dataset testing
        test_data_path = self.config.get('test_data_path', 'data/test')
        
        # Lakukan evaluasi
        return self.base_evaluator.evaluate_model(
            model_path=str(latest_checkpoint),
            dataset_path=test_data_path
        )
    
    def _evaluate_research(self) -> Dict:
        """
        Evaluasi model untuk skenario penelitian.
        
        Returns:
            Dict berisi hasil evaluasi penelitian
        """
        self.logger.info("🔬 Memulai evaluasi skenario penelitian...")
        
        return self.research_evaluator.run_all_scenarios()
    
    def list_checkpoints(self) -> Dict[str, Path]:
        """
        Dapatkan daftar checkpoint model yang tersedia.
        
        Returns:
            Dict berisi path checkpoint yang tersedia
        """
        try:
            checkpoints_dir = Path(self.config.get('checkpoints_dir', 'checkpoints'))
            
            # Pola pencarian checkpoint
            checkpoint_patterns = {
                'terbaik': '*_best.pth',
                'terakhir': '*_latest.pth',
                'epoch': '*_epoch_*.pth'
            }
            
            available_checkpoints = {}
            for label, pattern in checkpoint_patterns.items():
                matches = list(checkpoints_dir.glob(pattern))
                if matches:
                    # Ambil checkpoint terbaru untuk setiap tipe
                    latest_checkpoint = max(matches, key=os.path.getmtime)
                    available_checkpoints[label] = latest_checkpoint
            
            return available_checkpoints
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menemukan checkpoints: {str(e)}")
            return {}