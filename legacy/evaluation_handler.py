# File: smartcash/handlers/evaluation_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk evaluasi model dengan pendekatan modular

import os
import yaml
from typing import Dict, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.evaluator import Evaluator
from smartcash.handlers.research_scenario_handler import ResearchScenarioHandler


class EvaluationHandler:
    """
    Main handler for model evaluation with support for various scenarios.
    
    Features:
    - Regular model evaluation
    - Research scenario evaluation
    - Flexible configuration management
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        config_path: str = 'configs/base_config.yaml',
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize EvaluationHandler.
        
        Args:
            config: Custom configuration (optional)
            config_path: Path to config file (optional)
            logger: Custom logger (optional)
        """
        # Setup logger
        self.logger = logger or SmartCashLogger(__name__)
        
        # Load base configuration
        self.config = self._load_config(config_path)
        
        # Update with custom config if provided
        if config:
            self._update_config_dict(self.config, config)
        
        # Initialize evaluators
        try:
            self.base_evaluator = Evaluator(
                config=self.config, 
                logger=self.logger
            )
            
            self.research_evaluator = ResearchScenarioHandler(
                config=self.config, 
                logger=self.logger
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize evaluators: {str(e)}")
            raise
    
    def evaluate(
        self, 
        eval_type: str = 'regular',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model based on evaluation type.
        
        Args:
            eval_type: Type of evaluation ('regular' or 'research')
            **kwargs: Additional arguments for customizing evaluation
        
        Returns:
            Dict containing evaluation results
        """
        try:
            # Update configuration with additional arguments
            if kwargs:
                for key, value in kwargs.items():
                    self._set_nested_value(self.config, key, value)
            
            # Choose evaluator based on type
            if eval_type == 'regular':
                return self._evaluate_regular(**kwargs)
            elif eval_type == 'research':
                return self._evaluate_research()
            else:
                raise ValueError(f"Invalid evaluation type: {eval_type}")
        
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {str(e)}")
            raise
    
    def _evaluate_regular(self, **kwargs) -> Dict:
        """
        Regular model evaluation with latest checkpoint.
        
        Returns:
            Dict containing evaluation results
        """
        self.logger.info("🔍 Starting regular evaluation...")
        
        # Find latest model checkpoint
        checkpoints_dir = Path(self.config.get('checkpoints_dir', 'checkpoints'))
        
        # Checkpoint search patterns
        checkpoint_patterns = [
            '*_best.pth',    # Best checkpoint
            '*_latest.pth',  # Latest checkpoint
            '*_epoch_*.pth'  # Specific epoch checkpoint
        ]
        
        # Find valid checkpoint
        latest_checkpoint = None
        for pattern in checkpoint_patterns:
            matches = list(checkpoints_dir.glob(pattern))
            if matches:
                latest_checkpoint = max(matches, key=os.path.getmtime)
                break
        
        if not latest_checkpoint:
            raise FileNotFoundError("❌ No model checkpoint found")
        
        # Get test dataset path
        test_data_path = self.config.get('test_data_path', 'data/test')
        
        # Run evaluation
        return self.base_evaluator.evaluate_model(
            model_path=str(latest_checkpoint),
            dataset_path=test_data_path,
            **kwargs
        )
    
    def _evaluate_research(self) -> Dict:
        """
        Research scenario evaluation.
        
        Returns:
            Dict containing research evaluation results
        """
        self.logger.info("🔬 Starting research scenario evaluation...")
        
        results = {
            'research_results': self.research_evaluator.run_all_scenarios()
        }
        
        return results
    
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
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {str(e)}")
            raise
    
    def _update_config_dict(self, base: Dict, update: Dict) -> None:
        """Update configuration dictionary recursively."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_config_dict(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, d: Dict, key: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def get_config(self) -> Dict:
        """
        Dapatkan konfigurasi saat ini.
        
        Returns:
            Dictionary konfigurasi
        """
        return self.config