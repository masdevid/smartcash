"""
Simplified Evaluation Config Handler - No Shared Config Dependency
Handles configuration for 2×4 evaluation matrix (2 scenarios × 4 models = 8 tests)
"""

import os
import yaml
from typing import Dict, Any, Optional
from smartcash.ui.logger import get_module_logger
from smartcash.ui.model.evaluation.configs.evaluation_defaults import (
    get_default_evaluation_config,
    validate_evaluation_config
)
from smartcash.ui.model.evaluation.constants import (
    RESEARCH_SCENARIOS,
    MODEL_COMBINATIONS,
    EVALUATION_MATRIX
)

class EvaluationConfigHandler:
    """Configuration handler for evaluation module."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation config handler.
        
        Args:
            config: Optional configuration dictionary to use instead of loading from file
        """
        self.logger = get_module_logger("smartcash.ui.model.evaluation.configs")
        
        self._config_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'configs',
            'evaluation_config.yaml'
        )
        
        self._default_config = get_default_evaluation_config()
        self._loaded_config = config  # Store the provided config
        
        # If no config was provided, try to load from file
        if self._loaded_config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load evaluation configuration from YAML file.
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            config_file = config_path or self._config_file_path
            
            if not os.path.exists(config_file):
                self.logger.warning(f"Config file not found: {config_file}, using defaults")
                self._loaded_config = self._default_config.copy()
                return self._loaded_config
            
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Merge YAML config with defaults
            merged_config = self._merge_configs(self._default_config, yaml_config)
            
            # Validate merged configuration
            if not validate_evaluation_config(merged_config):
                self.logger.warning("Invalid configuration, using defaults")
                self._loaded_config = self._default_config.copy()
                return self._loaded_config
            
            self._loaded_config = merged_config
            self.logger.info(f"Evaluation config loaded from {config_file}")
            
            return self._loaded_config
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation config: {e}")
            self._loaded_config = self._default_config.copy()
            return self._loaded_config
    
    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
        """
        Save evaluation configuration to YAML file.
        
        Args:
            config: Configuration to save
            config_path: Optional path to save config
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_file = config_path or self._config_file_path
            
            # Validate configuration before saving
            if not validate_evaluation_config(config):
                self.logger.error("Invalid configuration, cannot save")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self._loaded_config = config.copy()
            self.logger.info(f"Evaluation config saved to {config_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation config: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        if self._loaded_config is None:
            return self.load_config()
        return self._loaded_config.copy()
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Args:
            updates: Configuration updates
            
        Returns:
            Updated configuration dictionary
        """
        try:
            current_config = self.get_config()
            updated_config = self._merge_configs(current_config, updates)
            
            # Validate updated configuration
            if not validate_evaluation_config(updated_config):
                self.logger.error("Invalid configuration update")
                return current_config
            
            self._loaded_config = updated_config
            self.logger.info("Evaluation configuration updated")
            
            return updated_config
            
        except Exception as e:
            self.logger.error(f"Failed to update evaluation config: {e}")
            return self.get_config()
    
    def get_scenarios_config(self) -> Dict[str, Any]:
        """
        Get scenarios configuration.
        
        Returns:
            Scenarios configuration dictionary
        """
        config = self.get_config()
        return config.get('evaluation', {}).get('scenarios', RESEARCH_SCENARIOS)
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """
        Get metrics configuration.
        
        Returns:
            Metrics configuration dictionary
        """
        config = self.get_config()
        return config.get('evaluation', {}).get('metrics', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """
        Get models configuration.
        
        Returns:
            Models configuration dictionary
        """
        config = self.get_config()
        return config.get('evaluation', {}).get('models', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get inference configuration.
        
        Returns:
            Inference configuration dictionary
        """
        config = self.get_config()
        return config.get('inference', {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """
        Get execution configuration.
        
        Returns:
            Execution configuration dictionary
        """
        config = self.get_config()
        return config.get('evaluation', {}).get('execution', {})
    
    def set_execution_mode(self, mode: str) -> bool:
        """
        Set evaluation execution mode.
        
        Args:
            mode: Execution mode ('all_scenarios', 'position_only', 'lighting_only')
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            valid_modes = ['all_scenarios', 'position_only', 'lighting_only']
            if mode not in valid_modes:
                self.logger.error(f"Invalid execution mode: {mode}")
                return False
            
            updates = {
                'evaluation': {
                    'execution': {
                        'run_mode': mode
                    }
                }
            }
            
            self.update_config(updates)
            self.logger.info(f"Execution mode set to: {mode}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set execution mode: {e}")
            return False
    
    def enable_scenario(self, scenario: str, enabled: bool = True) -> bool:
        """
        Enable or disable a specific scenario.
        
        Args:
            scenario: Scenario name
            enabled: Whether to enable the scenario
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if scenario not in RESEARCH_SCENARIOS:
                self.logger.error(f"Unknown scenario: {scenario}")
                return False
            
            updates = {
                'evaluation': {
                    'scenarios': {
                        scenario: {
                            'enabled': enabled
                        }
                    }
                }
            }
            
            self.update_config(updates)
            self.logger.info(f"Scenario {scenario} {'enabled' if enabled else 'disabled'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update scenario {scenario}: {e}")
            return False
    
    def enable_metric(self, metric: str, enabled: bool = True) -> bool:
        """
        Enable or disable a specific metric.
        
        Args:
            metric: Metric name
            enabled: Whether to enable the metric
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            updates = {
                'evaluation': {
                    'metrics': {
                        metric: {
                            'enabled': enabled
                        }
                    }
                }
            }
            
            self.update_config(updates)
            self.logger.info(f"Metric {metric} {'enabled' if enabled else 'disabled'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metric {metric}: {e}")
            return False
    
    def get_enabled_scenarios(self) -> list:
        """
        Get list of enabled scenarios.
        
        Returns:
            List of enabled scenario names
        """
        try:
            scenarios_config = self.get_scenarios_config()
            enabled_scenarios = []
            
            for scenario_name, scenario_config in scenarios_config.items():
                if scenario_config.get('enabled', True):
                    enabled_scenarios.append(scenario_name)
            
            return enabled_scenarios
            
        except Exception as e:
            self.logger.error(f"Failed to get enabled scenarios: {e}")
            return list(RESEARCH_SCENARIOS.keys())
    
    def get_enabled_metrics(self) -> list:
        """
        Get list of enabled metrics.
        
        Returns:
            List of enabled metric names
        """
        try:
            metrics_config = self.get_metrics_config()
            enabled_metrics = []
            
            for metric_name, metric_config in metrics_config.items():
                if metric_config.get('enabled', True):
                    enabled_metrics.append(metric_name)
            
            return enabled_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get enabled metrics: {e}")
            return ['map', 'precision_recall', 'f1_score', 'accuracy', 'inference_time']
    
    def get_evaluation_matrix(self) -> list:
        """
        Get evaluation matrix for enabled scenarios and models.
        
        Returns:
            List of evaluation test configurations
        """
        try:
            enabled_scenarios = self.get_enabled_scenarios()
            
            # Filter evaluation matrix by enabled scenarios
            filtered_matrix = []
            for test_config in EVALUATION_MATRIX:
                if test_config['scenario'] in enabled_scenarios:
                    filtered_matrix.append(test_config)
            
            return filtered_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to get evaluation matrix: {e}")
            return EVALUATION_MATRIX.copy()
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        try:
            merged = base_config.copy()
            
            for key, value in override_config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Failed to merge configs: {e}")
            return base_config.copy()
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Default configuration dictionary
        """
        try:
            self._loaded_config = self._default_config.copy()
            self.logger.info("Configuration reset to defaults")
            
            return self._loaded_config
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return self.get_config()
    
    def validate_current_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            current_config = self.get_config()
            return validate_evaluation_config(current_config)
            
        except Exception as e:
            self.logger.error(f"Failed to validate config: {e}")
            return False