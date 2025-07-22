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
    """Evaluation configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    
    Handles configuration for 2×4 evaluation matrix (2 scenarios × 4 models = 8 tests)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """Initialize evaluation config handler.
        
        Args:
            config: Optional configuration dictionary to use instead of loading from file
            logger: Optional logger instance
        """
        # Initialize pure delegation pattern (no mixin inheritance)
        self.logger = logger or get_module_logger('smartcash.ui.model.evaluation.config')
        self.module_name = 'evaluation'
        self.parent_module = 'model'
        
        # Initialize with defaults
        self._default_config = get_default_evaluation_config()
        self._config = self._default_config.copy()
        
        # Update with provided config if any
        if config:
            self._config.update(config)
        
        # Set up config file path
        self._config_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'configs',
            'evaluation_config.yaml'
        )
        
        self.logger.info("✅ Evaluation config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.validate_config(updates):
            self._config.update(updates)
            self.logger.debug(f"Configuration updated: {list(updates.keys())}")
        else:
            raise ValueError("Invalid configuration updates provided")

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_default_evaluation_config().copy()
        self.logger.info("Configuration reset to defaults")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            file_path = config_path or self._config_file_path
            with open(file_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default evaluation configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_evaluation_config()

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate evaluation configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Use the existing validation function
            return validate_evaluation_config(config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def get_scenarios_config(self) -> Dict[str, Any]:
        """
        Get scenarios configuration.
        
        Returns:
            Scenarios configuration dictionary
        """
        config = self.get_current_config()
        return config.get('evaluation', {}).get('scenarios', RESEARCH_SCENARIOS)
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """
        Get metrics configuration.
        
        Returns:
            Metrics configuration dictionary
        """
        config = self.get_current_config()
        return config.get('evaluation', {}).get('metrics', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """
        Get models configuration.
        
        Returns:
            Models configuration dictionary
        """
        config = self.get_current_config()
        return config.get('evaluation', {}).get('models', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get inference configuration.
        
        Returns:
            Inference configuration dictionary
        """
        config = self.get_current_config()
        return config.get('inference', {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """
        Get execution configuration.
        
        Returns:
            Execution configuration dictionary
        """
        config = self.get_current_config()
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
            return self.get_current_config()
    
    def validate_current_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            current_config = self.get_current_config()
            return validate_evaluation_config(current_config)
            
        except Exception as e:
            self.logger.error(f"Failed to validate config: {e}")
            return False
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for evaluation module."""
        from .evaluation_defaults import get_default_evaluation_config
        return get_default_evaluation_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'EvaluationConfigHandler':
        """Create config handler instance for evaluation module."""
        return EvaluationConfigHandler(config)
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components containing form data
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            # Initialize with current config as base
            extracted_config = self.get_current_config()
            
            # Extract evaluation settings from UI components
            if ui_components and 'widgets' in ui_components:
                widgets = ui_components['widgets']
                
                # Extract scenario settings
                if 'scenario_selector' in widgets:
                    scenario_widget = widgets['scenario_selector']
                    if hasattr(scenario_widget, 'value'):
                        selected_scenarios = scenario_widget.value if isinstance(scenario_widget.value, list) else [scenario_widget.value]
                        
                        # Update scenario configuration
                        for scenario in RESEARCH_SCENARIOS:
                            extracted_config.setdefault('evaluation', {}).setdefault('scenarios', {})[scenario] = {
                                'enabled': scenario in selected_scenarios
                            }
                
                # Extract metric settings
                if 'metrics_selector' in widgets:
                    metrics_widget = widgets['metrics_selector']
                    if hasattr(metrics_widget, 'value'):
                        selected_metrics = metrics_widget.value if isinstance(metrics_widget.value, list) else [metrics_widget.value]
                        
                        # Update metrics configuration
                        default_metrics = ['map', 'precision_recall', 'f1_score', 'accuracy', 'inference_time']
                        for metric in default_metrics:
                            extracted_config.setdefault('evaluation', {}).setdefault('metrics', {})[metric] = {
                                'enabled': metric in selected_metrics
                            }
                
                # Extract execution mode
                if 'execution_mode' in widgets:
                    mode_widget = widgets['execution_mode']
                    if hasattr(mode_widget, 'value'):
                        extracted_config.setdefault('evaluation', {}).setdefault('execution', {})['run_mode'] = mode_widget.value
                
                # Extract model selection
                if 'model_selector' in widgets:
                    model_widget = widgets['model_selector']
                    if hasattr(model_widget, 'value'):
                        extracted_config.setdefault('evaluation', {}).setdefault('models', {})['selected'] = model_widget.value
                
                # Extract inference settings
                if 'confidence_threshold' in widgets:
                    conf_widget = widgets['confidence_threshold']
                    if hasattr(conf_widget, 'value'):
                        extracted_config.setdefault('inference', {})['confidence_threshold'] = float(conf_widget.value)
                
                if 'iou_threshold' in widgets:
                    iou_widget = widgets['iou_threshold']
                    if hasattr(iou_widget, 'value'):
                        extracted_config.setdefault('inference', {})['iou_threshold'] = float(iou_widget.value)
                
                # Extract output settings
                if 'save_results' in widgets:
                    save_widget = widgets['save_results']
                    if hasattr(save_widget, 'value'):
                        extracted_config.setdefault('evaluation', {}).setdefault('output', {})['save_results'] = bool(save_widget.value)
                
                if 'output_format' in widgets:
                    format_widget = widgets['output_format']
                    if hasattr(format_widget, 'value'):
                        extracted_config.setdefault('evaluation', {}).setdefault('output', {})['format'] = format_widget.value
            
            self.logger.debug("✅ Configuration extracted from UI components")
            return extracted_config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract config from UI: {e}")
            # Return current config as fallback
            return self.get_current_config()