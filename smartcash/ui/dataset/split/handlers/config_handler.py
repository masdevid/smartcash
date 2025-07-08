"""
File: smartcash/ui/dataset/split/handlers/config_handler.py

Configuration handler for dataset split operations.

This module handles loading, validating, and saving configuration
for dataset splitting operations.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG, VALIDATION_RULES
from smartcash.ui.core.errors.handlers import handle_ui_errors

class SplitConfigHandler(ConfigHandler):
    """Handler for dataset split configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the configuration handler.
        
        Args:
            config: Optional initial configuration
            **kwargs: Additional keyword arguments for parent class
                - module_name: Name of the module (default: 'split')
                - default_config: Default configuration (default: DEFAULT_SPLIT_CONFIG)
        """
        # Store validation rules before passing to parent
        self._validation_rules = kwargs.pop('validation_rules', VALIDATION_RULES)
        
        # Extract module_name if provided in kwargs, otherwise use default
        module_name = kwargs.pop('module_name', 'split')
        
        # Initialize parent with remaining kwargs
        super().__init__(
            module_name=module_name,
            default_config=kwargs.pop('default_config', DEFAULT_SPLIT_CONFIG),
            **kwargs
        )
        
        # Initialize with empty config
        self._config = {}
        
        # Load any provided config or use default
        self.load_config(config if config is not None else DEFAULT_SPLIT_CONFIG)
            
    def load_config(self, config: Dict[str, Any]) -> None:
        """Load and validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to load
            
        Raises:
            ValueError: If the configuration is invalid
        """
        self.logger.debug(f"Loading config. Type: {type(config)}, Value: {config}")
        
        if not isinstance(config, dict):
            self.logger.error(f"Configuration must be a dictionary, got {type(config)}: {config}")
            raise ValueError("Configuration must be a dictionary")
            
        # Create a copy of the config with default values
        import copy
        loaded_config = copy.deepcopy(DEFAULT_SPLIT_CONFIG)
        
        # Update with provided values
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d
            
        loaded_config = update_dict(loaded_config, config)
        
        # Validate the config before setting it
        if not self.validate_config(loaded_config):
            raise ValueError("Invalid configuration provided")
            
        self._config = loaded_config
        self.logger.debug("Configuration loaded successfully")
        
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update the current configuration with new values.
        
        Args:
            config_updates: Dictionary containing configuration updates
            
        Raises:
            ValueError: If the updates would result in an invalid configuration
        """
        if not config_updates:
            return
            
        # Create a copy of current config with updates applied
        updated_config = self.config.copy()
        
        # Apply updates
        for key, value in config_updates.items():
            # Handle nested dictionaries
            if '.' in key:
                parts = key.split('.')
                current = updated_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                updated_config[key] = value
                
        # Validate the updated config
        if not self.validate_config(updated_config):
            raise ValueError("Configuration update would result in invalid configuration")
            
        # If we got here, the config is valid, so apply it
        self._config = updated_config
        self.logger.debug("Configuration updated successfully")
        
    def initialize(self, **kwargs) -> Dict[str, Any]:
        """Initialize the handler.
        
        Returns:
            Dict containing initialization status
        """
        try:
            self._is_initialized = True
            return {'status': 'success', 'message': 'SplitConfigHandler initialized'}
        except Exception as e:
            self.handle_error(f"Failed to initialize SplitConfigHandler: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if config is valid, False otherwise
            
        Raises:
            ValueError: If the configuration is invalid (specifically for invalid split ratios)
        """
        self.logger.debug(f"Validating config: {config}")
        
        # Check required fields
        if 'data' not in config:
            self.logger.error("Missing required section: 'data'")
            return False
            
        if 'output' not in config:
            self.logger.error("Missing required section: 'output'")
            return False
            
        data = config['data']
        output = config['output']
        
        # Check required data fields
        required_data_fields = ['split_ratios', 'seed', 'shuffle', 'stratify']
        for field in required_data_fields:
            if field not in data:
                self.logger.error(f"Missing required field: data.{field}")
                return False
                
        # Check required output fields
        required_output_fields = [
            'train_dir', 'val_dir', 'test_dir', 'create_subdirs',
            'overwrite', 'relative_paths', 'preserve_dir_structure',
            'use_symlinks', 'backup', 'backup_dir'
        ]
        for field in required_output_fields:
            if field not in output:
                self.logger.error(f"Missing required field: output.{field}")
                return False
        
        # Validate split ratios
        ratios = data.get('split_ratios', {})
        if not all(k in ratios for k in ['train', 'val', 'test']):
            self.logger.error("Missing required split ratios: train, val, test")
            return False
            
        # Check ratio types and values
        for name, ratio in ratios.items():
            if not isinstance(ratio, (int, float)):
                self.logger.error(f"Split ratio {name} must be a number")
                return False
            if not (0 <= ratio <= 1):
                self.logger.error(f"Split ratio {name} must be between 0 and 1")
                return False
        
        # Check ratios sum to 1.0 (with small tolerance for floating point)
        # This is the only validation that raises ValueError
        ratios_sum = sum(ratios.values())
        self.logger.debug(f"Validating ratios sum: {ratios_sum}")
        if not (0.999 <= ratios_sum <= 1.001):
            error_msg = f"Split ratios must sum to 1.0, got {ratios_sum:.3f}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug("Ratios validation passed")
            
        # Validate seed is an integer
        if not isinstance(data['seed'], int):
            self.logger.error("Seed must be an integer")
            return False
                
        # Validate boolean fields
        for field in ['shuffle', 'stratify', 'create_subdirs', 'overwrite', 
                     'relative_paths', 'preserve_dir_structure', 'use_symlinks', 'backup']:
            if field in data and not isinstance(data[field], bool):
                self.logger.error(f"{field} must be a boolean")
                return False
            if field in output and not isinstance(output[field], bool):
                self.logger.error(f"{field} must be a boolean")
                return False
        
        # Validate output directories are strings
        for field in ['train_dir', 'val_dir', 'test_dir', 'backup_dir']:
            if field in output and not isinstance(output[field], str):
                self.logger.error(f"{field} must be a string")
                return False
        
        # If we got here, all validations passed
        return True
    
    def extract_ui_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing the extracted configuration
        """
        form_components = ui_components.get('form_components', {})
        
        # Get values from form components
        config = {
            'data': {
                'split_ratios': {
                    'train': form_components.get('train_ratio', 0.7).value,
                    'val': form_components.get('val_ratio', 0.15).value,
                    'test': form_components.get('test_ratio', 0.15).value
                },
                'seed': form_components.get('seed', 42).value,
                'shuffle': form_components.get('shuffle', True).value,
                'stratify': form_components.get('stratify', False).value
            },
            'output': {
                'train_dir': form_components.get('train_dir', 'data/train').value,
                'val_dir': form_components.get('val_dir', 'data/val').value,
                'test_dir': form_components.get('test_dir', 'data/test').value,
                'create_subdirs': form_components.get('create_subdirs', True).value,
                'overwrite': form_components.get('overwrite', False).value
            },
            'advanced': {
                'use_relative_paths': form_components.get('use_relative_paths', True).value,
                'preserve_structure': form_components.get('preserve_structure', True).value,
                'symlink': form_components.get('symlink', False).value,
                'backup': form_components.get('backup', True).value,
                'backup_dir': form_components.get('backup_dir', 'backups').value
            }
        }
        
        return config
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dict containing the extracted configuration
        """
        config = {
            'data': {
                'split_ratios': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15
                },
                'seed': 42,
                'shuffle': True,
                'stratify': False
            },
            'output': {
                'train_dir': 'data/train',
                'val_dir': 'data/val',
                'test_dir': 'data/test',
                'create_subdirs': True,
                'overwrite': False,
                'relative_paths': True,
                'preserve_dir_structure': True,
                'use_symlinks': False,
                'backup': True,
                'backup_dir': 'data/backup'
            }
        }
        
        # Update with values from UI components if available
        if 'form_components' in ui_components:
            ui = ui_components['form_components']
            
            # Update split ratios
            if 'train_slider' in ui:
                config['data']['split_ratios']['train'] = ui['train_slider'].value
            if 'val_slider' in ui:
                config['data']['split_ratios']['val'] = ui['val_slider'].value
            if 'test_slider' in ui:
                config['data']['split_ratios']['test'] = ui['test_slider'].value
                
            # Update output paths
            if 'train_dir_input' in ui:
                config['output']['train_dir'] = ui['train_dir_input'].value
            if 'val_dir_input' in ui:
                config['output']['val_dir'] = ui['val_dir_input'].value
            if 'test_dir_input' in ui:
                config['output']['test_dir'] = ui['test_dir_input'].value
                
            # Update seed if available
            if 'seed_input' in ui:
                config['data']['seed'] = ui['seed_input'].value
                
            # Update boolean flags
            for field in ['shuffle', 'stratify']:
                checkbox_name = f"{field}_checkbox"
                if checkbox_name in ui:
                    config['data'][field] = ui[checkbox_name].value
                    
            # Update output boolean flags
            for field in ['create_subdirs', 'overwrite', 'relative_paths', 
                         'preserve_dir_structure', 'use_symlinks', 'backup']:
                checkbox_name = f"{field}_checkbox"
                if checkbox_name in ui:
                    config['output'][field] = ui[checkbox_name].value
                    
            if 'backup_dir_input' in ui:
                config['output']['backup_dir'] = ui['backup_dir_input'].value
        
        return config
        
    def update_ui_from_config(self, config: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            config: Configuration to apply
            ui_components: Dictionary of UI components to update
        """
        form_components = ui_components.get('form_components', {})
        log_output = ui_components.get('log_output')
        
        try:
            # Update split ratios
            ratios = config.get('data', {}).get('split_ratios', {})
            for ratio_name in ['train', 'val', 'test']:
                if ratio_name in ratios and f"{ratio_name}_ratio" in form_components:
                    form_components[f"{ratio_name}_ratio"].value = ratios[ratio_name]
            
            # Update other data settings
            data_settings = config.get('data', {})
            for setting in ['seed', 'shuffle', 'stratify']:
                if setting in data_settings and setting in form_components:
                    form_components[setting].value = data_settings[setting]
            
            # Update output settings
            output_settings = config.get('output', {})
            for setting in ['train_dir', 'val_dir', 'test_dir', 'create_subdirs', 'overwrite']:
                if setting in output_settings and setting in form_components:
                    form_components[setting].value = output_settings[setting]
            
            # Update advanced settings
            adv_settings = config.get('advanced', {})
            for setting in ['use_relative_paths', 'preserve_structure', 'symlink', 'backup', 'backup_dir']:
                if setting in adv_settings and setting in form_components:
                    form_components[setting].value = adv_settings[setting]
            
            if log_output:
                with log_output:
                    print("✅ Konfigurasi berhasil dimuat")
                    
        except Exception as e:
            if log_output:
                with log_output:
                    print(f"❌ Gagal memuat konfigurasi: {str(e)}")
            self.logger.error(f"Error updating UI from config: {str(e)}", exc_info=True)
    
    @handle_ui_errors(error_component_title="Save Config Error", log_error=True)
    def save_config(self, ui_components: Dict[str, Any]) -> bool:
        """Save konfigurasi dengan status update.
        
        Args:
            ui_components: UI components untuk extract config
            
        Returns:
            True jika berhasil save
        """
        # Extract current config
        config = self.extract_config(ui_components)
        
        # Validate sebelum save
        is_valid, error_msg = self.validate_config(config)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Save menggunakan parent method
        result = super().save_config(ui_components)
        
        if result:
            self.logger.info("💾 Konfigurasi berhasil disimpan")
            
        return result
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self._config
        
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration untuk handler ini.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_split_config()
        
    def execute_split(
        self,
        dataset_path: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int = 42,
        **kwargs
    ) -> Dict[str, list]:
        """Execute the dataset split operation.
        
        Args:
            dataset_path: Path to the dataset directory
            train_ratio: Ratio of training data (0-1)
            val_ratio: Ratio of validation data (0-1)
            test_ratio: Ratio of test data (0-1)
            random_seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to the split function
            
        Returns:
            Dictionary containing lists of filenames for each split
            
        Raises:
            ValueError: If the split ratios are invalid
            FileNotFoundError: If the dataset directory doesn't exist
        """
        from smartcash.dataset.split import split_dataset  # Import here to avoid circular imports
        
        # Validate ratios
        ratios_sum = train_ratio + val_ratio + test_ratio
        if not (0.999 <= ratios_sum <= 1.001):
            error_msg = f"Split ratios must sum to 1.0, got {ratios_sum:.3f}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check dataset path exists
        if not Path(dataset_path).exists():
            error_msg = f"Dataset directory not found: {dataset_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        # Execute the split
        self.logger.info(f"Splitting dataset at {dataset_path} with ratios: "
                        f"train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}, seed={random_seed}")
                        
        result = split_dataset(
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            **kwargs
        )
        
        self.logger.info(f"Split completed successfully. Files: train={len(result.get('train', []))}, "
                        f"val={len(result.get('val', []))}, test={len(result.get('test', []))}")
                        
        return result
