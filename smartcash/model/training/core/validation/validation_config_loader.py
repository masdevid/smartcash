#!/usr/bin/env python3
"""
Configuration loading for validation batch processing.

This module handles loading and validating configuration files for validation,
including class mappings and model configurations.
"""

import json
import os
from typing import Any, Dict, List, Optional

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ValidationConfigLoader:
    """Handles loading and validating configuration for validation processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config() if config_path else {}
        self.class_mapping = self.load_class_mapping()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate model configuration from file.
        
        Returns:
            Dictionary containing the loaded configuration
            
        Raises:
            FileNotFoundError: If config file is not found
            ValueError: If config file has invalid format or content
        """
        if not self.config_path or not os.path.isfile(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Validate required fields
            required_fields = ['class_mapping', 'model_config']
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Missing required field in config: {field}")
            
            # Load class mapping from loss.json format
            if 'class_mapping' not in config:
                logger.warning("No class_mapping found in config, creating from loss.json format")
                # Use mapping from loss.json: 0-6 (main), 7-13 (features), 14-16 (auth)
                config['class_mapping'] = {
                    str(i): i if i <= 13 else 'feature' for i in range(17)
                }
            
            # Convert string keys to integers for class mapping
            if isinstance(config['class_mapping'], dict):
                config['class_mapping'] = {
                    int(k): int(v) for k, v in config['class_mapping'].items()
                }
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in config file {self.config_path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error loading config from {self.config_path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def load_class_mapping(self) -> Dict[int, int]:
        """
        Load class mapping from loss.json or use SmartCash default mapping.
        
        Returns:
            Dictionary mapping fine class IDs to main class IDs
        """
        # SmartCash default mapping aligned with loss.json
        default_mapping = {
            # Main denominations (0-6) -> (0-6)
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
            # Nominal features (7-13) -> corresponding main class (0-6)
            7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6,
            # Authentication features (14-16) -> feature class (7)
            14: 7, 15: 7, 16: 7
        }
        
        if not self.config_path or not os.path.isfile(self.config_path):
            logger.info("Using SmartCash default class mapping (17->8 classes)")
            return default_mapping
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if 'class_mapping' in config and 'mapping_to_main' in config['class_mapping']:
                mapping = config['class_mapping']['mapping_to_main']
                # Convert to integer mapping, handle 'feature' entries
                loaded_mapping = {}
                for k, v in mapping.items():
                    key = int(k)
                    value = 7 if v == 'feature' else int(v) if isinstance(v, (int, str)) and str(v).isdigit() else 7
                    loaded_mapping[key] = value
                
                logger.info(f"Loaded SmartCash class mapping: {len(loaded_mapping)} fine -> 8 main classes")
                return loaded_mapping
            
            logger.warning(f"Invalid class_mapping structure in {self.config_path}, using default")
            return default_mapping
            
        except Exception as e:
            logger.warning(f"Error loading class mapping from {self.config_path}: {e}, using default")
            return default_mapping
    
    def get_class_names(self) -> List[str]:
        """
        Get class names aligned with loss.json specification.
        
        Returns:
            List of 17 fine-grained class names
        """
        return [
            '1000_whole', '2000_whole', '5000_whole', '10000_whole', 
            '20000_whole', '50000_whole', '100000_whole',
            '1000_nominal_feature', '2000_nominal_feature', '5000_nominal_feature', 
            '10000_nominal_feature', '20000_nominal_feature', '50000_nominal_feature', 
            '100000_nominal_feature', 'security_thread', 'watermark', 'special_sign'
        ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get loaded configuration."""
        return self.config
    
    def get_class_mapping(self) -> Dict[int, int]:
        """Get class mapping."""
        return self.class_mapping