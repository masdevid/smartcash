"""
File: smartcash/ui/dataset/augment/handlers/augment_config_handler.py
Description: Export for augment config handler

This module provides a clean export of the AugmentConfigHandler from the configs
directory, maintaining the separation of concerns between configs and handlers.
"""

# Import from configs directory
from ..configs.augment_config_handler import AugmentConfigHandler

# Export the class
__all__ = ['AugmentConfigHandler']