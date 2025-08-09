"""
SmartCash Model API

This module provides the main interface for working with SmartCash models.

Classes:
    SmartCashModelAPI: Main class for model operations

Functions:
    create_api: Factory function to create a SmartCashModelAPI instance
"""

from .core import SmartCashModelAPI, create_api

__all__ = [
    'SmartCashModelAPI',
    'create_api'
]
