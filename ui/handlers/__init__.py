"""
File: smartcash/ui/handlers/__init__.py
Deskripsi: Package untuk handlers UI
"""

from .error_handler import create_error_response, handle_ui_errors

__all__ = ['create_error_response', 'handle_ui_errors']