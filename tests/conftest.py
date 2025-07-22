"""
File: tests/conftest.py
Description: Pytest configuration and fixtures for the test suite.
"""
import pytest
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
