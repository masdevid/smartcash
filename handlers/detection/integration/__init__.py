# File: smartcash/handlers/detection/integration/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Package untuk integrasi detection handler dengan komponen lain

from smartcash.handlers.detection.integration.model_adapter import ModelAdapter
from smartcash.handlers.detection.integration.visualizer_adapter import VisualizerAdapter

__all__ = ['ModelAdapter', 'VisualizerAdapter']