# File: smartcash/handlers/model/integration/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Adapters untuk integrasi model dengan komponen lain

from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter

__all__ = [
    'CheckpointAdapter',
    'MetricsAdapter'
]