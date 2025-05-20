"""
File: smartcash/ui/dataset/visualization/handlers/tabs/__init__.py
Deskripsi: Modul handlers untuk tab visualisasi dataset
"""

from smartcash.ui.dataset.visualization.handlers.tabs.distribution_tab import on_distribution_click
from smartcash.ui.dataset.visualization.handlers.tabs.split_tab import on_split_click
from smartcash.ui.dataset.visualization.handlers.tabs.layer_tab import on_layer_click
from smartcash.ui.dataset.visualization.handlers.tabs.bbox_tab import on_bbox_click
from smartcash.ui.dataset.visualization.handlers.tabs.preprocessing_tab import on_preprocessing_samples_click
from smartcash.ui.dataset.visualization.handlers.tabs.augmentation_tab import on_augmentation_comparison_click

__all__ = [
    'on_distribution_click',
    'on_split_click',
    'on_layer_click',
    'on_bbox_click',
    'on_preprocessing_samples_click',
    'on_augmentation_comparison_click'
] 