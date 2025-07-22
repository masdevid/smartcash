"""
File: smartcash/dataset/augmentor/balancer/__init__.py
Deskripsi: Balancer module exports dengan preserved compatibility
"""

from smartcash.dataset.augmentor.balancer.balancer import (
    ClassBalancingStrategy,
    get_layer1_classes,
    get_layer2_classes,
    get_all_target_classes,
    is_target_class,
    calc_need,
    calc_priority_multiplier
)

from smartcash.dataset.augmentor.balancer.selector import FileSelectionStrategy

# Backward compatibility aliases
from smartcash.dataset.augmentor.balancer.balancer import ClassBalancingStrategy as BalancingStrategy
from smartcash.dataset.augmentor.balancer.selector import FileSelectionStrategy as SelectorStrategy

# One-liner utilities
calculate_balancing_needs = lambda class_counts, target_count=1000: ClassBalancingStrategy().calculate_balancing_needs(class_counts, target_count)
calculate_split_balancing_needs = lambda data_dir, target_split, target_count=1000: ClassBalancingStrategy().calculate_balancing_needs_split_aware(data_dir, target_split, target_count)
select_prioritized_files = lambda class_needs, files_metadata: FileSelectionStrategy().select_prioritized_files(class_needs, files_metadata)
select_split_files = lambda data_dir, target_split, class_needs: FileSelectionStrategy().select_prioritized_files_split_aware(data_dir, target_split, class_needs)

__all__ = [
    'ClassBalancingStrategy',
    'FileSelectionStrategy', 
    'BalancingStrategy',
    'SelectorStrategy',
    'get_layer1_classes',
    'get_layer2_classes', 
    'get_all_target_classes',
    'is_target_class',
    'calc_need',
    'calc_priority_multiplier',
    'calculate_balancing_needs',
    'calculate_split_balancing_needs',
    'select_prioritized_files',
    'select_split_files'
]