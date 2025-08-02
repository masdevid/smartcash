# === core/__init__.py ===
"""Core processing components"""
from .normalizer import YOLONormalizer
from .file_processor import FileProcessor  
from .stats_collector import StatsCollector
from .label_deduplicator import LabelDeduplicator, deduplicate_labels_in_directory

__all__ = ['YOLONormalizer', 'FileProcessor', 'StatsCollector', 'LabelDeduplicator', 'deduplicate_labels_in_directory']