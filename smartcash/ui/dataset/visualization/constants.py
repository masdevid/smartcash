"""
File: smartcash/ui/dataset/visualization/constants.py
Description: Constants and configuration for the visualization module

This module contains all constants, enums, and configuration data
used throughout the visualization module following the UI structure guidelines.
"""

from typing import Dict, Any, List
from enum import Enum

# =============================================================================
# ENUMS
# =============================================================================

class VisualizationOperation(Enum):
    """Available visualization operations."""
    ANALYZE = "analyze"
    REFRESH = "refresh"
    EXPORT = "export"
    COMPARE = "compare"

class ChartType(Enum):
    """Available chart types."""
    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"

class DataSplit(Enum):
    """Dataset splits."""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    ALL = "all"

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'module_name': 'visualization',
    'parent_module': 'dataset',
    'title': '📊 Dataset Visualization',
    'subtitle': 'Analyze and visualize your dataset statistics',
    'icon': '📊',
    'version': '2.0.0'
}

# =============================================================================
# BUTTON CONFIGURATION
# =============================================================================

BUTTON_CONFIG = {
    'analyze': {
        'text': '🔍 Analyze Dataset',
        'style': 'primary',
        'tooltip': 'Analyze current dataset statistics and generate visualizations',
        'order': 1
    },
    'refresh': {
        'text': '🔄 Refresh Data',
        'style': 'info',
        'tooltip': 'Refresh dataset statistics and update visualizations',
        'order': 2
    },
    'export': {
        'text': '📥 Export Charts',
        'style': 'success',
        'tooltip': 'Export generated charts and statistics to files',
        'order': 3
    },
    'compare': {
        'text': '📈 Compare Splits',
        'style': 'warning',
        'tooltip': 'Compare statistics across different data splits',
        'order': 4
    }
}

# =============================================================================
# FORM CONFIGURATION
# =============================================================================

CHART_TYPE_OPTIONS = [
    ('Bar Chart', 'bar'),
    ('Pie Chart', 'pie'),
    ('Line Chart', 'line'),
    ('Scatter Plot', 'scatter'),
    ('Histogram', 'histogram')
]

DATA_SPLIT_OPTIONS = [
    ('Training Set', 'train'),
    ('Validation Set', 'valid'),
    ('Test Set', 'test'),
    ('All Splits', 'all')
]

EXPORT_FORMAT_OPTIONS = [
    ('PNG Image', 'png'),
    ('PDF Document', 'pdf'),
    ('SVG Vector', 'svg'),
    ('HTML Report', 'html')
]

# =============================================================================
# STYLING CONSTANTS
# =============================================================================

VISUALIZATION_COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'train': '#4CAF50',
    'valid': '#2196F3',
    'test': '#FF9800'
}

SECTION_STYLES = {
    'statistics': {
        'border_color': VISUALIZATION_COLORS['info'],
        'background': '#f0f8ff'
    },
    'charts': {
        'border_color': VISUALIZATION_COLORS['success'],
        'background': '#f0fff0'
    },
    'comparison': {
        'border_color': VISUALIZATION_COLORS['warning'],
        'background': '#fffaf0'
    },
    'export': {
        'border_color': VISUALIZATION_COLORS['primary'],
        'background': '#f8f9ff'
    }
}

# =============================================================================
# CHART CONFIGURATION
# =============================================================================

CHART_CONFIG = {
    'default_width': 800,
    'default_height': 600,
    'color_palette': [
        '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
        '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
    ],
    'background_color': '#ffffff',
    'grid_alpha': 0.3,
    'title_font_size': 14,
    'axis_font_size': 12,
    'legend_font_size': 10
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_VISUALIZATION_PARAMS = {
    'chart_type': 'bar',
    'data_split': 'all',
    'export_format': 'png',
    'show_grid': True,
    'show_legend': True,
    'auto_refresh': False,
    'refresh_interval': 60
}

DEFAULT_STATISTICS_CONFIG = {
    'show_percentages': True,
    'show_totals': True,
    'show_class_distribution': True,
    'show_augmentation_stats': True,
    'decimal_places': 1
}

DEFAULT_EXPORT_CONFIG = {
    'dpi': 300,
    'transparent_background': False,
    'include_metadata': True,
    'compress_images': True
}

# =============================================================================
# MESSAGES
# =============================================================================

SUCCESS_MESSAGES = {
    'analysis_complete': '✅ Dataset analysis completed successfully',
    'refresh_complete': '✅ Data refresh completed',
    'export_complete': '✅ Charts exported successfully',
    'comparison_complete': '✅ Split comparison completed'
}

ERROR_MESSAGES = {
    'analysis_failed': '❌ Dataset analysis failed',
    'refresh_failed': '❌ Data refresh failed',
    'export_failed': '❌ Chart export failed',
    'comparison_failed': '❌ Split comparison failed',
    'no_data_found': '❌ No dataset found to analyze',
    'invalid_chart_type': '❌ Invalid chart type selected'
}

WARNING_MESSAGES = {
    'no_data_warning': '⚠️ No data available for selected split',
    'empty_dataset': '⚠️ Dataset appears to be empty',
    'missing_splits': '⚠️ Some data splits are missing'
}

# =============================================================================
# TIPS AND HELP TEXT
# =============================================================================

VISUALIZATION_TIPS = [
    "💡 Use bar charts for class distribution analysis",
    "📊 Pie charts work well for showing proportions",
    "📈 Line charts are great for trend analysis",
    "🔍 Compare splits to identify data imbalances",
    "📥 Export charts for documentation and reports",
    "🔄 Enable auto-refresh for real-time monitoring",
    "⚖️ Check augmentation statistics to validate data pipeline"
]

HELP_TEXT = {
    'chart_type': 'Select the type of chart to generate for visualization',
    'data_split': 'Choose which data split to analyze and visualize',
    'export_format': 'Select format for exporting generated charts',
    'show_grid': 'Display grid lines on charts for better readability',
    'show_legend': 'Include legend in charts to identify data series',
    'auto_refresh': 'Automatically refresh data at specified intervals',
    'refresh_interval': 'Time interval (seconds) for automatic data refresh'
}

# =============================================================================
# DATASET CLASSES (Business Logic)
# =============================================================================

BANKNOTE_CLASSES = [
    'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
    'rp_1000', 'rp_2000', 'rp_5000', 'rp_10000', 
    'rp_20000', 'rp_50000', 'rp_100000'
]

CLASS_COLORS = {
    'layer1': '#e74c3c',
    'layer2': '#3498db',
    'layer3': '#2ecc71',
    'layer4': '#f39c12',
    'layer5': '#9b59b6',
    'rp_1000': '#1abc9c',
    'rp_2000': '#e67e22',
    'rp_5000': '#34495e',
    'rp_10000': '#e91e63',
    'rp_20000': '#673ab7',
    'rp_50000': '#ff5722',
    'rp_100000': '#795548'
}

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

BACKEND_CONFIG = {
    'service_enabled': True,
    'cache_enabled': True,
    'cache_ttl_seconds': 300,
    'max_chart_size_mb': 10,
    'supported_formats': ['.png', '.jpg', '.pdf', '.svg', '.html'],
    'default_timeout': 30
}

PERFORMANCE_CONFIG = {
    'max_data_points': 10000,
    'chart_render_timeout': 30,
    'memory_limit_mb': 512,
    'enable_compression': True
}