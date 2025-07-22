"""
File: smartcash/ui/model/training/components/__init__.py
Training UI components package.
"""

from .training_ui import (
    create_training_ui,
    update_training_ui_from_config,
    get_training_form_values,
    update_metrics_display,
    update_chart_data,
    show_validation_results
)

from .training_charts import (
    create_dual_charts_layout,
    create_simple_chart_placeholder
)

from .training_form import (
    create_training_form,
    create_simple_training_form
)

from .training_metrics import (
    generate_metrics_table_html,
    get_initial_metrics_html,
    get_quality_indicator,
    create_metrics_summary,
    format_metric_value
)

from .training_config_summary import (
    create_config_summary,
    create_simple_config_summary
)

__all__ = [
    # Main UI
    'create_training_ui',
    'update_training_ui_from_config',
    'get_training_form_values',
    'update_metrics_display',
    'update_chart_data',
    'show_validation_results',
    
    # Charts
    'create_dual_charts_layout',
    'create_simple_chart_placeholder',
    
    # Forms
    'create_training_form',
    'create_simple_training_form',
    
    # Metrics
    'generate_metrics_table_html',
    'get_initial_metrics_html',
    'get_quality_indicator',
    'create_metrics_summary',
    'format_metric_value',
    
    # Config Summary
    'create_config_summary',
    'create_simple_config_summary'
]