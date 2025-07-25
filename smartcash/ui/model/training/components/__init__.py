"""
File: smartcash/ui/model/training/components/__init__.py
Training UI components package - Updated for unified training pipeline.
"""

# Unified training components (new)
from .unified_training_ui import (
    create_unified_training_ui,
    update_training_buttons_state,
    update_summary_display
)

from .unified_training_form import (
    create_unified_training_form
)

# Legacy components (kept for compatibility)
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
    # Unified training components (primary)
    'create_unified_training_ui',
    'update_training_buttons_state', 
    'update_summary_display',
    'create_unified_training_form',
    
    # Legacy components (compatibility)
    'create_training_ui',
    'update_training_ui_from_config',
    'get_training_form_values',
    'update_metrics_display',
    'update_chart_data',
    'show_validation_results',
    'create_dual_charts_layout',
    'create_simple_chart_placeholder',
    'create_training_form',
    'create_simple_training_form',
    'generate_metrics_table_html',
    'get_initial_metrics_html',
    'get_quality_indicator',
    'create_metrics_summary',
    'format_metric_value',
    'create_config_summary',
    'create_simple_config_summary'
]