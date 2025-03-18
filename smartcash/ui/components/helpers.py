"""
File: smartcash/ui/components/helpers.py
Deskripsi: Helper functions untuk komponen UI yang memanfaatkan ui_helpers untuk konsistensi
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Tuple, Union, Callable

# Import semua helper functions dari ui_helpers
from smartcash.ui.utils.ui_helpers import (
    create_tab_view,
    create_loading_indicator,
    update_output_area,
    register_observer_callback,
    display_file_info,
    create_progress_updater,
    run_task,
    create_button_group,
    create_confirmation_dialog
)

# Alias untuk backward compatibility
# Semua fungsi berikut adalah alias untuk fungsi dari ui_helpers
# untuk mempertahankan backward compatibility dengan kode yang sudah ada

def run_task_with_progress(*args, **kwargs):
    """Alias untuk run_task dari ui_helpers."""
    return run_task(*args, **kwargs)