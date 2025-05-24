from smartcash.ui.components.progress_tracking import create_progress_tracking
import ipywidgets as widgets

def create_progress_section():
    return create_progress_tracking(
        module_name='download',
        show_step_progress=True,
        show_overall_progress=True,
        show_current_progress=False,
        width='100%'
    ) 