from smartcash.ui.components.progress_tracking import create_progress_tracking
import ipywidgets as widgets

def create_progress_section():
    progress_components = create_progress_tracking(
        module_name='download',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    progress_components['progress_container'].layout.margin = '5px 0'
    progress_components['progress_container'].layout.padding = '5px 0'
    progress_components['progress_container'].layout.border_radius = '5px'
    progress_components['progress_container'].layout.display = 'none'
    return progress_components 