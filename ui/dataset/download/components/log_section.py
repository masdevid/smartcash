from smartcash.ui.components.log_accordion import create_log_accordion
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS

def create_log_section():
    log_components = create_log_accordion(
        module_name='download',
        height='200px',
        width='100%'
    )
    log_components['log_accordion'].layout.margin = '5px 0'
    log_components['log_accordion'].layout.border_radius = '5px'
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border=f'1px solid {COLORS.get("border", "#ddd")}',
            padding='15px',
            margin='15px 0',
            display='none',
            border_radius='5px',
            min_height='50px'
        )
    )
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            margin='5px 0',
            width='100%',
            min_height='30px',
            padding='2px 0'
        )
    )
    return {
        **log_components,
        'summary_container': summary_container,
        'confirmation_area': confirmation_area
    } 