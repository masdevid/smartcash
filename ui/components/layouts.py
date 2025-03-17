"""
File: smartcash/ui/components/layouts.py
Deskripsi: Definisi layout standar untuk widgets UI
"""

import ipywidgets as widgets

# Standard layouts
STANDARD_LAYOUTS = {
    'header': widgets.Layout(margin='0 0 15px 0'),
    'section': widgets.Layout(margin='15px 0 10px 0'),
    'container': widgets.Layout(width='100%', padding='10px'),
    'output': widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        min_height='100px',
        max_height='300px',
        margin='10px 0',
        overflow='auto'
    ),
    'button': widgets.Layout(margin='10px 0'),
    'button_small': widgets.Layout(margin='5px'),
    'hbox': widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        width='100%'
    ),
    'vbox': widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    )
}

# Layout untuk container utama
MAIN_CONTAINER = widgets.Layout(
    width='100%',
    padding='10px'
)

# Layout untuk output widget
OUTPUT_WIDGET = widgets.Layout(
    width='100%',
    border='1px solid #ddd',
    min_height='100px',
    margin='10px 0',
    padding='8px 4px'
)

# Layout untuk tombol
BUTTON = widgets.Layout(
    margin='10px 0'
)

# Layout untuk tombol yang disembunyikan
HIDDEN_BUTTON = widgets.Layout(
    margin='10px 0',
    display='none'
)

# Layout untuk input text
TEXT_INPUT = widgets.Layout(
    width='60%',
    margin='10px 0'
)

# Layout untuk textarea
TEXT_AREA = widgets.Layout(
    width='60%',
    height='150px',
    margin='10px 0'
)

# Layout untuk radio dan checkbox
SELECTION = widgets.Layout(
    margin='10px 0'
)

# Layout untuk grup widget horizontal
HORIZONTAL_GROUP = widgets.Layout(
    display='flex',
    flex_flow='row wrap',
    align_items='center',
    width='100%'
)

# Layout untuk grup widget vertikal
VERTICAL_GROUP = widgets.Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%'
)

# Layout untuk divider
DIVIDER = widgets.Layout(
    height='1px',
    border='0',
    border_top='1px solid #eee',
    margin='15px 0'
)

# Layout untuk kartu/card
CARD = widgets.Layout(
    border='1px solid #ddd',
    border_radius='4px',
    padding='15px',
    margin='10px 0',
    width='100%'
)

# Layout untuk tab container
TABS = widgets.Layout(
    width='100%',
    margin='10px 0'
)

# Layout untuk accordion
ACCORDION = widgets.Layout(
    width='100%',
    margin='10px 0'
)

def create_divider():
    """Buat divider horizontal."""
    return widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>")