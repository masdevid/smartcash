"""
File: smartcash/ui/components/card.py
Deskripsi: Komponen kartu UI yang reusable dengan berbagai variasi
"""

from typing import Optional, Dict, Any, Union, List, Callable
import ipywidgets as widgets
from IPython.display import HTML, display

from smartcash.ui.utils.constants import COLORS


def create_card(
    title: str, 
    value: str, 
    icon: str, 
    color: str, 
    description: Optional[str] = None,
    footer: Optional[str] = None,
    width: str = "250px",
    height: str = "auto",
    on_click: Optional[Callable] = None,
    extra_content: Optional[widgets.Widget] = None
) -> widgets.Widget:
    """
    Membuat komponen kartu statistik yang reusable.
    
    Args:
        title: Judul kartu
        value: Nilai utama yang ditampilkan
        icon: Ikon emoji
        color: Warna aksen kartu (bisa nama warna atau kode hex)
        description: Deskripsi opsional di bawah nilai
        footer: Teks footer opsional
        width: Lebar kartu (contoh: "250px" atau "100%")
        height: Tinggi kartu (default: "auto")
        on_click: Fungsi callback ketika kartu diklik
        extra_content: Widget tambahan untuk dimasukkan ke dalam kartu
        
    Returns:
        Widget kartu yang siap digunakan
    """
    # Style untuk kartu
    card_style = f"""
    <style>
        .card-container {{
            width: {width}; 
            height: {height}; 
            background-color: {COLORS['card']}; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-top: 3px solid {color};
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: {'pointer' if on_click else 'default'};
        }}
        .card-container:hover {{
            transform: {'translateY(-2px)' if on_click else 'none'};
            box-shadow: {'0 4px 8px rgba(0,0,0,0.15)' if on_click else '0 2px 5px rgba(0,0,0,0.1)'};
        }}
        .card-icon {{
            font-size: 1.5em; 
            margin-bottom: 5px;
        }}
        .card-title {{
            font-size: 0.9em; 
            color: {COLORS['dark']}; 
            margin-bottom: 5px;
        }}
        .card-value {{
            font-size: 1.4em; 
            font-weight: bold; 
            color: {color}; 
            margin-bottom: 5px;
        }}
        .card-description {{
            font-size: 0.8em; 
            color: {COLORS['muted']}; 
            margin-bottom: 5px;
        }}
        .card-footer {{
            font-size: 0.8em; 
            color: {COLORS['muted']}; 
            margin-top: 10px; 
            padding-top: 5px; 
            border-top: 1px solid {COLORS['border']};
        }}
    </style>
    """
    
    # Membangun konten HTML
    html_parts = [f'<div class="card-icon">{icon}</div>']
    html_parts.append(f'<div class="card-title">{title}</div>')
    html_parts.append(f'<div class="card-value">{value}</div>')
    
    if description:
        html_parts.append(f'<div class="card-description">{description}</div>')
    
    if footer:
        html_parts.append(f'<div class="card-footer">{footer}</div>')
    
    # Membuat widget HTML
    html_widget = widgets.HTML(
        value=card_style + "".join(html_parts),
        layout=widgets.Layout(width="100%", height="auto")
    )
    
    # Membungkus dalam container
    container = widgets.VBox(
        [html_widget],
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='0',
            margin='10px',
            border='none',
            flex_flow='column',
            align_items='stretch'
        )
    )
    
    # Menambahkan konten tambahan jika ada
    if extra_content:
        container.children += (extra_content,)
    
    # Menambahkan event handler jika on_click disediakan
    if on_click:
        def on_click_handler(change):
            on_click()
        
        # Tambahkan button transparan di atas kartu untuk menangani klik
        click_handler = widgets.Button(
            description='',
            layout=widgets.Layout(
                position='absolute',
                width='100%',
                height='100%',
                padding='0',
                border='none',
                background='transparent'
            ),
            style={'button_color': 'transparent'}
        )
        click_handler.on_click(lambda b: on_click())
        
        # Gabungkan dengan container yang sudah ada
        container = widgets.Box(
            [container, click_handler],
            layout=widgets.Layout(
                position='relative',
                width=width,
                height=height,
                padding='0',
                margin='10px',
                border='none',
                display='flex',
                flex_flow='column',
                align_items='stretch'
            )
        )
    
    return container


def create_card_row(
    cards: List[Dict[str, Any]],
    columns: int = 3,
    justify_content: str = 'space-between',
    width: str = '100%'
) -> widgets.Widget:
    """
    Membuat baris kartu dengan layout yang rapi.
    
    Args:
        cards: Daftar kamus berisi parameter untuk create_card
        columns: Jumlah kolom per baris
        justify_content: Penyelarasan horizontal ('flex-start', 'center', 'flex-end', 'space-between', 'space-around')
        width: Lebar total container
        
    Returns:
        Widget yang berisi baris kartu
    """
    # Membuat grid container
    grid = widgets.GridBox(
        children=[create_card(**card) for card in cards],
        layout=widgets.Layout(
            width=width,
            grid_template_columns=f'repeat({columns}, 1fr)',
            grid_gap='20px',
            justify_content=justify_content,
            padding='10px'
        )
    )
    
    return grid


# Fungsi bantuan untuk membuat kartu dengan style yang umum
def create_info_card(
    title: str, 
    value: str, 
    icon: str = "ℹ️", 
    description: Optional[str] = None,
    on_click: Optional[Callable] = None
) -> widgets.Widget:
    """Membuat kartu dengan style info."""
    return create_card(
        title=title,
        value=value,
        icon=icon,
        color=COLORS['info'],
        description=description,
        on_click=on_click
    )


def create_success_card(
    title: str, 
    value: str, 
    icon: str = "✅", 
    description: Optional[str] = None,
    on_click: Optional[Callable] = None
) -> widgets.Widget:
    """Membuat kartu dengan style sukses."""
    return create_card(
        title=title,
        value=value,
        icon=icon,
        color=COLORS['success'],
        description=description,
        on_click=on_click
    )


def create_warning_card(
    title: str, 
    value: str, 
    icon: str = "⚠️", 
    description: Optional[str] = None,
    on_click: Optional[Callable] = None
) -> widgets.Widget:
    """Membuat kartu dengan style peringatan."""
    return create_card(
        title=title,
        value=value,
        icon=icon,
        color=COLORS['warning'],
        description=description,
        on_click=on_click
    )


def create_error_card(
    title: str, 
    value: str, 
    icon: str = "❌", 
    description: Optional[str] = None,
    on_click: Optional[Callable] = None
) -> widgets.Widget:
    """Membuat kartu dengan style error."""
    return create_card(
        title=title,
        value=value,
        icon=icon,
        color=COLORS['error'],
        description=description,
        on_click=on_click
    )
