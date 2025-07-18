"""
File: smartcash/ui/dataset/visualization/components/visualization_ui.py
Deskripsi: Antarmuka pengguna visualisasi dataset mengikuti template standar SmartCash.

Modul ini menyediakan antarmuka untuk memvisualisasikan statistik dataset,
distribusi kelas, dan menghasilkan berbagai chart untuk analisis dataset.

Urutan Kontainer:
1. Header Container (Judul, Status)
2. Form Container (Opsi Visualisasi)
3. Action Container (Tombol Analisis/Refresh/Ekspor/Bandingkan)
4. Summary Container (Ringkasan Statistik)
5. Operation Container (Progres + Log)
6. Footer Container (Tips dan Informasi)
"""

from typing import Optional, Dict, Any, List
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.decorators import handle_ui_errors

# Module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


def create_data_card(title: str, content: widgets.Widget, width: str = "100%") -> widgets.VBox:
    """Membuat container kartu bergaya untuk visualisasi data.
    
    Args:
        title: Judul kartu
        content: Widget yang akan ditempatkan di dalam kartu
        width: Lebar kartu (default: "100%")
        
    Returns:
        Widget VBox yang berisi kartu
    """
    card_header = widgets.HTML(
        value=f'<div style="padding: 10px; background-color: #f8f9fa; border-bottom: 1px solid #dee2e6; border-radius: 5px 5px 0 0; font-weight: bold;">{title}</div>',
        layout=widgets.Layout(width='100%')
    )
    
    card_content = widgets.VBox(
        [content],
        layout=widgets.Layout(
            padding='10px',
            border='1px solid #dee2e6',
            border_top='none',
            border_radius='0 0 5px 5px',
            width='100%',
            overflow='auto'
        )
    )
    
    return widgets.VBox(
        [card_header, card_content],
        layout=widgets.Layout(
            width=width,
            margin='0 0 15px 0',
            box_shadow='0 2px 4px rgba(0,0,0,0.1)'
        )
    )


@handle_ui_errors(error_component_title="Kesalahan UI Visualisasi")
def create_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Membuat antarmuka pengguna visualisasi dataset mengikuti standar SmartCash.
    
    Fungsi ini membuat UI lengkap untuk memvisualisasikan statistik dataset
    dengan bagian-bagian berikut:
    - Pemilihan tipe chart dan pembagian data
    - Opsi analisis dan ekspor
    - Ringkasan statistik
    - Chart dan visualisasi interaktif
    """
    # Inisialisasi konfigurasi dan kamus komponen
    current_config = config or {}
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Buat Header Container ===
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='📊'  # Chart emoji for visualization
    )
    # Store both the container object and its widget
    ui_components['containers']['header'] = {
        'container': header_container.container,
        'widget': header_container
    }
    
    # === 2. Buat Form Container ===
    # Buat widget form dengan tata letak dua kolom
    form_widgets = _create_module_form_widgets(current_config)
    
    # Buat container form dengan widget
    form_container = create_form_container(
        form_rows=form_widgets['form_rows'],
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px",
        layout_kwargs={
            'width': '100%',
            'max_width': '100%',
            'margin': '0',
            'padding': '0',
            'justify_content': 'flex-start',
            'align_items': 'flex-start'
        }
    )
    
    # Simpan referensi
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Buat Action Container ===
    # Tentukan tombol aksi dengan ikon dan tooltip
    action_buttons = [
        {
            'id': 'refresh',
            'text': 'Segarkan',
            'style': 'success',
            'tooltip': 'Segarkan visualisasi dengan pengaturan saat ini',
            'icon': 'refresh'
        },
        {
            'id': 'preprocessed',
            'text': 'Contoh Praproses',
            'style': 'info',
            'tooltip': 'Lihat contoh data yang telah diproses',
            'icon': 'filter'
        },
        {
            'id': 'augmented',
            'text': 'Contoh Augmentasi',
            'style': 'info',
            'tooltip': 'Lihat contoh data yang telah diaugmentasi',
            'icon': 'magic'
        },
    ]
    
    # Buat container aksi dengan tombol
    action_container = create_action_container(
        show_save_reset=False,
        buttons=action_buttons,
        title="📊 Aksi Visualisasi"
    )
    
    # Simpan container aksi
    ui_components['containers']['actions'] = action_container
    
    # Simpan referensi tombol dengan ID dasar saja
    if hasattr(action_container, 'get_button'):
        # Jika action container memiliki method get_button
        ui_components['refresh'] = action_container.get_button('refresh')
        ui_components['preprocessed'] = action_container.get_button('preprocessed')
        ui_components['augmented'] = action_container.get_button('augmented')
    elif hasattr(action_container, 'buttons') and isinstance(action_container.buttons, dict):
        # Jika action container memiliki atribut buttons yang berupa dictionary
        buttons = action_container.buttons
        
        # Gunakan ID dasar saja tanpa menambahkan _button
        for btn_id in ['refresh', 'preprocessed', 'augmented']:
            # Coba dapatin button dengan ID dasar
            button = buttons.get(btn_id)
            if button is not None:
                ui_components[btn_id] = button
    
    # === 4. Buat Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    summary_container = create_summary_container(
        title="📋 Statistik Dataset",
        theme="info",
        icon="📊"
    )
    summary_container.set_content(summary_content)
    
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Buat Operation Container ===
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='single',
        log_module_name=UI_CONFIG['module_name'],
        # log_namespace_filter='visualization',  # Temporarily disabled
        log_height="150px",
        log_entry_style='compact',
        collapsible=True,
        collapsed=False
    )
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Buat Footer Container ===
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        show_tips=True,
        show_version=True
    )
    # Simpan container dan widget-nya
    ui_components['containers']['footer'] = {
        'container': footer_container.container,
        'widget': footer_container
    }
    
    # === 5.1 Buat Dashboard Container ===
    # Buat container untuk dashboard cards
    dashboard_container = widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            margin='10px 0',
            border='1px solid #e0e0e0',
            border_radius='5px',
            display='flex',
            flex_flow='row wrap',
            justify_content='space-between',
            align_items='stretch',
            align_content='stretch',
            overflow='hidden'
        )
    )
    
    # Simpan dashboard container
    ui_components['containers']['dashboard'] = dashboard_container
    
    # === 7. Buat Main Container ===
    # Siapkan komponen untuk container utama
    components = [
        # Header container (objek dengan atribut .container)
        {'type': 'header', 'component': header_container.container, 'order': 0},
        # Form container (dictionary dengan kunci 'container')
        {'type': 'form', 'component': form_container['container'], 'order': 2},
        # Action container (dictionary dengan kunci 'container')
        {'type': 'action', 'component': action_container['container'], 'order': 3},
        # Dashboard container
        {'type': 'dashboard', 'component': dashboard_container, 'order': 1},
        # Summary container (objek dengan atribut .container)
        {'type': 'summary', 'component': summary_container.container, 'order': 4},
        # Operation container (dictionary dengan kunci 'container')
        {'type': 'operation', 'component': operation_container['container'], 'order': 5},
        # Footer container (objek dengan atribut .container)
        {'type': 'footer', 'component': footer_container.container, 'order': 6}
    ]
    
    # Buat container utama dengan semua komponen
    main_container = create_main_container(
        components=components,
        **kwargs
    )
    
    # Simpan referensi UI utama
    ui_components['ui'] = main_container
    ui_components['main_container'] = main_container
    
    # Tambahkan semua container ke ui_components untuk akses mudah
    ui_components['containers']['main'] = main_container
    
    # Buat kamus hasil dengan semua komponen
    result = {
        'ui_components': ui_components,
        'ui': main_container.container,  # Gunakan widget sebenarnya, bukan objek MainContainer
        'main_container': main_container.container,
        'containers': ui_components['containers'],
        'widgets': ui_components['widgets']
    }
    
    # Tambahkan semua komponen ke root untuk kompatibilitas ke belakang
    result.update(ui_components['containers'])
    result.update(ui_components['widgets'])
    
    # Tambahkan referensi langsung ke semua container untuk akses lebih mudah
    for container_name, container in ui_components['containers'].items():
        result[container_name] = container
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat widget form khusus modul untuk opsi visualisasi dengan tata letak dua kolom.
    
    Args:
        config: Kamus konfigurasi untuk widget form
        
    Returns:
        Kamus yang berisi UI form dan referensi widget
    """
    from ..constants import CHART_TYPE_OPTIONS, DATA_SPLIT_OPTIONS, EXPORT_FORMAT_OPTIONS
    
    # Common layout for form elements
    dropdown_layout = widgets.Layout(
        width='90%',
        margin='5px 0',
        padding='5px 0'
    )
    
    checkbox_layout = widgets.Layout(
        width='100%',
        margin='8px 0',
        padding='5px 0'
    )
    
    # Chart type selection
    chart_type_dropdown = widgets.Dropdown(
        options=CHART_TYPE_OPTIONS,
        value=config.get('chart_type', 'bar'),
        description='Chart Type:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Data split selection
    data_split_dropdown = widgets.Dropdown(
        options=DATA_SPLIT_OPTIONS,
        value=config.get('data_split', 'all'),
        description='Data Split:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Export format selection
    export_format_dropdown = widgets.Dropdown(
        options=EXPORT_FORMAT_OPTIONS,
        value=config.get('export_format', 'png'),
        description='Export Format:',
        style={'description_width': '120px'},
        layout=dropdown_layout
    )
    
    # Checkbox options
    show_grid_checkbox = widgets.Checkbox(
        value=config.get('show_grid', True),
        description='Show Grid',
        layout=checkbox_layout
    )
    
    show_legend_checkbox = widgets.Checkbox(
        value=config.get('show_legend', True),
        description='Show Legend',
        layout=checkbox_layout
    )
    
    auto_refresh_checkbox = widgets.Checkbox(
        value=config.get('auto_refresh', False),
        description='Auto Refresh',
        layout=checkbox_layout
    )
    
    # Refresh interval
    refresh_interval_int = widgets.IntText(
        value=config.get('refresh_interval', 60),
        description='Refresh (s):',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='80%', margin='8px 0')
    )
    
    # Create form sections with two-column layout
    chart_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>📊 Chart Configuration</h4>"),
        chart_type_dropdown,
        data_split_dropdown,
        export_format_dropdown
    ], layout=widgets.Layout(width='48%', margin='0 1% 10px 0'))
    
    display_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>⚙️ Display</h4>"),
        show_grid_checkbox,
        show_legend_checkbox
    ], layout=widgets.Layout(width='48%', margin='0 0 10px 1%'))
    
    refresh_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>🔄 Refresh</h4>"),
        widgets.HBox([
            auto_refresh_checkbox,
            refresh_interval_int
        ], layout=widgets.Layout(width='100%', justify_content='space-between'))
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Combine all sections in a two-column layout
    form_content = widgets.VBox([
        widgets.HBox([chart_section, display_section], 
                    layout=widgets.Layout(justify_content='space-between')),
        refresh_section
    ])
    
    return {
        'form_rows': [[form_content]],  # Single row containing our custom layout
        'widgets': {
            'chart_type_dropdown': chart_type_dropdown,
            'data_split_dropdown': data_split_dropdown,
            'export_format_dropdown': export_format_dropdown,
            'show_grid_checkbox': show_grid_checkbox,
            'show_legend_checkbox': show_legend_checkbox,
            'auto_refresh_checkbox': auto_refresh_checkbox,
            'refresh_interval_int': refresh_interval_int
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> str:
    """
    Buat konten ringkasan untuk modul.
    
    Args:
        config: Kamus konfigurasi
        
    Returns:
        String HTML yang berisi konten ringkasan
    """
    return """
    <div style="padding: 10px;">
        <h5>📊 Ringkasan Dataset</h5>
        <p>Statistik dan visualisasi dataset akan ditampilkan di sini setelah analisis.</p>
        <ul>
            <li>Total sampel: <span id="total-samples">-</span></li>
            <li>Distribusi kelas: <span id="class-distribution">-</span></li>
            <li>Pembagian data: <span id="data-splits">-</span></li>
            <li>Status augmentasi: <span id="augmentation-status">-</span></li>
        </ul>
    </div>
    """


def _create_module_info_box() -> widgets.Widget:
    """
    Buat konten kotak info untuk footer.
    
    Returns:
        Widget yang berisi konten kotak info
    """
    return widgets.HTML(
        value="""
        <div style="padding: 12px; background: #e3f2fd; border-radius: 4px; margin: 8px 0;">
            <h4 style="margin-top: 0; color: #0d47a1;">📊 Panduan Visualisasi</h4>
            <p>Modul ini membantu Anda menganalisis dan memvisualisasikan statistik dataset.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Pilih tipe chart dan pembagian data</li>
                <li>Konfigurasi opsi tampilan</li>
                <li>Klik 'Analisis Dataset' untuk menghasilkan visualisasi</li>
                <li>Gunakan 'Ekspor Chart' untuk menyimpan hasil</li>
            </ol>
        </div>
        """
    )