"""
File: smartcash/ui_components/dataset_components.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk pengelolaan dataset mata uang Rupiah
"""

import ipywidgets as widgets
from IPython.display import HTML, display
from smartcash.ui_components.widget_layouts import (
    MAIN_CONTAINER, BUTTON, OUTPUT_WIDGET, 
    TEXT_INPUT, SELECTION, CARD, ACCORDION,
    TABS, create_divider
)

def create_dataset_ui():
    """Buat komponen UI untuk dataset management."""
    main = widgets.VBox(layout=MAIN_CONTAINER)
    
    # Header
    header = widgets.HTML(
        "<h1>📊 Dataset Management</h1>" +
        "<p>Pengelolaan dataset untuk deteksi mata uang Rupiah</p>"
    )
    
    # Tab untuk berbagai operasi dataset
    tabs = widgets.Tab(layout=TABS)
    
    # ==== Tab 1: Download Dataset ====
    download_tab = widgets.VBox()
    
    # Pilihan sumber dataset
    source_group = widgets.VBox([
        widgets.HTML("<h3>🌐 Sumber Dataset</h3>")
    ])
    
    source_radio = widgets.RadioButtons(
        options=[('Roboflow', 'roboflow'), ('Local', 'local')],
        value='roboflow',
        description='Source:',
        layout=SELECTION
    )
    
    # Konfigurasi Roboflow
    roboflow_card = widgets.VBox(layout=CARD)
    roboflow_card_header = widgets.HTML("<h4>🔑 Konfigurasi Roboflow</h4>")
    
    roboflow_api_key = widgets.Text(
        placeholder='Masukkan Roboflow API Key',
        description='API Key:',
        layout=TEXT_INPUT,
        style={'description_width': 'initial'}
    )
    
    roboflow_workspace = widgets.Text(
        value='smartcash-wo2us',
        description='Workspace:',
        layout=TEXT_INPUT,
        style={'description_width': 'initial'}
    )
    
    roboflow_project = widgets.Text(
        value='rupiah-emisi-2022',
        description='Project:',
        layout=TEXT_INPUT,
        style={'description_width': 'initial'}
    )
    
    roboflow_version = widgets.Text(
        value='3',
        description='Version:',
        layout=TEXT_INPUT,
        style={'description_width': 'initial'}
    )
    
    # Tombol download
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download',
        layout=BUTTON
    )
    
    # Status output untuk download
    download_status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Progress bar
    download_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Downloading:',
        bar_style='info',
        orientation='horizontal',
        layout={'width': '100%', 'margin': '10px 0', 'visibility': 'hidden'}
    )
    
    # Susun komponen download tab
    roboflow_card.children = [
        roboflow_card_header,
        roboflow_api_key,
        roboflow_workspace,
        roboflow_project,
        roboflow_version
    ]
    
    source_group.children = [
        source_radio,
        roboflow_card
    ]
    
    download_tab.children = [
        source_group,
        create_divider(),
        download_button,
        download_progress,
        download_status
    ]
    
    # ==== Tab 2: Dataset Validation ====
    validation_tab = widgets.VBox()
    
    # Header validasi
    validation_header = widgets.HTML("<h3>🔍 Validasi Dataset</h3>")
    
    # Pilihan split data
    split_selector = widgets.Dropdown(
        options=[('Training', 'train'), ('Validation', 'valid'), ('Testing', 'test')],
        value='train',
        description='Dataset Split:',
        layout=SELECTION,
        style={'description_width': 'initial'}
    )
    
    # Opsi validasi
    fix_checkbox = widgets.Checkbox(
        value=True,
        description='Perbaiki masalah otomatis',
        layout=SELECTION
    )
    
    visualize_checkbox = widgets.Checkbox(
        value=True,
        description='Visualisasikan masalah',
        layout=SELECTION
    )
    
    move_invalid_checkbox = widgets.Checkbox(
        value=False,
        description='Pindahkan file tidak valid',
        layout=SELECTION
    )
    
    # Tombol validasi
    validate_button = widgets.Button(
        description='Validasi Dataset',
        button_style='primary',
        icon='check-circle',
        layout=BUTTON
    )
    
    # Progress bar validasi
    validation_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Validating:',
        bar_style='info',
        orientation='horizontal',
        layout={'width': '100%', 'margin': '10px 0', 'visibility': 'hidden'}
    )
    
    # Status output validasi
    validation_status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Susun komponen validation tab
    validation_tab.children = [
        validation_header,
        split_selector,
        fix_checkbox,
        visualize_checkbox,
        move_invalid_checkbox,
        create_divider(),
        validate_button,
        validation_progress,
        validation_status
    ]
    
    # ==== Tab 3: Augmentation ====
    augmentation_tab = widgets.VBox()
    
    # Header augmentation
    augmentation_header = widgets.HTML("<h3>🔄 Augmentasi Dataset</h3>")
    
    # Tipe augmentasi
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('Lighting (variasi pencahayaan)', 'lighting'),
            ('Position (variasi posisi)', 'position'),
            ('Combined (kombinasi)', 'combined'),
            ('Extreme Rotation (rotasi ekstrim)', 'extreme_rotation')
        ],
        value=['combined'],
        description='Tipe Augmentasi:',
        layout=widgets.Layout(width='100%', margin='10px 0'),
        style={'description_width': 'initial'}
    )
    
    # Jumlah variasi
    variations_slider = widgets.IntSlider(
        value=2,
        min=1,
        max=5,
        step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=SELECTION,
        style={'description_width': 'initial'}
    )
    
    # Opsi tambahan
    resume_checkbox = widgets.Checkbox(
        value=True,
        description='Resume jika terganggu',
        layout=SELECTION
    )
    
    validate_results_checkbox = widgets.Checkbox(
        value=True,
        description='Validasi hasil',
        layout=SELECTION
    )
    
    # Tombol augmentasi
    augment_button = widgets.Button(
        description='Augmentasi Dataset',
        button_style='primary',
        icon='random',
        layout=BUTTON
    )
    
    # Progress bar augmentasi
    augmentation_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Augmenting:',
        bar_style='info',
        orientation='horizontal',
        layout={'width': '100%', 'margin': '10px 0', 'visibility': 'hidden'}
    )
    
    # Status output augmentasi
    augmentation_status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Susun komponen augmentation tab
    augmentation_tab.children = [
        augmentation_header,
        augmentation_types,
        variations_slider,
        resume_checkbox,
        validate_results_checkbox,
        create_divider(),
        augment_button,
        augmentation_progress,
        augmentation_status
    ]
    
    # ==== Tab 4: Analisis Dataset ====
    analysis_tab = widgets.VBox()
    
    # Header analisis
    analysis_header = widgets.HTML("<h3>📈 Analisis Dataset</h3>")
    
    # Pilihan split untuk analisis
    analysis_split_selector = widgets.Dropdown(
        options=[('Training', 'train'), ('Validation', 'valid'), ('Testing', 'test')],
        value='train',
        description='Dataset Split:',
        layout=SELECTION,
        style={'description_width': 'initial'}
    )
    
    # Opsi analisis
    detailed_checkbox = widgets.Checkbox(
        value=True,
        description='Analisis detail',
        layout=SELECTION
    )
    
    visualize_distribution_checkbox = widgets.Checkbox(
        value=True,
        description='Visualisasi distribusi',
        layout=SELECTION
    )
    
    # Tombol analisis
    analyze_button = widgets.Button(
        description='Analisis Dataset',
        button_style='primary',
        icon='bar-chart',
        layout=BUTTON
    )
    
    # Status output analisis
    analysis_status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Susun komponen analysis tab
    analysis_tab.children = [
        analysis_header,
        analysis_split_selector,
        detailed_checkbox,
        visualize_distribution_checkbox,
        create_divider(),
        analyze_button,
        analysis_status
    ]
    
    # Susun tab
    tabs.children = [download_tab, validation_tab, augmentation_tab, analysis_tab]
    tabs.set_title(0, "📥 Download")
    tabs.set_title(1, "🔍 Validasi")
    tabs.set_title(2, "🔄 Augmentasi")
    tabs.set_title(3, "📈 Analisis")
    
    # Help accordion
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Pengelolaan Dataset</h4>
            <ol>
                <li><b>Download:</b> Unduh dataset dari Roboflow atau gunakan dataset lokal.</li>
                <li><b>Validasi:</b> Pastikan dataset memiliki format yang benar dan tidak ada masalah.</li>
                <li><b>Augmentasi:</b> Perkaya dataset dengan variasi gambar baru.</li>
                <li><b>Analisis:</b> Lihat distribusi dan statistik dataset.</li>
            </ol>
            
            <h4>Tips</h4>
            <ul>
                <li>Validasi dataset sebelum augmentasi untuk menghindari masalah.</li>
                <li>Gunakan kombinasi augmentasi untuk hasil terbaik.</li>
                <li>Analisis dataset setelah augmentasi untuk memastikan distribusi yang baik.</li>
            </ul>
        </div>
    """)], selected_index=None, layout=ACCORDION)
    
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Susun UI utama
    main.children = [header, tabs, help_info]
    
    # Return dictionary of components for handler
    return {
        'ui': main,
        'tabs': tabs,
        # Download components
        'source_radio': source_radio,
        'roboflow_card': roboflow_card,
        'roboflow_api_key': roboflow_api_key,
        'roboflow_workspace': roboflow_workspace,
        'roboflow_project': roboflow_project,
        'roboflow_version': roboflow_version,
        'download_button': download_button,
        'download_progress': download_progress,
        'download_status': download_status,
        # Validation components
        'split_selector': split_selector,
        'fix_checkbox': fix_checkbox,
        'visualize_checkbox': visualize_checkbox,
        'move_invalid_checkbox': move_invalid_checkbox,
        'validate_button': validate_button,
        'validation_progress': validation_progress,
        'validation_status': validation_status,
        # Augmentation components
        'augmentation_types': augmentation_types,
        'variations_slider': variations_slider,
        'resume_checkbox': resume_checkbox,
        'validate_results_checkbox': validate_results_checkbox,
        'augment_button': augment_button,
        'augmentation_progress': augmentation_progress,
        'augmentation_status': augmentation_status,
        # Analysis components
        'analysis_split_selector': analysis_split_selector,
        'detailed_checkbox': detailed_checkbox,
        'visualize_distribution_checkbox': visualize_distribution_checkbox,
        'analyze_button': analyze_button,
        'analysis_status': analysis_status
    }