"""
File: ui_components/augmentation_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk augmentasi data, termasuk kontrol parameter augmentasi dan visualisasi hasil.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import gc

def create_augmentation_controls():
    """
    Buat kontrol untuk parameter augmentasi data.
    
    Returns:
        Dictionary berisi widget untuk kontrol augmentasi
    """
    # Widget untuk memilih tipe augmentasi
    augmentation_type = widgets.RadioButtons(
        options=[
            ('Posisi (rotasi, perspektif, flip)', 'position'),
            ('Pencahayaan (kontras, bayangan, saturasi)', 'lighting'),
            ('Kombinasi (posisi dan pencahayaan)', 'combined'),
            ('Rotasi Ekstrim (untuk orientasi bervariasi)', 'extreme_rotation')
        ],
        value='combined',
        description='Tipe Augmentasi:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )
    
    # Widget untuk memilih split dataset
    split_selection = widgets.SelectMultiple(
        options=[('Train', 'train'), ('Validation', 'valid'), ('Test', 'test')],
        value=['train'],
        description='Dataset Split:',
        style={'description_width': 'initial'},
        layout={'width': '300px'}
    )
    
    # Slider untuk jumlah workers
    num_workers_slider = widgets.IntSlider(
        value=2,  # Nilai default lebih kecil untuk Colab
        min=1,
        max=4,   # Nilai maksimum lebih kecil untuk Colab
        step=1,
        description='Jumlah Workers:',
        style={'description_width': 'initial'}
    )
    
    # Slider untuk jumlah variasi
    num_variations_slider = widgets.IntSlider(
        value=2,
        min=1,
        max=5,
        step=1,
        description='Jumlah Variasi:',
        style={'description_width': 'initial'},
        tooltip='Jumlah variasi gambar yang akan dihasilkan untuk setiap gambar input'
    )
    
    return {
        'augmentation_type': augmentation_type,
        'split_selection': split_selection,
        'num_workers_slider': num_workers_slider,
        'num_variations_slider': num_variations_slider
    }

def create_augmentation_buttons():
    """
    Buat tombol-tombol untuk kontrol augmentasi data.
    
    Returns:
        Dictionary berisi tombol augmentasi
    """
    # Tombol untuk augmentasi data
    augment_button = widgets.Button(
        description='Augmentasi Data',
        button_style='primary',
        icon='plus'
    )
    
    # Tombol untuk membersihkan hasil augmentasi
    clean_button = widgets.Button(
        description='Bersihkan Augmentasi',
        button_style='danger',
        icon='trash-alt'
    )
    
    return {
        'augment_button': augment_button, 
        'clean_button': clean_button
    }

def create_augmentation_ui(is_colab=True):
    """
    Buat UI lengkap untuk augmentasi data.
    
    Args:
        is_colab: Boolean untuk menunjukkan apakah berjalan di Google Colab
    
    Returns:
        Dictionary berisi semua komponen UI untuk augmentasi data
    """
    # Buat kontrol parameter
    controls = create_augmentation_controls()
    
    # Buat tombol
    buttons = create_augmentation_buttons()
    
    # Buat output area
    augmentation_output = widgets.Output()
    
    # Buat peringatan khusus untuk Google Colab
    colab_warning = widgets.HTML(
        value="""
        <div style='background-color: #FFF5E4; color:#FFA725; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
            <b>⚠️ Catatan Khusus untuk Google Colab:</b><br>
            - Augmentasi dapat memakan waktu & memori lebih banyak dalam lingkungan Colab<br>
            - Gunakan worker dengan jumlah <b>1-2</b> untuk performa optimal<br>
            - Batasi augmentasi pada split kecil (Testing) terlebih dahulu sebelum memproses dataset besar<br>
            - Gunakan jumlah variasi yang lebih kecil (1-2) untuk mengurangi beban memori
        </div>
        """
    ) if is_colab else widgets.HTML("")
    
    # Gabungkan elemen-elemen dalam UI yang lengkap
    header = widgets.HTML("<h2>🔀 Augmentasi Data</h2>")
    description = widgets.HTML("<p>Augmentasi data membantu meningkatkan variasi dataset dan membantu model belajar lebih baik.</p>")
    
    settings_header = widgets.HTML("<h3>🔄 Pengaturan Augmentasi</h3>")
    
    # Tata letak UI
    control_layout = widgets.VBox([
        controls['augmentation_type'],
        widgets.HBox([
            controls['split_selection'], 
            widgets.VBox([
                controls['num_workers_slider'], 
                controls['num_variations_slider']
            ])
        ])
    ])
    
    button_layout = widgets.HBox([
        buttons['augment_button'],
        buttons['clean_button']
    ])
    
    # Gabungkan semuanya dalam layout utama
    main_ui = widgets.VBox([
        header,
        description,
        colab_warning,
        settings_header,
        control_layout,
        button_layout,
        augmentation_output
    ])
    
    # Return struktur UI dan komponen untuk handler
    return {
        'ui': main_ui,
        'output': augmentation_output,
        'augmentation_type': controls['augmentation_type'],
        'split_selection': controls['split_selection'],
        'num_workers_slider': controls['num_workers_slider'],
        'num_variations_slider': controls['num_variations_slider'],
        'augment_button': buttons['augment_button'],
        'clean_button': buttons['clean_button']
    }