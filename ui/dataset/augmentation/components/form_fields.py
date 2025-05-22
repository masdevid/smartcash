"""
File: smartcash/ui/dataset/augmentation/components/form_fields.py
Deskripsi: Form fields untuk konfigurasi augmentasi dataset (tanpa move to preprocessed)
"""

import ipywidgets as widgets

def num_variations_field(config: dict) -> widgets.IntSlider:
    """Field untuk jumlah variasi augmentasi."""
    return widgets.IntSlider(
        value=config.get('num_variations', 3),
        min=1,
        max=10,
        step=1,
        description='Jumlah Variasi:',
        tooltip='Jumlah variasi augmentasi per gambar',
        layout=widgets.Layout(width='100%')
    )

def target_count_field(config: dict) -> widgets.IntSlider:
    """Field untuk target jumlah gambar per kelas."""
    return widgets.IntSlider(
        value=config.get('target_count', 1000),
        min=100,
        max=5000,
        step=100,
        description='Target per Kelas:',
        tooltip='Target jumlah gambar per kelas setelah augmentasi',
        layout=widgets.Layout(width='100%')
    )

def output_prefix_field(config: dict) -> widgets.Text:
    """Field untuk prefix output file."""
    return widgets.Text(
        value=config.get('output_prefix', 'aug'),
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        tooltip='Prefix yang ditambahkan ke nama file hasil augmentasi',
        layout=widgets.Layout(width='100%')
    )

def augmentation_types_field(config: dict) -> widgets.SelectMultiple:
    """Field untuk pemilihan jenis augmentasi."""
    return widgets.SelectMultiple(
        options=[
            ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
            ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
            ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
        ],
        value=config.get('types', ['combined']),
        description='Jenis Augmentasi:',
        tooltip='Pilih jenis augmentasi yang akan diterapkan',
        layout=widgets.Layout(width='100%', height='100px')
    )

def split_target_field(config: dict) -> widgets.Dropdown:
    """Field untuk target split dataset."""
    return widgets.Dropdown(
        options=['train', 'valid', 'test'],
        value=config.get('target_split', 'train'),
        description='Target Split:',
        tooltip='Split dataset yang akan diaugmentasi',
        layout=widgets.Layout(width='100%')
    )

def balance_classes_field(config: dict) -> widgets.Checkbox:
    """Field untuk balancing kelas."""
    return widgets.Checkbox(
        value=config.get('balance_classes', True),
        description='Balance Classes',
        tooltip='Seimbangkan jumlah gambar per kelas',
        layout=widgets.Layout(width='auto')
    )

def validate_results_field(config: dict) -> widgets.Checkbox:
    """Field untuk validasi hasil."""
    return widgets.Checkbox(
        value=config.get('validate_results', True),
        description='Validasi Hasil',
        tooltip='Validasi hasil augmentasi setelah proses selesai',
        layout=widgets.Layout(width='auto')
    )