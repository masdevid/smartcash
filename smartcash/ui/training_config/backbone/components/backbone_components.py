"""
File: smartcash/ui/training_config/backbone/components/backbone_components.py
Deskripsi: Komponen UI untuk pemilihan backbone model
"""

from typing import Dict, Any, List, Optional
import ipywidgets as widgets

def create_backbone_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk pemilihan backbone model.
    
    Args:
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, create_divider
    
    # Inisialisasi komponen
    ui_components = {}
    
    # Tambahkan komponen status
    ui_components['status'] = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Import ModelManager untuk mendapatkan model yang dioptimalkan
    try:
        # Coba impor dari lokasi yang benar
        try:
            from smartcash.model.manager import ModelManager
            from smartcash.model.config.backbone_config import BackboneConfig
            
            # Daftar backbone yang didukung - hanya menggunakan yang tersedia di BackboneConfig
            backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
            
            # Pastikan backbone_options tidak kosong
            if not backbone_options:
                backbone_options = ['efficientnet_b4', 'cspdarknet_s']
                # Log ke UI components status jika tersedia
                if 'status' in ui_components:
                    with ui_components['status']:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        from IPython.display import display
                        display(create_status_indicator("warning", "BackboneConfig.BACKBONE_CONFIGS kosong, menggunakan opsi default"))
                else:
                    print(f"⚠️ BackboneConfig.BACKBONE_CONFIGS kosong, menggunakan opsi default")
            
            # Daftar model yang dioptimalkan - hanya menyimpan model yang diperlukan
            optimized_models = {}
            
            # Salin model dari ModelManager.OPTIMIZED_MODELS dengan validasi backbone (hanya model yang dibutuhkan)
            allowed_models = ['yolov5s', 'efficient_basic', 'efficient_optimized', 'efficient_advanced']
            for model_key, model_config in ModelManager.OPTIMIZED_MODELS.items():
                # Hanya menyimpan model yang diizinkan (menghapus efficient_experiment)
                if model_key not in allowed_models:
                    continue
                    
                # Salin konfigurasi model
                optimized_models[model_key] = model_config.copy()
                
                # Validasi backbone ada dalam opsi yang tersedia
                if 'backbone' in model_config and model_config['backbone'] not in backbone_options:
                    # Jika backbone tidak valid, gunakan default yang tersedia
                    # Log ke UI components status jika tersedia
                    if 'status' in ui_components:
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("warning", f"Backbone '{model_config['backbone']}' untuk model '{model_key}' tidak tersedia dalam opsi"))
                    else:
                        print(f"⚠️ Backbone '{model_config['backbone']}' untuk model '{model_key}' tidak tersedia dalam opsi")
                    if 'efficientnet_b4' in backbone_options:
                        optimized_models[model_key]['backbone'] = 'efficientnet_b4'
                        print(f"  ℹ️ Menggunakan 'efficientnet_b4' sebagai pengganti")
                    elif 'cspdarknet_s' in backbone_options:
                        optimized_models[model_key]['backbone'] = 'cspdarknet_s'
                        print(f"  ℹ️ Menggunakan 'cspdarknet_s' sebagai pengganti")
                    elif backbone_options:
                        optimized_models[model_key]['backbone'] = backbone_options[0]
                        # Log ke UI components status jika tersedia
                    if 'status' in ui_components:
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("info", f"Menggunakan '{backbone_options[0]}' sebagai pengganti"))
                    else:
                        print(f"  ℹ️ Menggunakan '{backbone_options[0]}' sebagai pengganti")
            
            # Jika tidak ada model yang dioptimalkan, buat default
            if not optimized_models:
                # Log ke UI components status jika tersedia
                if 'status' in ui_components:
                    with ui_components['status']:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        from IPython.display import display
                        display(create_status_indicator("warning", "ModelManager.OPTIMIZED_MODELS kosong, menggunakan model default"))
                else:
                    print(f"⚠️ ModelManager.OPTIMIZED_MODELS kosong, menggunakan model default")
                optimized_models = {
                    'efficient_optimized': {
                        'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                        'backbone': 'efficientnet_b4' if 'efficientnet_b4' in backbone_options else backbone_options[0],
                        'use_attention': True,
                        'use_residual': False,
                        'use_ciou': False
                    },
                    'yolov5s': {
                        'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
                        'backbone': 'cspdarknet_s' if 'cspdarknet_s' in backbone_options else backbone_options[0],
                        'use_attention': False,
                        'use_residual': False,
                        'use_ciou': False
                    }
                }
        except ImportError as ie:
            # Coba impor alternatif jika struktur modul berubah
            try:
                from smartcash.model.config.backbone_config import BackboneConfig
                backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
                
                # Pastikan backbone_options tidak kosong
                if not backbone_options:
                    backbone_options = ['efficientnet_b4', 'cspdarknet_s']
                    print(f"⚠️ BackboneConfig.BACKBONE_CONFIGS kosong, menggunakan opsi default")
                
                # Definisi model yang dioptimalkan secara manual dengan backbone yang valid
                optimized_models = {
                    'efficient_optimized': {
                        'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                        'backbone': 'efficientnet_b4' if 'efficientnet_b4' in backbone_options else backbone_options[0],
                        'use_attention': True,
                        'use_residual': False,
                        'use_ciou': False
                    },
                    'yolov5s': {
                        'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
                        'backbone': 'cspdarknet_s' if 'cspdarknet_s' in backbone_options else backbone_options[0],
                        'use_attention': False,
                        'use_residual': False,
                        'use_ciou': False
                    },
                    'efficient_advanced': {
                        'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
                        'backbone': 'efficientnet_b4' if 'efficientnet_b4' in backbone_options else backbone_options[0],
                        'use_attention': True,
                        'use_residual': True,
                        'use_ciou': True
                    }
                }
                import_success = False
                # Log ke UI components status jika tersedia
                if 'status' in ui_components:
                    with ui_components['status']:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        from IPython.display import display
                        display(create_status_indicator("warning", f"Menggunakan definisi model alternatif karena ModelManager tidak dapat diimpor: {str(ie)}"))
                else:
                    print(f"⚠️ Menggunakan definisi model alternatif karena ModelManager tidak dapat diimpor: {str(ie)}")
            except Exception as inner_e:
                # Log ke UI components status jika tersedia
                if 'status' in ui_components:
                    with ui_components['status']:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        from IPython.display import display
                        display(create_status_indicator("error", f"Error saat mengakses BackboneConfig: {str(inner_e)}"))
                else:
                    print(f"⚠️ Error saat mengakses BackboneConfig: {str(inner_e)}")
                backbone_options = ['efficientnet_b4', 'cspdarknet_s']
                optimized_models = {
                    'efficient_optimized': {
                        'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                        'backbone': 'efficientnet_b4',
                        'use_attention': True,
                        'use_residual': False,
                        'use_ciou': False
                    },
                    'yolov5s': {
                        'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
                        'backbone': 'cspdarknet_s',
                        'use_attention': False,
                        'use_residual': False,
                        'use_ciou': False
                    }
                }
                import_success = False
    except Exception as e:
        # Fallback jika terjadi error saat mengakses ModelManager dan BackboneConfig
        # Log ke UI components status jika tersedia
        if 'status' in ui_components:
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                from IPython.display import display
                display(create_status_indicator("error", f"Error mengakses konfigurasi backbone: {str(e)}"))
        else:
            print(f"⚠️ Error mengakses konfigurasi backbone: {str(e)}")
        backbone_options = ['efficientnet_b4', 'cspdarknet_s']
        
        # Buat optimized_models fallback dengan backbone yang valid
        optimized_models = {
            'efficient_optimized': {
                'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': False,
                'use_ciou': False
            },
            'yolov5s': {
                'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
                'backbone': 'cspdarknet_s',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            }
        }
        import_success = False
    
    # Import konstanta untuk styling yang konsisten
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Buat komponen UI
    ui_components['title'] = widgets.HTML(
        value=f"<h3>{ICONS['model']} Konfigurasi Backbone Model</h3>"
    )
    
    # Pastikan model yang dioptimalkan memiliki kunci yang valid
    if 'yolov5s' not in optimized_models:
        # Tambahkan model yolov5s jika tidak ada
        optimized_models['yolov5s'] = {
            'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
            'backbone': 'cspdarknet_s',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        }
    
    # Pastikan efficient_optimized juga tersedia
    if 'efficient_optimized' not in optimized_models:
        optimized_models['efficient_optimized'] = {
            'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
            'backbone': 'efficientnet_b4',
            'use_attention': True,
            'use_residual': False,
            'use_ciou': False
        }
        
    # Tambahkan model efficient_basic sebagai model default
    if 'efficient_basic' not in optimized_models:
        optimized_models['efficient_basic'] = {
            'description': 'Model dasar dengan EfficientNet-B4 tanpa optimasi tambahan',
            'backbone': 'efficientnet_b4',
            'use_attention': False,
            'use_residual': False,
            'use_ciou': False
        }
    
    # Dropdown untuk memilih model yang dioptimalkan
    model_options = list(optimized_models.keys())
    
    # Pastikan model_options tidak kosong
    if not model_options or len(model_options) == 0:
        model_options = ['efficient_optimized', 'yolov5s']
        with ui_components['status']:
            print(f"⚠️ Tidak ada opsi model yang tersedia, menggunakan opsi default: {model_options}")
        
        # Jika optimized_models kosong, tambahkan model default
        if not optimized_models:
            optimized_models = {
                'efficient_optimized': {
                    'description': 'Model dengan EfficientNet-B4 dan FeatureAdapter',
                    'backbone': 'efficientnet_b4',
                    'use_attention': True,
                    'use_residual': False,
                    'use_ciou': False
                },
                'yolov5s': {
                    'description': 'YOLOv5s dengan CSPDarknet sebagai backbone',
                    'backbone': 'cspdarknet_s',
                    'use_attention': False,
                    'use_residual': False,
                    'use_ciou': False
                }
            }
    
    # Pastikan nilai default selalu ada dalam opsi
    default_model = config.get('model_type', 'efficient_basic')
    if default_model not in model_options:
        default_model = 'efficient_basic' if 'efficient_basic' in model_options else ('efficient_optimized' if 'efficient_optimized' in model_options else model_options[0])
        with ui_components['status']:
            print(f"⚠️ Model default tidak ditemukan dalam opsi, menggunakan: {default_model}")
    
    ui_components['model_type'] = widgets.Dropdown(
        options=model_options,
        value=default_model,
        description='Model:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Pastikan backbone_options berisi semua backbone yang digunakan
    if 'cspdarknet_s' not in backbone_options:
        backbone_options.append('cspdarknet_s')
    if 'efficientnet_b4' not in backbone_options:
        backbone_options.append('efficientnet_b4')
    
    # Dropdown backbone akan otomatis diupdate berdasarkan model yang dipilih
    # Pastikan opsi backbone tidak kosong dan selalu memiliki nilai default yang valid
    if not backbone_options or len(backbone_options) == 0:
        backbone_options = ['efficientnet_b4', 'cspdarknet_s']
        with ui_components['status']:
            print(f"⚠️ Tidak ada opsi backbone yang tersedia, menggunakan opsi default: {backbone_options}")
    
    # Pastikan backbone options hanya menggunakan format lowercase_underscore
    # Ini untuk mengatasi masalah dengan 'EfficientNet-B4' vs 'efficientnet_b4'
    backbone_options = [option.lower().replace('-', '_') for option in backbone_options]
    
    # Pastikan nilai default selalu ada dalam opsi
    default_backbone = config.get('backbone', 'efficientnet_b4')
    if default_backbone not in backbone_options:
        default_backbone = 'efficientnet_b4' if 'efficientnet_b4' in backbone_options else backbone_options[0]
        with ui_components['status']:
            print(f"⚠️ Backbone default tidak ditemukan dalam opsi, menggunakan: {default_backbone}")
    
    ui_components['backbone_type'] = widgets.Dropdown(
        options=backbone_options,
        value=default_backbone,
        description='Backbone:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%'),
        disabled=False  # Diubah menjadi tidak dinonaktifkan untuk menghindari masalah saat update nilai
    )
    
    ui_components['pretrained'] = widgets.Checkbox(
        value=True,
        description='Gunakan pretrained weights',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['freeze_backbone'] = widgets.Checkbox(
        value=True,
        description='Freeze backbone layers',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    ui_components['freeze_layers'] = widgets.IntSlider(
        value=3,
        min=0,
        max=5,
        step=1,
        description='Freeze layers:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Informasi backbone
    ui_components['backbone_info'] = widgets.HTML(
        value="<p>Informasi backbone akan ditampilkan di sini</p>"
    )
    
    # Buat tombol aksi
    ui_components['save_button'] = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='success',
        icon='save',
        tooltip='Simpan konfigurasi backbone',
        layout=widgets.Layout(width='200px')
    )
    
    ui_components['reset_button'] = widgets.Button(
        description='Reset',
        button_style='warning',
        icon='refresh',
        tooltip='Reset ke default',
        layout=widgets.Layout(width='100px')
    )
    
    # Status indicator
    ui_components['status'] = widgets.Output(
        layout=widgets.Layout(width='100%', padding='10px')
    )
    
    # Susun layout dengan cards
    
    # Card untuk model selection
    model_selection_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['model']} Pilihan Model</h4>"),
        ui_components['model_type']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Card untuk konfigurasi backbone
    backbone_config_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['settings']} Konfigurasi Backbone</h4>"),
        ui_components['backbone_type'],
        ui_components['pretrained'],
        ui_components['freeze_backbone'],
        ui_components['freeze_layers']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Buat info_box untuk informasi backbone
    ui_components['info_box'] = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS['info']} Informasi Model</h4>"),
        ui_components['backbone_info']
    ], layout=widgets.Layout(
        padding='10px', 
        border='1px solid #ddd', 
        border_radius='5px',
        margin='0 0 15px 0',
        background_color='#f8f9fa'
    ))
    
    # Buat layout yang lebih compact
    # Baris pertama: Model dan backbone selection
    model_row = widgets.HBox([
        widgets.VBox([
            ui_components['model_type']
        ], layout=widgets.Layout(width='50%', padding='5px')),
        widgets.VBox([
            ui_components['backbone_type']
        ], layout=widgets.Layout(width='50%', padding='5px'))
    ], layout=widgets.Layout(margin='0 0 10px 0'))
    
    # Baris kedua: Pretrained dan freeze backbone
    config_row = widgets.HBox([
        widgets.VBox([
            ui_components['pretrained']
        ], layout=widgets.Layout(width='50%', padding='5px')),
        widgets.VBox([
            ui_components['freeze_backbone']
        ], layout=widgets.Layout(width='50%', padding='5px'))
    ], layout=widgets.Layout(margin='0 0 10px 0'))
    
    # Baris ketiga: Freeze layers
    feature_row = widgets.HBox([
        widgets.VBox([
            ui_components['freeze_layers']
        ], layout=widgets.Layout(width='50%', padding='5px'))
    ], layout=widgets.Layout(margin='0 0 10px 0'))
    
    # Gabungkan semua dalam layout compact
    ui_components['form'] = widgets.VBox([
        model_row,
        config_row,
        feature_row,
        ui_components['info_box']
    ], layout=widgets.Layout(padding='5px'))
    
    # Placeholder untuk tombol konfigurasi yang akan ditambahkan dari initializer
    ui_components['buttons_placeholder'] = widgets.HBox(
        [],
        layout=widgets.Layout(padding='10px')
    )
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['model']} Backbone Model Selection", 
                          "Pilih backbone model dan konfigurasi layer untuk deteksi mata uang")
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi backbone model</p>
        </div>"""
    )
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[ui_components['status']], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Backbone Selection Logs")
    
    # Container utama
    ui_components['main_container'] = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Model Configuration</h4>"),
        ui_components['form'],
        create_divider(),
        ui_components['buttons_placeholder'],
        log_accordion
    ])
    
    # Tambahkan referensi komponen tambahan ke ui_components
    ui_components.update({
        'header': header,
        'status_panel': status_panel,
        'log_accordion': log_accordion,
        'module_name': 'backbone'
    })
    
    # Handler untuk mengupdate UI berdasarkan model yang dipilih
    def on_model_change(change):
        try:
            model_key = change['new']
            
            # Validasi model_key ada dalam optimized_models
            if model_key not in optimized_models:
                with ui_components['status']:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    from IPython.display import display
                    display(create_status_indicator("warning", f"Model {model_key} tidak ditemukan dalam daftar model yang dioptimalkan"))
                return
                
            model_config = optimized_models[model_key]
            
            # Update backbone dengan penanganan error yang lebih kuat
            try:
                backbone_value = model_config.get('backbone', 'efficientnet_b4')
                available_options = ui_components['backbone_type'].options
                
                # Periksa apakah backbone ada dalam opsi dropdown
                if backbone_value in available_options:
                    # Simpan status disabled saat ini
                    was_disabled = ui_components['backbone_type'].disabled
                    
                    try:
                        # Aktifkan sementara jika dinonaktifkan
                        if was_disabled:
                            ui_components['backbone_type'].disabled = False
                            
                        # Update nilai dengan penanganan error
                        ui_components['backbone_type'].value = backbone_value
                        
                        # Kembalikan status disabled
                        if was_disabled:
                            ui_components['backbone_type'].disabled = was_disabled
                            
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                    except Exception as set_error:
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("warning", f"Error saat mengatur nilai backbone: {str(set_error)}"))
                            display(create_status_indicator("info", "Mencoba metode alternatif..."))
                            
                        # Jika gagal, coba metode alternatif dengan membuat ulang dropdown
                        try:
                            # Buat dropdown baru dengan nilai yang benar
                            new_dropdown = widgets.Dropdown(
                                options=available_options,
                                value=backbone_value,
                                description='Backbone:',
                                style={'description_width': 'initial'},
                                layout=widgets.Layout(width='100%'),
                                disabled=was_disabled
                            )
                            
                            # Ganti dropdown lama dengan yang baru
                            ui_components['backbone_type'] = new_dropdown
                            with ui_components['status']:
                                from smartcash.ui.utils.alert_utils import create_status_indicator
                                from IPython.display import display
                                display(create_status_indicator("success", "Berhasil mengatur backbone dengan metode alternatif"))
                        except Exception as alt_error:
                            with ui_components['status']:
                                from smartcash.ui.utils.alert_utils import create_status_indicator
                                from IPython.display import display
                                display(create_status_indicator("error", f"Gagal mengatur backbone dengan metode alternatif: {str(alt_error)}"))
                else:
                    # Jika backbone tidak ada dalam opsi, gunakan nilai yang tersedia
                    with ui_components['status']:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        from IPython.display import display
                        display(create_status_indicator("warning", f"Backbone '{backbone_value}' tidak ditemukan dalam opsi dropdown"))
                    
                    # Cari backbone alternatif yang valid
                    if 'efficientnet_b4' in available_options:
                        alternative = 'efficientnet_b4'
                    elif 'cspdarknet_s' in available_options:
                        alternative = 'cspdarknet_s'
                    elif available_options:
                        alternative = available_options[0]
                    else:
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("error", "Tidak ada opsi backbone yang tersedia"))
                        return
                    
                    # Update model_config dengan backbone alternatif
                    model_config['backbone'] = alternative
                    optimized_models[model_key]['backbone'] = alternative
                    
                    # Coba set nilai dengan backbone alternatif
                    try:
                        was_disabled = ui_components['backbone_type'].disabled
                        if was_disabled:
                            ui_components['backbone_type'].disabled = False
                        ui_components['backbone_type'].value = alternative
                        if was_disabled:
                            ui_components['backbone_type'].disabled = was_disabled
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("info", f"Menggunakan backbone alternatif: {alternative}"))
                    except Exception as alt_error:
                        with ui_components['status']:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            from IPython.display import display
                            display(create_status_indicator("error", f"Error saat mengatur backbone alternatif: {str(alt_error)}"))
            except Exception as e:
                with ui_components['status']:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    from IPython.display import display
                    display(create_status_indicator("error", f"Error umum saat mengupdate backbone: {str(e)}"))
            
            # Fitur optimasi tidak lagi diupdate melalui UI karena checkbox dihapus
            # Namun kita tetap perlu mendapatkan nilai-nilai ini dari model_config untuk ditampilkan di informasi backbone
            use_attention = model_config.get('use_attention', False)
            use_residual = model_config.get('use_residual', False)
            use_ciou = model_config.get('use_ciou', False)
                    
            # Update informasi backbone
            try:
                backbone_info = f"""
                <div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;'>
                    <h4>{model_key.replace('_', ' ').title()}</h4>
                    <p><strong>Deskripsi:</strong> {model_config['description']}</p>
                    <p><strong>Backbone:</strong> {model_config['backbone']}</p>
                    <p><strong>Fitur Optimasi:</strong></p>
                    <ul>
                        <li>FeatureAdapter (Attention): {'✅ Aktif' if use_attention else '❌ Tidak aktif'}</li>
                        <li>ResidualAdapter: {'✅ Aktif' if use_residual else '❌ Tidak aktif'}</li>
                        <li>CIoU Loss: {'✅ Aktif' if use_ciou else '❌ Tidak aktif'}</li>
                    </ul>
                </div>
                """
                ui_components['backbone_info'].value = backbone_info
            except Exception as e:
                with ui_components['status']:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    from IPython.display import display
                    display(create_status_indicator("error", f"Error saat mengupdate informasi backbone: {str(e)}"))
        except Exception as e:
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                from IPython.display import display
                display(create_status_indicator("error", f"Error umum saat mengubah model: {str(e)}"))
        
        # Catatan: Bagian ini telah dipindahkan ke dalam blok try-except di atas
    
    # Daftarkan handler
    ui_components['model_type'].observe(on_model_change, names='value')
    
    # Trigger handler untuk inisialisasi awal
    on_model_change({'new': ui_components['model_type'].value})
    
    return ui_components
