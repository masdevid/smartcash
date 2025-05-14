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
            
            # Daftar backbone yang didukung
            backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
            
            # Daftar model yang dioptimalkan
            optimized_models = ModelManager.OPTIMIZED_MODELS
            import_success = True
        except ImportError:
            # Coba impor alternatif jika struktur modul berubah
            from smartcash.model.config.backbone_config import BackboneConfig
            backbone_options = list(BackboneConfig.BACKBONE_CONFIGS.keys())
            
            # Definisi model yang dioptimalkan secara manual
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
                },
                'efficient_advanced': {
                    'description': 'Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
                    'backbone': 'efficientnet_b4',
                    'use_attention': True,
                    'use_residual': True,
                    'use_ciou': True
                }
            }
            import_success = False
            print(f"⚠️ Menggunakan definisi model alternatif karena ModelManager tidak dapat diimpor")
    except Exception as e:
        # Fallback jika terjadi error saat mengakses ModelManager dan BackboneConfig
        print(f"⚠️ Error mengakses konfigurasi backbone: {str(e)}")
        backbone_options = ['efficientnet_b4', 'cspdarknet_s']
        
        # Buat optimized_models fallback
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
            },
            'efficient_basic': {
                'description': 'Model dasar tanpa optimasi khusus',
                'backbone': 'efficientnet_b4',
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
    default_model = config.get('model_type', 'efficient_optimized')
    if default_model not in model_options:
        default_model = 'efficient_optimized' if 'efficient_optimized' in model_options else model_options[0]
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
    
    # Fitur optimasi
    ui_components['use_attention'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_attention'],
        description='Gunakan FeatureAdapter (Attention)',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
    )
    
    ui_components['use_residual'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_residual'],
        description='Gunakan ResidualAdapter',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
    )
    
    ui_components['use_ciou'] = widgets.Checkbox(
        value=optimized_models['efficient_optimized']['use_ciou'],
        description='Gunakan CIoU Loss',
        style={'description_width': '250px'},
        layout=widgets.Layout(width='400px'),
        disabled=True  # Dinonaktifkan karena akan otomatis diupdate
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
    
    # Card untuk fitur optimasi
    optimization_card = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin-top:0;'>{ICONS.get('optimization', '⚙️')} Fitur Optimasi</h4>"),
        ui_components['use_attention'],
        ui_components['use_residual'],
        ui_components['use_ciou']
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='5px',
        padding='10px',
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
    
    # Baris ketiga: Freeze layers dan fitur optimasi
    feature_row = widgets.HBox([
        widgets.VBox([
            ui_components['freeze_layers']
        ], layout=widgets.Layout(width='50%', padding='5px')),
        widgets.VBox([
            widgets.HBox([
                ui_components['use_attention'],
                ui_components['use_residual'],
                ui_components['use_ciou']
            ])
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
                    print(f"⚠️ Model {model_key} tidak ditemukan dalam daftar model yang dioptimalkan")
                return
                
            model_config = optimized_models[model_key]
            
            # Update backbone dengan penanganan error
            try:
                backbone_value = model_config.get('backbone', 'efficientnet_b4')
                
                # Pastikan backbone ada dalam opsi dropdown
                if backbone_value in ui_components['backbone_type'].options:
                    # Simpan status disabled saat ini
                    was_disabled = ui_components['backbone_type'].disabled
                    
                    # Aktifkan sementara jika dinonaktifkan
                    if was_disabled:
                        ui_components['backbone_type'].disabled = False
                        
                    # Update nilai
                    ui_components['backbone_type'].value = backbone_value
                    
                    # Kembalikan status disabled
                    if was_disabled:
                        ui_components['backbone_type'].disabled = was_disabled
                else:
                    with ui_components['status']:
                        print(f"⚠️ Backbone {backbone_value} tidak ditemukan dalam opsi dropdown")
            except Exception as e:
                with ui_components['status']:
                    print(f"❌ Error saat mengupdate backbone: {str(e)}")
            
            # Update fitur optimasi dengan penanganan error
            try:
                ui_components['use_attention'].value = model_config.get('use_attention', False)
                ui_components['use_residual'].value = model_config.get('use_residual', False)
                ui_components['use_ciou'].value = model_config.get('use_ciou', False)
            except Exception as e:
                with ui_components['status']:
                    print(f"❌ Error saat mengupdate fitur optimasi: {str(e)}")
                    
            # Update informasi backbone
            try:
                backbone_info = f"""
                <div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;'>
                    <h4>{model_key.replace('_', ' ').title()}</h4>
                    <p><strong>Deskripsi:</strong> {model_config['description']}</p>
                    <p><strong>Backbone:</strong> {model_config['backbone']}</p>
                    <p><strong>Fitur Optimasi:</strong></p>
                    <ul>
                        <li>FeatureAdapter (Attention): {'✅ Aktif' if model_config.get('use_attention', False) else '❌ Tidak aktif'}</li>
                        <li>ResidualAdapter: {'✅ Aktif' if model_config.get('use_residual', False) else '❌ Tidak aktif'}</li>
                        <li>CIoU Loss: {'✅ Aktif' if model_config.get('use_ciou', False) else '❌ Tidak aktif'}</li>
                    </ul>
                </div>
                """
                ui_components['backbone_info'].value = backbone_info
            except Exception as e:
                with ui_components['status']:
                    print(f"❌ Error saat mengupdate informasi backbone: {str(e)}")
        except Exception as e:
            with ui_components['status']:
                print(f"❌ Error umum saat mengubah model: {str(e)}")
        
        # Catatan: Bagian ini telah dipindahkan ke dalam blok try-except di atas
    
    # Daftarkan handler
    ui_components['model_type'].observe(on_model_change, names='value')
    
    # Trigger handler untuk inisialisasi awal
    on_model_change({'new': ui_components['model_type'].value})
    
    return ui_components
