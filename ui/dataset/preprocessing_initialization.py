"""
File: smartcash/ui/dataset/preprocessing_initialization.py
Deskripsi: Inisialisasi komponen untuk preprocessing dataset dengan penanganan path yang lebih baik dan UI yang terorganisir
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen preprocessing dataset dengan utilitas standar."""
    logger = ui_components.get('logger')
    
    try:
        # Deteksi environment dan konfigurasi awal
        drive_mounted = False
        drive_path = None
        smartcash_dir = None
        
        # Gunakan environment manager jika tersedia
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
            drive_mounted = True
            drive_path = str(env.drive_path) if hasattr(env, 'drive_path') else '/content/drive/MyDrive'
            smartcash_dir = f"{drive_path}/SmartCash"
        else:
            # Fallback ke deteksi manual
            try:
                from smartcash.ui.utils.drive_utils import detect_drive_mount
                drive_mounted, drive_path = detect_drive_mount()
                if drive_mounted and drive_path:
                    smartcash_dir = f"{drive_path}/SmartCash"
            except ImportError:
                pass
        
        # Dapatkan path dari config dengan prioritas
        if config and 'preprocessing' in config:
            data_dir = config.get('data', {}).get('dir', 'data')
            preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            
            # Gunakan Drive path jika tersedia dan config belum menggunakan path absolute
            if drive_mounted and smartcash_dir and not os.path.isabs(data_dir):
                data_dir = os.path.join(smartcash_dir, data_dir)
            
            if drive_mounted and smartcash_dir and not os.path.isabs(preprocessed_dir):
                preprocessed_dir = os.path.join(smartcash_dir, preprocessed_dir)
        elif drive_mounted and smartcash_dir:
            # Default ke Drive path jika tidak ada config
            data_dir = f"{smartcash_dir}/data"
            preprocessed_dir = f"{smartcash_dir}/data/preprocessed"
        else:
            # Fallback ke local path
            data_dir = "data"
            preprocessed_dir = "data/preprocessed"
        
        # Create path input container
        path_accordion = widgets.Accordion([
            widgets.VBox([
                widgets.Text(
                    value=data_dir,
                    description='Dataset Path:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='100%', margin='5px 0')
                ),
                widgets.Text(
                    value=preprocessed_dir,
                    description='Preprocessed Path:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='100%', margin='5px 0')
                ),
                widgets.Button(
                    description='Update Paths',
                    button_style='info',
                    icon='refresh',
                    layout=widgets.Layout(width='auto', margin='5px 0')
                )
            ])
        ])
        path_accordion.set_title(0, f"{ICONS['folder']} Data Paths (Klik untuk Edit)")
        path_accordion.selected_index = None  # Collapsed by default
        
        # Get input widgets
        path_input = path_accordion.children[0].children[0]
        preprocessed_input = path_accordion.children[0].children[1]
        update_path_button = path_accordion.children[0].children[2]
        
        # Handler untuk tombol update path
        def on_update_path(b):
            new_data_dir = path_input.value
            new_preprocessed_dir = preprocessed_input.value
            
            # Update ui_components dengan path baru (absolute untuk UI)
            ui_components['data_dir'] = new_data_dir
            ui_components['preprocessed_dir'] = new_preprocessed_dir
            
            # Konversi ke path relative untuk disimpan ke config jika absolute
            rel_data_dir = new_data_dir
            rel_preprocessed_dir = new_preprocessed_dir
            
            if drive_mounted and smartcash_dir and new_data_dir.startswith(smartcash_dir):
                rel_data_dir = os.path.relpath(new_data_dir, smartcash_dir)
            
            if drive_mounted and smartcash_dir and new_preprocessed_dir.startswith(smartcash_dir):
                rel_preprocessed_dir = os.path.relpath(new_preprocessed_dir, smartcash_dir)
                
            # Update config jika tersedia
            if config:
                if 'data' not in config:
                    config['data'] = {}
                config['data']['dir'] = rel_data_dir
                
                if 'preprocessing' not in config:
                    config['preprocessing'] = {}
                config['preprocessing']['output_dir'] = rel_preprocessed_dir
                
                # Coba simpan config
                try:
                    from smartcash.ui.dataset.preprocessing_config_handler import save_preprocessing_config
                    save_preprocessing_config(config)
                    if logger: logger.success(f"{ICONS['success']} Konfigurasi path berhasil disimpan")
                except Exception as e:
                    if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi: {str(e)}")
            
            # Update status panel
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Path dataset diperbarui: {new_data_dir}"
            )
            
            # Log perubahan
            if logger: 
                logger.success(f"{ICONS['success']} Path dataset diperbarui:")
                logger.info(f"🔍 Dataset Path: {new_data_dir}")
                logger.info(f"🔍 Preprocessed Path: {new_preprocessed_dir}")
        
        update_path_button.on_click(on_update_path)
        
        # Tambahkan info box tentang path dataset
        status_info = f"""
        <div style="margin:10px 0; padding:10px; border-left:4px solid {COLORS['info']}; background-color:{COLORS['alert_info_bg']}">
            <h4 style="margin-top:0; color:{COLORS['alert_info_text']}">{ICONS['info']} Informasi Dataset Path</h4>
            <p><strong>Dataset Path:</strong> {data_dir}</p>
            <p><strong>Preprocessed Path:</strong> {preprocessed_dir}</p>
            <p style="font-size:0.9em; margin-top:10px; color:{COLORS['muted']}">
                {ICONS['folder']} Dataset akan diproses dari sumber di atas. Klik pada panel "Data Paths" untuk mengubah.
            </p>
        </div>
        """
        
        # Tambahkan info path ke status panel
        update_status_panel(ui_components, "info", status_info, True)
        
        # Cek keberadaan dataset
        is_data_exist = os.path.exists(data_dir)
        is_preprocessed_exist = os.path.exists(preprocessed_dir)
        
        # Tambahkan peringatan jika path tidak valid
        if not is_data_exist:
            with ui_components.get('status'):
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator("warning", f"{ICONS['warning']} Dataset path tidak ditemukan: {data_dir}"))
        
        if is_preprocessed_exist:
            # Tampilkan tombol cleanup
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
                
            # Tampilkan tombol visualisasi
            for btn_name in ['visualize_button', 'compare_button', 'summary_button']:
                if btn_name in ui_components:
                    ui_components[btn_name].layout.display = 'inline-flex'
            
            if logger: logger.info(f"{ICONS['folder']} Dataset preprocessed terdeteksi di: {preprocessed_dir}")
        
        # Store path di ui_components
        ui_components.update({
            'data_dir': data_dir,
            'preprocessed_dir': preprocessed_dir,
            'path_input': path_input,
            'preprocessed_input': preprocessed_input,
            'update_path_button': update_path_button,
            'path_accordion': path_accordion,
            'on_update_path': on_update_path
        })
        
        # Tambahkan path panel ke UI jika belum ada
        if 'path_accordion' not in ui_components:
            if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
                children = list(ui_components['ui'].children)
                # Temukan posisi tepat setelah status_panel
                for i, child in enumerate(children):
                    if child == ui_components.get('status_panel'):
                        children.insert(i + 1, path_accordion)
                        ui_components['ui'].children = children
                        break
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error inisialisasi path: {str(e)}")
    
    return ui_components

def update_status_panel(ui_components, status_type, message, is_html=False):
    """Update status panel dengan pesan dan jenis status."""
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        if 'status_panel' in ui_components:
            if is_html:
                ui_components['status_panel'].value = message
            else:
                ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika alert_utils tidak tersedia
        if 'status_panel' in ui_components:
            if is_html:
                ui_components['status_panel'].value = message
            else:
                style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
                bg_color = style.get('bg_color', '#d1ecf1')
                text_color = style.get('text_color', '#0c5460')
                border_color = style.get('border_color', '#0c5460') 
                icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
                
                ui_components['status_panel'].value = f"""
                <div style="padding: 10px; background-color: {bg_color}; 
                            color: {text_color}; margin: 10px 0; border-radius: 4px; 
                            border-left: 4px solid {border_color};">
                    <p style="margin:5px 0">{icon} {message}</p>
                </div>
                """