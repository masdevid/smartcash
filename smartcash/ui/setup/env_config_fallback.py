"""
File: smartcash/ui/01_setup/env_config_fallback.py
Deskripsi: Mode fallback untuk konfigurasi environment SmartCash jika terjadi error saat loading komponen utama
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any

def create_fallback_ui(error: Exception = None) -> Dict[str, Any]:
    """
    Buat UI fallback untuk kasus dimana komponen utama tidak dapat dimuat.
    
    Args:
        error: Exception yang terjadi (opsional)
    
    Returns:
        Dictionary berisi widget UI fallback
    """
    # Header dengan informasi error
    header = widgets.HTML(
        value=f"""<h2 style="color: #2c3e50; margin: 20px 0 10px 0;">🛠️ Konfigurasi Environment (Mode Terbatas)</h2>
               <p style="color: #7f8c8d;">Beberapa fitur mungkin tidak tersedia karena error saat memuat komponen.</p>"""
    )
    
    # Tambahkan pesan error jika ada
    error_widget = None
    if error:
        error_widget = widgets.HTML(
            value=f"""<div style="padding: 10px; background-color: #f8d7da; 
                    color: #721c24; border-left: 4px solid #721c24; 
                    border-radius: 4px; margin: 10px 0;">
                    <p><strong>Error:</strong> {str(error)}</p>
                </div>"""
        )
    
    # Cek apakah di Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    # Output widget
    output = widgets.Output()
    
    # Container untuk tombol
    button_container = widgets.HBox([])
    
    # Tombol mount drive untuk Colab
    if is_colab:
        mount_button = widgets.Button(
            description="🔗 Mount Google Drive", 
            button_style="info",
            layout=widgets.Layout(margin='5px')
        )
        
        def on_mount_click(b):
            try:
                with output:
                    clear_output(wait=True)
                    from google.colab import drive
                    drive.mount('/content/drive')
                    display(HTML("<p style='color: green'>✅ Google Drive berhasil terhubung</p>"))
            except Exception as e:
                with output:
                    clear_output(wait=True)
                    display(HTML(f"<p style='color: red'>❌ Error: {str(e)}</p>"))
        
        mount_button.on_click(on_mount_click)
        button_container.children = [mount_button]
    
    # Tombol setup direktori
    setup_dir_button = widgets.Button(
        description="📁 Setup Direktori", 
        button_style="primary",
        layout=widgets.Layout(margin='5px')
    )
    
    def on_setup_dir_click(b):
        try:
            with output:
                clear_output(wait=True)
                import os
                
                # Direktori yang diperlukan
                dirs = [
                    'data/train', 'data/valid', 'data/test',
                    'data/preprocessed/train', 'data/preprocessed/valid', 'data/preprocessed/test',
                    'configs', 'logs', 'runs', 'visualizations'
                ]
                
                # Buat direktori
                for directory in dirs:
                    os.makedirs(directory, exist_ok=True)
                    display(HTML(f"<p>✅ Direktori <code>{directory}</code> berhasil dibuat</p>"))
                
                display(HTML("<p style='color: green; font-weight: bold;'>✅ Semua direktori berhasil dibuat</p>"))
        except Exception as e:
            with output:
                clear_output(wait=True)
                display(HTML(f"<p style='color: red'>❌ Error: {str(e)}</p>"))
    
    setup_dir_button.on_click(on_setup_dir_click)
    
    # Tambahkan tombol setup direktori ke container
    children = list(button_container.children)
    children.append(setup_dir_button)
    button_container.children = tuple(children)
    
    # Buat widgets list untuk ui
    widgets_list = [header]
    if error_widget:
        widgets_list.append(error_widget)
    
    widgets_list.extend([button_container, output])
    
    # Gabungkan semua komponen
    ui = widgets.VBox(widgets_list)
    
    # Return komponen UI
    return {
        'ui': ui,
        'output': output,
        'header': header,
        'button_container': button_container
    }

def handle_fallback_environment(error: Exception = None) -> Dict[str, Any]:
    """
    Handler fallback untuk konfigurasi environment.
    
    Args:
        error: Exception yang terjadi (opsional)
        
    Returns:
        Dictionary berisi widget UI fallback yang sudah di-display
    """
    # Buat UI fallback
    ui_components = create_fallback_ui(error)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components