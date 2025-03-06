import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

def create_dataset_ui(drive_path=None, is_colab=False):
    """Create UI components for dataset management.
    
    Args:
        drive_path: Path to Google Drive if mounted
        is_colab: Boolean indicating if running in Colab
        
    Returns:
        Dictionary of UI components
    """
    # Storage options radio buttons
    storage_options = widgets.RadioButtons(
        options=[('Lokal', 'local'), ('Google Drive', 'drive')],
        value='local' if drive_path is None else 'drive',
        description='Simpan dataset di:',
        disabled=not is_colab or drive_path is None,
        layout={'width': 'max-content'},
        style={'description_width': 'initial'}
    )
    
    # Force download checkbox
    force_download_checkbox = widgets.Checkbox(
        value=False,
        description='Paksa download ulang',
        disabled=False
    )
    
    # Download button
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='info',
        icon='download'
    )
    
    # Cleanup button
    cleanup_button = widgets.Button(
        description='Bersihkan File Sementara',
        button_style='warning',
        icon='trash'
    )
    
    # Check status button
    check_status_button = widgets.Button(
        description='Cek Status Dataset',
        button_style='primary',
        icon='search'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Create headers and descriptions
    storage_header = widgets.HTML("<h3>🗄️ Konfigurasi Penyimpanan Dataset</h3>")
    download_header = widgets.HTML("<h3>📥 Download dan Pengelolaan Dataset</h3>")
    tips_text = widgets.HTML("""
    <div style="margin-top: 20px;">
    <p><strong>💡 Tips:</strong></p>
    <ol>
        <li>Menggunakan Google Drive akan menyimpan dataset di Drive sehingga tersedia di sesi Colab berikutnya</li>
        <li>Centang 'Paksa download ulang' jika ingin mengunduh dataset baru meskipun sudah ada yang tersedia</li>
        <li>Gunakan 'Cek Status Dataset' untuk memverifikasi keberadaan dan ukuran dataset</li>
    </ol>
    </div>
    """)
    
    # Assemble UI components
    ui = widgets.VBox([
        storage_header,
        storage_options,
        download_header,
        widgets.VBox([
            force_download_checkbox,
            widgets.HBox([download_button, cleanup_button, check_status_button])
        ]),
        output_area,
        tips_text
    ])
    
    # Return all components in a dictionary
    return {
        'ui': ui,
        'storage_options': storage_options,
        'force_download_checkbox': force_download_checkbox,
        'download_button': download_button,
        'cleanup_button': cleanup_button,
        'check_status_button': check_status_button,
        'output_area': output_area
    }