"""
Cell: 0.1 - Sinkronisasi Konfigurasi dengan Google Drive
Deskripsi: Setup awal minimal dengan logika didelegasikan ke module
"""

# Import yang diperlukan
import sys
from IPython.display import display
import ipywidgets as widgets

# Tambahkan direktori proyek ke path
if '.' not in sys.path: sys.path.append('.')

# Coba import setup helper, atau install dependencies jika gagal
try:
    from smartcash.ui.setup.colab_initializer import initialize_environment
except ImportError:
    # Fallback: Install package dasar
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets", "tqdm", "pyyaml"])
    
    # Buat UI sederhana untuk status
    status = widgets.HTML("<h3 style='color:#2F58CD'>🚀 Menginisialisasi SmartCash...</h3>")
    progress = widgets.IntProgress(value=0, min=0, max=3, description='Mempersiapkan environment:', 
                                  style={'description_width': 'initial'})
    message = widgets.HTML("Menginstall modul dasar...")
    display(widgets.VBox([status, progress, message]))
    
    # Coba mount Drive jika di Colab
    try:
        if 'google.colab' in sys.modules:
            progress.value = 1
            message.value = "Menghubungkan Google Drive..."
            from google.colab import drive
            drive.mount('/content/drive')
            
        progress.value = 2
        message.value = "Silakan jalankan kembali cell ini setelah instalasi."
        progress.value = 3
    except:
        message.value = "⚠️ Gagal setup awal. Silakan jalankan cell instalasi awal terlebih dahulu."
else:
    # Jika import berhasil, jalankan inisialisasi
    initialize_environment()