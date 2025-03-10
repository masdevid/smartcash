# Cell 2.1 - Dataset Download
# Persiapan dataset untuk training model SmartCash dengan implementasi ObserverManager
# %%capture
import sys
from pathlib import Path
from IPython.display import display

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Baca konfigurasi jika tersedia
try:
    from smartcash.utils.config_manager import ConfigManager
    config_dir = Path("configs")
    config_file = config_dir / "base_config.yaml"
    
    # Pastikan direktori config ada
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Gunakan method statis dari ConfigManager untuk load config
    config = ConfigManager.load_config(
        filename=str(config_file), 
        fallback_to_pickle=True
    )
    
    if not config:
        print("⚠️ Konfigurasi tidak ditemukan, menggunakan konfigurasi default")
        config = {}
except Exception as e:
    print(f"ℹ️ Menggunakan konfigurasi default ({str(e)})")
    config = {}  # Gunakan konfigurasi default dalam handler

# Import komponen UI dan handler
try:
    from smartcash.ui_components.dataset_download import create_dataset_download_ui
    from smartcash.ui_handlers.dataset_download import setup_download_handlers
except ImportError as e:
    print(f"❌ Error: {str(e)}")
    print("🔄 Memuat fallback UI...")
    
    # Fallback jika komponen tidak tersedia
    import ipywidgets as widgets
    def create_dataset_download_ui():
        return {'ui': widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}
    
    def setup_download_handlers(ui_components, config=None):
        return ui_components

# Buat dan setup komponen UI
ui_components = create_dataset_download_ui()
ui_components = setup_download_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])

# Register cleanup function untuk dipanggil saat notebook dibersihkan
try:
    if 'cleanup' in ui_components:
        import atexit
        atexit.register(ui_components['cleanup'])
except:
    pass