"""
File: smartcash/ui/training_config/cell_3_3_training_strategy.py
Deskripsi: Cell untuk konfigurasi strategi training model SmartCash dengan fallback sederhana
"""

# Import dari utility cell
from IPython.display import display, HTML

# Setup environment dan load config
try:
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    env, config = setup_notebook_environment("training_strategy", "configs/training_config.yaml")
    ui_components = setup_ui_component(env, config, "training_strategy")
    
    # Setup handler secara manual
    try:
        from smartcash.ui.training_config.training_strategy_handler import setup_training_strategy_handlers
        ui_components = setup_training_strategy_handlers(ui_components, env, config)
    except ImportError as e:
        print(f"⚠️ Tidak dapat setup handler training_strategy: {e}")
        
    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    # Fallback sederhana satu baris jika modul tidak tersedia
    print(f"⚠️ Error: {e}")
    display(HTML("""
    <div style="border:1px solid #ffc107; border-radius:5px; padding:15px; margin:10px 0; background:#fff3cd">
        <h3 style="margin-top:0">⚠️ Mode Terbatas</h3>
        <p>Komponen UI untuk konfigurasi strategi training tidak dapat dimuat.</p>
        <p>Pastikan repository SmartCash telah di-clone dengan benar dan semua dependencies telah diinstall.</p>
    </div>
    """))