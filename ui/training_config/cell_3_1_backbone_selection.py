"""
File: smartcash/ui/training_config/cell_3_1_backbone_selection.py
Deskripsi: Cell untuk pemilihan backbone dan konfigurasi layer model SmartCash dengan fallback sederhana
"""

# Import dari utility cell
from IPython.display import display, HTML

# Setup environment dan load config dengan error handling sederhana
try:
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    env, config = setup_notebook_environment("backbone_selection", "configs/model_config.yaml")
    ui_components = setup_ui_component(env, config, "backbone_selection")
    
    # Setup handler secara manual
    try:
        from smartcash.ui.training_config.backbone_selection_handler import setup_backbone_selection_handlers
        ui_components = setup_backbone_selection_handlers(ui_components, env, config)
    except ImportError as e:
        print(f"⚠️ Tidak dapat setup handler backbone_selection: {e}")
        
    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    # Fallback sederhana satu baris jika modul tidak tersedia
    print(f"⚠️ Error: {e}")
    display(HTML("""
    <div style="border:1px solid #ffc107; border-radius:5px; padding:15px; margin:10px 0; background:#fff3cd">
        <h3 style="margin-top:0">⚠️ Mode Terbatas</h3>
        <p>Komponen UI untuk pemilihan backbone tidak dapat dimuat.</p>
        <p>Pastikan repository SmartCash telah di-clone dengan benar dan semua dependencies telah diinstall.</p>
    </div>
    """))