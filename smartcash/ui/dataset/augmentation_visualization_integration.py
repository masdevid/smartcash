"""
File: smartcash/ui/dataset/augmentation_visualization_integration.py
Deskripsi: Integrasi visualisasi distribusi kelas dan class balancer ke dalam modul augmentasi
"""

from typing import Dict, Any

def integrate_visualization_to_augmentation(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Integrasi visualisasi distribusi kelas ke dalam modul augmentasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import handler visualisasi
    from smartcash.ui.visualization.visualization_integrator import setup_visualization_handlers
    
    # Setup handlers visualisasi
    ui_components = setup_visualization_handlers(ui_components, env, config)
    
    return ui_components

def update_augmentation_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Update augmentation handler dengan integrasi class balancer yang diperbaiki.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Tambahkan informasi balancing ke UI jika belum ada
        if 'aug_options' in ui_components and len(ui_components['aug_options'].children) <= 6:
            # Import widget untuk pembuatan UI
            import ipywidgets as widgets
            from smartcash.ui.utils.constants import COLORS, ICONS
            
            # Ambil existing children
            existing_children = list(ui_components['aug_options'].children)
            
            # Buat widget balancing
            balance_checkbox = widgets.Checkbox(
                value=True,
                description='Balance classes (target 1000 objek per kelas)',
                style={'description_width': 'initial'}
            )
            
            # Tambahkan checkbox balancing ke aug_options
            ui_components['aug_options'].children = tuple(existing_children + [balance_checkbox])
            
            if logger: logger.info(f"{ICONS.get('success', '✅')} Opsi balancing kelas berhasil ditambahkan ke UI")
            
        # Integrasikan visualisasi
        ui_components = integrate_visualization_to_augmentation(ui_components, env, config)
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Error integrasi class balancer: {str(e)}")
    
    return ui_components