"""
File: smartcash/ui/setup/dependency/components/input_options.py
Deskripsi: Komponen input untuk dependency installer
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

def create_package_selector_grid(config: Optional[Dict[str, Any]] = None) -> widgets.GridBox:
    """Membuat grid selector untuk package"""
    config = config or {}
    packages = config.get('packages', {})
    
    # Default layout untuk checkbox
    checkbox_layout = widgets.Layout(
        width='auto', 
        margin='0 10px 0 0'
    )
    
    # Default layout untuk label
    label_layout = widgets.Layout(
        width='auto'
    )
    
    # Buat checkbox untuk setiap package
    checkboxes = []
    labels = []
    descriptions = []
    
    for pkg_name, pkg_info in packages.items():
        if isinstance(pkg_info, dict):
            description = pkg_info.get('description', '')
            required = pkg_info.get('required', False)
            default = pkg_info.get('default', True)
        else:
            description = ''
            required = False
            default = True
        
        # Buat checkbox
        checkbox = widgets.Checkbox(
            value=default,
            description='',
            disabled=required,
            indent=False,
            layout=checkbox_layout
        )
        
        # Buat label
        label = widgets.HTML(
            f"<div style='font-weight: {'bold' if required else 'normal'};'>{pkg_name}</div>",
            layout=label_layout
        )
        
        # Buat deskripsi
        desc = widgets.HTML(
            f"<div style='color: #666; font-size: 0.9em;'>{description}</div>" if description else "",
            layout=widgets.Layout(width='auto')
        )
        
        checkboxes.append(checkbox)
        labels.append(label)
        descriptions.append(desc)
    
    # Buat grid
    items = []
    for i in range(len(checkboxes)):
        items.extend([checkboxes[i], labels[i], descriptions[i]])
    
    grid = widgets.GridBox(
        items,
        layout=widgets.Layout(
            grid_template_columns='auto auto 1fr',
            grid_gap='8px 12px',
            width='100%'
        )
    )
    
    return grid

def get_selected_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Mendapatkan daftar package yang dipilih"""
    selected_packages = []
    
    # Dapatkan grid dan config
    grid = ui_components.get('package_grid')
    config = ui_components.get('config', {})
    packages = config.get('packages', {})
    
    if not grid or not hasattr(grid, 'children'):
        return selected_packages
    
    # Dapatkan checkbox dari grid
    checkboxes = [child for child in grid.children if isinstance(child, widgets.Checkbox)]
    
    # Dapatkan nama package yang dipilih
    package_names = list(packages.keys())
    for i, checkbox in enumerate(checkboxes):
        if i < len(package_names) and checkbox.value:
            selected_packages.append(package_names[i])
    
    # Tambahkan custom packages jika ada
    custom_packages_input = ui_components.get('custom_packages')
    if custom_packages_input and hasattr(custom_packages_input, 'value'):
        custom_text = custom_packages_input.value.strip()
        if custom_text:
            custom_packages = [pkg.strip() for pkg in custom_text.split('\n') if pkg.strip()]
            selected_packages.extend(custom_packages)
    
    return selected_packages
