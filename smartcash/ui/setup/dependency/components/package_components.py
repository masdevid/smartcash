# =============================================================================
# File: smartcash/ui/setup/dependency/components/package_components.py
# Deskripsi: Komponen khusus untuk package management
# =============================================================================

import ipywidgets as widgets
from typing import Dict, Any, List
from smartcash.ui.components import create_card, create_checkbox
from smartcash.ui.setup.dependency.handlers.defaults import get_package_by_key

def create_package_item(package: Dict[str, Any], is_selected: bool = False) -> widgets.Widget:
    """Create single package item dengan status indicators"""
    checkbox = create_checkbox(f"pkg_{package['key']}", package['name'], is_selected, disabled=package.get('required', False))
    
    indicators = []
    if package.get('required', False): indicators.append("🔒 Required")
    if package.get('default', False): indicators.append("⭐ Default")
    if package.get('installed', False): indicators.append("✅ Installed")
    if package.get('update_available', False): indicators.append("🆙 Update Available")
    
    description = widgets.HTML(value=f"""
    <div style='margin-left: 25px; padding: 8px; background: #f8f9fa; border-radius: 4px;'>
        <p style='margin: 0 0 5px 0; color: #495057; font-size: 0.85em;'>{package['description']}</p>
        <span style='color: #6c757d; font-family: monospace; font-size: 0.75em;'>{package['pip_name']}</span>
        {' | ' + ' | '.join(indicators) if indicators else ''}
    </div>
    """)
    
    return widgets.VBox([checkbox, description])

def create_category_section(category: Dict[str, Any], selected_packages: List[str]) -> widgets.Widget:
    """Create category section dengan packages"""
    header = widgets.HTML(value=f"""
    <h4 style='margin: 10px 0 5px 0; color: #333;'>{category['icon']} {category['name']}</h4>
    <p style='margin: 0 0 10px 0; color: #666; font-size: 0.9em;'>{category['description']}</p>
    """)
    
    package_widgets = [create_package_item(pkg, pkg['key'] in selected_packages) for pkg in category['packages']]
    return widgets.VBox([header, widgets.VBox(package_widgets, layout=widgets.Layout(padding='0 0 0 20px'))])
