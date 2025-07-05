"""
File: smartcash/ui/setup/dependency/components/custom_packages_tab.py
Deskripsi: Tab untuk custom packages management
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.components.form_container import create_form_container, LayoutType

def create_custom_packages_tab(config: Dict[str, Any], logger) -> widgets.VBox:
    """Create tab untuk custom packages"""
    
    custom_packages = config.get('custom_packages', '')
    
    # Create form container with column layout
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding='20px',
        gap='16px'
    )
    
    # Add header
    header_html = """
    <div style="margin-bottom: 10px;">
        <h3 style="color: #333; margin: 0 0 10px 0;">🛠️ Custom Packages</h3>
        <p style="color: #666; margin: 0;">Tambahkan packages custom yang tidak tersedia di categories default.</p>
    </div>
    """
    form_container['add_item'](widgets.HTML(header_html), height='auto')
    
    # Add instructions
    instructions_html = """
    <div style="
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
    ">
        <h4 style="color: #495057; margin: 0 0 10px 0;">📋 Format Input</h4>
        <ul style="margin: 0; padding-left: 20px; color: #6c757d;">
            <li>Satu package per baris</li>
            <li>Format: <code>package_name</code> atau <code>package_name==version</code></li>
            <li>Contoh: <code>numpy>=1.20.0</code>, <code>matplotlib</code></li>
            <li>Untuk git: <code>git+https://github.com/user/repo.git</code></li>
        </ul>
    </div>
    """
    form_container['add_item'](widgets.HTML(instructions_html), height='auto')
    
    # Create text area with button row
    text_area_container = widgets.VBox([
        widgets.Textarea(
            value=custom_packages,
            placeholder="Masukkan packages custom di sini...\nContoh:\nnumpy>=1.20.0\nmatplotlib\ngit+https://github.com/user/repo.git",
            layout=widgets.Layout(
                width='100%',
                height='300px',
                border='1px solid #ddd',
                border_radius='8px',
                padding='10px',
                margin='0 0 10px 0'
            )
        ),
        widgets.HBox([
            widgets.Button(
                description='Parse Packages',
                button_style='info',
                icon='check',
                layout=widgets.Layout(width='150px')
            )
        ], layout=widgets.Layout(justify_content='flex-end'))
    ])
    
    # Add text area container to form
    form_container['add_item'](text_area_container, height='auto')
    
    # Get references to widgets
    packages_textarea = text_area_container.children[0]
    parse_button = text_area_container.children[1].children[0]
    
    # Output area for parsed packages
    output_area = widgets.Output()
    form_container['add_item'](output_area, height='auto')
    
    def on_parse_button_clicked(b):
        with output_area:
            output_area.clear_output()
            packages = parse_custom_packages(packages_textarea.value)
            if packages:
                output_area.append_display_data(widgets.HTML(create_parsed_packages_html(packages)))
    
    parse_button.on_click(on_parse_button_clicked)
    # Button handlers
    
    # Create container with max width
    container = widgets.VBox([
        form_container['container']
    ], layout=widgets.Layout(
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='0'
    ))
    
    # Store references for external access
    container.packages_textarea = packages_textarea
    container.output_area = output_area
    return container

def parse_custom_packages(packages_text: str) -> list:
    """Parse custom packages dari text input"""
    packages = []
    
    for line in packages_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Basic validation
            if any(char in line for char in ['>', '<', '=', '!']) or line.startswith('git+'):
                packages.append({
                    'raw': line,
                    'name': extract_package_name(line),
                    'version_spec': extract_version_spec(line),
                    'is_git': line.startswith('git+')
                })
            else:
                # Simple package name
                packages.append({
                    'raw': line,
                    'name': line,
                    'version_spec': '',
                    'is_git': False
                })
    
    return packages

def extract_package_name(package_spec: str) -> str:
    """Extract package name dari spec"""
    if package_spec.startswith('git+'):
        # Git package
        return package_spec.split('/')[-1].replace('.git', '')
    
    # Regular package
    for separator in ['==', '>=', '<=', '>', '<', '!=']:
        if separator in package_spec:
            return package_spec.split(separator)[0].strip()
    
    return package_spec.strip()

def extract_version_spec(package_spec: str) -> str:
    """Extract version specification"""
    if package_spec.startswith('git+'):
        return 'git'
    
    for separator in ['==', '>=', '<=', '>', '<', '!=']:
        if separator in package_spec:
            return package_spec.split(separator, 1)[1].strip()
    
    return ''

def create_parsed_packages_html(packages: list) -> str:
    """Create HTML untuk parsed packages"""
    if not packages:
        return ""
    
    html = "<div style='background: #f8f9fa; border-radius: 8px; padding: 15px;'>"
    html += "<h4 style='color: #495057; margin: 0 0 15px 0;'>📦 Parsed Packages</h4>"
    
    for pkg in packages:
        icon = "🔗" if pkg['is_git'] else "📦"
        version_text = f" ({pkg['version_spec']})" if pkg['version_spec'] else ""
        
        html += f"""
        <div style='
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            padding: 8px;
            background: white;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        '>
            <span style='font-size: 16px;'>{icon}</span>
            <div>
                <strong>{pkg['name']}</strong>{version_text}
                <div style='color: #6c757d; font-size: 12px;'>{pkg['raw']}</div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html