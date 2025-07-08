"""
File: smartcash/ui/setup/dependency/components/custom_packages_tab.py
Deskripsi: Tab untuk custom packages management
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.components.form_container import create_form_container, LayoutType

def create_custom_packages_tab(config: Dict[str, Any], logger=None) -> widgets.VBox:
    """Create enhanced tab for custom packages with modern responsive design."""
    
    custom_packages = config.get('custom_packages', '')
    
    # Create responsive form container
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding='16px',
        gap='12px'
    )
    
    # Add compact header
    header_html = """
    <div style="margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 24px;">🛠️</span>
            <div>
                <h3 style="color: #333; margin: 0; font-size: 1.25rem;">Custom Packages</h3>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Add custom packages not available in default categories.</p>
            </div>
        </div>
    </div>
    """
    form_container['add_item'](widgets.HTML(header_html), height='auto')
    
    # Add compact instructions
    instructions_html = """
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #f1f3f4);
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    ">
        <div style="display: flex; align-items: flex-start; gap: 8px;">
            <span style="font-size: 16px; margin-top: 2px;">📋</span>
            <div style="flex: 1;">
                <h4 style="color: #495057; margin: 0 0 8px 0; font-size: 1rem;">Input Format</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; font-size: 0.85rem; color: #6c757d;">
                    <div>• One package per line</div>
                    <div>• Format: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">package==version</code></div>
                    <div>• Example: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">numpy>=1.20.0</code></div>
                    <div>• Git: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">git+https://...</code></div>
                </div>
            </div>
        </div>
    </div>
    """
    form_container['add_item'](widgets.HTML(instructions_html), height='auto')
    
    # Create enhanced text area with improved styling
    text_area_container = widgets.VBox([
        widgets.Textarea(
            value=custom_packages,
            placeholder="Enter custom packages here...\nExamples:\nnumpy>=1.20.0\nmatplotlib\ntensorflow==2.8.0\ngit+https://github.com/user/repo.git",
            layout=widgets.Layout(
                width='100%',
                height='280px',
                border='1px solid #e1e5e9',
                border_radius='8px',
                padding='12px',
                margin='0 0 8px 0',
                font_family='monospace'
            )
        ),
        widgets.HBox([
            widgets.Button(
                description='📦 Parse Packages',
                button_style='primary',
                icon='check-circle',
                layout=widgets.Layout(width='160px', height='32px')
            ),
            widgets.Button(
                description='🗑️ Clear',
                button_style='warning',
                icon='trash',
                layout=widgets.Layout(width='100px', height='32px')
            )
        ], layout=widgets.Layout(justify_content='flex-end', gap='8px'))
    ])
    
    # Add text area container to form
    form_container['add_item'](text_area_container, height='auto')
    
    # Get references to widgets
    packages_textarea = text_area_container.children[0]
    parse_button = text_area_container.children[1].children[0]
    clear_button = text_area_container.children[1].children[1]
    
    # Enhanced output area for parsed packages
    output_area = widgets.Output(
        layout=widgets.Layout(
            max_height='300px',
            overflow_y='auto',
            border='1px solid #e1e5e9',
            border_radius='8px',
            padding='8px'
        )
    )
    form_container['add_item'](output_area, height='auto')
    
    # Enhanced button handlers
    def on_parse_button_clicked(b):
        with output_area:
            output_area.clear_output()
            packages_text = packages_textarea.value.strip()
            
            if not packages_text:
                output_area.append_display_data(widgets.HTML("""
                    <div style="color: #6c757d; font-style: italic; text-align: center; padding: 20px;">
                        No packages to parse. Enter some package specifications above.
                    </div>
                """))
                return
                
            packages = parse_custom_packages(packages_text)
            if packages:
                output_area.append_display_data(widgets.HTML(create_enhanced_parsed_packages_html(packages)))
            else:
                output_area.append_display_data(widgets.HTML("""
                    <div style="color: #dc3545; text-align: center; padding: 20px;">
                        No valid packages found. Please check your input format.
                    </div>
                """))
    
    def on_clear_button_clicked(b):
        packages_textarea.value = ''
        output_area.clear_output()
    
    parse_button.on_click(on_parse_button_clicked)
    clear_button.on_click(on_clear_button_clicked)
    
    # Create responsive container
    container = widgets.VBox([
        form_container['container']
    ], layout=widgets.Layout(
        width='100%',
        max_width='1000px',
        margin='0 auto',
        padding='0'
    ))
    
    # Store references for external access
    container.packages_textarea = packages_textarea
    container.output_area = output_area
    container.parse_button = parse_button
    container.clear_button = clear_button
    
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

def create_enhanced_parsed_packages_html(packages: list) -> str:
    """Create enhanced HTML for parsed packages with modern design."""
    if not packages:
        return ""
    
    html = """
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 8px; 
        padding: 16px;
        border: 1px solid #dee2e6;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 18px;">📦</span>
            <h4 style="color: #495057; margin: 0;">Parsed Packages ({len(packages)})</h4>
        </div>
        <div style="display: grid; gap: 8px;">
    """
    
    for pkg in packages:
        icon = "🔗" if pkg['is_git'] else "📦"
        version_text = f" <span style='color: #6c757d;'>({pkg['version_spec']})</span>" if pkg['version_spec'] and pkg['version_spec'] != 'git' else ""
        git_badge = "<span style='background: #17a2b8; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.7rem; margin-left: 8px;'>GIT</span>" if pkg['is_git'] else ""
        
        html += f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        ">
            <span style="font-size: 16px;">{icon}</span>
            <div style="flex: 1; min-width: 0;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <strong style="color: #333; font-size: 0.9rem;">{pkg['name']}</strong>
                    {version_text}
                    {git_badge}
                </div>
                <div style="color: #6c757d; font-size: 0.8rem; font-family: monospace; margin-top: 2px;">
                    {pkg['raw']}
                </div>
            </div>
            <div style="
                background: #28a745; 
                color: white; 
                padding: 2px 8px; 
                border-radius: 12px; 
                font-size: 0.7rem;
                font-weight: 500;
            ">
                VALID
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    return html

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