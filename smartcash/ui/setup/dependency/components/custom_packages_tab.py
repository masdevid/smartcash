"""
File: smartcash/ui/setup/dependency/components/custom_packages_tab.py
Deskripsi: Tab untuk custom packages management
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.components.form_container import create_form_container, LayoutType

def create_custom_packages_tab(config: Dict[str, Any], logger=None) -> widgets.VBox:
    """Create enhanced tab for custom packages with modern full-width design."""
    
    custom_packages = config.get('custom_packages', '')
    
    # Create main container with full width
    main_container = widgets.VBox(layout=widgets.Layout(
        width='100%',
        padding='16px',
        overflow='hidden'
    ))
    
    # Add header
    header = widgets.HTML("""
    <div style="margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span style="font-size: 24px;">🛠️</span>
            <div>
                <h2 style="color: #333; margin: 0; font-size: 1.5rem;">Custom Packages</h2>
                <p style="color: #666; margin: 0; font-size: 0.95rem;">Add custom packages not available in default categories</p>
            </div>
        </div>
    </div>
    """)
    
    # Create form container with card styling
    form_card = widgets.VBox(layout=widgets.Layout(
        width='100%',
        border='1px solid #e0e0e0',
        border_radius='8px',
        padding='0',
        background_color='white',
        margin='0 0 16px 0',
        overflow='hidden'
    ))
    
    # Add form header
    form_header = widgets.HTML("""
    <div style="
        background: linear-gradient(135deg, #6c757d08, #6c757d04);
        border-bottom: 1px solid #e0e0e0;
        padding: 12px 16px;
    ">
        <h3 style="margin: 0; color: #495057; font-size: 1.1rem;">
            <span style="margin-right: 8px;">📝</span>Package Specifications
        </h3>
    </div>
    """)
    
    # Create form content
    form_content = widgets.VBox(layout=widgets.Layout(
        padding='16px',
        width='100%'
    ))
    
    # Add instructions
    instructions = widgets.HTML("""
    <div style="
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #495057;
    ">
        <div style="display: flex; align-items: flex-start; gap: 8px;">
            <span style="font-size: 1.2em;">ℹ️</span>
            <div>
                <div style="font-weight: 600; margin-bottom: 6px;">Input Format:</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 6px; font-size: 0.85em;">
                    <div>• One package per line</div>
                    <div>• Format: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">package==version</code></div>
                    <div>• Example: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">numpy>=1.20.0</code></div>
                    <div>• Git: <code style="background: #e9ecef; padding: 1px 4px; border-radius: 3px;">git+https://...</code></div>
                </div>
            </div>
        </div>
    </div>
    """)
    
    # Create text area
    text_area = widgets.Textarea(
        value=custom_packages,
        placeholder="Enter custom packages here...\nExamples:\nnumpy>=1.20.0\nmatplotlib\ntensorflow==2.8.0\ngit+https://github.com/user/repo.git",
        layout=widgets.Layout(
            width='100%',
            height='200px',
            border='1px solid #e1e5e9',
            border_radius='6px',
            padding='12px',
            margin='0 0 12px 0',
            font_family='monospace',
            font_size='0.9em'
        )
    )
    
    # Create buttons
    button_row = widgets.HBox(layout=widgets.Layout(
        width='100%',
        justify_content='flex-end',
        gap='8px',
        margin='8px 0 0 0'
    ))
    
    parse_button = widgets.Button(
        description='📦 Parse Packages',
        button_style='primary',
        layout=widgets.Layout(width='180px', height='36px')
    )
    
    clear_button = widgets.Button(
        description='🗑️ Clear',
        button_style='warning',
        layout=widgets.Layout(width='120px', height='36px')
    )
    
    button_row.children = [parse_button, clear_button]
    
    # Add widgets to form content
    form_content.children = [instructions, text_area, button_row]
    
    # Add to form card
    form_card.children = [form_header, form_content]
    
    # Create output card
    output_card = widgets.VBox(layout=widgets.Layout(
        width='100%',
        border='1px solid #e0e0e0',
        border_radius='8px',
        padding='0',
        background_color='white',
        margin='0',
        overflow='hidden'
    ))
    
    # Output header
    output_header = widgets.HTML("""
    <div style="
        background: linear-gradient(135deg, #6c757d08, #6c757d04);
        border-bottom: 1px solid #e0e0e0;
        padding: 12px 16px;
    ">
        <h3 style="margin: 0; color: #495057; font-size: 1.1rem;">
            <span style="margin-right: 8px;">📋</span>Parsed Packages
        </h3>
    </div>
    """)
    
    # Output content
    output_content = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            max_height='300px',
            overflow_y='auto',
            padding='12px',
            margin='0'
        )
    )
    
    # Add empty state
    with output_content:
        output_content.append_display_data(widgets.HTML("""
            <div style="
                color: #6c757d;
                font-style: italic;
                text-align: center;
                padding: 30px 20px;
                font-size: 0.95em;
            ">
                No packages parsed yet. Enter package specifications above and click "Parse Packages".
            </div>
        """))
    
    # Assemble output card
    output_card.children = [output_header, output_content]
    
    # Add everything to main container
    main_container.children = [header, form_card, output_card]
    
    # Button handlers
    def on_parse_button_clicked(b):
        with output_content:
            output_content.clear_output()
            packages_text = text_area.value.strip()
            
            if not packages_text:
                output_content.append_display_data(widgets.HTML("""
                    <div style="
                        color: #6c757d;
                        font-style: italic;
                        text-align: center;
                        padding: 30px 20px;
                        font-size: 0.95em;
                    ">
                        No packages to parse. Enter some package specifications above.
                    </div>
                """))
                return
                
            packages = parse_custom_packages(packages_text)
            if packages:
                output_content.append_display_data(widgets.HTML(create_enhanced_parsed_packages_html(packages)))
            else:
                output_content.append_display_data(widgets.HTML("""
                    <div style="
                        color: #dc3545;
                        text-align: center;
                        padding: 30px 20px;
                        font-size: 0.95em;
                    ">
                        No valid packages found. Please check your input format.
                    </div>
                """))
    
    def on_clear_button_clicked(b):
        text_area.value = ''
        with output_content:
            output_content.clear_output()
            output_content.append_display_data(widgets.HTML("""
                <div style="
                    color: #6c757d;
                    font-style: italic;
                    text-align: center;
                    padding: 30px 20px;
                    font-size: 0.95em;
                ">
                    No packages parsed yet. Enter package specifications above and click "Parse Packages".
                </div>
            """))
    
    parse_button.on_click(on_parse_button_clicked)
    clear_button.on_click(on_clear_button_clicked)
    
    # Store references for external access
    main_container.packages_textarea = text_area
    main_container.output_area = output_content
    main_container.parse_button = parse_button
    main_container.clear_button = clear_button
    
    return main_container

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
    
    html_parts = ["""
    <style>
        .packages-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 10px;
            margin: 0;
            padding: 0;
        }
        .package-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 12px 14px;
            transition: all 0.2s ease;
        }
        .package-card:hover {
            border-color: #dee2e6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .package-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .package-name {
            font-weight: 600;
            color: #212529;
            margin: 0;
            font-size: 0.92rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .package-type {
            font-size: 0.75rem;
            background: #f1f3f5;
            color: #495057;
            padding: 2px 8px;
            border-radius: 10px;
            text-transform: capitalize;
        }
        .package-spec {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            color: #495057;
            font-size: 0.8rem;
            margin: 0 0 8px 0;
            word-break: break-all;
            background: #f8f9fa;
            padding: 6px 8px;
            border-radius: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .package-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: #6c757d;
        }
        .package-version {
            background: #e9ecef;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        .package-actions {
            display: flex;
            gap: 6px;
        }
        .action-btn {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            cursor: pointer;
            border: 1px solid #dee2e6;
            background: white;
            color: #495057;
            transition: all 0.2s;
        }
        .action-btn:hover {
            background: #f8f9fa;
            border-color: #ced4da;
        }
    </style>
    <div class="packages-grid">
    """]
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