"""
File: smartcash/ui/setup/dependency/components/custom_packages_tab.py
Deskripsi: Tab untuk custom packages management
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_custom_packages_tab(config: Dict[str, Any], logger) -> widgets.VBox:
    """Create tab untuk custom packages"""
    
    custom_packages = config.get('custom_packages', '')
    
    # Header
    header = widgets.HTML("""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #333; margin: 0 0 10px 0;">🛠️ Custom Packages</h3>
        <p style="color: #666; margin: 0;">Tambahkan packages custom yang tidak tersedia di categories default.</p>
    </div>
    """)
    
    # Instructions
    instructions = widgets.HTML("""
    <div style="
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    ">
        <h4 style="color: #495057; margin: 0 0 10px 0;">📋 Format Input</h4>
        <ul style="margin: 0; padding-left: 20px; color: #6c757d;">
            <li>Satu package per baris</li>
            <li>Format: <code>package_name</code> atau <code>package_name==version</code></li>
            <li>Contoh: <code>numpy>=1.20.0</code>, <code>matplotlib</code></li>
            <li>Untuk git: <code>git+https://github.com/user/repo.git</code></li>
        </ul>
    </div>
    """)
    
    # Text area for custom packages
    packages_textarea = widgets.Textarea(
        value=custom_packages,
        placeholder="Masukkan packages custom di sini...\nContoh:\nnumpy>=1.20.0\nmatplotlib\ngit+https://github.com/user/repo.git",
        layout=widgets.Layout(
            width='100%',
            height='300px',
            border='1px solid #ddd',
            border_radius='8px',
            padding='10px'
        )
    )
    
    # Parse and validate button
    parse_btn = widgets.Button(
        description="🔍 Parse & Validate",
        button_style='info',
        layout=widgets.Layout(width='150px', margin='10px 0')
    )
    
    # Clear button
    clear_btn = widgets.Button(
        description="🗑️ Clear All",
        button_style='danger',
        layout=widgets.Layout(width='120px', margin='10px 0 10px 10px')
    )
    
    # Status output
    status_output = widgets.HTML(
        value="<p style='color: #666; margin: 10px 0;'>Status: Siap menerima input packages</p>"
    )
    
    # Parsed packages display
    parsed_display = widgets.HTML(
        value="",
        layout=widgets.Layout(
            width='100%',
            max_height='200px',
            overflow='auto',
            border='1px solid #ddd',
            border_radius='8px',
            padding='10px',
            margin='10px 0'
        )
    )
    
    # Button handlers
    def on_parse_click(btn):
        """Handler untuk parse button"""
        try:
            packages_text = packages_textarea.value.strip()
            if not packages_text:
                status_output.value = "<p style='color: #ffa500;'>⚠️ Tidak ada packages untuk diparse</p>"
                parsed_display.value = ""
                return
            
            # Parse packages
            parsed_packages = parse_custom_packages(packages_text)
            
            if parsed_packages:
                status_output.value = f"<p style='color: #4CAF50;'>✅ Berhasil parse {len(parsed_packages)} packages</p>"
                parsed_display.value = create_parsed_packages_html(parsed_packages)
            else:
                status_output.value = "<p style='color: #f44336;'>❌ Tidak ada packages valid ditemukan</p>"
                parsed_display.value = ""
                
        except Exception as e:
            status_output.value = f"<p style='color: #f44336;'>❌ Error parsing: {str(e)}</p>"
            logger.error(f"Error parsing custom packages: {e}")
    
    def on_clear_click(btn):
        """Handler untuk clear button"""
        packages_textarea.value = ""
        status_output.value = "<p style='color: #666;'>🗑️ Packages cleared</p>"
        parsed_display.value = ""
    
    parse_btn.on_click(on_parse_click)
    clear_btn.on_click(on_clear_click)
    
    # Container
    container = widgets.VBox([
        header,
        instructions,
        packages_textarea,
        widgets.HBox([parse_btn, clear_btn]),
        status_output,
        parsed_display
    ], layout=widgets.Layout(padding='20px'))
    
    # Store reference untuk access dari handler
    container.packages_textarea = packages_textarea
    container.status_output = status_output
    container.parsed_display = parsed_display
    
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