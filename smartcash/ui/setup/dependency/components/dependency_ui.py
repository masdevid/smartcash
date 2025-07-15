"""
Dependency UI Components Creation

This module contains the UI component creation logic for the dependency management interface.
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.logger import get_module_logger

def create_dependency_ui_components(module_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all UI components for dependency management.
    
    Args:
        module_config: Module configuration containing package categories
        
    Returns:
        Dictionary containing all UI components
    """
    try:
        from smartcash.ui.components.header_container import create_header_container
        from smartcash.ui.components.form_container import create_form_container, LayoutType
        from smartcash.ui.components.action_container import create_action_container
        from smartcash.ui.components.operation_container import create_operation_container
        from smartcash.ui.components.footer_container import create_footer_container, PanelConfig, PanelType
        from smartcash.ui.components.main_container import create_main_container
        
        # 1. Header Container
        header_container = create_header_container(
            title="📦 Manajemen Paket",
            subtitle="Kelola paket Python untuk lingkungan SmartCash",
            status_message="Siap untuk manajemen paket",
            status_type="info"
        )
        
        # 2. Form Container with Package Grid
        form_container = create_form_container(
            layout_type=LayoutType.COLUMN,
            container_margin="0",
            container_padding="10px",
            gap="10px"
        )
        
        # Create package grid and additional components
        package_grid, package_checkboxes = _create_package_grid(module_config)
        custom_packages = _create_custom_packages_area()
        
        # Add items to form (removed duplicate status_output)
        packages_label = widgets.HTML("<h3 style='margin: 10px 0;'>📦 Kategori Paket</h3>")
        custom_label = widgets.HTML("<h4>➕ Paket Tambahan</h4>")
        
        form_items = [packages_label, package_grid, custom_label, custom_packages]
        
        for item in form_items:
            form_container['add_item'](item, height="auto")
        
        # 3. Action Container with corrected button labels
        action_container = create_action_container(
            buttons=[
                {
                    'id': 'install',
                    'text': '📥 Instal Terpilih',
                    'style': 'success',
                    'tooltip': 'Instal paket yang dipilih'
                },
                {
                    'id': 'check_status', 
                    'text': '🔍 Cek Status',
                    'style': 'info',
                    'tooltip': 'Periksa status instalasi dan temukan paket yang hilang'
                },
                {
                    'id': 'update',
                    'text': '⬆️ Update Semua',
                    'style': 'warning',
                    'tooltip': 'Update paket yang terinstal'
                },
                {
                    'id': 'uninstall',
                    'text': '🗑️ Uninstal',
                    'style': 'danger', 
                    'tooltip': 'Uninstal paket yang dipilih'
                }
            ],
            title="🔧 Operasi Paket",
            show_save_reset=True
        )
        
        # 4. Operation Container with consistent logging and progress
        operation_container = create_operation_container(
            show_progress=True,
            show_dialog=False,  # Disable dialog to reduce clutter
            show_logs=True,
            log_module_name="Manajemen Paket",
            log_height="200px",
            log_entry_style='compact',  # Ensure consistent hover behavior
            progress_style="prominent"  # Make progress more visible
        )
        
        # 5. Footer Container
        footer_container = create_footer_container(
            panels=[
                PanelConfig(
                    panel_type=PanelType.INFO_ACCORDION,
                    title="💡 Tips Manajemen Paket",
                    content="""
                    <div style="padding: 10px;">
                        <ul>
                            <li><strong>Tambah Paket:</strong> Tambahkan paket yang dipilih ke konfigurasi</li>
                            <li><strong>Instal Terpilih:</strong> Instal paket yang dipilih dan paket tambahan</li>
                            <li><strong>Cek Status:</strong> Verifikasi paket mana yang terinstal dan temukan yang hilang</li>
                            <li><strong>Update Semua:</strong> Update semua paket terinstal ke versi terbaru</li>
                            <li><strong>Uninstal:</strong> Hapus paket yang dipilih dari lingkungan</li>
                        </ul>
                    </div>
                    """,
                    style="info",
                    open_by_default=False
                )
            ]
        )
        
        # 6. Main Container
        components = [
            {'type': 'header', 'component': header_container.container, 'order': 0},
            {'type': 'form', 'component': form_container['container'], 'order': 1},
            {'type': 'action', 'component': action_container['container'], 'order': 2},
            {'type': 'operation', 'component': operation_container['container'], 'order': 3},
            {'type': 'footer', 'component': footer_container.container, 'order': 4}
        ]
        
        main_container = create_main_container(components=components)
        
        # Return UI components dictionary
        return {
            # Container components
            'main_container': main_container.container,
            'ui': main_container.container,
            'header_container': header_container,
            'form_container': form_container,
            'action_container': action_container,
            'footer_container': footer_container,
            'operation_container': operation_container,
            
            # Form elements
            'package_checkboxes': package_checkboxes,
            'custom_packages': custom_packages,
            
            # Buttons (updated names)
            'add_button': action_container['buttons'].get('add_packages'),
            'install_button': action_container['buttons'].get('install'),
            'check_button': action_container['buttons'].get('check_status'),
            'update_button': action_container['buttons'].get('update'),
            'uninstall_button': action_container['buttons'].get('uninstall'),
            
            # Operation components
            'operation_container': operation_container.get('container'),
            'progress_tracker': operation_container.get('progress_tracker'),
            'log_accordion': operation_container.get('log_accordion')
        }
        
    except Exception as e:
        get_module_logger("smartcash.ui.setup.dependency.components.dependency_ui").error(f"❌ Failed to create dependency UI components: {e}")
        raise

def _create_package_grid(module_config: Dict[str, Any]) -> tuple:
    """Create the 4-column package grid layout.
    
    Args:
        module_config: Module configuration containing package categories
        
    Returns:
        Tuple of (package_grid_widget, package_checkboxes_dict)
    """
    # Package categories for grid layout
    package_categories = module_config.get("package_categories", {})
    
    # Create grid layout for package categories (4 columns)
    package_checkboxes = {}
    category_containers = []
    
    # Get all categories
    categories = [
        'core_requirements',
        'ml_ai_libraries', 
        'data_processing',
        'additional_packages'
    ]
    
    for category_key in categories:
        category = package_categories.get(category_key, {})
        if not category:
            continue
            
        packages = category.get("packages", [])
        if not packages:
            continue
        
        # Create category header
        category_header = widgets.HTML(
            value=f"<h4 style='margin: 5px 0; color: {category.get('color', '#333')};'>"
                  f"{category.get('icon', '📦')} {category.get('name', category_key)}</h4>"
        )
        
        # Create checkboxes for this category
        category_checkboxes = []
        for pkg in packages:
            # All core packages (from all categories) should be selected by default
            # Core packages are those marked as is_default=True or from requirements.txt
            is_core_package = pkg.get('is_default', False) or pkg.get('source') == 'requirements.txt'
            checkbox = widgets.Checkbox(
                value=is_core_package,
                description=f"{pkg['name']} ({pkg.get('version', '')})",
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='1px 0')
            )
            # Store the package name as a custom attribute for easy access
            checkbox.package_name = pkg['name']
            category_checkboxes.append(checkbox)
        
        # Store checkboxes by category
        package_checkboxes[category_key] = category_checkboxes
        
        # Create container for this category
        category_container = widgets.VBox(
            [category_header] + category_checkboxes,
            layout=widgets.Layout(
                margin='5px',
                padding='10px',
                border='1px solid #ddd',
                border_radius='5px',
                width='22%'  # 4 columns with some margin
            )
        )
        category_containers.append(category_container)
    
    # Create 4-column grid using HBox
    grid_rows = []
    for i in range(0, len(category_containers), 4):
        row = widgets.HBox(
            category_containers[i:i+4],
            layout=widgets.Layout(
                justify_content='space-between',
                width='100%'
            )
        )
        grid_rows.append(row)
    
    # Package grid container
    package_grid = widgets.VBox(
        grid_rows,
        layout=widgets.Layout(width='100%')
    )
    
    return package_grid, package_checkboxes

def _create_custom_packages_area() -> widgets.Textarea:
    """Create the custom packages text area.
    
    Returns:
        Textarea widget for custom packages
    """
    return widgets.Textarea(
        placeholder="Masukkan paket tambahan (satu per baris)\nContoh:\nnumpy>=1.21.0\npandas>=1.3.0",
        layout=widgets.Layout(height='100px', width='100%')
    )