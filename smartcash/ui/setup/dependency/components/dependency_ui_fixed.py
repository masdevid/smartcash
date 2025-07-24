"""
Dependency UI Components Creation - Fixed Version

This module contains the UI component creation logic for the dependency management interface.
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import ipywidgets as widgets
import traceback
from smartcash.ui.logger import get_module_logger

# Cache for UI components to prevent recreation
_ui_components_cache = None
logger = get_module_logger(__name__)

def create_dependency_ui_components(module_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create all UI components for dependency management.
    
    This function is optimized with caching to prevent unnecessary re-renders.
    
    Args:
        module_config: Configuration dictionary for the module
        
    Returns:
        Dictionary containing all UI components
    """
    global _ui_components_cache
    
    # Return cached components if available
    if _ui_components_cache is not None:
        return _ui_components_cache
        
    logger = get_module_logger("smartcash.ui.setup.dependency.components.dependency_ui")
    logger.debug("Creating dependency UI components...")
    
    # Initialize with default values
    package_checkboxes = {}
    custom_packages = _create_custom_packages_area()
    
    # Import UI components
    from smartcash.ui.components.header_container import create_header_container
    from smartcash.ui.components.form_container import create_form_container, LayoutType
    from smartcash.ui.components.action_container import create_action_container
    from smartcash.ui.components.operation_container import create_operation_container
    from smartcash.ui.components.footer_container import create_footer_container
    from smartcash.ui.components.main_container import create_main_container
    
    # Ensure module_config is a dictionary
    if module_config is None:
        module_config = {}
        logger.warning("No module_config provided, using empty configuration")
    
    logger.info("🔄 Creating dependency UI components...")
    
    # 1. Header Container
    header_container = create_header_container(
        title="Manajemen Dependensi",
        subtitle="Kelola paket dan dependensi proyek Anda",
        icon="📦"
    )
    logger.debug("✅ Header container created successfully")
    
    # 2. Form Container - Package Grid
    package_grid, package_checkboxes = _create_package_grid(module_config)
    
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        padding='10px',
        spacing='10px'
    )
    
    # Add form items
    form_items = [
        widgets.HTML("<h3>Pilih Paket yang Diperlukan</h3>"),
        widgets.HTML("<p>Centang paket yang ingin Anda instal:</p>"),
        package_grid,
        widgets.HTML("<h4 style='margin-top: 20px;'>Paket Tambahan</h4>"),
        widgets.HTML("<p>Masukkan paket tambahan (format: nama==versi, satu per baris):</p>"),
        custom_packages
    ]
    
    # Add items to form container
    for item in form_items:
        if item is not None:  # Skip None items
            form_container['add_item'](item, height="auto")
            
    logger.debug("✅ Form container created successfully")
    
    # 3. Action Container
    action_container = create_action_container(
        buttons=[
            {
                'id': 'install',
                'text': '📥 Instal Paket Terpilih',
                'style': 'success',
                'tooltip': 'Instal paket yang dipilih'
            },
            {
                'id': 'check_status', 
                'text': '🔍 Periksa Status',
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
    logger.debug("✅ Action container created successfully")
    
    # 4. Operation Container with dual progress tracking
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=False,  # Disable dialog to reduce clutter
        show_logs=True,
        log_module_name="Manajemen Paket",
        log_height="200px",
        progress_style="prominent",  # Make progress more visible
        progress_levels='dual'  # Enable dual progress tracking
    )
    logger.debug("✅ Operation container created successfully")
    
    # 5. Footer Container
    from smartcash.ui.components.footer_container import PanelConfig, PanelType
    
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
                is_open=False
            )
        ],
        module_info={
            'name': 'Dependency Management',
            'version': '1.0.0',
            'description': 'Kelola paket dan dependensi proyek SmartCash'
        }
    )
    logger.debug("✅ Footer container created successfully")
    
    # 6. Main Container
    main_container = create_main_container(
        header=header_container,
        form=form_container,
        action=action_container,
        operation=operation_container,
        footer=footer_container
    )
    logger.debug("✅ Main container created successfully")
    
    # Store components in cache
    _ui_components_cache = {
        'main_container': main_container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        '_package_checkboxes': package_checkboxes,
        '_custom_packages': custom_packages
    }
    
    logger.info("✅ Successfully created all dependency UI components")
    return _ui_components_cache

def _create_package_grid(module_config: Dict[str, Any]) -> Tuple[widgets.Widget, Dict[str, List[widgets.Checkbox]]]:
    """Create the 4-column package grid layout.
    
    Args:
        module_config: Module configuration containing package categories
        
    Returns:
        Tuple of (package_grid_widget, package_checkboxes_dict)
    """
    # Initialize with empty values in case of early return
    package_checkboxes = {}
    category_containers = []
    
    # Safely get package categories with default empty dict
    package_categories = module_config.get("package_categories", {})
        
    if not package_categories:
        logger.warning("No package categories found in module_config. Using default categories.")
        # Provide default categories if none found
        package_categories = {
            'core_requirements': {
                'name': 'Persyaratan Inti',
                'icon': '⭐',
                'color': '#4CAF50',
                'packages': [
                    {'name': 'numpy', 'version': '>=1.19.0', 'is_default': True},
                    {'name': 'pandas', 'version': '>=1.3.0', 'is_default': True}
                ]
            }
        }
    
    # Define default categories to ensure consistent ordering
    default_categories = [
        'core_requirements',
        'ml_ai_libraries', 
        'data_processing',
        'additional_packages'
    ]
    
    # Get all categories from config, but ensure default categories are included
    categories = list(dict.fromkeys(default_categories + list(package_categories.keys())))

    for category_key in categories:
        try:
            category = package_categories.get(category_key, {})
            if not category:
                # Skip if category is empty and not in default categories
                if category_key not in default_categories:
                    continue
                # Create empty category for default categories
                category = {
                    'name': f'{category_key.replace("_", " ").title()}',
                    'icon': '📦',
                    'color': '#757575',
                    'packages': []
                }
                
            category_name = category.get('name', category_key.replace('_', ' ').title())
            category_icon = category.get('icon', '📦')
            category_color = category.get('color', '#757575')
            packages = category.get('packages', [])
            
            # Create checkboxes for this category
            category_checkboxes = []
            for pkg in packages:
                if isinstance(pkg, dict):
                    pkg_name = pkg.get('name', '')
                    pkg_version = pkg.get('version', '')
                    is_default = pkg.get('is_default', False)
                    
                    # Create checkbox description
                    description = f"{pkg_name}"
                    if pkg_version:
                        description += f" ({pkg_version})"
                    
                    checkbox = widgets.Checkbox(
                        value=is_default,
                        description=description,
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(margin='1px 0')
                    )
                    checkbox.package_name = pkg_name  # Store package name for easy access
                    category_checkboxes.append(checkbox)
                elif isinstance(pkg, str):
                    # Handle simple string package names
                    checkbox = widgets.Checkbox(
                        value=False,
                        description=pkg,
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(margin='1px 0')
                    )
                    checkbox.package_name = pkg
                    category_checkboxes.append(checkbox)
            
            # Store checkboxes for this category
            package_checkboxes[category_key] = category_checkboxes
            
            # Create category container
            if category_checkboxes:
                # Create category header
                category_header = widgets.HTML(
                    value=f"<h4 style='margin: 5px 0; color: {category_color};'>{category_icon} {category_name}</h4>"
                )
                
                # Create category container with border
                category_container = widgets.VBox(
                    children=[category_header] + category_checkboxes,
                    layout=widgets.Layout(
                        border='1px solid #ddd',
                        margin='5px',
                        padding='10px',
                        width='22%'
                    )
                )
                category_containers.append(category_container)
            
        except Exception as e:
            logger.error(f"Error processing category {category_key}: {str(e)}")
            continue
    
    if category_containers:
        # Create horizontal layout with categories
        package_grid = widgets.HBox(
            children=category_containers,
            layout=widgets.Layout(
                justify_content='space-between',
                width='100%'
            )
        )
    else:
        # Fallback if no categories
        empty_message = widgets.HTML(
            value="<p>Tidak ada kategori paket yang tersedia dalam konfigurasi.</p>"
        )
        return empty_message, {}
    
    return package_grid, package_checkboxes

def _create_custom_packages_area() -> widgets.Textarea:
    """Create the custom packages text area.
    
    Returns:
        Textarea widget for custom package input
    """
    return widgets.Textarea(
        value='',
        placeholder='Masukkan nama paket tambahan (satu per baris)\nContoh:\nnumpy>=1.19.0\npandas',
        style={'description_width': 'initial'},
        layout=widgets.Layout(height='auto')
    )