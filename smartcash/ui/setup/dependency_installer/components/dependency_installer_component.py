"""
File: smartcash/ui/setup/dependency_installer/components/dependency_installer_component.py
Deskripsi: Fixed dependency installer component menggunakan log accordion dan center alignment
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """Create UI components untuk dependency installer dengan pendekatan modular dan DRY"""
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
    from smartcash.ui.setup.dependency_installer.utils.package_utils import get_package_categories
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
    import logging
    
    # Setup logger dan konstanta
    MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, "DEPS")
    logger = logging.getLogger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    logger.info("Creating dependency installer UI components")
    
    try:
        # Header
        header = create_header(f"📦 {MODULE_LOGGER_NAME}", "Setup package yang diperlukan untuk SmartCash")
        
        # Import konstanta terpusat
        from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
        
        # Dapatkan konfigurasi untuk level info
        info_config = get_status_config('info')
        
        # Status panel dengan style dari konstanta terpusat
        status_panel = widgets.HTML(
            value=f"""
            <div style="padding:8px 12px; background-color:{info_config['bg']}; 
                       color:{info_config['color']}; border-radius:4px; margin:10px 0;
                       border-left:4px solid {info_config['border']};">
                <p style="margin:3px 0">{info_config['emoji']} Pilih packages yang akan diinstall dan klik "Mulai Instalasi"</p>
            </div>
            """,
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        
        # Package categories
        package_categories = get_package_categories()
        checkboxes = {}
        category_boxes = []
        
        # Create category boxes
        for category in package_categories:
            category_box = create_category_box(category, checkboxes)
            category_boxes.append(category_box)
        
        # Packages container - 3 columns full width dengan space-between dan tinggi yang sama
        packages_container = widgets.HBox(
            category_boxes,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row nowrap',
                justify_content='space-between',
                align_items='stretch', # Mengubah dari flex-start ke stretch agar tinggi sama
                width='100%',
                margin='10px 0',
                overflow='hidden'
            )
        )
        
        # Custom packages
        custom_packages = widgets.Textarea(
            placeholder='Package tambahan (satu per baris)',
            layout=widgets.Layout(width='100%', height='80px')
        )
        
        custom_section = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('edit', '📝')} Custom Packages</h3>"),
            custom_packages
        ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
        
        # Install button
        install_button = widgets.Button(
            description='Mulai Instalasi',
            button_style='primary',
            icon='download',
            layout=widgets.Layout(margin='10px 0')
        )
        
        # Progress tracking container dengan visibilitas yang selalu terlihat
        progress_components = create_progress_tracking_container()
        
        # Pastikan progress container selalu terlihat saat diinisialisasi
        if hasattr(progress_components['container'], 'layout'):
            progress_components['container'].layout.visibility = 'visible'
            logger.info("Progress container visibility set to visible during initialization")
        
        # Log accordion untuk output
        log_components = create_log_accordion(
            module_name=MODULE_LOGGER_NAME,
            height='200px'
        )
        
        # Info box
        info_box = get_dependencies_info()
        
        # Main container
        main = widgets.VBox([
            header,
            status_panel,
            packages_container,
            custom_section,
            widgets.HBox([install_button], layout=widgets.Layout(justify_content='center')),
            progress_components['container'],
            log_components['log_accordion'],
            info_box
        ], layout=widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'))
        
        # Pastikan semua komponen kritis tersedia
        logger.info("Memastikan semua komponen kritis tersedia")
        
        # UI components - pastikan semua komponen kritis tersedia
        ui_components = {
            'ui': main,
            'status': log_components['log_output'],
            'log_output': log_components['log_output'],
            'status_panel': status_panel,
            'install_button': install_button,
            'custom_packages': custom_packages,
            'progress_tracker': progress_components['tracker'],
            'progress_container': progress_components['container'],
            'module_name': MODULE_LOGGER_NAME,
            'logger_namespace': DEPENDENCY_INSTALLER_LOGGER_NAMESPACE,
            **checkboxes
        }
        
        # Validasi komponen kritis secara manual
        critical_components = ['ui', 'install_button', 'status', 'log_output', 'progress_container', 'status_panel']
        missing_components = [comp for comp in critical_components if comp not in ui_components]
        
        if missing_components:
            error_msg = f"Komponen kritis tidak tersedia: {', '.join(missing_components)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"UI components created successfully with {len(ui_components)} components")
        return ui_components
    except Exception as e:
        logger.error(f"Error creating UI components: {str(e)}")
        # Buat fallback UI yang minimal namun memenuhi kriteria komponen kritis
        error_widget = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', min_height='100px', padding='10px', width='100%'))
        
        # Buat komponen minimal yang diperlukan
        fallback_container = widgets.VBox([
            widgets.HTML(value='<h3>⚠️ Deps (Fallback Mode)</h3>'),
            error_widget
        ], layout=widgets.Layout(padding='10px', width='100%'))
        
        # Install button dummy
        install_button = widgets.Button(
            description='Coba Lagi',
            button_style='danger',
            icon='refresh',
            layout=widgets.Layout(margin='10px 0')
        )
        
        # Tampilkan error di widget
        with error_widget:
            print(f"Error saat membuat UI components: {str(e)}")
            print("Silakan restart kernel dan coba lagi.")
        
        # Buat komponen minimal yang memenuhi kriteria validasi
        return {
            'ui': fallback_container,
            'main_container': fallback_container,
            'install_button': install_button,
            'status': error_widget,
            'log_output': error_widget,
            'progress_container': error_widget,
            'status_panel': error_widget,
            'error_widget': error_widget,
            'error': f"Error saat membuat UI components: {str(e)}"
        }

def create_category_box(category: Dict[str, Any], checkboxes: Dict[str, Any]) -> widgets.VBox:
    """Create category box dengan center alignment untuk package items"""
    from smartcash.ui.utils.constants import COLORS
    import logging
    
    # Setup logger
    logger = logging.getLogger('dependency_installer')
    
    # Header dengan center alignment
    header = widgets.HTML(f"""
    <div style="padding:8px 0; border-bottom:1px solid {COLORS['border']}; margin-bottom:8px; text-align:center;">
        <h4 style="margin:0; color:{COLORS['primary']}">{category['icon']} {category['name']}</h4>
        <small style="color:{COLORS['muted']}">{category['description']}</small>
    </div>
    """)
    
    # Package checkboxes dengan center alignment
    package_widgets = []
    for package in category['packages']:
        try:
            # Membuat status widget dengan format yang konsisten dengan UI logger
            status_widget = widgets.HTML(
                f"<span style='color:{COLORS['info']};font-size:11px;white-space:nowrap;'>🔍 Checking...</span>",
                layout=widgets.Layout(width='90px', margin='0')
            )
            
            # Cek apakah kunci 'default' ada, jika tidak, gunakan True sebagai default
            # Ini untuk menangani kasus di mana objek paket tidak memiliki kunci 'default'
            default_value = True  # Default ke True jika tidak ada
            if 'default' in package:
                default_value = package['default']
            else:
                # Gunakan default berdasarkan kunci paket
                # Secara default, pilih paket inti dan torch
                default_packages = ['yolov5_req', 'smartcash_req', 'torch_req']
                default_value = package['key'] in default_packages
                logger.info(f"Package {package['key']} tidak memiliki kunci 'default', menggunakan {default_value}")
            
            # Buat checkbox dengan nilai default yang tepat
            checkbox = widgets.Checkbox(
                description=package['name'],
                value=default_value,
                tooltip=package['description'],
                layout=widgets.Layout(width='calc(100% - 80px)', margin='2px 0')
            )
            
            # Horizontal row dengan center alignment dan proper width
            row = widgets.HBox([checkbox, status_widget], 
                              layout=widgets.Layout(
                                  width='90%',
                                  justify_content='space-between',
                                  align_items='center',
                                  margin='3px 0',
                                  padding='0',
                                  overflow='hidden'
                              ))
            package_widgets.append(row)
            
            # Store references
            checkboxes[package['key']] = checkbox
            checkboxes[f"{package['key']}_status"] = status_widget
        except Exception as e:
            logger.error(f"Error saat membuat checkbox untuk package {package.get('name', 'unknown')}: {str(e)}")
            # Buat widget error sebagai fallback
            error_widget = widgets.HTML(
                f"<div style='color:red;'>Error: {str(e)}</div>",
                layout=widgets.Layout(width='100%', margin='5px 0')
            )
            package_widgets.append(error_widget)
    
    # Category container dengan proper responsive width dan flex-grow untuk tinggi yang sama
    return widgets.VBox([header] + package_widgets, 
                       layout=widgets.Layout(
                           width='32%',
                           max_width='32%',
                           margin='0',
                           padding='10px',
                           border=f'1px solid {COLORS["border"]}',
                           border_radius='6px',
                           overflow='hidden',
                           box_sizing='border-box',
                           flex_grow='1', # Menambahkan flex-grow agar tinggi sama
                           display='flex',
                           flex_direction='column'
                       ))