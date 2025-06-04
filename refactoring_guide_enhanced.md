# SmartCash UI Refactoring Guide - Enhanced
**File: docs/refactoring_guide_enhanced.md**  
**Deskripsi: Panduan refactoring dengan parent module support dan complex component architecture**

## 🏗️ Enhanced Directory Structure

```
smartcash/ui/{parent_domain}/
├── {parent}_init.py              # Parent module initializer
├── {module}/
│   ├── {module}_init.py          # Child module initializer
│   ├── handlers/
│   │   ├── config_extractor.py
│   │   ├── config_updater.py  
│   │   ├── defaults.py
│   │   └── {module}_handler.py
│   ├── components/
│   │   ├── ui_components.py      # Main entry point (max 500 lines)
│   │   ├── ui_layout.py          # Layout ensemble & responsiveness
│   │   ├── ui_forms.py           # Form collections
│   │   ├── ui_forms_basic.py     # Basic forms (if >500 lines)
│   │   └── ui_forms_advanced.py  # Advanced forms (if >500 lines)
│   └── utils/
│       └── constants.py
└── shared/
    ├── components/               # Shared across parent modules
    ├── handlers/
    └── utils/
```

## 🎯 Enhanced Naming Conventions

### Parent-Child Module Pattern
```python
# Parent module
smartcash/ui/model/model_init.py                    # ModelInitializer
smartcash/ui/model/training/training_init.py       # TrainingInitializer(parent='model')
smartcash/ui/model/evaluation/evaluation_init.py   # EvaluationInitializer(parent='model')

# Domain grouping
smartcash/ui/dataset/dataset_init.py               # DatasetInitializer  
smartcash/ui/dataset/download/download_init.py     # DownloadInitializer(parent='dataset')
smartcash/ui/dataset/preprocessing/preprocessing_init.py # PreprocessingInitializer(parent='dataset')
```

### ConfigHandler with Parent Support
```python
class {Module}ConfigHandler(ConfigHandler):
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)

# Usage
TrainingConfigHandler('training', 'model')
DownloadConfigHandler('download', 'dataset')
```

## 📁 Enhanced File Templates

### 1. Parent Module Init Template
```python
"""
File: smartcash/ui/{parent_domain}/{parent}_init.py
Deskripsi: Parent module initializer untuk {parent_domain}
"""

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.handlers.config_handlers import ConfigHandler

class {Parent}ConfigHandler(ConfigHandler):
    def __init__(self, module_name: str = '{parent}'):
        super().__init__(module_name)
    
    def extract_config(self, ui_components): return extract_{parent}_config(ui_components)
    def update_ui(self, ui_components, config): update_{parent}_ui(ui_components, config)
    def get_default_config(self): return get_default_{parent}_config()

class {Parent}Initializer(CommonInitializer):
    def __init__(self): super().__init__('{parent}', {Parent}ConfigHandler)
    def _create_ui_components(self, config, env=None, **kwargs): return create_{parent}_main_ui(config)
    def _setup_module_handlers(self, ui_components, config, env=None, **kwargs): return setup_{parent}_handlers(ui_components, config, env)
    def _get_default_config(self): return get_default_{parent}_config()
    def _get_critical_components(self): return ['ui', 'main_container', 'child_modules']

_parent_initializer = {Parent}Initializer()

def initialize_{parent}_ui(env=None, config=None, **kwargs): return _parent_initializer.initialize(env=env, config=config, **kwargs)
def get_{parent}_config(): return _parent_initializer.get_current_config()
def get_{parent}_child_modules(): return _parent_initializer.get_child_modules()
```

### 2. Child Module Init with Parent Template
```python
"""
File: smartcash/ui/{parent_domain}/{module}/{module}_init.py
Deskripsi: {Module} initializer dengan parent module support
"""

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.handlers.config_handlers import ConfigHandler

class {Module}ConfigHandler(ConfigHandler):
    def __init__(self, module_name: str = '{module}', parent_module: str = '{parent}'):
        super().__init__(module_name, parent_module)
    
    def extract_config(self, ui_components): return extract_{module}_config(ui_components)
    def update_ui(self, ui_components, config): update_{module}_ui(ui_components, config)
    def get_default_config(self): return get_default_{module}_config()

class {Module}Initializer(CommonInitializer):
    def __init__(self): super().__init__('{module}', {Module}ConfigHandler, '{parent}')
    def _create_ui_components(self, config, env=None, **kwargs): return create_{module}_main_ui(config)
    def _setup_module_handlers(self, ui_components, config, env=None, **kwargs): return setup_{module}_handlers(ui_components, config, env)
    def _get_default_config(self): return get_default_{module}_config()
    def _get_critical_components(self): return ['ui', 'main_button', 'save_button', 'reset_button', 'log_output']

_child_initializer = {Module}Initializer()

def initialize_{module}_ui(env=None, config=None, **kwargs): return _child_initializer.initialize(env=env, config=config, **kwargs)
def get_{module}_config(): return _child_initializer.get_current_config()
def connect_to_parent(parent_ui_components): return connect_config_to_parent(_child_initializer, '{parent}', parent_ui_components)
```

### 3. UI Components Main Entry Template
```python
"""
File: smartcash/ui/{parent_domain}/{module}/components/ui_components.py
Deskripsi: Main UI components entry point untuk {module} (max 500 lines)
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from .ui_layout import create_{module}_layout
from .ui_forms import create_{module}_forms
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking_container

def create_{module}_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create main UI untuk {module} dengan responsive layout"""
    
    # Forms dan input components
    forms_components = create_{module}_forms(config)
    
    # Layout assembly dengan responsiveness
    layout_components = create_{module}_layout(forms_components, config)
    
    # Action buttons standardized
    action_buttons = create_action_buttons(
        primary_label="Execute {Module}",
        primary_icon="play", 
        secondary_buttons=[("Validate", "check", "info"), ("Preview", "search", "")],
        cleanup_enabled=True
    )
    
    # Progress tracking standardized
    progress_components = create_progress_tracking_container()
    
    # Assembly final UI dengan responsive grid
    ui_components = {
        'ui': layout_components['main_container'],
        'main_container': layout_components['main_container'],
        'forms': forms_components,
        'layout': layout_components,
        'action_buttons': action_buttons,
        'progress': progress_components,
        
        # Button mappings standardized
        'main_button': action_buttons['download_button'],
        'validate_button': action_buttons['check_button'], 
        'preview_button': action_buttons.get('cleanup_button'),
        'save_button': forms_components.get('save_button'),
        'reset_button': forms_components.get('reset_button'),
        
        # Progress mappings
        'progress_container': progress_components['container'],
        'update_progress': progress_components.get('update_progress'),
        'complete_operation': progress_components.get('complete_operation'),
        
        # Form mappings
        **{k: v for k, v in forms_components.items() if k.endswith(('_input', '_slider', '_checkbox', '_dropdown'))},
        
        'module_name': '{module}',
        'parent_module': '{parent}'
    }
    
    return ui_components

def get_{module}_critical_components() -> list: return ['ui', 'main_container', 'main_button', 'forms']
def validate_{module}_ui_structure(ui_components: Dict[str, Any]) -> Dict[str, Any]: 
    return {'valid': all(comp in ui_components for comp in get_{module}_critical_components())}
```

### 4. UI Layout Template with Flex/Grid
```python
"""
File: smartcash/ui/{parent_domain}/{module}/components/ui_layout.py
Deskripsi: Responsive layout assembly dengan full flex dan grid support
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.status_panel import create_status_panel

def create_{module}_layout(forms_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create responsive layout dengan flex dan grid"""
    
    # Header section
    header = create_header(f"🎯 {module.title()}", f"Advanced {module} configuration dan processing")
    
    # Status panel
    status_panel = create_status_panel("Ready untuk {module} processing", "info")
    
    # Main content grid - responsive two column
    left_content = create_responsive_container([
        forms_components.get('primary_form_container'),
        forms_components.get('config_form_container')
    ], container_type="vbox", width="100%")
    
    right_content = create_responsive_container([
        forms_components.get('advanced_form_container'),
        forms_components.get('preview_container')
    ], container_type="vbox", width="100%")
    
    content_grid = create_responsive_two_column(left_content, right_content, left_width="58%", right_width="40%")
    
    # Action section dengan centered layout
    action_section = create_responsive_container([
        forms_components.get('action_buttons_container'),
        forms_components.get('progress_container')
    ], container_type="vbox", width="100%", justify_content="center")
    
    # Output section dengan flexible height
    output_section = create_responsive_container([
        forms_components.get('log_container'),
        forms_components.get('result_container')
    ], container_type="vbox", width="100%")
    
    # Main container assembly dengan flex layout
    main_container = widgets.VBox([
        header,
        status_panel, 
        content_grid,
        widgets.HTML("<hr style='margin: 20px 0; border: 1px solid #e0e0e0;'>"),
        action_section,
        output_section
    ], layout=widgets.Layout(
        width='100%', max_width='100%', margin='0', padding='10px',
        display='flex', flex_direction='column', align_items='stretch',
        overflow='hidden', box_sizing='border-box'
    ))
    
    return {
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'content_grid': content_grid,
        'left_content': left_content,
        'right_content': right_content,
        'action_section': action_section,
        'output_section': output_section,
        'layout_type': 'responsive_grid'
    }

def create_{module}_mobile_layout(forms_components: Dict[str, Any]) -> Dict[str, Any]:
    """Mobile-optimized single column layout"""
    return create_responsive_container([
        forms_components.get('primary_form_container'),
        forms_components.get('config_form_container'), 
        forms_components.get('advanced_form_container'),
        forms_components.get('action_buttons_container')
    ], container_type="vbox", width="100%")

def apply_responsive_styling(widget: widgets.Widget, mobile_breakpoint: str = "768px"):
    """Apply responsive CSS styling"""
    widget.add_class('responsive-widget')
    return widget
```

### 5. UI Forms Template
```python
"""
File: smartcash/ui/{parent_domain}/{module}/components/ui_forms.py  
Deskripsi: Form collections untuk {module} (split jika >500 lines)
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.split_config import create_split_config
from smartcash.ui.utils.layout_utils import create_responsive_container

def create_{module}_forms(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create all form collections untuk {module}"""
    
    # Primary forms
    primary_forms = create_{module}_primary_forms(config)
    
    # Configuration forms  
    config_forms = create_{module}_config_forms(config)
    
    # Advanced forms
    advanced_forms = create_{module}_advanced_forms(config) if _should_show_advanced(config) else {}
    
    # Action forms
    action_forms = create_{module}_action_forms()
    
    # Container assembly dengan responsive grid
    forms_containers = _assemble_form_containers(primary_forms, config_forms, advanced_forms, action_forms)
    
    return {
        **primary_forms,
        **config_forms, 
        **advanced_forms,
        **action_forms,
        **forms_containers,
        'form_sections': ['primary', 'config', 'advanced', 'action']
    }

def create_{module}_primary_forms(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Primary input forms"""
    
    # Input widgets dengan responsive layout
    input_widgets = {
        'input_path': widgets.Text(description="Input Path:", value=config.get('input_path', ''), 
                                  layout=widgets.Layout(width='100%'), style={'description_width': '120px'}),
        'output_path': widgets.Text(description="Output Path:", value=config.get('output_path', ''),
                                   layout=widgets.Layout(width='100%'), style={'description_width': '120px'}),
        'batch_size': widgets.IntSlider(description="Batch Size:", value=config.get('batch_size', 32), min=1, max=256,
                                       layout=widgets.Layout(width='100%'), style={'description_width': '120px'})
    }
    
    # Primary form container
    primary_container = create_responsive_container(list(input_widgets.values()), 
                                                   title="📝 Primary Configuration", container_type="vbox")
    
    return {**input_widgets, 'primary_form_container': primary_container}

def create_{module}_config_forms(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Configuration forms with split support"""
    
    # Dataset split configuration
    split_config = create_split_config(
        title="Dataset Split", 
        train_value=config.get('train_split', 0.7),
        val_value=config.get('val_split', 0.2),
        test_value=config.get('test_split', 0.1)
    )
    
    # Processing options
    processing_widgets = {
        'enable_augmentation': widgets.Checkbox(description="Enable Augmentation", value=config.get('enable_augmentation', True)),
        'enable_validation': widgets.Checkbox(description="Enable Validation", value=config.get('enable_validation', True)),
        'parallel_processing': widgets.Checkbox(description="Parallel Processing", value=config.get('parallel_processing', False))
    }
    
    config_container = create_responsive_container([
        split_config['container'],
        *list(processing_widgets.values())
    ], title="⚙️ Processing Configuration", container_type="vbox")
    
    return {
        'split_config': split_config,
        **processing_widgets, 
        'config_form_container': config_container
    }

def create_{module}_advanced_forms(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Advanced configuration forms"""
    
    advanced_widgets = {
        'learning_rate': widgets.FloatLogSlider(description="Learning Rate:", value=config.get('learning_rate', 1e-3), 
                                               min=-6, max=-1, step=0.1, layout=widgets.Layout(width='100%')),
        'optimizer_dropdown': widgets.Dropdown(description="Optimizer:", options=['Adam', 'SGD', 'RMSprop'], 
                                              value=config.get('optimizer', 'Adam'), layout=widgets.Layout(width='100%')),
        'scheduler_dropdown': widgets.Dropdown(description="Scheduler:", options=['None', 'StepLR', 'CosineAnnealingLR'],
                                              value=config.get('scheduler', 'None'), layout=widgets.Layout(width='100%'))
    }
    
    advanced_container = create_responsive_container(list(advanced_widgets.values()),
                                                    title="🔬 Advanced Settings", container_type="vbox")
    
    return {**advanced_widgets, 'advanced_form_container': advanced_container}

def create_{module}_action_forms() -> Dict[str, Any]:
    """Action forms dengan save/reset buttons"""
    
    save_reset = create_save_reset_buttons("Save Config", "Reset", with_sync_info=True)
    
    return {
        'save_reset_buttons': save_reset,
        'save_button': save_reset['save_button'],
        'reset_button': save_reset['reset_button'],
        'action_buttons_container': save_reset['container']
    }

def _assemble_form_containers(primary: Dict, config: Dict, advanced: Dict, action: Dict) -> Dict[str, Any]:
    """Assemble form containers dengan responsive layout"""
    return {
        'forms_main_container': create_responsive_container([
            primary.get('primary_form_container'),
            config.get('config_form_container'),
            advanced.get('advanced_form_container'),
            action.get('action_buttons_container')
        ], container_type="vbox")
    }

def _should_show_advanced(config: Optional[Dict[str, Any]] = None) -> bool:
    """Determine if advanced forms should be shown"""
    return config.get('show_advanced', False) if config else False

def extract_forms_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration dari semua forms"""
    return {k.replace('_input', '').replace('_slider', '').replace('_checkbox', '').replace('_dropdown', ''): 
            getattr(v, 'value', None) for k, v in ui_components.items() 
            if k.endswith(('_input', '_slider', '_checkbox', '_dropdown')) and hasattr(v, 'value')}

def update_forms_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update forms dari configuration"""
    [setattr(ui_components[f"{key}_input"], 'value', value) if f"{key}_input" in ui_components else
     setattr(ui_components[f"{key}_slider"], 'value', value) if f"{key}_slider" in ui_components else
     setattr(ui_components[f"{key}_checkbox"], 'value', value) if f"{key}_checkbox" in ui_components else
     setattr(ui_components[f"{key}_dropdown"], 'value', value) if f"{key}_dropdown" in ui_components else None
     for key, value in config.items()]
```

## 🔧 Enhanced Logger Namespace with Parent Support

### Enhanced Namespace Registration
```python
# File: smartcash/ui/utils/ui_logger_namespace.py
KNOWN_NAMESPACES = {
    # Parent modules
    "smartcash.ui.model": "MODEL",
    "smartcash.ui.dataset": "DATASET", 
    
    # Child modules dengan parent context
    "smartcash.ui.model.training": "TRAIN",
    "smartcash.ui.model.evaluation": "EVAL",
    "smartcash.ui.dataset.download": "DOWNLOAD",
    "smartcash.ui.dataset.preprocessing": "PREPROC",
}

def register_parent_child_namespace(parent: str, module: str, parent_id: str, module_id: str):
    """Register parent-child namespace relationship"""
    KNOWN_NAMESPACES[f"smartcash.ui.{parent}"] = parent_id
    KNOWN_NAMESPACES[f"smartcash.ui.{parent}.{module}"] = module_id
```

## ⚡ Enhanced Operation Context with Parent Support

### Parent-Child Communication Pattern
```python
from smartcash.ui.initializers.config_cell_initializer import connect_config_to_parent

# Connect child config ke parent UI
def setup_parent_child_connection(parent_ui_components, child_initializer, child_type):
    """Setup parent-child communication"""
    connect_config_to_parent(child_initializer, child_type, parent_ui_components)
    
    # Auto-update parent saat child config berubah
    child_initializer.add_parent_callback(child_type, lambda cfg: update_parent_display(parent_ui_components, cfg, child_type))

def update_parent_display(parent_ui, child_config, child_type):
    """Update parent UI display dari child config changes"""
    display_key = f'{child_type}_status_display'
    parent_ui.get(display_key) and setattr(parent_ui[display_key], 'value', f"✅ {child_type.title()} configured")
```

## 📊 Complex Component Split Guidelines

### When to Split Components (>500 lines)

**UI Forms Split:**
```python
# Original: ui_forms.py (>500 lines)
ui_forms_basic.py      # Basic input forms
ui_forms_advanced.py   # Advanced configuration
ui_forms_validation.py # Validation forms
```

**UI Layout Split:**
```python
# Original: ui_layout.py (>500 lines)  
ui_layout_grid.py      # Grid layouts
ui_layout_responsive.py # Responsive utilities
ui_layout_mobile.py    # Mobile optimizations
```

## ✅ Enhanced Migration Checklist

### Parent Module Support
- [ ] Create parent module init file
- [ ] Update child modules dengan parent parameter
- [ ] Setup parent-child communication
- [ ] Register parent-child namespaces

### Complex Component Architecture  
- [ ] Split UI components jika >500 lines
- [ ] Implement responsive flex/grid layouts
- [ ] Create dedicated forms collections
- [ ] Add mobile-optimized layouts

### DRY & One-liner Implementation
- [ ] Consolidate repetitive code
- [ ] Use existing components dari `ui/components/**`
- [ ] Implement one-liner patterns
- [ ] Remove duplicate implementations

### Responsiveness & Layout
- [ ] Use `create_responsive_container()`
- [ ] Implement flex dan grid layouts
- [ ] Add mobile breakpoint handling
- [ ] Test responsive behavior

Pattern ini memastikan scalability untuk complex parent-child relationships sambil maintaining responsive design dan DRY principles.