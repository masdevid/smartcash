# 📚 Dokumentasi Komponen UI Shared SmartCash

## 🎯 Struktur Direktori

```
smartcash/ui/components/
├── __init__.py                 # Ekspor komponen utama + lazy loading
├── card.py                     # Komponen kartu untuk info dan metrics
├── status_panel.py             # Panel status dengan update capabilities
├── log_accordion.py            # Accordion untuk log output
├── tabs.py                     # Widget tab untuk organisasi UI
├── save_reset_buttons.py       # Tombol standar simpan/reset
├── action_buttons.py           # Tombol aksi yang reusable
├── info_accordion.py           # Accordion untuk informasi
├── layout/                     # Komponen layout
│   ├── __init__.py            # Ekspor layout components
│   └── layout_components.py   # Layout utils & responsive containers
├── widgets/                    # Form widgets
│   ├── dropdown.py            # Dropdown selection widget
│   ├── checkbox.py            # Checkbox dengan custom styling
│   ├── text_input.py          # Text input dengan validasi
│   ├── slider.py              # Slider dengan configurasi
│   └── log_slider.py          # Logarithmic slider untuk ranges
├── alerts/                     # Alert components
│   ├── __init__.py            # Ekspor alert functions
│   ├── alert_components.py    # Alert boxes dan notifications
│   └── constants.py           # Alert constants (colors, icons)
├── dialog/                     # Dialog components
│   ├── __init__.py            # Ekspor dialog functions
│   └── dialog_components.py   # Confirmation & info dialogs
├── header/                     # Header components
│   ├── __init__.py            # Ekspor header functions
│   └── header_components.py   # Section headers dan titles
├── progress_tracker/          # Progress tracking system
│   ├── __init__.py            # Ekspor progress components
│   ├── types.py               # Type definitions
│   ├── progress_tracker.py    # Main progress tracker
│   ├── ui_components.py       # UI components manager
│   └── factory.py            # Factory functions untuk trackers
└── info/                      # Info components
    ├── __init__.py            # Ekspor info functions
    └── info_components.py     # Info accordion & tabbed info
```

## 🔧 Komponen Utama

### 📋 Core Components

#### **Card Components** (`card.py`)
```python
from smartcash.ui.components import create_card, create_info_card

# Kartu basic dengan metrics
card = create_card(title="Total Data", value="1,234", icon="📊", color="#4CAF50")

# Kartu dengan tipe predefined
info_card = create_info_card("Info", "Pesan penting", "ℹ️")
success_card = create_success_card("Sukses", "Operasi berhasil", "✅")
warning_card = create_warning_card("Peringatan", "Perhatian", "⚠️")
error_card = create_error_card("Error", "Terjadi kesalahan", "❌")

# Baris kartu responsive
cards = create_card_row([
    {"title": "Data 1", "value": "123", "icon": "🔢"},
    {"title": "Data 2", "value": "456", "icon": "📈"}
], columns=2)
```

#### **Status Panel** (`status_panel.py`)
```python
from smartcash.ui.components import create_status_panel, update_status_panel

# Panel status dengan update capabilities
status = create_status_panel("Memuat data...", "info")
update_status_panel(status, "Data berhasil dimuat!", "success")
```

#### **Log Accordion** (`log_accordion.py`)
```python
from smartcash.ui.components import create_log_accordion, update_log

# Log accordion dengan expandable UI
log_ui = create_log_accordion("proses")
update_log(log_ui, "🚀 Memulai proses...")
update_log(log_ui, "✅ Proses selesai", expand=True)
```

#### **Tabs** (`tabs.py`)
```python
from smartcash.ui.components import create_tabs, create_tab_widget

# Tab widget untuk organisasi UI
tabs = create_tabs([
    ("Konfigurasi", config_widget),
    ("Hasil", results_widget)
])
```

#### **Action Buttons** (`action_buttons.py`)
```python
from smartcash.ui.components import create_action_buttons

# Tombol aksi standar
buttons = create_action_buttons(
    actions=['download', 'check', 'cleanup'],
    layout='horizontal'  # atau 'vertical'
)
```

### 🎛️ Form Widgets

#### **Dropdown** (`widgets/dropdown.py`)
```python
from smartcash.ui.components import create_dropdown

# Dropdown dengan options
dropdown = create_dropdown(
    options=['Option 1', 'Option 2', 'Option 3'],
    value='Option 1',
    description='Pilih opsi:',
    style='info'
)
```

#### **Checkbox** (`widgets/checkbox.py`)
```python
from smartcash.ui.components import create_checkbox

# Checkbox dengan custom styling
checkbox = create_checkbox(
    value=True,
    description='Enable fitur',
    style='success'
)
```

#### **Text Input** (`widgets/text_input.py`)
```python
from smartcash.ui.components import create_text_input

# Text input dengan validasi
text_input = create_text_input(
    value='default_value',
    placeholder='Masukkan teks...',
    description='Input teks:',
    validation_type='path'  # atau 'email', 'number'
)
```

#### **Sliders** (`widgets/slider.py`, `widgets/log_slider.py`)
```python
from smartcash.ui.components import create_slider, create_log_slider

# Slider biasa
slider = create_slider(
    value=50,
    min=0,
    max=100,
    step=1,
    description='Nilai:'
)

# Logarithmic slider untuk ranges besar
log_slider = create_log_slider(
    value=1000,
    min=1,
    max=10000,
    description='Learning Rate:'
)
```

### 🔀 Layout Components

#### **Layout Utilities** (`layout/layout_components.py`)
```python
from smartcash.ui.components.layout import (
    create_responsive_container,
    create_responsive_two_column,
    create_divider,
    get_responsive_config
)

# Container responsive
container = create_responsive_container([widget1, widget2])

# Two-column layout
two_col = create_responsive_two_column(left_widgets, right_widgets)

# Divider
divider = create_divider()

# Responsive config
config = get_responsive_config()
```

### 🚨 Alert Components

#### **Alert System** (`alerts/alert_components.py`)
```python
from smartcash.ui.components.alerts import (
    create_alert,
    create_status_indicator,
    create_info_box
)

# Alert dengan berbagai tipe
alert = create_alert("Pesan sukses", "success", dismissible=True)
status = create_status_indicator("Loading...", "info")
info_box = create_info_box("Informasi penting", "warning")
```

### 💬 Dialog Components

#### **Dialog System** (`dialog/dialog_components.py`)
```python
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area
)

# Confirmation dialog
def on_confirm():
    print("Dikonfirmasi!")

show_confirmation_dialog(
    "Yakin ingin menghapus?",
    on_confirm,
    title="Konfirmasi Hapus"
)

# Info dialog
show_info_dialog("Operasi berhasil!", "Sukses")

# Clear dialog
clear_dialog_area()
```

### 📊 Progress Tracker

#### **Progress System** (`progress_tracker/`)
```python
from smartcash.ui.components.progress_tracker import (
    create_single_progress_tracker,
    create_dual_progress_tracker,
    create_triple_progress_tracker,
    ProgressLevel,
    ProgressConfig
)

# Single progress tracker
tracker = create_single_progress_tracker(
    title="Memproses Data",
    auto_hide=True
)

# Dual progress tracker
dual_tracker = create_dual_progress_tracker(
    title="Training Model",
    overall_label="Epochs",
    step_label="Batches"
)

# Triple progress tracker
triple_tracker = create_triple_progress_tracker(
    title="Preprocessing",
    overall_label="Splits",
    step_label="Files",
    current_label="Operations"
)

# Update progress
tracker.update_overall(50, "Epoch 5/10")
tracker.update_step(75, "Batch 750/1000")
```

### 📋 Info Components

#### **Info System** (`info/info_components.py`)
```python
from smartcash.ui.components.info import (
    create_info_accordion,
    create_tabbed_info,
    style_info_content
)

# Info accordion
info_accordion = create_info_accordion(
    title="Petunjuk Penggunaan",
    content="Ini adalah petunjuk...",
    icon="help",
    open_by_default=False
)

# Tabbed info
tabbed_info = create_tabbed_info([
    ("Overview", "Penjelasan umum..."),
    ("Details", "Detail teknis...")
])
```

## 🎨 Styling & Theming

### Color Constants
```python
# Dari alerts/constants.py
COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

ICONS = {
    'success': '✅',
    'warning': '⚠️',
    'error': '❌',
    'info': 'ℹ️',
    'loading': '⏳'
}
```

### Layout Presets
```python
# Dari layout/layout_components.py
LAYOUTS = {
    'responsive_container': widgets.Layout(
        width='100%', 
        max_width='100%', 
        padding='10px', 
        overflow='hidden'
    ),
    'responsive_button': widgets.Layout(
        width='auto', 
        max_width='150px', 
        height='32px', 
        margin='2px'
    ),
    'two_column_left': widgets.Layout(
        width='47%', 
        margin='0', 
        padding='4px'
    ),
    'two_column_right': widgets.Layout(
        width='47%', 
        margin='0', 
        padding='4px'
    )
}
```

## 🔧 Pola Penggunaan

### 1. **Lazy Loading Pattern**
```python
# Komponen widgets di-load secara lazy untuk avoid circular imports
def __getattr__(name):
    if name == 'create_dropdown':
        from smartcash.ui.components.widgets.dropdown import create_dropdown
        return create_dropdown
    # ... more lazy imports
```

### 2. **Consistent Naming Convention**
```python
# Semua komponen mengikuti pattern create_[component_name]
create_card()
create_dropdown()
create_slider()
create_progress_tracker()
```

### 3. **Responsive Design**
```python
# Semua layout components support responsive design
create_responsive_container()
create_responsive_two_column()
get_responsive_config()
```

### 4. **Error Handling**
```python
# Consistent error handling dengan contextual messages
try:
    component = create_component()
except Exception as e:
    logger.error(f"🚨 Error creating component: {e}")
    return fallback_component()
```

## 🚀 Best Practices

### ✅ **DRY Principle**
- Reuse komponen dari `ui/components/**`
- Consistent styling menggunakan LAYOUTS dan COLORS
- Shared utilities untuk common operations

### ✅ **One-Liner Style**
```python
# Prefer concise, readable code
dropdown = create_dropdown(options=['A', 'B'], value='A', description='Select:')
```

### ✅ **Contextual Logging**
```python
# Use contextual emojis dalam logs
logger.info("🚀 Starting process...")
logger.success("✅ Process completed!")
logger.warning("⚠️ Warning detected")
logger.error("🚨 Error occurred")
```

### ✅ **Progressive Enhancement**
- Support untuk auto-hide pada progress trackers
- Expandable accordions untuk advanced options
- Responsive layouts untuk berbagai screen sizes

### ✅ **Integration Ready**
- Komponen dirancang untuk integrasi dengan handlers
- Consistent return types untuk easy chaining
- Support untuk callback functions

## 🎯 Pola Integrasi

### **Dengan Handlers**
```python
# Komponen mengembalikan dict dengan keys yang konsisten
ui_components = create_form_components(config)
setup_handlers(ui_components)  # Handlers expect standard keys
```

### **Dengan Progress Tracking**
```python
# Progress components support callback integration
def progress_callback(level, current, total, message):
    tracker.update_progress(level, (current/total)*100, message)

process_data(data, progress_callback=progress_callback)
```

### **Dengan Configuration**
```python
# Komponen support configuration-driven creation
config = get_component_config()
components = create_components_from_config(config)
```

## 📝 Kesimpulan

Shared components SmartCash dirancang dengan prinsip:
- **🔄 Reusability**: Komponen dapat digunakan di berbagai modul
- **📱 Responsiveness**: Layout adaptif untuk berbagai ukuran layar
- **🎨 Consistency**: Styling dan behavior yang seragam
- **⚡ Performance**: Lazy loading dan optimized rendering
- **🔧 Maintainability**: Clear separation of concerns dan modular design

Sistem ini memungkinkan rapid development UI yang konsisten dan maintainable untuk YOLOv5-EfficientNet-B4 SmartCash detection system. 🎉