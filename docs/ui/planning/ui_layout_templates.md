# SmartCash UI Layout Templates

## Konvensi Penamaan

Penting untuk mengikuti konvensi penamaan yang konsisten untuk komponen UI:

- **Komponen UI Utama**: Gunakan format `[module]_ui` (contoh: `preprocessing_ui`, `augmentation_ui`)
- **Komponen Section**: Gunakan format `[nama]_section` (contoh: `normalization_section`, `validation_section`)
- **Komponen Form**: Gunakan format `[nama]_form` (contoh: `input_form`, `config_form`)
- **Komponen Lainnya**: Gunakan penamaan deskriptif yang jelas (contoh: `progress_tracker`, `log_accordion`)

## Custom Layout vs Standard Layout

SmartCash mendukung dua pendekatan layout:

1. **Standard Layout**: Mengikuti struktur container-based yang dijelaskan di bawah, dengan komponen yang disusun secara vertikal.
2. **Custom Layout**: Untuk kasus khusus yang memerlukan susunan komponen yang berbeda, seperti pada modul `env_config` yang menggunakan grid layout dan card-based UI.

## Struktur Utama Container-Based Layout

SmartCash menggunakan pendekatan container-based layout untuk memastikan konsistensi UI di seluruh aplikasi. Berikut adalah struktur utama yang digunakan:

### Komponen Container Utama

- **Header Container**
  - Header (must)
  - Status Panel (must)

- **Form Container**
  - Custom to each module
  - Input fields and form controls
  - Validation messages

- **Action Container**
  - Save/Reset Buttons (only if need persistence config) | (float right)
  - Big Primary Buttons (for single operation only) | (float center)
  - Action Buttons (for multiple operations) | (float left)

- **Summary Container** (Nice to have)
  - Custom to each module
  - Displays summary or preview of settings

- **Operation Container**
  - Progress Tracker (must)
  - Dialog Confirmation Area (Opsional)
  - Log Accordion (must)

- **Footer Container**
  - Info Accordion(s) (Nice to have)
  - Tips Panel (opsional)

### Struktur File UI

- `[module]_ui.py`: Berisi definisi komponen UI utama
- `*_section.py`: Berisi definisi komponen UI untuk section
- `*_panel.py`: Berisi komponen UI panel
- `*_widget.py`: Berisi komponen widget spesifik
- `*[what]_info.py`: Berisi informasi atau panduan yang bisa di-collapse (harus diletakkan di folder `info_box/`)

### Komponen UI Standar

1. **Header Container**
   - Menunjukkan modul yang sedang aktif
   - Menampilkan status operasi terbaru
   - Dapat menyertakan tombol aksi cepat

2. **Form Container**
   - Input fields yang terorganisir dalam section
   - Validasi input inline
   - Tombol aksi yang relevan dengan konteks

3. **Visualization Container**
   - Menampilkan output atau visualisasi data
   - Dapat berupa tabel, grafik, atau preview
   - Mendukung interaksi pengguna

4. **Operation Container**
   - Progress bar untuk operasi yang berjalan
   - Confirmation dialog area
   - Logging real-time dengan filter level
   - Notifikasi untuk event penting

### Tipe Layout Khusus

1. **Wizard/Stepper**
   - Untuk alur multi-langkah
   - Navigasi antara step
   - Validasi antar step

2. **Dashboard**
   - Koleksi widget dan metrik
   - Layout grid yang responsif
   - Customizable views

3. **Explorer/Viewer**
   - Fokus pada tampilan data
   - Navigasi dan filter yang kuat
   - Detail view yang interaktif

## Komponen Container dan Isinya

Setiap container memiliki peran khusus dan berisi komponen-komponen tertentu. Berikut adalah penjelasan detail untuk setiap container:

### 1. Header Container

Container untuk menampilkan judul halaman, subtitle, dan panel status yang konsisten di seluruh aplikasi.

**Fitur Utama:**
- Menampilkan judul utama dan subjudul
- Panel status yang dapat diperbarui dengan berbagai tipe pesan
- Dukungan untuk ikon opsional
- Tampilan yang responsif

**Contoh Penggunaan Dasar:**
```python
from smartcash.ui.components.header_container import create_header_container

# Membuat header container dengan status panel
header = create_header_container(
    title="🚀 Environment Setup",
    subtitle="Configure environment for SmartCash YOLOv5-EfficientNet",
    status_message="Ready to configure environment",
    status_type="info"
)

# Menampilkan header
display(header.container)

# Memperbarui status
header.update_status(
    message="Environment configured successfully",
    status_type="success"
)

# Menyembunyikan/menampilkan status panel
header.toggle_status_panel(show=False)
```

**Metode yang Tersedia:**
- `update_status(message: str, status_type: str = "info")` - Memperbarui pesan status
- `toggle_status_panel(show: bool = None)` - Menyembunyikan/menampilkan panel status
- `set_title(title: str)` - Mengatur judul
- `set_subtitle(subtitle: str)` - Mengatur subjudul

**Parameter Pembuatan:**
- `title`: Judul utama (string)
- `subtitle`: Subjudul (string, opsional)
- `icon`: Ikon opsional (emoji atau karakter unicode)
- `status_message`: Pesan status awal
- `status_type`: Tipe status ("info", "success", "warning", "error")
- `show_status_panel`: Menampilkan/menyembunyikan panel status (default: True)
- `**style_options`: Opsi gaya tambahan (margin, padding, dll.)

### 2. Form Container

Container fleksibel untuk membuat form dengan berbagai tipe input dan tata letak yang responsif.

**Fitur Utama:**
- Mendukung tata letak kolom, baris, atau grid
- Validasi input otomatis
- Pengelompokan field yang logis
- Tampilan yang responsif
- Dukungan untuk berbagai tipe input

**Contoh Penggunaan Dasar:**
```python
from smartcash.ui.components.form_container import create_form_container, LayoutType
import ipywidgets as widgets

# Membuat form container dengan tata letak kolom
form = create_form_container(
    layout_type=LayoutType.COLUMN,
    container_margin="16px 0",
    container_padding="16px",
    gap="12px"
)

# Menambahkan field ke form
form['add_item'](
    widgets.IntText(description="Batch Size:", value=32),
    align_items='center',
    margin='0 0 8px 0'
)

form['add_item'](
    widgets.Dropdown(
        options=['CPU', 'GPU', 'TPU'],
        value='GPU',
        description='Device:'
    ),
    align_items='center'
)

# Menampilkan form
display(form['container'])
```

**Contoh dengan Grid Layout:**
```python
# Membuat form dengan tata letak grid
grid_form = create_form_container(
    layout_type=LayoutType.GRID,
    grid_columns=2,  # 2 kolom
    gap="12px"
)

# Menambahkan field ke grid
for i in range(4):
    grid_form['add_item'](
        widgets.FloatText(description=f"Parameter {i+1}:", value=0.5),
        grid_area=f"item{i+1}"  # Posisi di grid
    )

# Mengubah tata letak secara dinamis
grid_form['set_layout'](grid_columns=3)  # Ubah ke 3 kolom
```

**Metode yang Tersedia:**
- `add_item(widget, **layout_options)` - Menambahkan widget ke form
- `set_layout(**layout_options)` - Mengubah tata letak form
- `clear()` - Menghapus semua item dari form

**Opsi Tata Letak:**
- `layout_type`: `LayoutType.COLUMN`, `LayoutType.ROW`, atau `LayoutType.GRID`
- `gap`: Jarak antar item (contoh: "8px")
- `container_margin`: Margin container luar
- `container_padding`: Padding container dalam
- `grid_columns`: Jumlah kolom untuk tata letak grid
- `grid_template_areas`: Template area untuk tata letak grid
- `align_items`: Penyelarasan vertikal item
- `justify_content`: Penyelarasan horizontal item

**Kelas FormItem:**
Membungkus widget form dengan opsi tata letak tambahan:
- `widget`: Widget yang dibungkus
- `align_items`: Penyelarasan vertikal
- `margin`: Margin di sekitar item
- `grid_area`: Area grid untuk penempatan (hanya untuk tata letak grid)

### 3. Summary Container

Container ini menampilkan ringkasan informasi penting atau hasil dari operasi.

**Komponen di dalamnya:**
- **Summary Cards** - Kartu yang menampilkan statistik atau informasi penting
- **Summary Table** - Tabel yang menampilkan data ringkasan

**Contoh penggunaan:**
```python
summary_container = create_summary_container()
summary_container.children = (summary_stats_widget,)
```

### 4. Action Container

Container untuk mengelola tombol aksi dengan dukungan fase dan status yang berbeda. Khususnya berguna untuk alur kerja multi-tahap seperti penyiapan lingkungan atau pelatihan model.

**Fitur Utama:**
- Tombol aksi dengan dukungan fase (phases)
- Status tombol yang dapat disesuaikan (enabled/disabled)
- Tooltip dan ikon opsional
- Tampilan yang konsisten di seluruh aplikasi

**Contoh Penggunaan Dasar:**
```python
from smartcash.ui.components.action_container import create_action_container

# Membuat action container
action_container = create_action_container(
    buttons=[
        {
            "button_id": "setup_env",
            "text": "🚀 Setup Environment",
            "style": "primary",
            "order": 1,
            "tooltip": "Initialize the development environment"
        },
        {
            "button_id": "train_model",
            "text": "🤖 Train Model",
            "style": "success",
            "order": 2,
            "disabled": True
        }
    ]
)

# Mendapatkan referensi ke tombol
setup_btn = action_container.get_button('setup_env')
train_btn = action_container.get_button('train_model')

# Menambahkan event handler
def on_setup_click(button):
    button.disabled = True
    # Lakukan inisialisasi...
    train_btn.disabled = False  # Aktifkan tombol train setelah setup selesai

setup_btn.on_click(on_setup_click)

# Menampilkan container
display(action_container.container)
```

**Contoh dengan Fase Otomatis:**
```python
# Mendefinisikan fase-fase untuk tombol
phases = {
    'initial': {
        'text': '🚀 Initialize',
        'style': 'primary',
        'disabled': False
    },
    'processing': {
        'text': '⏳ Processing...',
        'style': 'info',
        'disabled': True
    },
    'completed': {
        'text': '✅ Completed',
        'style': 'success',
        'disabled': False
    }
}

# Membuat action container dengan fase
action = create_action_container(phases=phases)

# Mengubah fase secara dinamis
action.set_phase('processing')
# ... lakukan pekerjaan ...
action.set_phase('completed')
```

**Metode yang Tersedia:**
- `get_button(button_id)` - Mendapatkan referensi ke tombol
- `set_phase(phase_id)` - Mengubah fase tombol
- `enable_all()` - Mengaktifkan semua tombol
- `disable_all()` - Menonaktifkan semua tombol
- `set_all_buttons_enabled(enabled)` - Mengatur status aktif/tidak aktif semua tombol

**Fase Bawaan untuk Setup Environment:**
1. `initial` - Fase awal
2. `init` - Sedang menginisialisasi
3. `drive` - Sedang memounting Google Drive
4. `symlink` - Membuat symlink
5. `folders` - Membuat struktur folder
6. `dependencies` - Menginstal dependensi
7. `models` - Mengunduh model
8. `verifying` - Memverifikasi setup
9. `ready` - Siap digunakan
10. `error` - Terjadi kesalahan

**Contoh Penggunaan Fase Bawaan:**
```python
# Menggunakan fase bawaan untuk setup environment
action = create_action_container()

# Atur fase secara berurutan
action.set_phase('initial')
# ... lakukan inisialisasi ...
action.set_phase('drive')
# ... mount drive ...
action.set_phase('ready')

# Atau langsung ke fase error jika terjadi masalah
action.set_phase('error')
```
# Progress Tracker dalam Main Container
progress_tracker = ProgressTracker(
    operation="Dataset Preprocessing",
    level=ProgressLevel.DUAL,
    auto_hide=False
)

# Menambahkan progress tracker ke action container
action_container = create_action_container(...)
action_container.add_component('progress_tracker', progress_tracker)

# Atau menambahkan progress tracker langsung ke main container
main_container = create_main_container(
    header_container=header_container.container,
    form_container=form_container['container'],
    action_container=action_container.container,
    progress_container=progress_tracker,  # Sebagai parameter terpisah
    footer_container=footer_container.container
)

# Dialog Konfirmasi menggunakan Action Container
# Contoh penggunaan dialog konfirmasi
def on_preprocess_click(b):
    action_container['show_dialog'](
        title="Konfirmasi Preprocessing",
        message="Apakah Anda yakin ingin memulai preprocessing?",
        on_confirm=lambda: start_preprocessing(),
        on_cancel=lambda: print("Dibatalkan"),
        confirm_text="Ya, Mulai",
        cancel_text="Batal"
    )

# Membuat progress indicator
progress_indicator = ProgressIndicator(
    value=0,
    min=0,
    max=100,
    description='Progress:',
    bar_style='info',
    style={'bar_color': '#0078d4'},
    orientation='horizontal'
)

# Membuat log accordion
log_accordion = LogAccordion(
    title="Training Logs",
    max_lines=1000,
    auto_scroll=True,
    timestamp_format="%H:%M:%S"
)

# Membuat confirmation area
confirmation_area = ConfirmationArea(
    confirm_text="Confirm",
    cancel_text="Cancel",
    style="warning"
)

# Menambahkan komponen ke operation container
operation_container.add_component(progress_indicator, "progress")
operation_container.add_component(log_accordion, "logs")
operation_container.add_component(confirmation_area, "confirmation")

# Contoh penggunaan
log_accordion.append("Starting training process...")
progress_indicator.value = 10

# Update progress secara bertahap
def update_progress():
    for i in range(10, 100, 10):
        progress_indicator.value = i
        log_accordion.append(f"Progress: {i}%")
        time.sleep(0.5)
    log_accordion.append("Training completed successfully!")
    progress_indicator.bar_style = 'success'

# Menjalankan update progress
import threading
progress_thread = threading.Thread(target=update_progress)
progress_thread.start()

# Contoh penggunaan confirmation area
def on_confirm():
    log_accordion.append("Action confirmed!")

def on_cancel():
    log_accordion.append("Action cancelled!")

confirmation_area.on_confirm = on_confirm
confirmation_area.on_cancel = on_cancel

# Menampilkan operation container
display(operation_container.container)

### 5. Operation Container

Container untuk mengelola operasi yang membutuhkan pelacakan progress, logging, dan dialog konfirmasi. Sangat berguna untuk operasi yang berjalan lama seperti pelatihan model atau pemrosesan data.

**Fitur Utama:**
- Multiple progress levels (primary, secondary, tertiary)
- Logging terintegrasi dengan berbagai level
- Dialog konfirmasi dan informasi
- Tampilan yang responsif dan terorganisir

**Contoh Penggunaan Dasar:**
```python
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.log_accordion import LogLevel
from IPython.display import display

# Membuat operation container
operation = create_operation_container(
    show_progress=True,
    show_dialog=True,
    show_logs=True,
    log_module_name="Training"
)

# Menampilkan container
display(operation['container'])

# Contoh penggunaan progress tracking
operation.update_progress(
    progress=25,
    message="Loading training data...",
    level="primary"
)

# Contoh logging
operation.log("Starting training process...", LogLevel.INFO)
operation.warning("Low GPU memory detected!")

# Contoh dialog konfirmasi
def on_confirm():
    operation.log("Operation confirmed!")
    
operation['show_dialog'](
    title="Confirm Action",
    message="Are you sure you want to continue?",
    on_confirm=on_confirm,
    confirm_text="Yes, Continue",
    cancel_text="Cancel"
)
```

**Contoh dengan Multiple Progress Levels:**
```python
# Update progress level primer
operation.update_progress(
    progress=30,
    message="Processing dataset...",
    level="primary"
)

# Update progress level sekunder
operation.update_progress(
    progress=75,
    message="Augmenting images...",
    level="secondary"
)

# Sembunyikan progress level tertentu
operation.set_progress_visibility("secondary", False)

# Tandai progress selesai
operation.complete_progress(
    message="Dataset processed successfully!",
    level="primary"
)
```

**Metode yang Tersedia:**
- `update_progress(progress, message, level)` - Memperbarui progress
- `complete_progress(message, level)` - Menandai progress selesai
- `error_progress(message, level)` - Menandai error pada progress
- `reset_progress(level)` - Mereset progress
- `log(message, level)` - Mencatat pesan log
- `show_dialog()` - Menampilkan dialog konfirmasi
- `show_info()` - Menampilkan dialog informasi
- `clear_dialog()` - Menutup dialog yang sedang aktif

**Level Log yang Didukung:**
- `LogLevel.DEBUG` - Pesan debugging
- `LogLevel.INFO` - Informasi umum
- `LogLevel.WARNING` - Peringatan
- `LogLevel.ERROR` - Kesalahan
- `LogLevel.CRITICAL` - Kesalahan kritis

**Contoh Penanganan Error:**
```python
try:
    # Kode yang mungkin menyebabkan error
    operation.update_progress(50, "Processing...")
    raise ValueError("Something went wrong!")
except Exception as e:
    operation.error_progress(f"Error: {str(e)}")
    operation.log(f"Error details: {str(e)}", LogLevel.ERROR)
```

### 6. Footer Container

Container fleksibel untuk menampilkan berbagai panel informasi di bagian bawah antarmuka. Mendukung berbagai tipe panel seperti InfoBox dan InfoAccordion dengan tata letak yang dapat disesuaikan.

**Fitur Utama:**
- Mendukung multiple panel dengan tipe berbeda
- Tata letak fleksibel dengan flexbox
- Responsif terhadap ukuran layar
- Dukungan untuk accordion yang dapat diperluas
- Gaya yang dapat disesuaikan

**Contoh Penggunaan Dasar:**
```python
from smartcash.ui.components.footer_container import (
    create_footer_container,
    PanelConfig,
    PanelType
)

# Membuat footer dengan dua panel
footer = create_footer_container(
    panels=[
        PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="ℹ️ Informasi",
            content="""
            <div style="padding: 8px;">
                <p>Versi Aplikasi: <strong>1.0.0</strong></p>
                <p>Status: <span style="color: green;">✓ Berjalan dengan baik</span></p>
            </div>
            """,
            style="info",
            flex="1",
            min_width="250px"
        ),
        PanelConfig(
            panel_type=PanelType.INFO_ACCORDION,
            title="📊 Statistik",
            content="""
            <div style="padding: 8px;">
                <p>Total Data: <strong>1,245</strong> sampel</p>
                <p>Training: <strong>1,000</strong> (80%)</p>
                <p>Validation: <strong>200</strong> (16%)</p>
                <p>Test: <strong>45</strong> (4%)</p>
            </div>
            """,
            style="primary",
            flex="1",
            min_width="250px",
            open_by_default=False
        )
    ],
    # Gaya tambahan untuk container
    style={
        "border_top": "1px solid #dee2e6",
        "padding": "12px",
        "background": "#f8f9fa"
    },
    # Konfigurasi tata letak
    flex_flow="row wrap",
    justify_content="space-between",
    align_items="flex-start"
)

# Menambahkan panel secara dinamis
footer.add_panel(
    PanelConfig(
        panel_type=PanelType.INFO_ACCORDION,
        title="🔧 Pengaturan Cepat",
        content="""
        <div style="padding: 8px;">
            <button class="btn btn-sm btn-outline-secondary">Reset Pengaturan</button>
            <button class="btn btn-sm btn-outline-primary">Simpan Konfigurasi</button>
        </div>
        """,
        style="secondary",
        flex="1",
        min_width="300px"
    )
)

# Menampilkan footer
display(footer.container)
```

**Metode yang Tersedia:**
- `add_panel(panel_config)` - Menambahkan panel baru
- `remove_panel(panel_id)` - Menghapus panel berdasarkan ID
- `update_panel(panel_id, **updates)` - Memperbarui panel yang ada
- `clear_panels()` - Menghapus semua panel
- `toggle_panel(panel_id, is_open)` - Membuka/menutup panel accordion

**Opsi Panel:**
- `panel_type`: `PanelType.INFO_BOX` atau `PanelType.INFO_ACCORDION`
- `title`: Judul panel
- `content`: Konten HTML panel
- `style`: Gaya tampilan ("info", "primary", "success", "warning", "danger")
- `flex`: Nilai flex untuk tata letak
- `min_width`: Lebar minimum panel
- `open_by_default`: Buka accordion secara default (hanya untuk INFO_ACCORDION)

**Contoh Lanjutan dengan Event Handling:**
```python
# Membuat panel dengan konten interaktif
def on_reset_clicked(b):
    print("Pengaturan direset!")

def on_save_clicked(b):
    print("Konfigurasi disimpan!")

# Buat tombol dengan event handler
reset_btn = widgets.Button(description="Reset Pengaturan", 
                          layout={"width": "48%", "margin": "0 1% 0 0"})
save_btn = widgets.Button(description="Simpan Konfigurasi",
                         button_style="primary",
                         layout={"width": "48%", "margin": "0 0 0 1%"})

reset_btn.on_click(on_reset_clicked)
save_btn.on_click(on_save_clicked)

# Tambahkan ke dalam panel
footer.add_panel(
    PanelConfig(
        panel_type=PanelType.INFO_BOX,
        title="🎛️ Kontrol Cepat",
        content=widgets.HBox([reset_btn, save_btn]),
        style="light",
        flex="1",
        min_width="350px"
    )
)
```
```

## Contoh Layout dari Preprocessing Components

Berikut adalah contoh implementasi layout dari modul preprocessing dengan mengikuti konvensi penamaan yang disarankan:

```python
def create_preprocessing_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """🎨 Create preprocessing UI using shared container components"""
    config = config or {}
    
    # Initialize UI components dictionary
    ui_components = {}
    
    # === CORE COMPONENTS ===
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="Dataset Preprocessing",
        subtitle="Preprocessing dataset dengan YOLO normalization",
        icon="🚀"
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Create Form Components
    preprocessing_form = create_preprocessing_form(config)
    
    # 3. Create Form Container
    form_container = create_form_container()
    form_container['form_container'].children = (preprocessing_form,)
    form_container['save_button'] = widgets.Button(description="💾 Simpan Konfigurasi")
    form_container['reset_button'] = widgets.Button(description="🔄 Reset")
    ui_components['form_container'] = form_container['container']
    
    # 4. Create Summary Container
    summary_section = create_preprocessing_summary_section()
    ui_components['summary_container'] = summary_section
    
    # 5. Create Action Container dengan Dialog Konfirmasi
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "preprocess",
                "text": "🚀 Mulai Preprocessing",
                "style": "primary",
                "order": 1
            },
            {
                "button_id": "check",
                "text": "🔍 Check Dataset",
                "style": "info",
                "order": 2
            },
            {
                "button_id": "cleanup",
                "text": "🧹 Cleanup",
                "style": "warning",
                "order": 3
            }
        ],
        title="🚀 Preprocessing Operations",
        alignment="left"
    )
    
    # Simpan referensi ke dialog area dan fungsi dialog
    ui_components['confirmation_area'] = action_container['dialog_area']
    ui_components['show_dialog'] = action_container['show_dialog']
    ui_components['show_info'] = action_container['show_info']
    ui_components['clear_dialog'] = action_container['clear_dialog']
    ui_components['is_dialog_visible'] = action_container['is_dialog_visible']
    
    # Tambahkan progress tracker ke action container
    progress_tracker = ProgressTracker(
        operation="Dataset Preprocessing",
        level=ProgressLevel.DUAL,
        auto_hide=False
    )
    action_container.add_component('progress_tracker', progress_tracker)
    
    # Contoh penggunaan dialog konfirmasi
    def on_preprocess_click(b):
        action_container['show_dialog'](
            title="Konfirmasi Preprocessing",
            message="Apakah Anda yakin ingin memulai preprocessing?",
            on_confirm=lambda: start_preprocessing(),
            on_cancel=lambda: print("Dibatalkan"),
            confirm_text="Ya, Mulai",
            cancel_text="Batal"
        )
    
    # Hubungkan handler dengan tombol
    preprocess_btn = action_container['buttons']['preprocess']
    preprocess_btn.on_click(on_preprocess_click)
    
    ui_components['action_container'] = action_container.container
    ui_components['progress_tracker'] = progress_tracker
    
    # 6. Create Footer Container with Log Accordion and Info Box
    log_accordion = create_log_accordion(
        module_name='preprocessing',
        height='200px'
    )
    
    footer_container = create_footer_container(
        show_buttons=False,
        log_accordion=log_accordion,
        info_box=widgets.HTML(
            value="""<div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8; margin: 10px 0;">
                <h5>ℹ️ Tips</h5>
                <ul>
                    <li>Gunakan resolusi yang sesuai dengan model target</li>
                    <li>Min-Max normalization (0-1) direkomendasikan untuk YOLO</li>
                    <li>Aktifkan validasi untuk memastikan dataset berkualitas</li>
                </ul>
            </div>"""
        )
    )
    ui_components['footer_container'] = footer_container.container
    ui_components['log_accordion'] = log_accordion
    
    # 7. Create Main Container with all components
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=form_container['container'],
        summary_container=summary_section,
        action_container=action_container.container,
        progress_container=progress_tracker,  # Progress tracker sebagai parameter terpisah
        footer_container=footer_container.container
    )
    ui_components['main_container'] = main_container.container
    ui_components['ui'] = main_container.container  # Alias for compatibility
    
    # === BUTTON MAPPING ===
    
    # Extract buttons from action container using standard approach
    preprocess_btn = action_container.get_button('preprocess')
    check_btn = action_container.get_button('check')
    cleanup_btn = action_container.get_button('cleanup')
    
    # Add buttons to ui_components
    ui_components.update({
        'preprocess_btn': preprocess_btn,
        'check_btn': check_btn, 
        'cleanup_btn': cleanup_btn,
    })
    
    return ui_components
```

## Form Layout Pattern

Form layout menggunakan pola section-based dengan HBox dan VBox dengan GRID LAYOUT untuk membuat layout yang responsif:

```
┌─────────────────────────────────────────────────────────┐
│ Form Header                                             │
├───────────────────────────┬─────────────────────────────┤
│ Section 1                 │ Section 2                   │
│ - Input 1                 │ - Input 1                   │
│ - Input 2                 │ - Input 2                   │
│ - Input 3                 │ - Input 3                   │
├───────────────────────────┼─────────────────────────────┤
│ Section 3                 │ Section 4                   │
│ - Input 1                 │ - Input 1                   │
│ - Input 2                 │ - Input 2                   │
├───────────────────────────┴─────────────────────────────┤
│ Save/Reset Buttons                                      │
└─────────────────────────────────────────────────────────┘
```

**Contoh implementasi dengan konvensi penamaan yang disarankan:**

```python
def create_preprocessing_form(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Create preprocessing form dengan section-based layout"""
    if not config:
        config = {}
    
    # === NORMALIZATION SECTION ===
    normalization_section = create_normalization_section(config)
    
    # === PROCESSING SECTION ===
    processing_section = create_processing_section(config)
    
    # === VALIDATION SECTION ===
    validation_section = create_validation_section(config)
    
    # === CLEANUP SECTION ===
    cleanup_section = create_cleanup_section(config)
    
    # === LAYOUT ASSEMBLY ===
    
    # Top row dengan dua section
    top_row = widgets.HBox([normalization_section, processing_section], 
        layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    # Bottom row dengan dua section
    bottom_row = widgets.HBox([validation_section, cleanup_section],
        layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    # Container utama yang menggabungkan semua section
    form_container = widgets.VBox([
        widgets.HTML("<h5 style='margin:8px 0;color:#495057;border-bottom:2px solid #28a745;padding-bottom:4px;'>⚙️ Konfigurasi Preprocessing</h5>"),
        top_row,
        bottom_row
    ], layout=widgets.Layout(
        padding='12px', width='100%', border='1px solid #dee2e6',
        border_radius='6px', background_color='#f8f9fa'
    ))
    
    return form_container
```

## Area Konfirmasi

Area konfirmasi adalah komponen penting untuk operasi yang memerlukan konfirmasi pengguna sebelum dijalankan. Ini biasanya ditampilkan dalam Action Container:

```
┌─────────────────────────────────────────────────────────┐
│ Confirmation Area                                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Konfirmasi Operasi                                  │ │
│ │                                                     │ │
│ │ Apakah Anda yakin ingin melanjutkan?                │ │
│ │                                                     │ │
│ │ [Ya, Lanjutkan]        [Batal]                      │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Contoh implementasi dialog konfirmasi menggunakan ActionContainer:**

```python
# 1. Buat action container dengan tombol-tombol aksi
action_container = create_action_container(
    buttons=[
        {
            "button_id": "preprocess",
            "text": "🚀 Mulai Preprocessing",
            "style": "primary",
            "order": 1
        },
        {
            "button_id": "check",
            "text": "🔍 Check Dataset",
            "style": "info",
            "order": 2
        }
    ],
    title="🚀 Preprocessing Operations",
    alignment="left"
)

# 2. Tambahkan handler untuk tombol preprocess
def on_preprocess_click(b):
    # Tampilkan dialog konfirmasi menggunakan metode bawaan action_container
    action_container['show_dialog'](
        title="Konfirmasi Preprocessing",
        message="Apakah Anda yakin ingin memulai preprocessing?",
        on_confirm=lambda: start_preprocessing(),
        on_cancel=lambda: print("Dibatalkan"),
        confirm_text="Ya, Mulai",
        cancel_text="Batal",
        danger_mode=True  # Optional: untuk styling merah pada tombol konfirmasi
    )

# 3. Hubungkan handler dengan tombol
preprocess_btn = action_container['buttons']['preprocess']
preprocess_btn.on_click(on_preprocess_click)

# 4. Untuk menampilkan dialog info (tanpa tombol cancel)
def show_info_message():
    action_container['show_info'](
        title="Informasi",
        message="Preprocessing selesai dengan sukses!",
        on_confirm=lambda: print("Info ditutup"),
        confirm_text="OK"
    )

# 5. Untuk memeriksa apakah dialog sedang ditampilkan
is_visible = action_container['is_dialog_visible']()

# 6. Untuk menghapus dialog yang sedang ditampilkan
action_container['clear_dialog']()
```

## Keuntungan Container-Based Layout

1. **Konsistensi** - Memastikan tampilan yang konsisten di seluruh aplikasi
2. **Modularitas** - Memudahkan penggantian atau pembaruan komponen individual
3. **Pemeliharaan** - Memudahkan pemeliharaan dan perubahan UI di masa depan
4. **Reusabilitas** - Komponen dapat digunakan kembali di berbagai bagian aplikasi
5. **Responsivitas** - Layout dapat beradaptasi dengan ukuran layar yang berbeda

## Custom Layout Implementation

Untuk kasus khusus yang memerlukan layout yang berbeda dari standar, SmartCash menyediakan fleksibilitas untuk membuat custom layout. Contoh implementasi custom layout dapat dilihat pada modul `env_config`.

### Contoh Custom Layout: Environment Config

Modul `env_config` menggunakan pendekatan card-based UI dengan grid layout untuk menampilkan informasi environment dan setup summary dalam format yang lebih visual.

```python
def create_env_config_ui() -> Dict[str, Any]:
    """🎨 Buat komponen UI untuk environment configuration"""
    # Initialize components dictionary
    ui_components = {}
    
    # 1. Create standard containers
    header_container = create_header_container(...)
    form_components = create_form_container(...)
    summary_container = create_summary_container(...)
    footer_container = create_footer_container(...)
    
    # 2. Create custom components specific to env_config
    setup_summary = create_setup_summary()
    env_info_panel = create_env_info_panel()
    tips_requirements = create_tips_requirements()
    
    # 3. Create the main container with custom component order
    main_container = create_main_container(...)
    
    # 4. Override the default vertical layout with custom component ordering
    all_components = [
        header_container.container,
        form_components['container'],
        summary_container.container,
        env_info_panel,           # Custom component
        tips_requirements,        # Custom component
        footer_container.container
    ]
    
    # Filter out any None components
    all_components = [c for c in all_components if c is not None]
    
    # Replace the main container's children with our custom ordered components
    main_container.container.children = all_components
    
    return ui_components
```

### Komponen Custom Layout

1. **Environment Info Panel** - Panel informasi yang menampilkan detail tentang environment dalam format grid dengan 2 kolom

   ```python
   def create_env_info_panel() -> widgets.HTML:
       # Detect environment info
       env_info = detect_environment_info()
       
       # Return HTML widget with grid layout
       return widgets.HTML(
           value=f"""
           <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
               <div><!-- System Info --></div>
               <div><!-- Resources Info --></div>
           </div>
           """
       )
   ```

2. **Setup Summary** - Card-based summary dengan status updates dan visual indicators

3. **Tips Panel** - Panel tips dengan multi-column layout

   ```python
   def create_tips_requirements() -> widgets.HTML:
       # Define tips content dalam format multi-column
       tips_content = [
           ["Tip 1", "Tip 2", "Tip 3"],  # Column 1
           ["Tip 4", "Tip 5", "Tip 6"]   # Column 2
       ]
       
       return create_tips_panel(
           title="💡 Tips & Requirements",
           tips=tips_content,
           columns=2  # Multi-column layout
       )
   ```

## Praktik Terbaik

1. **Gunakan komponen bersama** - Manfaatkan komponen yang sudah ada di `smartcash.ui.components`
2. **Ikuti konvensi penamaan** - Gunakan format `[module]_ui` untuk UI utama dan `[nama]_section` atau `[nama]_form` untuk komponen
3. **Kelompokkan input terkait** - Kelompokkan input yang terkait dalam section yang sama
4. **Berikan feedback visual** - Gunakan status panel dan progress tracker untuk memberikan feedback
5. **Implementasikan area konfirmasi** - Gunakan area konfirmasi untuk operasi penting
6. **Dokumentasikan komponen** - Berikan docstring dan komentar untuk komponen yang dibuat

## Referensi

### Standard Container Components

- `smartcash.ui.components.main_container` - Container utama untuk layout
- `smartcash.ui.components.header_container` - Container untuk header dan status panel
- `smartcash.ui.components.form_container` - Container untuk form input
- `smartcash.ui.components.action_container` - Container untuk tombol aksi dan dialog konfirmasi
- `smartcash.ui.components.footer_container` - Container untuk footer dan informasi tambahan
- `smartcash.ui.components.dialog.confirmation_dialog` - Komponen untuk dialog konfirmasi
- `smartcash.ui.components.progress_tracker.progress_tracker` - Komponen untuk progress tracking

### Custom Layout Components

- `smartcash.ui.setup.env_config.components` - Contoh implementasi custom layout
  - `env_info_panel.py` - Panel informasi environment dengan grid layout
  - `setup_summary.py` - Card-based summary dengan status updates
  - `tips_panel.py` - Panel tips dengan multi-column layout
  - `ui_components.py` - Implementasi custom layout dengan komponen standard dan custom
