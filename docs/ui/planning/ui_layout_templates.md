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

```
┌─────────────────────────────────────────────────────────┐
│ Main Container                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Header Container                                    │ │
│ │ - Header (Title, Subtitle, Icon)                    │ │
│ │ - Status Panel                                      │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Form Container                                      │ │
│ │ - Form Sections                                     │ │
│ │ - Save/Reset Buttons                                │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Summary Container                                   │ │
│ │ - Summary Cards                                     │ │
│ │ - Summary Table                                     │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Action Container                                    │ │
│ │ - Action Buttons                                    │ │
│ │ - Progress Tracker                                  │ │
│ │ - Confirmation Area                                 │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Footer Container                                    │ │
│ │ - Log Accordion                                     │ │
│ │ - Info Box                                          │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Komponen Container dan Isinya

Setiap container memiliki peran khusus dan berisi komponen-komponen tertentu. Berikut adalah penjelasan detail untuk setiap container:

### 1. Header Container

Container ini menampilkan judul halaman, subtitle, dan panel status.

**Komponen di dalamnya:**
- **Header** - Menampilkan judul utama, subtitle, dan ikon opsional
- **Status Panel** - Menampilkan pesan status dengan tipe (info, success, warning, error)

**Contoh penggunaan:**
```python
header_container = create_header_container(
    title="Dataset Preprocessing",
    subtitle="Preprocessing dataset dengan YOLO normalization dan real-time progress",
    icon="🚀"
)
```

### 2. Form Container

Container ini berisi form input dan opsi konfigurasi yang dikelompokkan secara logis, serta tombol save/reset jika diperlukan.

**Komponen di dalamnya:**
- **Form** - Form input yang dikelompokkan dalam section
- **Save/Reset Buttons** - Tombol untuk menyimpan atau mengatur ulang konfigurasi

**Contoh penggunaan:**
```python
form_container = create_form_container()
form_container['form_container'].children = (input_options,)
```

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

Container ini berisi tombol-tombol aksi utama, progress tracker, dan area konfirmasi.

**Komponen di dalamnya:**
- **Action Buttons** - Tombol-tombol untuk melakukan operasi utama
- **Progress Tracker** - Menampilkan progress operasi yang sedang berjalan
- **Confirmation Area** - Area untuk konfirmasi tindakan penting

**Contoh penggunaan:**
```python
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

# Hubungkan handler dengan tombol
preprocess_btn = action_container['buttons']['preprocess']
preprocess_btn.on_click(on_preprocess_click)
```

### 5. Footer Container

Container ini berisi log accordion dan info box.

**Komponen di dalamnya:**
- **Log Accordion** - Menampilkan log operasi dalam format accordion yang dapat diperluas
- **Info Box** - Menampilkan tips dan informasi penting

**Contoh penggunaan:**
```python
# Log Accordion
log_accordion = create_log_accordion(
    module_name='preprocessing',
    height='200px'
)

# Footer Container dengan Log Accordion dan Info Box
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
