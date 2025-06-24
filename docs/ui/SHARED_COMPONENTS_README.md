# UI Components Documentation

Modul komponen UI yang dapat digunakan kembali untuk aplikasi SmartCash. Dibangun dengan `ipywidgets` dan dirancang untuk konsistensi dan kemudahan penggunaan.

## Daftar Isi
1. [Cara Menggunakan](#cara-menggunakan)
2. [Komponen Inti](#komponen-inti)
   - [Kartu (Cards)](#kartu-cards)
   - [Panel Status](#panel-status)
   - [Log Accordion](#log-accordion)
   - [Tab](#tab)
   - [Tombol Simpan/Reset](#tombol-simpanreset)
   - [Info Accordion](#info-accordion)
3. [Komponen Form](#komponen-form)
   - [Input Teks](#input-teks)
   - [Checkbox](#checkbox)
   - [Dropdown](#dropdown)
   - [Slider](#slider)
   - [Logarithmic Slider](#logarithmic-slider)
4. [Submodul](#submodul)
   - [Alerts](#alerts)
   - [Dialog](#dialog)
   - [Header](#header)
   - [Progress Tracker](#progress-tracker)
   - [Layout](#layout)
   - [Info](#info)

## Cara Menggunakan

```python
# Import komponen yang dibutuhkan
from smartcash.ui.components import (
    create_card, create_status_panel, 
    create_log_accordion, create_tabs
)

# Atau import semua komponen
from smartcash.ui.components import *
```

## Komponen Inti

### Kartu (Cards)

Komponen kartu untuk menampilkan informasi dalam bentuk yang rapi.

```python
# Membuat kartu sederhana
card = create_card(
    title="Total Data",
    value="1,234",
    icon="📊",
    color="#4CAF50"
)

# Membuat kartu dengan tipe bawaan
info_card = create_info_card("Info", "Ini adalah info penting", "ℹ️")
success_card = create_success_card("Sukses", "Operasi berhasil", "✅")
warning_card = create_warning_card("Peringatan", "Ada yang perlu diperhatikan", "⚠️")
error_card = create_error_card("Error", "Terjadi kesalahan", "❌")

# Membuat baris kartu
cards = create_card_row([
    {"title": "Data 1", "value": "123", "icon": "🔢"},
    {"title": "Data 2", "value": "456", "icon": "📈"},
], columns=2)
```

### Panel Status

Menampilkan pesan status dengan gaya yang konsisten.

```python
# Membuat panel status
status = create_status_panel("Memuat data...", "info")

# Memperbarui panel status
update_status_panel(status, "Data berhasil dimuat!", "success")
```

### Log Accordion

Komponen untuk menampilkan log dengan format yang rapi.

```python
# Membuat log accordion
log_ui = create_log_accordion("proses")

# Menambahkan pesan ke log
update_log(log_ui, "Memulai proses...")
update_log(log_ui, "Proses selesai", expand=True)
```

### Tab

Membuat antarmuka tab yang rapi.

```python
# Membuat tab
tabs = create_tabs([
    ("Tab 1", widget1),
    ("Tab 2", widget2)
])

# Atau menggunakan fungsi alternatif
tabs = create_tab_widget([("Tab 1", widget1), ("Tab 2", widget2)])
```

### Tombol Simpan/Reset

Komponen tombol standar untuk aksi simpan dan reset.

```python
buttons = create_save_reset_buttons(
    save_label="Simpan Perubahan",
    reset_label="Reset",
    save_tooltip="Simpan konfigurasi saat ini",
    reset_tooltip="Kembalikan ke pengaturan awal"
)
```

### Info Accordion

Komponen accordion untuk menampilkan informasi tambahan.

```python
info = create_info_accordion(
    title="Petunjuk",
    content="<p>Ini adalah petunjuk penggunaan.</p>",
    open_by_default=False
)
```

## Komponen Form

### Input Teks

Komponen input teks dengan validasi opsional.

```python
from smartcash.ui.components import create_text_input

text_input = create_text_input(
    name="username",
    value="",
    description="Username:",
    placeholder="Masukkan username",
    tooltip="Username untuk login ke sistem"
)
```

### Checkbox

Komponen checkbox untuk input boolean.

```python
from smartcash.ui.components import create_checkbox

checkbox = create_checkbox(
    name="enable_feature",
    value=True,
    description="Aktifkan fitur",
    tooltip="Centang untuk mengaktifkan fitur khusus"
)
```

### Dropdown

Komponen dropdown untuk memilih dari beberapa opsi.

```python
from smartcash.ui.components import create_dropdown

dropdown = create_dropdown(
    name="model_type",
    value="yolov5s",
    options=["yolov5s", "yolov5m", "yolov5l", "yolov5x"],
    description="Model:",
    tooltip="Pilih model yang akan digunakan"
)
```

### Slider

Komponen slider untuk memilih nilai numerik dalam rentang tertentu.

```python
from smartcash.ui.components import create_slider

slider = create_slider(
    name="threshold",
    value=0.5,
    min_val=0.0,
    max_val=1.0,
    step=0.05,
    description="Threshold:",
    tooltip="Nilai threshold untuk deteksi objek"
)
```

### Logarithmic Slider

Komponen slider dengan skala logaritmik, cocok untuk rentang nilai yang lebar.

```python
from smartcash.ui.components import create_log_slider

log_slider = create_log_slider(
    name="learning_rate",
    value=0.001,
    min_val=0.0001,
    max_val=0.1,
    step=0.0001,
    description="Learning Rate:",
    tooltip="Nilai learning rate untuk training model"
)
```

## Submodul

### Alerts

Komponen untuk menampilkan notifikasi dan pesan.

```python
from smartcash.ui.components.alerts import create_alert

alert = create_alert(
    message="Ini adalah pesan penting",
    alert_type="success",
    title="Berhasil",
    icon="✅"
)
```

### Dialog

Komponen dialog interaktif.

```python
from smartcash.ui.components.dialog import show_confirmation_dialog

show_confirmation_dialog(
    ui_components=ui,
    title="Konfirmasi",
    message="Apakah Anda yakin?",
    on_confirm=lambda: print("Dikonfirmasi"),
    on_cancel=lambda: print("Dibatalkan")
)
```

### Header

Komponen header untuk antarmuka pengguna.

```python
from smartcash.ui.components.header import create_header

header = create_header(
    title="Dashboard",
    description="Ringkasan data dan statistik",
    icon="📊"
)
```

### Progress Tracker

Komponen untuk melacak kemajuan operasi.

```python
from smartcash.ui.components.progress_tracker import create_triple_progress_tracker

tracker = create_triple_progress_tracker()
tracker.update_overall(50, "Sedang memproses...")
tracker.update_step(1, "Tahap 1 selesai")
tracker.complete("Proses selesai!")
```

### Layout

Komponen untuk tata letak responsif.

```python
from smartcash.ui.components.layout import (
    create_divider,
    create_responsive_container,
    create_responsive_two_column,
    get_responsive_button_layout
)

# Membuat pembatas visual
divider = create_divider()

# Membuat container responsif
container = create_responsive_container(
    children=[widget1, widget2],
    container_type='vbox',
    layout={'padding': '10px'}
)

# Membuat tata letak dua kolom
two_col = create_responsive_two_column(
    left_content=left_widget,
    right_content=right_widget,
    left_width='60%',
    right_width='40%'
)

# Mendapatkan tata letak tombol responsif
button_layout = get_responsive_button_layout(width='200px', max_width='100%')
```

### Info

Komponen untuk menampilkan informasi tambahan dan dokumentasi.

```python
from smartcash.ui.components.info.info_components import create_info_section

info_section = create_info_section(
    title="Panduan Penggunaan",
    content="""
    <h4>Langkah-langkah:</h4>
    <ol>
        <li>Pilih dataset yang akan digunakan</li>
        <li>Konfigurasi parameter training</li>
        <li>Klik tombol 'Mulai Training'</li>
    </ol>
    """,
    icon="ℹ️"
)
```

## Panduan Gaya

1. **Konsistensi**: Gunakan komponen yang ada untuk memastikan konsistensi UI
2. **Dokumentasi**: Selalu lihat dokumentasi fungsi untuk parameter yang tersedia
3. **Responsif**: Komponen didesain untuk responsif, tetapi selalu uji di berbagai ukuran layar
4. **Aksesibilitas**: Pastikan kontras warna cukup dan teks mudah dibaca

