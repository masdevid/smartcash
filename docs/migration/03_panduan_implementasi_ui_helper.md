# Panduan Implementasi ui_helpers.py

## Pendahuluan
File `smartcash/ui/utils/ui_helpers.py` berisi kumpulan fungsi helper yang distandarisasi untuk digunakan oleh semua komponen UI dalam SmartCash. Panduan ini menjelaskan bagaimana menggunakan modul ini dengan konsisten.

## Tujuan
1. Memastikan konsistensi tampilan dan perilaku di seluruh aplikasi
2. Mengurangi duplikasi kode
3. Menyederhanakan pemeliharaan dan update UI
4. Menjamin tampilan yang konsisten pada semua platform dan notebook

## Fungsi-fungsi yang Tersedia

### Pengaturan Tema dan Style
- `set_active_theme(theme_name)` - Mengubah tema aktif
- `inject_css_styles()` - Menambahkan CSS global

### Komponen UI Dasar
- `create_header(title, description, icon)` - Membuat header
- `create_section_title(title, icon)` - Membuat judul section

### Alert dan Status
- `create_status_indicator(status, message)` - Membuat indikator status
- `create_info_alert(message, alert_type, icon)` - Membuat alert box
- `create_info_box(title, content, style, icon, collapsed)` - Membuat info box

### Komponen Interaktif
- `create_tab_view(tabs)` - Membuat tab view
- `create_loading_indicator(message)` - Membuat indikator loading
- `create_button_group(buttons, layout)` - Membuat grup tombol
- `create_confirmation_dialog(title, message, on_confirm, on_cancel)` - Membuat dialog konfirmasi

### Pengaturan Output
- `update_output_area(output_widget, message, status, clear)` - Update area output
- `create_progress_updater(progress_bar)` - Buat fungsi updater progress

### Utilitas
- `register_observer_callback(observer_manager, event_type, output_widget, group_name)` - Register observer callback
- `display_file_info(file_path, description)` - Tampilkan info file
- `format_file_size(size_bytes)` - Format ukuran file
- `run_task(task_func, on_complete, on_error, with_output)` - Jalankan task dengan error handling

### Elemen UI Tambahan
- `create_divider()` - Buat divider horizontal
- `create_spacing(height)` - Buat elemen spacing

## Cara Penggunaan

### 1. Import yang Benar

```python
# Import semua helper yang diperlukan
from smartcash.ui.utils.ui_helpers import (
    create_header,
    create_status_indicator,
    update_output_area,
    create_button_group
)
```

### 2. Menggunakan Helper dalam Komponen UI

```python
def create_my_component(title, description):
    # Buat header dengan helper
    header = create_header(title, description, icon="🔍")
    
    # Buat tombol dengan helper
    buttons = create_button_group([
        ("Simpan", "primary", "save", on_save_clicked),
        ("Batal", "warning", "times", on_cancel_clicked)
    ])
    
    # Buat output area
    output = widgets.Output()
    
    # Update output dengan helper
    update_output_area(output, "Komponen siap digunakan", "info")
    
    return {
        'ui': widgets.VBox([header, buttons, output]),
        'output': output
    }
```

### 3. Menangani Error dan Loading

```python
# Buat indikator loading
loading, toggle_loading = create_loading_indicator("Memproses data...")

# Gunakan dalam fungsi
def process_data():
    toggle_loading(True, "Membaca file...")
    try:
        # Lakukan proses
        result = run_task(
            task_func=compute_heavy_task,
            with_output=output_widget
        )
        toggle_loading(False)
        return result
    except Exception as e:
        toggle_loading(False)
        update_output_area(output_widget, f"Error: {str(e)}", "error")
```

## Catatan Penting

1. **Konsistensi**: Selalu gunakan fungsi dari `ui_helpers.py` daripada mengimplementasikan ulang
2. **Fallback**: Jika perlu fallback, cukup buat elemen UI dengan HTML sederhana satu baris.
3. **Backward Compatibility**: Gunakan alias jika perlu untuk mempertahankan compatibility

## Contoh Implementasi Lengkap

Lihat file berikut untuk contoh implementasi yang baik:
- `smartcash/ui/components/headers.py`
- `smartcash/ui/components/helpers.py`
- `smartcash/ui/training_config/config_handler.py`
- `smartcash/ui/dataset/download_initialization.py`