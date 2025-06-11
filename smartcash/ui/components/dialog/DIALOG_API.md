# 📋 SmartCash Dialog Components API

## 🎯 Overview

Komponen dialog yang reusable untuk confirmation, info, dan user interaction dengan callback support.

## 🚀 Basic Usage

### Confirmation Dialog

```python
from smartcash.ui.components.dialog import show_confirmation_dialog

def handle_confirm():
    print("✅ User confirmed!")
    # Execute your operation here

def handle_cancel():
    print("🚫 User cancelled")

show_confirmation_dialog(
    ui_components,
    title="Konfirmasi Operasi",
    message="Apakah Anda yakin ingin melanjutkan?",
    on_confirm=handle_confirm,
    on_cancel=handle_cancel,
    confirm_text="Ya, Lanjutkan",
    cancel_text="Batal"
)
```

### Info Dialog

```python
from smartcash.ui.components.dialog import show_info_dialog

def handle_close():
    print("ℹ️ Dialog closed")

show_info_dialog(
    ui_components,
    title="Informasi",
    message="Operasi telah selesai dengan sukses!",
    on_close=handle_close,
    close_text="Tutup",
    dialog_type="success"  # 'info', 'success', 'warning', 'error'
)
```

## 📋 API Reference

### `show_confirmation_dialog()`

Menampilkan dialog konfirmasi dengan dua tombol (confirm/cancel).

**Parameters:**
- `ui_components: Dict[str, Any]` - Dictionary UI components yang berisi 'confirmation_area' atau 'dialog_area'
- `title: str` - Judul dialog
- `message: str` - Pesan dialog (mendukung HTML)
- `on_confirm: Callable = None` - Callback untuk tombol konfirmasi
- `on_cancel: Callable = None` - Callback untuk tombol batal
- `confirm_text: str = "Ya"` - Text tombol konfirmasi
- `cancel_text: str = "Batal"` - Text tombol batal
- `danger_mode: bool = False` - Menggunakan style merah untuk operasi berbahaya

**Example:**
```python
show_confirmation_dialog(
    ui_components,
    title="⚠️ Hapus Data",
    message="Data yang dihapus tidak dapat dikembalikan!",
    on_confirm=lambda: delete_data(),
    on_cancel=lambda: print("Penghapusan dibatalkan"),
    confirm_text="Ya, Hapus",
    cancel_text="Batal",
    danger_mode=True
)
```

### `show_info_dialog()`

Menampilkan dialog informasi dengan satu tombol close.

**Parameters:**
- `ui_components: Dict[str, Any]` - Dictionary UI components
- `title: str` - Judul dialog
- `message: str` - Pesan dialog
- `on_close: Callable = None` - Callback untuk tombol close
- `close_text: str = "Tutup"` - Text tombol close
- `dialog_type: str = "info"` - Type dialog untuk styling

**Dialog Types:**
- `'info'` - Biru (informasi)
- `'success'` - Hijau (sukses)
- `'warning'` - Kuning (peringatan)
- `'error'` - Merah (error)

**Example:**
```python
show_info_dialog(
    ui_components,
    title="✅ Berhasil",
    message="Preprocessing selesai: 1,234 gambar diproses",
    on_close=lambda: refresh_ui(),
    dialog_type="success"
)
```

### `clear_dialog_area()`

Membersihkan area dialog dan reset visibility flag.

```python
from smartcash.ui.components.dialog import clear_dialog_area

clear_dialog_area(ui_components)
```

### `is_dialog_visible()`

Mengecek apakah dialog sedang ditampilkan.

```python
from smartcash.ui.components.dialog import is_dialog_visible

if is_dialog_visible(ui_components):
    print("Dialog sedang ditampilkan")
```

## 🏗️ Integration Patterns

### Dengan Operation Handlers

```python
def preprocessing_handler(ui_components):
    if should_execute_operation(ui_components, 'preprocessing'):
        # Execute setelah konfirmasi
        return execute_preprocessing(ui_components)
    
    if not is_dialog_visible(ui_components):
        # Show confirmation jika belum ada dialog
        show_confirmation_dialog(
            ui_components,
            title="🚀 Mulai Preprocessing",
            message="Memulai preprocessing dataset dengan API baru?",
            on_confirm=lambda: set_operation_confirmed(ui_components, 'preprocessing'),
            on_cancel=lambda: log_operation_cancelled(ui_components, 'preprocessing')
        )
```

### Dengan Progress Tracking

```python
def show_operation_result(ui_components, success, message):
    dialog_type = "success" if success else "error"
    title = "✅ Berhasil" if success else "❌ Gagal"
    
    show_info_dialog(
        ui_components,
        title=title,
        message=message,
        dialog_type=dialog_type,
        on_close=lambda: reset_ui_state(ui_components)
    )
```

### Dengan State Management

```python
# Set confirmation flags
def set_operation_confirmed(ui_components, operation_type):
    ui_components[f'_{operation_type}_confirmed'] = True
    print(f"✅ {operation_type} dikonfirmasi")

def should_execute_operation(ui_components, operation_type):
    return ui_components.pop(f'_{operation_type}_confirmed', False)

# Usage
show_confirmation_dialog(
    ui_components,
    title="Konfirmasi",
    message="Lanjutkan operasi?",
    on_confirm=lambda: set_operation_confirmed(ui_components, 'cleanup'),
    on_cancel=lambda: print("Operasi dibatalkan")
)
```

## 🎨 Styling & Appearance

### Confirmation Dialog Styles

- **Normal Mode**: Border biru, background abu-abu muda
- **Danger Mode**: Border merah, background merah muda, tombol confirm merah

### Info Dialog Styles

- **Info**: Biru (#17a2b8) dengan background biru muda
- **Success**: Hijau (#28a745) dengan background hijau muda  
- **Warning**: Kuning (#ffc107) dengan background kuning muda
- **Error**: Merah (#dc3545) dengan background merah muda

### Dialog Layout

- Width: 100% dengan max-width 500px
- Centered positioning dengan margin auto
- Box shadow untuk depth
- Responsive padding dan spacing
- Clean typography dengan proper line-height

## 🔧 Setup Requirements

### UI Components Structure

Dialog membutuhkan area container dalam `ui_components`:

```python
ui_components = {
    'confirmation_area': widgets.Output(),  # Primary
    'dialog_area': widgets.Output(),        # Fallback
    # ... other components
}
```

### Container Widget

```python
# Create dialog area
confirmation_area = widgets.Output(layout=widgets.Layout(
    width='100%', 
    min_height='50px', 
    max_height='200px',
    margin='10px 0',
    padding='5px',
    border='1px solid #e0e0e0',
    border_radius='4px',
    background_color='#fafafa'
))

ui_components['confirmation_area'] = confirmation_area
```

## 🚀 Advanced Usage

### Chained Dialogs

```python
def show_step1_confirmation(ui_components):
    show_confirmation_dialog(
        ui_components,
        title="Step 1",
        message="Lanjut ke step 2?",
        on_confirm=lambda: show_step2_confirmation(ui_components),
        on_cancel=lambda: print("Workflow dibatalkan")
    )

def show_step2_confirmation(ui_components):
    show_confirmation_dialog(
        ui_components,
        title="Step 2", 
        message="Eksekusi final step?",
        on_confirm=lambda: execute_final_step(ui_components),
        on_cancel=lambda: show_step1_confirmation(ui_components)
    )
```

### Dynamic Message Generation

```python
def show_cleanup_confirmation(ui_components):
    # Get dynamic info
    config = extract_config(ui_components)
    cleanup_target = config.get('cleanup_target', 'preprocessed')
    
    # Build dynamic message
    message = f"""
    <strong>Target:</strong> {cleanup_target}<br>
    <strong>Action:</strong> Permanent deletion<br>
    <br>
    <span style='color: #dc3545;'>⚠️ This action cannot be undone!</span>
    """
    
    show_confirmation_dialog(
        ui_components,
        title="🗑️ Konfirmasi Cleanup",
        message=message,
        on_confirm=lambda: execute_cleanup(ui_components, cleanup_target),
        danger_mode=True
    )
```

### Error Handling

```python
def safe_show_dialog(ui_components, dialog_config):
    try:
        show_confirmation_dialog(ui_components, **dialog_config)
    except Exception as e:
        # Fallback ke console
        print(f"⚠️ Dialog error: {str(e)}")
        print(f"📋 {dialog_config['title']}: {dialog_config['message']}")
        
        # Auto-execute default action atau skip
        if 'on_cancel' in dialog_config:
            dialog_config['on_cancel']()
```

### Custom Callbacks dengan Context

```python
class OperationHandler:
    def __init__(self, ui_components):
        self.ui_components = ui_components
        self.operation_context = {}
    
    def show_preprocessing_confirmation(self, dataset_info):
        self.operation_context = {'dataset_info': dataset_info}
        
        show_confirmation_dialog(
            self.ui_components,
            title="🚀 Preprocessing Dataset",
            message=f"Process {dataset_info['total_images']} images?",
            on_confirm=self.handle_preprocessing_confirm,
            on_cancel=self.handle_operation_cancel
        )
    
    def handle_preprocessing_confirm(self):
        dataset_info = self.operation_context.get('dataset_info', {})
        execute_preprocessing_with_context(self.ui_components, dataset_info)
    
    def handle_operation_cancel(self):
        print("🚫 Preprocessing cancelled by user")
        self.operation_context.clear()
```

## 📝 Best Practices

### 1. Clear Messaging
```python
# ✅ Good - Clear and specific
show_confirmation_dialog(
    ui_components,
    title="🗑️ Hapus 1,234 Files",
    message="Menghapus semua data preprocessed. Data tidak dapat dikembalikan!"
)

# ❌ Avoid - Vague messaging  
show_confirmation_dialog(
    ui_components,
    title="Konfirmasi",
    message="Lanjutkan?"
)
```

### 2. Consistent Callbacks
```python
# ✅ Good - Always provide both callbacks
show_confirmation_dialog(
    ui_components,
    title="Operasi",
    message="Lanjutkan?",
    on_confirm=lambda: handle_confirm(),
    on_cancel=lambda: handle_cancel()  # Always handle cancel
)
```

### 3. Error Recovery
```python
# ✅ Good - Handle callback errors
def safe_confirm_handler():
    try:
        execute_operation()
    except Exception as e:
        show_info_dialog(
            ui_components,
            title="❌ Error",
            message=f"Operasi gagal: {str(e)}",
            dialog_type="error"
        )
```

### 4. State Cleanup
```python
# ✅ Good - Always cleanup after dialog
def operation_complete_handler():
    execute_operation()
    clear_dialog_area(ui_components)  # Explicit cleanup
    reset_operation_state(ui_components)
```

## 🔍 Troubleshooting

### Dialog Tidak Muncul
- Pastikan `confirmation_area` atau `dialog_area` ada dalam `ui_components`
- Check console untuk error messages
- Verify widget layout dan visibility

### Callback Tidak Berjalan
- Pastikan callback function tidak `None`
- Check untuk exceptions dalam callback
- Verify lambda syntax untuk inline callbacks

### Layout Issues  
- Adjust container `max_width` jika dialog terlalu besar
- Check parent container overflow settings
- Verify responsive layout pada berbagai screen sizes

## 📦 Export Summary

```python
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,  # Main confirmation dialog
    show_info_dialog,          # Info/success/warning/error dialog  
    clear_dialog_area,         # Clear dialog area
    is_dialog_visible          # Check dialog visibility
)
```