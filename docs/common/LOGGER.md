# Logger Module

## Overview
Modul `logger` menyediakan fungsionalitas logging yang ditingkatkan untuk aplikasi SmartCash dengan dukungan integrasi UI dan penanganan level log yang aman. Modul ini dibangun di atas modul `logging` standar Python dengan beberapa fitur tambahan seperti:

- Dukungan level log yang dinormalisasi
- Integrasi dengan komponen UI (seperti Jupyter Notebook/Colab)
- Format pesan log yang konsisten dengan emoji
- Penanganan error yang aman untuk operasi logging

## Daftar Isi
- [LogLevel](#loglevel)
- [SmartCashLogger](#smartcashlogger)
- [Fungsi Utilitas](#fungsi-utilitas)
  - [get_logger](#get_logger)
  - [safe_log_to_ui](#safe_log_to_ui)
  - [create_ui_logger](#create_ui_logger)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Best Practices](#best-practices)

## LogLevel

Enum yang mendefinisikan level-level log yang tersedia beserta alias-nya.

### Nilai-nilai Level
- `DEBUG`: Informasi debugging rinci
- `INFO`: Konfirmasi bahwa segala sesuatunya berfungsi seperti yang diharapkan
- `WARNING`: Indikasi bahwa sesuatu yang tidak terduga terjadi
- `ERROR`: Kesalahan serius yang menyebabkan kegagalan fungsi tertentu
- `CRITICAL`: Kesalahan kritis yang mungkin menyebabkan program berhenti

### Alias
- `WARN`: Alias untuk `WARNING`
- `FATAL`: Alias untuk `CRITICAL`

## SmartCashLogger

Kelas logger utama yang menyediakan fungsionalitas logging dengan dukungan UI.

### Inisialisasi
```python
logger = SmartCashLogger(
    name: str, 
    level: Union[str, int, LogLevel] = LogLevel.INFO, 
    ui_components: Optional[Dict[str, Any]] = None
)
```

#### Parameter
- `name`: Nama logger (biasanya `__name__`)
- `level`: Level log default (default: `LogLevel.INFO`)
- `ui_components`: Komponen UI untuk integrasi logging (opsional)

### Metode

#### debug(message: str, **kwargs)
Mencatat pesan debug.

#### info(message: str, **kwargs)
Mencatat pesan informasi.

#### warning(message: str, **kwargs)
Mencatat pesan peringatan.

#### warn(message: str)
Alias untuk `warning()`.

#### error(message: str, **kwargs)
Mencatat pesan error.

#### critical(message: str, **kwargs)
Mencatat pesan kritis.

#### fatal(message: str)
Alias untuk `critical()`.

#### set_ui_components(ui_components: Dict[str, Any])
Mengatur komponen UI untuk integrasi logging.

#### set_level(level: Union[str, int, LogLevel])
Mengatur level logging.

## Fungsi Utilitas

### get_logger
```python
get_logger(
    name: str, 
    level: Union[str, int, LogLevel] = LogLevel.INFO, 
    ui_components: Optional[Dict[str, Any]] = None
) -> SmartCashLogger
```

Fungsi factory untuk membuat instance `SmartCashLogger`.

### safe_log_to_ui
```python
safe_log_to_ui(
    ui_components: Dict[str, Any], 
    message: str, 
    level: str = 'info'
) -> bool
```

Mencatat pesan ke komponen UI dengan penanganan error yang aman.

### create_ui_logger
```python
create_ui_logger(
    name: str, 
    ui_components: Dict[str, Any], 
    level: Union[str, int, LogLevel] = LogLevel.INFO
) -> SmartCashLogger
```

Membuat logger dengan integrasi komponen UI.

## Contoh Penggunaan

### Contoh Dasar
```python
from smartcash.common.logger import get_logger

# Membuat logger
logger = get_logger(__name__)

# Mencatat pesan
logger.info("Aplikasi dimulai")
try:
    # Kode yang mungkin menghasilkan error
    result = 10 / 0
except Exception as e:
    logger.error("Terjadi kesalahan", exc_info=True)
```

### Contoh dengan UI Integration (Jupyter/Colab)
```python
from IPython.display import display
import ipywidgets as widgets
from smartcash.common.logger import create_ui_logger

# Membuat output widget untuk logging
log_output = widgets.Output()
display(log_output)

# Membuat logger dengan UI integration
logger = create_ui_logger(
    name=__name__,
    ui_components={'log_output': log_output},
    level='DEBUG'
)

# Sekarang log akan muncul di widget output
logger.info("Ini adalah pesan info")
logger.warning("Ini adalah peringatan")
logger.error("Ini adalah error")
```

## Best Practices

1. **Gunakan Nama Logger yang Deskriptif**
   - Gunakan `__name__` sebagai nama logger untuk memudahkan pelacakan asal log

2. **Pilih Level yang Tepat**
   - `DEBUG`: Untuk informasi debugging
   - `INFO`: Untuk informasi operasional
   - `WARNING`: Untuk kondisi yang tidak diharapkan tapi tidak menghentikan eksekusi
   - `ERROR`: Untuk kesalahan yang menyebabkan kegagalan fungsi
   - `CRITICAL`: Untuk kesalahan yang menghentikan aplikasi

3. **Gunakan Parameter `exc_info` untuk Exception**
   ```python
   try:
       # Kode yang mungkin error
   except Exception as e:
       logger.error("Terjadi kesalahan", exc_info=True)
   ```

4. **Hindari String Formatting di Parameter Log**
   ```python
   # Buruk
   logger.debug("Nilai x adalah " + str(x))
   
   # Baik
   logger.debug("Nilai x adalah %s", x)
   ```

5. **Gunakan UI Components untuk Lingkungan Interaktif**
   - Di Jupyter/Colab, gunakan `create_ui_logger` dengan widget output untuk tampilan log yang lebih baik

6. **Atur Level Log Sesuai Kebutuhan**
   - Di production, set level ke `INFO` atau `WARNING`
   - Saat debugging, set ke `DEBUG`

7. **Gunakan Emoji untuk Visual Feedback**
   - Logger secara otomatis menambahkan emoji yang sesuai untuk level log yang berbeda
   - ℹ️ INFO, ⚠️ WARNING, ❌ ERROR, 🚨 CRITICAL, ✅ SUCCESS, 🔍 DEBUG

Dokumentasi ini mencakup semua fitur utama dari modul logger SmartCash. Untuk informasi lebih lanjut, lihat kode sumber di `smartcash/common/logger.py`.
