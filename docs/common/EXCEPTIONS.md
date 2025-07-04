# Exceptions Module

## Overview
Modul `exceptions` mendefinisikan hierarki exception terpadu untuk seluruh komponen SmartCash. Setiap exception mewarisi dari `SmartCashError` dan menyertakan konteks yang kaya untuk memudahkan penanganan dan pelaporan error.

## Daftar Isi
- [Struktur Error](#struktur-error)
  - [ErrorContext](#errorcontext)
  - [SmartCashError](#smartcasherror)
- [Kategori Error](#kategori-error)
  - [ConfigError](#configerror)
  - [DatasetError](#dataseterror)
  - [ModelError](#modelerror)
  - [DetectionError](#detectionerror)
  - [UIError](#uierror)
  - [FileError](#fileerror)
  - [APIError](#apierror)
  - [ValidationError](#validationerror)
  - [NotSupportedError](#notsupportederror)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Best Practices](#best-practices)

## Struktur Error

### ErrorContext
```python
@dataclass
class ErrorContext:
    component: str = ""
    operation: str = ""
    details: Optional[Dict[str, Any]] = None
    ui_components: Optional[Dict[str, Any]] = None
```

Kelas ini menyimpan konteks tambahan untuk error, termasuk:
- `component`: Nama komponen yang menyebabkan error
- `operation`: Operasi yang sedang dilakukan saat error terjadi
- `details`: Detail tambahan dalam bentuk dictionary
- `ui_components`: Referensi ke komponen UI terkait

### SmartCashError
```python
class SmartCashError(Exception):
    def __init__(
        self, 
        message: str = "Terjadi error pada sistem SmartCash",
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None
    )
```

Base class untuk semua exception di SmartCash. Mendukung:
- Pesan error yang deskriptif
- Kode error opsional
- Konteks error yang kaya

## Kategori Error

### ConfigError
Error terkait konfigurasi aplikasi.

### DatasetError
Base class untuk error terkait dataset.

#### Turunan DatasetError:
- `DatasetFileError`: Error operasi file dataset
- `DatasetValidationError`: Error validasi dataset
- `DatasetProcessingError`: Error saat memproses dataset
- `DatasetCompatibilityError`: Inkompatibilitas dataset dengan model

### ModelError
Base class untuk error terkait model.

#### Turunan ModelError:
- `ModelConfigurationError`: Error konfigurasi model
- `ModelTrainingError`: Error saat training model
- `ModelInferenceError`: Error saat inferensi model
- `ModelCheckpointError`: Error terkait checkpoint model
- `ModelExportError`: Error saat mengekspor model
- `ModelEvaluationError`: Error saat evaluasi model
- `ModelServiceError`: Error pada model service

#### Error Komponen Model:
- `ModelComponentError`: Error dasar komponen model
- `BackboneError`: Error pada backbone model
- `UnsupportedBackboneError`: Backbone tidak didukung
- `NeckError`: Error pada neck model
- `HeadError`: Error pada detection head model

### DetectionError
Error terkait proses deteksi objek.

#### Turunan DetectionError:
- `DetectionInferenceError`: Error saat inferensi deteksi
- `DetectionPostprocessingError`: Error saat post-processing deteksi

### UIError
Error terkait antarmuka pengguna.

#### Turunan UIError:
- `UIComponentError`: Error inisialisasi/operasi komponen UI
- `UIActionError`: Error saat menjalankan aksi UI

### FileError
Error operasi file I/O.

### APIError
Error terkait pemanggilan API.

### ValidationError
Error validasi input.

### NotSupportedError
Fitur atau operasi yang diminta tidak didukung.

## Contoh Penggunaan

### Menangkap Exception Spesifik
```python
try:
    # Kode yang mungkin melempar exception
    dataset = load_dataset("path/to/dataset")
except DatasetFileError as e:
    print(f"Error memuat dataset: {e}")
    if e.context.details and 'file_path' in e.context.details:
        print(f"File yang bermasalah: {e.context.details['file_path']}")
```

### Menambahkan Konteks ke Exception
```python
try:
    # Kode yang mungkin error
    model.train()
except ModelTrainingError as e:
    # Menambahkan konteks tambahan
    raise e.with_context(
        component="training_pipeline",
        operation="model_training",
        details={"epoch": current_epoch, "batch": current_batch}
    ) from e
```

### Membuat Custom Exception
```python
class CustomModelError(ModelError):
    """Custom error untuk model khusus."""
    def __init__(self, message="Terjadi kesalahan pada model khusus", custom_param=None):
        self.custom_param = custom_param
        super().__init__(message)
```

## Best Practices

1. **Gunakan Exception yang Spesifik**
   - Gunakan exception yang paling spesifik yang tersedia
   - Hindari menangkap `Exception` secara langsung

2. **Sertakan Konteks yang Cukup**
   - Gunakan `ErrorContext` untuk menyertakan informasi tambahan
   - Sertakan detail yang berguna untuk debugging

3. **Chain Exception dengan Benar**
   - Gunakan `raise ... from ...` untuk mempertahankan traceback asli

4. **Buat Pesan Error yang Deskriptif**
   - Jelaskan apa yang salah dan mengapa
   - Sertakan nilai-nilai kunci yang relevan

5. **Logging**
   - Log exception dengan level yang sesuai
   - Sertakan konteks tambahan dalam log

6. **Dokumentasi**
   - Dokumentasikan exception yang mungkin dilempar oleh fungsi/method
   - Jelaskan kondisi yang menyebabkan exception dilempar

7. **Error Recovery**
   - Tangani exception yang dapat dipulihkan
   - Biarkan exception yang tidak dapat ditangani menyebar ke lapisan yang lebih tinggi

8. **Testing**
   - Tulis test case untuk skenario error yang diharapkan
   - Verifikasi pesan error dan konteks yang disertakan

Dokumentasi ini mencakup semua exception utama yang didefinisikan di SmartCash. Untuk informasi lebih lanjut, lihat kode sumber di `smartcash/common/exceptions.py`.
