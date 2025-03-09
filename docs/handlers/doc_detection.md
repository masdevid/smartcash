# Dokumentasi DetectionManager SmartCash

## Deskripsi

`DetectionManager` adalah komponen pusat untuk proses deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi deteksi. Implementasi telah dioptimasi dengan pendekatan berbasis pipeline dan strategy yang modular.

## Struktur File Lengkap

```
smartcash/handlers/detection/
├── __init__.py                         # Export komponen utama (DetectionManager)
├── detection_manager.py                # Entry point minimal (facade)
│
├── core/                               # Komponen inti deteksi
│   ├── __init__.py                     # Export DefaultDetector, ImagePreprocessor, DetectionPostprocessor
│   ├── detector.py                     # Komponen untuk proses deteksi inti
│   ├── preprocessor.py                 # Komponen untuk preprocessing gambar 
│   └── postprocessor.py                # Komponen untuk postprocessing hasil deteksi
│
├── strategies/                         # Strategi-strategi deteksi
│   ├── __init__.py                     # Export BaseDetectionStrategy, ImageDetectionStrategy, DirectoryDetectionStrategy
│   ├── base_strategy.py                # Kelas abstrak untuk strategi deteksi
│   ├── image_strategy.py               # Implementasi strategi untuk deteksi gambar tunggal
│   └── directory_strategy.py           # Implementasi strategi untuk deteksi folder
│
├── pipeline/                           # Pipeline deteksi
│   ├── __init__.py                     # Export BasePipeline, DetectionPipeline, BatchDetectionPipeline
│   ├── base_pipeline.py                # Kelas abstrak untuk pipeline deteksi
│   ├── detection_pipeline.py           # Implementasi pipeline untuk deteksi gambar tunggal
│   └── batch_pipeline.py               # Implementasi pipeline untuk deteksi batch gambar
│
├── integration/                        # Adapter untuk integrasi
│   ├── __init__.py                     # Export ModelAdapter, VisualizerAdapter
│   ├── model_adapter.py                # Adapter untuk integrasi dengan ModelManager
│   └── visualizer_adapter.py           # Adapter untuk integrasi dengan ResultVisualizer
│
├── output/                             # Pengelolaan output
│   ├── __init__.py                     # Export OutputManager
│   └── output_manager.py               # Komponen untuk pengelolaan hasil deteksi
│
└── observers/                          # Observer pattern (future implementation)
    ├── __init__.py                     # Export untuk observer pattern
    ├── base_observer.py                # Kelas dasar untuk observer
    ├── progress_observer.py            # Observer untuk monitoring progres
    └── metrics_observer.py             # Observer untuk monitoring metrik
```

## Fitur Utama

### 1. Deteksi Multi-Format
- Mendukung deteksi dari gambar tunggal, batch gambar, dan folder
- Pemilihan strategi secara otomatis berdasarkan tipe input
- Deteksi otomatis format file gambar (jpg, jpeg, png, bmp)

### 2. Pipeline Deteksi Komprehensif
- Alur kerja lengkap: preprocessing, deteksi, postprocessing
- Visualisasi hasil deteksi
- Penyimpanan hasil dalam format JSON
- Mekanisme observer untuk tracking progres

### 3. Integrasi dengan Komponen Lain
- Integrasi dengan ModelManager untuk loading model
- Integrasi dengan ResultVisualizer untuk visualisasi
- Dukungan untuk konfigurasi dari ConfigManager
- Support untuk berbagai layer deteksi (banknote, nominal, security)
- Dukungan khusus untuk Google Colab

### 4. Performa dan Optimasi
- Lazy-loading komponen untuk inisialisasi cepat
- Penggunaan GPU otomatis jika tersedia
- Half precision (FP16) untuk inferensi cepat
- Pengelolaan memori efisien dengan torch.no_grad()
- Progress tracking dengan tqdm

## Kelas-Kelas Utama

### 1. handlers/detection/detection_manager.py

**DetectionManager**: Facade yang menyediakan antarmuka sederhana untuk semua operasi deteksi.

**Metode utama**:
- `detect()`: Deteksi objek dari berbagai sumber (otomatis memilih strategi)
- `detect_image()`: Deteksi objek dari gambar tunggal
- `detect_batch()`: Deteksi objek dari batch gambar
- `detect_directory()`: Deteksi objek dari direktori berisi gambar

### 2. handlers/detection/core/

**DefaultDetector**: Menggunakan model untuk mendeteksi objek dalam gambar yang telah dipreprocess.

**ImagePreprocessor**: Memproses gambar input menjadi tensor yang siap untuk model.

**DetectionPostprocessor**: Memproses hasil deteksi mentah menjadi format yang mudah digunakan.

### 3. handlers/detection/strategies/

**BaseDetectionStrategy**: Kelas abstrak yang mendefinisikan interface untuk semua strategi deteksi.

**ImageDetectionStrategy**: Implementasi strategi untuk deteksi pada gambar tunggal.

**DirectoryDetectionStrategy**: Implementasi strategi untuk deteksi pada direktori berisi gambar.

### 4. handlers/detection/pipeline/

**BasePipeline**: Kelas abstrak untuk semua pipeline deteksi.

**DetectionPipeline**: Pipeline lengkap untuk deteksi gambar tunggal.

**BatchDetectionPipeline**: Pipeline untuk batch processing deteksi pada banyak gambar.

### 5. handlers/detection/integration/

**ModelAdapter**: Adapter untuk integrasi dengan ModelManager.

**VisualizerAdapter**: Adapter untuk integrasi dengan ResultVisualizer.

### 6. handlers/detection/output/

**OutputManager**: Mengelola penyimpanan hasil deteksi dan visualisasi.

## Format Hasil Deteksi

### Hasil Deteksi Gambar Tunggal

```python
{
    'source': '/path/to/image.jpg',        # Path ke gambar sumber
    'detections': [                        # List deteksi
        {
            'bbox': [0.32, 0.45, 0.15, 0.2],    # [x, y, w, h] normalisasi
            'bbox_pixels': [205, 288, 96, 128],  # [x, y, w, h] dalam piksel
            'bbox_xyxy': [157, 224, 253, 352],   # [xmin, ymin, xmax, ymax]
            'class_id': 3,                       # ID kelas
            'class_name': '010',                 # Nama kelas
            'layer': 'banknote',                 # Nama layer
            'confidence': 0.92                   # Skor konfidiensi
        },
        # ... deteksi lainnya
    ],
    'detections_by_layer': {              # Deteksi dikelompokkan berdasarkan layer
        'banknote': [...],                # Deteksi untuk layer banknote
        'nominal': [...]                  # Deteksi untuk layer nominal
    },
    'num_detections': 5,                  # Jumlah deteksi
    'inference_time': 0.023,              # Waktu inferensi dalam detik
    'execution_time': 0.12,               # Waktu eksekusi total dalam detik
    'visualization_path': '/path/to/output/result.jpg',  # Path hasil visualisasi
    'output_paths': {
        'json': '/path/to/output/result.json'
    }
}
```

### Hasil Deteksi Direktori/Batch

```python
{
    'source': '/path/to/directory',       # Path ke direktori (untuk directory detection)
    'total_images': 10,                   # Jumlah total gambar
    'processed_images': 10,               # Jumlah gambar yang berhasil diproses
    'results': [                          # List hasil deteksi per gambar
        # ... hasil deteksi untuk setiap gambar seperti format di atas
    ],
    'errors': [                           # List error jika ada
        {'source': '/path/to/image.jpg', 'error': 'Error message'}
    ],
    'detections_by_layer': {              # Semua deteksi dikelompokkan berdasarkan layer
        'banknote': [...],                # Deteksi untuk layer banknote dari semua gambar
        'nominal': [...]                  # Deteksi untuk layer nominal dari semua gambar
    },
    'execution_time': 1.25,               # Waktu eksekusi total dalam detik
    'success_rate': 1.0,                  # Tingkat keberhasilan (processed/total)
    'num_errors': 0,                      # Jumlah error
    'total_detections': 25                # Jumlah total deteksi dari semua gambar
}
```

def get_colab_path(self, path: Union[str, Path]) -> str:
       """Dapatkan path yang user-friendly untuk Google Colab."""
       if not self.colab_mode:
           return str(path)
           
       # Konversi ke path absolut
       abs_path = Path(path).absolute()
       
       # Cek apakah di directory Google Drive
       if '/content/drive/' in str(abs_path):
           # Format path untuk menampilkan di Colab
           return f"📂 Google Drive: {str(abs_path).replace('/content/drive/MyDrive/', '')}"
       else:
           # Path lokal di Colab
           return f"📂 Colab: {str(abs_path).replace('/content/', '')}"
   ```

3. **Visualisasi Adaptif**: Penyesuaian visualisasi untuk notebook Colab

## Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `DetectionManager` sebagai entry point yang menyembunyikan kompleksitas
   - Menyediakan antarmuka sederhana untuk berbagai operasi deteksi

2. **Strategy Pattern**:
   - `BaseDetectionStrategy` dengan implementasi konkrit untuk berbagai skenario
   - `ImageDetectionStrategy` untuk deteksi gambar tunggal
   - `DirectoryDetectionStrategy` untuk deteksi folder

3. **Pipeline Pattern**:
   - `BasePipeline` dengan implementasi berbeda untuk alur kerja deteksi
   - `DetectionPipeline` untuk deteksi gambar tunggal
   - `BatchDetectionPipeline` untuk deteksi batch

4. **Adapter Pattern**:
   - `ModelAdapter` untuk integrasi dengan ModelManager
   - `VisualizerAdapter` untuk integrasi dengan ResultVisualizer

5. **Observer Pattern**:
   - Dukungan untuk observer dalam strategi dan pipeline untuk monitoring progres
   - Notifikasi events seperti 'start', 'progress', dan 'complete'

6. **Lazy-loading Pattern**:
   - Components diinisialisasi hanya saat dibutuhkan
   - Menggunakan private methods dengan prefix `_get_` untuk loading komponen

## Optimasi Performa

### 1. Lazy-Loading

Semua komponen dimuat secara lazy untuk mengurangi overhead inisialisasi:

```python
def _get_model(self) -> torch.nn.Module:
    """Lazy-load model."""
    if self._model is None:
        model_adapter = self._get_model_adapter()
        self._model = model_adapter.get_model()
    return self._model
```

### 2. Deteksi GPU

Otomatis mendeteksi dan menggunakan GPU jika tersedia:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3. Half Precision

Dukungan untuk inferensi half precision (FP16) untuk performa lebih baik:

```python
if self.half_precision and device.type == 'cuda':
    model = model.half()
```

### 4. Pengelolaan Memori

Menggunakan context manager `torch.no_grad()` untuk efisiensi memori saat inferensi:

```python
with torch.no_grad():
    predictions = self.model(image)
```

### 5. Batching

Pipeline batch untuk pemrosesan paralel:

```python
batch_pipeline = self._get_batch_pipeline()
return batch_pipeline.run(
    sources=image_files,
    conf_threshold=conf_threshold,
    visualize=visualize,
    **kwargs
)
```

## Kesimpulan

DetectionManager SmartCash menawarkan:

1. **Antarmuka Terpadu**: Entry point sederhana untuk semua operasi deteksi
2. **Modularitas**: Pemisahan komponen dengan tanggung jawab yang jelas
3. **Fleksibilitas**: Dukungan untuk berbagai strategi dan pipeline deteksi
4. **Integrasi**: Integrasi mulus dengan komponen lain di SmartCash
5. **Performa**: Optimasi untuk performa dengan lazy-loading dan GPU acceleration
6. **Robustness**: Penanganan error yang komprehensif dan logging yang baik
7. **Colab Support**: Dukungan khusus untuk Google Colab

DetectionManager memfasilitasi deteksi mata uang Rupiah dengan mudah dan akurat, mendukung berbagai skenario penggunaan dari deteksi gambar tunggal hingga pemrosesan batch folder besar.