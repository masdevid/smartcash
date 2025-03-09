# Rencana Restrukturisasi Unified Preprocessing Handler SmartCash

## Tujuan

Restrukturisasi `unified_preprocessing.py` untuk mengikuti prinsip Single Responsibility dengan membuat komponen lebih atomic, modular, dan dapat diuji. Refaktorisasi akan menerapkan pola desain modern seperti Factory, Strategy, dan Observer.

## ⚠️ Peringatan Duplikasi

**HINDARI menduplikasi implementasi yang sudah ada di folder `utils`!** Berdasarkan `UTILS_DOCS.md`, beberapa komponen yang sudah direfaktor di utils:

- `EnhancedDatasetValidator` - Gunakan adapter pattern untuk integrasi
- `AugmentationManager` - Gunakan eksisting daripada reimplementasi
- `EnhancedCache` / `CacheManager` - Gunakan untuk caching
- `LayerConfigManager` - Gunakan untuk konfigurasi layer

## Struktur Folder dan File

```
smartcash/handlers/preprocessing/
├── __init__.py                          # Export komponen utama
├── preprocessing_manager.py             # Entry point minimal (facade)

├── core/                                # Komponen inti preprocessing
│   ├── preprocessing_component.py       # Komponen dasar
│   ├── validation_component.py          # Validasi dataset
│   ├── augmentation_component.py        # Augmentasi dataset
│   ├── cleaning_component.py            # Pembersihan dataset
│   ├── analysis_component.py            # Analisis dataset
│   └── reporting_component.py           # Pelaporan preprocessing

├── pipeline/                            # Pipeline dan workflow
│   ├── preprocessing_pipeline.py        # Pipeline lengkap
│   ├── validation_pipeline.py           # Pipeline validasi
│   ├── augmentation_pipeline.py         # Pipeline augmentasi
│   └── analysis_pipeline.py             # Pipeline analisis

├── strategies/                          # Strategi-strategi preprocessing
│   ├── base_strategy.py                 # Strategi dasar
│   ├── validation/
│   │   ├── basic_validation.py          # Validasi dasar
│   │   ├── advanced_validation.py       # Validasi lanjutan
│   │   └── auto_fix.py                  # Fitur perbaikan otomatis
│   ├── augmentation/
│   │   ├── position_strategy.py         # Augmentasi posisi
│   │   ├── lighting_strategy.py         # Augmentasi pencahayaan
│   │   ├── combined_strategy.py         # Augmentasi gabungan
│   │   └── extreme_strategy.py          # Augmentasi ekstrem
│   └── analysis/
│       ├── class_distribution.py        # Analisis distribusi kelas
│       ├── layer_distribution.py        # Analisis distribusi layer
│       └── bbox_statistics.py           # Statistik bounding box

├── integration/                         # Adapter untuk integrasi
│   ├── validator_adapter.py             # Adapter untuk EnhancedDatasetValidator
│   ├── augmentation_adapter.py          # Adapter untuk AugmentationManager
│   ├── cache_adapter.py                 # Adapter untuk CacheManager
│   ├── drive_adapter.py                 # Adapter untuk Google Drive
│   └── colab_utils.py                   # Utilitas Colab

├── observers/                           # Observer pattern untuk monitoring
│   ├── base_observer.py                 # Observer dasar
│   ├── progress_observer.py             # Monitoring progres
│   ├── metrics_observer.py              # Monitoring metrik
│   └── colab_observer.py                # Observer khusus Colab

└── visualizations/                       # Visualisasi preprocessing
    ├── validation_visualizer.py          # Visualisasi validasi
    ├── augmentation_visualizer.py        # Visualisasi augmentasi
    ├── class_distribution_visualizer.py  # Visualisasi distribusi kelas
    └── report_visualizer.py              # Visualisasi laporan lengkap
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `preprocessing_manager.py` sebagai entry point dengan antarmuka sederhana
   - Menyembunyikan kompleksitas subsistem pipeline, component, strategy

2. **Strategy Pattern**: 
   - Strategi untuk validasi, augmentasi, dan analisis yang dapat diganti
   - Mendukung penambahan strategi baru tanpa mengubah kode yang sudah ada

3. **Adapter Pattern**: 
   - Adapter untuk komponen dari `utils`
   - Adapter khusus untuk Google Drive di Colab

4. **Observer Pattern**: 
   - Monitoring progres tanpa mengganggu fungsi utama
   - Observer khusus untuk antarmuka Colab

5. **Factory Pattern**: 
   - Pembuatan komponen dengan konfigurasi yang tepat
   - Pembuatan pipeline dan strategi secara terstruktur

6. **Pipeline Pattern**: 
   - Proses preprocessing sebagai pipeline
   - Mendukung eksekusi sebagian pipeline

## Integrasi dengan Google Colab + Drive

### Deteksi Environment

```python
def is_running_in_colab():
    """Deteksi apakah kode berjalan dalam Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False
```

### Adapter Drive

- `drive_adapter.py` akan menangani integrasi dengan Google Drive
- Mencakup:
  - Mount/unmount Drive secara otomatis
  - Konversi path lokal ke path Drive
  - Handling symlink di Drive
  - Cache status mount Drive

### Colab Observer

- `colab_observer.py` untuk memberikan feedback di lingkungan Colab
- Mendukung:
  - Progress bar yang kompatibel dengan Colab
  - Visualisasi real-time
  - Status preprocessing dalam bentuk widget

### Modifikasi Path untuk Colab

```python
def get_adjusted_path(path, mount_point="/content/drive"):
    """Sesuaikan path untuk Google Drive jika berjalan di Colab."""
    if is_running_in_colab():
        if str(path).startswith('/'):
            return Path(f"{mount_point}/MyDrive/{str(path).lstrip('/')}")
        return Path(f"{mount_point}/MyDrive/{path}")
    return Path(path)
```

## Kelas Kunci

### PreprocessingManager (Facade)

```python
class PreprocessingManager:
    """
    Manager utama preprocessing sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability.
    """
    
    def __init__(self, config=None, logger=None, colab_mode=None):
        # Deteksi otomatis colab jika tidak diberikan
        self.colab_mode = is_running_in_colab() if colab_mode is None else colab_mode
        
        # Setup adapter dan komponen
        # ...
        
    def run_full_pipeline(self, **kwargs):
        """Jalankan pipeline preprocessing lengkap."""
        pass
        
    def validate_dataset(self, **kwargs):
        """Jalankan validasi dataset saja."""
        pass
        
    def augment_dataset(self, **kwargs):
        """Jalankan augmentasi dataset saja."""
        pass
        
    def analyze_dataset(self, **kwargs):
        """Jalankan analisis dataset saja."""
        pass
        
    def generate_report(self, **kwargs):
        """Hasilkan laporan preprocessing."""
        pass
```

### PreprocessingPipeline

```python
class PreprocessingPipeline:
    """
    Pipeline lengkap preprocessing dengan tahap yang dapat dikonfigurasi.
    """
    
    def __init__(self, config, observers=None):
        self.config = config
        self.observers = observers or []
        self.components = []
        
    def add_component(self, component):
        """Tambahkan komponen ke pipeline."""
        self.components.append(component)
        return self
        
    def add_observer(self, observer):
        """Tambahkan observer untuk monitoring pipeline."""
        self.observers.append(observer)
        return self
        
    def run(self, **kwargs):
        """Jalankan pipeline lengkap."""
        # Notifikasi observer
        # Proses setiap komponen
        # Kompilasi hasil
        pass
```

### ValidationComponent

```python
class ValidationComponent(PreprocessingComponent):
    """Komponen validasi dataset dengan menggunakan EnhancedDatasetValidator."""
    
    def __init__(self, config, validator_adapter=None):
        super().__init__(config)
        self.validator_adapter = validator_adapter or ValidatorAdapter(config)
        
    def process(self, split='train', **kwargs):
        """Proses validasi dataset."""
        # Gunakan adapter untuk validator dari utils
        # Konversi hasil ke format standar
        pass
```

## Langkah Migrasi

1. **Ekstraksi Komponen**: Memecah `UnifiedPreprocessingHandler` menjadi komponen-komponen kecil
2. **Implementasi Adapter**: Membuat adapter untuk komponen dari `utils`
3. **Implementasi Pipeline**: Membuat pipeline preprocessing modular
4. **Implementasi Manager**: Membuat facade manager sebagai entry point
5. **Implementasi Observer**: Menambahkan observer untuk monitoring
6. **Integrasi Colab**: Menambahkan fitur khusus untuk Colab dan Drive

## Catatan Migrasi

- **Mempertahankan API Publik**: Pastikan API publik dari `UnifiedPreprocessingHandler` tetap didukung
- **Kompatibilitas Mundur**: Pastikan kode yang sudah ada tetap berfungsi dengan benar
- **Unit Tests**: Buat unit tests untuk setiap komponen
- **Dukungan Colab**: Pastikan semua komponen berfungsi di lingkungan Colab
- **Progress Reporting**: Gunakan tqdm untuk progress bar yang konsisten

## Implementasi UI di Colab

Komponen ini akan mendukung UI yang akan diimplementasikan di Colab dengan:

1. **Widget Interaktif**: Observer akan memberikan data untuk widget Colab
2. **Visualisasi Real-time**: Proses preprocessing akan menampilkan visualisasi real-time
3. **Progress Bar**: Status progres akan ditampilkan secara konsisten
4. **Error Handling**: Pesan error yang jelas dan informatif
5. **Integrasi Drive**: Operasi file yang mulus dengan Google Drive

## Contoh Penggunaan Setelah Restrukturisasi

```python
# Inisialisasi di Colab
from smartcash.handlers.preprocessing import PreprocessingManager

# Deteksi otomatis mode Colab
preprocessor = PreprocessingManager(config=config)

# Jalankan pipeline lengkap
results = preprocessor.run_full_pipeline(
    splits=['train', 'valid'],
    validate_dataset=True,
    fix_issues=True,
    augment_data=True,
    report_format='html'
)

# Visualisasi hasil
preprocessor.visualize_results(results)
```