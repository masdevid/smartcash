# Rencana Restrukturisasi Detection Handler SmartCash

## Tujuan

Restrukturisasi `detection_handler.py` untuk mengikuti prinsip Single Responsibility dengan membuat komponen lebih atomic, modular, dan dapat diuji. Refaktorisasi akan menerapkan pola desain modern seperti Factory, Strategy, dan Observer Pattern untuk meningkatkan keterbacaan dan pemeliharaan kode.

## ⚠️ Peringatan Duplikasi

**HINDARI menduplikasi implementasi yang sudah ada di folder `utils`, `data_manager`, `model_manager`, dan `checkpoint_manager`!** Beberapa komponen yang sudah ada:

- `ResultVisualizer` - Gunakan untuk visualisasi hasil deteksi
- `YOLOv5Model` - Gunakan untuk model deteksi
- `MetricsCalculator` - Gunakan untuk perhitungan metrik
- `LayerConfigManager` - Gunakan untuk konfigurasi layer

## Struktur Folder dan File

```
smartcash/handlers/detection/
├── __init__.py                       # Export komponen utama
├── detection_manager.py              # Entry point minimal (facade)

├── core/                             # Komponen inti deteksi
│   ├── detection_component.py        # Kelas dasar komponen deteksi
│   ├── model_loader.py               # Loader model untuk deteksi
│   ├── detector.py                   # Proses deteksi inti
│   ├── preprocessor.py               # Preprocessing gambar untuk deteksi
│   └── postprocessor.py              # Postprocessing hasil deteksi

├── strategies/                       # Strategi-strategi deteksi
│   ├── base_strategy.py              # Strategi deteksi dasar
│   ├── image_strategy.py             # Strategi deteksi gambar
│   └── directory_strategy.py         # Strategi deteksi folder

├── observers/                        # Observer untuk monitoring
│   ├── base_observer.py              # Observer dasar
│   ├── progress_observer.py          # Monitoring progress
│   └── metrics_observer.py           # Mengumpulkan metrik deteksi

├── pipeline/                         # Pipeline deteksi
│   ├── detection_pipeline.py         # Pipeline deteksi dasar
│   ├── image_pipeline.py             # Pipeline deteksi gambar
│   └── batch_pipeline.py             # Pipeline deteksi batch

├── integration/                      # Adapter untuk integrasi
│   ├── model_adapter.py              # Adapter untuk model dari model_manager
│   ├── visualizer_adapter.py         # Adapter untuk visualizer dari utils
│   └── drive_adapter.py              # Adapter untuk Google Drive

└── output/                           # Pengelolaan output
    ├── output_manager.py             # Pengelolaan hasil deteksi
    ├── image_output_handler.py       # Handler output gambar
    ├── json_output_handler.py        # Handler output JSON
    └── drive_output_handler.py       # Handler output ke Google Drive
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `detection_manager.py` sebagai entry point dengan antarmuka sederhana
   - Menyembunyikan kompleksitas subsistem pipeline, component, strategy

2. **Strategy Pattern**: 
   - Strategi untuk deteksi berbagai sumber (gambar, folder)
   - Mendukung penambahan strategi baru tanpa mengubah kode yang sudah ada

3. **Adapter Pattern**: 
   - Adapter untuk komponen dari `utils` dan manager lainnya
   - Adapter khusus untuk Google Drive di Colab

4. **Observer Pattern**: 
   - Monitoring progres dan metrik deteksi

5. **Factory Pattern**: 
   - Pembuatan komponen dengan konfigurasi yang tepat
   - Pembuatan strategi secara terstruktur

6. **Pipeline Pattern**: 
   - Proses deteksi sebagai pipeline terintegrasi

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

### DriveAdapter

- `drive_adapter.py` akan menangani integrasi dengan Google Drive
- Mencakup:
  - Mount/unmount Drive secara otomatis
  - Konversi path lokal ke path Drive
  - Handling symlink di Drive
  - Cache status mount Drive

### DriveOutputHandler

- `drive_output_handler.py` untuk menyimpan hasil deteksi ke Google Drive
- Mendukung:
  - Penyimpanan output dengan struktur folder yang rapi
  - Auto-create folder jika belum ada
  - Handling path relatif dan absolut

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

### DetectionManager (Facade)

```python
class DetectionManager:
    """
    Manager utama deteksi sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability.
    """
    
    def __init__(self, config=None, logger=None, colab_mode=None):
        # Deteksi otomatis colab jika tidak diberikan
        self.colab_mode = is_running_in_colab() if colab_mode is None else colab_mode
        
        # Setup adapter dan komponen
        # ...
        
    def detect(self, source, **kwargs):
        """Deteksi objek dari berbagai sumber."""
        # Pilih strategi berdasarkan tipe source
        # ...
        
    def detect_image(self, image_path, **kwargs):
        """Deteksi objek dari gambar."""
        # ...
        
    def detect_directory(self, dir_path, **kwargs):
        """Deteksi objek dari folder."""
        # ...
```

### BaseDetectionStrategy

```python
class BaseDetectionStrategy:
    """
    Strategi dasar untuk deteksi objek dari berbagai sumber.
    """
    
    def __init__(self, config, detector, preprocessor, postprocessor, output_manager, observers=None):
        self.config = config
        self.detector = detector
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.output_manager = output_manager
        self.observers = observers or []
        
    def add_observer(self, observer):
        """Tambahkan observer untuk monitoring."""
        self.observers.append(observer)
        return self
        
    def detect(self, source, **kwargs):
        """Metode abstract untuk deteksi objek."""
        raise NotImplementedError("Subclass must implement abstract method.")
        
    def notify_observers(self, event_type, data=None):
        """Notifikasi semua observer tentang event."""
        for observer in self.observers:
            observer.update(event_type, data)
```

### ImageDetectionStrategy

```python
class ImageDetectionStrategy(BaseDetectionStrategy):
    """
    Strategi deteksi untuk gambar.
    """
    
    def detect(self, source, **kwargs):
        """Deteksi objek dari gambar."""
        # Notify start
        self.notify_observers('start', {'source': source})
        
        # Preprocessing
        image = self.preprocessor.process(source)
        
        # Detection
        detections = self.detector.detect(image, **kwargs)
        
        # Postprocessing
        results = self.postprocessor.process(detections, **kwargs)
        
        # Save output
        output_path = self.output_manager.save(source, results, **kwargs)
        
        # Notify complete
        self.notify_observers('complete', {
            'source': source,
            'results': results,
            'output_path': output_path
        })
        
        return {
            'source': source,
            'detections': results,
            'output_path': output_path
        }
```

### DirectoryDetectionStrategy

```python
class DirectoryDetectionStrategy(BaseDetectionStrategy):
    """
    Strategi deteksi untuk direktori berisi gambar.
    """
    
    def detect(self, source, **kwargs):
        """Deteksi objek dari folder berisi gambar."""
        # Cari semua file gambar dalam direktori
        image_files = self._find_image_files(source)
        
        # Inisialisasi hasil
        results = {
            'source': source,
            'total_images': len(image_files),
            'processed_images': 0,
            'results': []
        }
        
        # Notify start
        self.notify_observers('start', {
            'source': source,
            'total_images': len(image_files)
        })
        
        # Proses setiap gambar
        with tqdm(image_files, desc="📷 Memproses gambar") as pbar:
            for img_path in pbar:
                # Proses satu gambar
                img_result = self._process_single_image(img_path, **kwargs)
                
                # Tambahkan ke hasil
                results['results'].append(img_result)
                results['processed_images'] += 1
                
                # Update progress
                self.notify_observers('progress', {
                    'current': results['processed_images'],
                    'total': results['total_images'],
                    'latest_result': img_result
                })
                
                # Update tqdm description
                pbar.set_description(f"📷 Memproses gambar ({results['processed_images']}/{results['total_images']})")
        
        # Notify complete
        self.notify_observers('complete', results)
        
        return results
    
    def _find_image_files(self, directory):
        """Cari semua file gambar dalam direktori."""
        dir_path = Path(directory)
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(dir_path.glob(ext)))
            
        return image_files
    
    def _process_single_image(self, img_path, **kwargs):
        """Proses satu gambar."""
        # Preprocessing
        image = self.preprocessor.process(img_path)
        
        # Detection
        detections = self.detector.detect(image, **kwargs)
        
        # Postprocessing
        results = self.postprocessor.process(detections, **kwargs)
        
        # Save output
        output_path = self.output_manager.save(img_path, results, **kwargs)
        
        return {
            'source': str(img_path),
            'detections': results,
            'output_path': output_path
        }
```

### DefaultDetector

```python
class DefaultDetector:
    """
    Komponen inti untuk deteksi objek.
    """
    
    def __init__(self, model, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger
        
    def detect(self, image, conf_thres=0.25, **kwargs):
        """
        Deteksi objek dalam gambar.
        
        Args:
            image: Image tensor
            conf_thres: Confidence threshold
            
        Returns:
            Detections
        """
        # Jalankan model untuk deteksi
        with torch.no_grad():
            predictions = self.model(image)
            
        # Return raw predictions
        return predictions
```

### ImagePreprocessor

```python
class ImagePreprocessor:
    """
    Preprocessing gambar untuk deteksi.
    """
    
    def __init__(self, config, img_size=(640, 640)):
        self.config = config
        self.img_size = img_size
        
    def process(self, image_path):
        """
        Preprocess gambar untuk model deteksi.
        
        Args:
            image_path: Path ke gambar
            
        Returns:
            Tensor gambar yang siap untuk deteksi
        """
        # Baca gambar
        if isinstance(image_path, str) or isinstance(image_path, Path):
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's a numpy array
            img = image_path
            
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalisasi
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, 0)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        
        # Move to device
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            
        return img_tensor
```

### DetectionPostprocessor

```python
class DetectionPostprocessor:
    """
    Postprocessing hasil deteksi.
    """
    
    def __init__(self, config, class_config=None):
        self.config = config
        self.class_config = class_config or {}
        
    def process(self, predictions, original_size=None, conf_thres=0.25, **kwargs):
        """
        Postprocess prediksi model.
        
        Args:
            predictions: Raw predictions dari model
            original_size: Ukuran asli gambar (h, w)
            conf_thres: Confidence threshold
            
        Returns:
            List deteksi terformat
        """
        # ... implementasi postprocessing
        # ... konversi koordinat ke format yang diinginkan
        # ... filtering berdasarkan confidence
        pass
```

## Menghadapi Keterbatasan Colab

1. **Memori Terbatas**:
   - Implementasi `batch_size` yang cerdas saat memproses folder besar
   - Cleaning cache secara berkala

2. **Keterbatasan GPU**:
   - Deteksi otomatis ketersediaan GPU
   - Fallback ke CPU dengan peringatan performa
   - Optimasi model untuk mode CPU (torch.quantize)

3. **Timeout Session**:
   - Checkpoint untuk deteksi batch/folder
   - Resume detection dari checkpoint
   - Status penyimpanan otomatis ke Drive

## Contoh Penggunaan Setelah Restrukturisasi

```python
from smartcash.handlers.detection import DetectionManager

# Inisialisasi
detector = DetectionManager(config=config)

# Deteksi dari gambar
results = detector.detect_image("path/to/image.jpg", conf_thres=0.4)

# Deteksi dari folder
results = detector.detect_directory("path/to/folder")

# Atau dengan detect() otomatis pilih strategi
results = detector.detect("path/to/image.jpg")
results = detector.detect("path/to/folder")
```

## Rencana Implementasi dan Migrasi

1. **Fase 1: Persiapan**
   - Buat struktur folder dan file dasar
   - Definisikan interface kunci untuk komponen

2. **Fase 2: Ekstraksi Komponen**
   - Ekstraksi dan refaktorisasi DetectionHandler menjadi komponen lebih kecil
   - Implementasi BaseDetectionStrategy dan turunannya

3. **Fase 3: Implementasi Pipeline**
   - Buat pipeline detection untuk gambar dan folder
   - Implementasi observer untuk monitoring

4. **Fase 4: Integrasi dengan Utils dan Manager Lain**
   - Buat adapter untuk integration dengan utils yang sudah ada
   - Integrasi dengan data_manager dan model_manager

5. **Fase 5: Dukungan Colab**
   - Implementasi DriveAdapter
   - Optimasi untuk Colab

6. **Fase 6: Optimasi dan Uji**
   - Optimasi performa
   - Pengujian komprehensif
   - Dokumentasi lengkap