# Dokumentasi EvaluationManager SmartCash

## Deskripsi

`EvaluationManager` adalah komponen pusat untuk evaluasi model deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi evaluasi model. Implementasi telah dioptimasi dengan pendekatan pipeline yang modular dan mendukung berbagai skenario pengujian.

## Struktur dan Komponen

`EvaluationManager` mengadopsi struktur modular berikut:

```
smartcash/handlers/evaluation/
├── __init__.py                          # Export komponen utama
├── evaluation_manager.py                # Entry point sebagai facade
├── core/                                # Komponen inti evaluasi
│   ├── evaluation_component.py          # Komponen dasar
│   ├── model_evaluator.py               # Evaluasi model
│   └── report_generator.py              # Generator laporan
├── pipeline/                            # Pipeline dan workflow
│   ├── base_pipeline.py                 # Pipeline dasar
│   ├── evaluation_pipeline.py           # Pipeline evaluasi standar
│   ├── batch_evaluation_pipeline.py     # Pipeline evaluasi batch
│   └── research_pipeline.py             # Pipeline penelitian
├── integration/                         # Adapter untuk integrasi
│   ├── metrics_adapter.py               # Adapter untuk MetricsCalculator
│   ├── model_manager_adapter.py         # Adapter untuk ModelManager
│   ├── dataset_adapter.py               # Adapter untuk DatasetManager
│   ├── checkpoint_manager_adapter.py    # Adapter untuk CheckpointManager
│   ├── visualization_adapter.py         # Adapter untuk visualisasi
│   └── adapters_factory.py              # Factory untuk adapter
└── observers/                           # Observer pattern
    ├── base_observer.py                 # Observer dasar
    ├── progress_observer.py             # Monitoring progres
    └── metrics_observer.py              # Monitoring metrik
```

`EvaluationManager` menggabungkan beberapa pipeline terspesialisasi menjadi satu antarmuka terpadu:

- **EvaluationPipeline**: Evaluasi model tunggal
- **BatchEvaluationPipeline**: Evaluasi batch model secara paralel
- **ResearchPipeline**: Evaluasi skenario penelitian dan perbandingan model

## Fitur Utama

### 1. Evaluasi Model Tunggal

- Evaluasi model dengan berbagai metrik standar (mAP, F1, precision, recall)
- Pengukuran waktu inferensi dan performa
- Perhitungan metrik per kelas dan layer
- Validasi model dan dataset otomatis
- Dukungan untuk model multilayer dengan evaluasi detil per layer

### 2. Evaluasi Batch

- Evaluasi beberapa model secara paralel dengan dataset yang sama
- Perbandingan performa berbagai model
- Analisis model terbaik berdasarkan metrik yang dipilih
- Visualisasi perbandingan dengan berbagai plot
- Thread-safety untuk eksekusi paralel

### 3. Evaluasi Skenario Penelitian

- Evaluasi model dalam skenario penelitian yang berbeda
- Analisis performa model dengan backbone berbeda (EfficientNet vs CSPDarknet)
- Perbandingan ketahanan model pada berbagai kondisi (posisi, pencahayaan)
- Analisis statistik dengan perhitungan rata-rata dan standar deviasi
- Multiple runs untuk mengukur stabilitas hasil

### 4. Pembuatan Laporan

- Generasi laporan dalam berbagai format (JSON, CSV, Markdown, HTML)
- Visualisasi hasil dengan berbagai jenis plot
- Analisis komprehensif termasuk insight dan rekomendasi
- Dukungan untuk eksport dan sharing hasil
- Penyimpanan metrik untuk analisis jangka panjang

### 5. Integrasi dengan Komponen Lain

- Adapter pattern untuk integrasi dengan komponan lain
- Factory pattern untuk inisialisasi komponen
- Observer pattern untuk progress monitoring dan metrics tracking
- Integrasi dengan logger untuk informatif logs
- Pemanfaatan berbagai utilitas SmartCash

## Kelas Utama

### EvaluationManager

```python
class EvaluationManager:
    """
    Manager utama evaluasi sebagai facade.
    Menyederhanakan antarmuka untuk evaluasi model dengan menggunakan berbagai
    adapter dan pipeline.
    """
    
    def __init__(
        self,
        config: Dict,
        logger = None,
        colab_mode: bool = False
    ):
        """
        Inisialisasi evaluation manager dengan berbagai adapter dan pipeline.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
            colab_mode: Mode Google Colab
        """
```

### Pipeline-Pipeline Evaluasi

#### EvaluationPipeline

```python
class EvaluationPipeline(BasePipeline):
    """
    Pipeline evaluasi dengan berbagai komponen yang dapat dikonfigurasi.
    Menggabungkan beberapa komponen evaluasi menjadi satu alur kerja.
    """
    
    def run(
        self,
        model_path: str,
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline evaluasi.
        
        Args:
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi
        """
```

#### BatchEvaluationPipeline

```python
class BatchEvaluationPipeline(BasePipeline):
    """
    Pipeline untuk evaluasi batch model.
    Evaluasi beberapa model dengan dataset yang sama secara paralel.
    """
    
    def run(
        self,
        model_paths: List[str],
        dataset_path: str,
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi batch untuk beberapa model.
        
        Args:
            model_paths: List path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            parallel: Evaluasi secara paralel
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap model
        """
```

#### ResearchPipeline

```python
class ResearchPipeline(BasePipeline):
    """
    Pipeline untuk evaluasi skenario penelitian.
    Evaluasi model dalam konteks perbandingan skenario penelitian dengan
    visualisasi hasil.
    """
    
    def run(
        self,
        scenarios: Dict[str, Dict],
        metrics_adapter = None,
        model_adapter = None,
        dataset_adapter = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan evaluasi skenario penelitian.
        
        Args:
            scenarios: Dictionary skenario penelitian
                Format: {
                    'Skenario-1': {
                        'desc': 'Deskripsi skenario',
                        'model': 'path/ke/model.pt',
                        'data': 'path/ke/dataset'
                    },
                    ...
                }
            metrics_adapter: Adapter untuk MetricsCalculator (opsional)
            model_adapter: Adapter untuk ModelManager (opsional)
            dataset_adapter: Adapter untuk DatasetManager (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil evaluasi untuk setiap skenario
        """
```

### Core Components

#### ModelEvaluator

```python
class ModelEvaluator(EvaluationComponent):
    """
    Komponen untuk evaluasi model dengan berbagai strategi.
    Melakukan proses evaluasi pada model dengan dataset yang diberikan.
    """
    
    def process(
        self,
        model_path: str,
        dataset_path: str,
        observers: Optional[List] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses evaluasi model.
        
        Args:
            model_path: Path ke checkpoint model
            dataset_path: Path ke dataset evaluasi
            observers: List observer (opsional)
            **kwargs: Parameter tambahan
                - batch_size: Ukuran batch
                - num_workers: Jumlah worker
                - device: Device untuk evaluasi ('cuda', 'cpu')
                - half_precision: Gunakan half precision
            
        Returns:
            Dictionary hasil evaluasi
        """
```

#### ReportGenerator

```python
class ReportGenerator:
    """
    Generator laporan hasil evaluasi model.
    Menghasilkan laporan dalam berbagai format (JSON, CSV, Markdown, HTML).
    """
    
    def generate(
        self,
        results: Dict[str, Any],
        format: str = 'json',
        output_path: Optional[str] = None,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Generate laporan evaluasi.
        
        Args:
            results: Hasil evaluasi
            format: Format laporan ('json', 'csv', 'markdown', 'html')
            output_path: Path output laporan (opsional)
            include_plots: Sertakan visualisasi (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan yang dihasilkan
        """
```

## Metode Utama di EvaluationManager

### evaluate_model

```python
def evaluate_model(
    self,
    model_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Evaluasi model tunggal menggunakan pipeline standar.
    
    Args:
        model_path: Path ke file model (opsional, gunakan terbaik jika None)
        dataset_path: Path ke dataset evaluasi (opsional, gunakan test_dir dari config)
        **kwargs: Parameter tambahan untuk pipeline
        
    Returns:
        Dictionary berisi hasil evaluasi
    """
```

Mengevaluasi satu model tertentu menggunakan dataset tertentu. Jika tidak diberikan model_path, akan menggunakan checkpoint terbaik yang tersedia. Jika tidak diberikan dataset_path, akan menggunakan test_dir dari konfigurasi.

### evaluate_batch

```python
def evaluate_batch(
    self,
    model_paths: Optional[List[str]] = None,
    dataset_path: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Evaluasi batch model menggunakan pipeline batch.
    
    Args:
        model_paths: List path ke file model (opsional, gunakan semua checkpoint jika None)
        dataset_path: Path ke dataset evaluasi (opsional, gunakan test_dir dari config)
        **kwargs: Parameter tambahan untuk pipeline
        
    Returns:
        Dictionary berisi hasil evaluasi untuk setiap model
    """
```

Mengevaluasi beberapa model secara paralel dengan dataset yang sama. Memberikan perbandingan komprehensif antar model.

### evaluate_research_scenarios

```python
def evaluate_research_scenarios(
    self,
    scenarios: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    Evaluasi skenario penelitian menggunakan pipeline penelitian.
    
    Args:
        scenarios: Dictionary skenario penelitian (opsional, gunakan default jika None)
        **kwargs: Parameter tambahan untuk pipeline
        
    Returns:
        Dictionary berisi hasil evaluasi untuk setiap skenario
    """
```

Mengevaluasi berbagai skenario penelitian, seperti model dengan backbone berbeda (EfficientNet vs CSPDarknet) pada berbagai kondisi pengujian (variasi posisi, pencahayaan dll).

### generate_report

```python
def generate_report(
    self,
    results: Dict,
    format: str = 'json',
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Buat laporan hasil evaluasi.
    
    Args:
        results: Hasil evaluasi 
        format: Format laporan ('json', 'csv', 'markdown', 'html')
        output_path: Path output laporan (opsional)
        **kwargs: Parameter tambahan untuk generator laporan
        
    Returns:
        Path ke file laporan
    """
```

Membuat laporan hasil evaluasi dalam berbagai format. Format yang didukung: JSON, CSV, Markdown, dan HTML.

### visualize_results

```python
def visualize_results(
    self,
    results: Dict,
    prefix: str = "",
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Visualisasikan hasil evaluasi menggunakan VisualizationAdapter.
    
    Args:
        results: Hasil evaluasi (dari evaluate_model, evaluate_batch, atau 
                evaluate_research_scenarios)
        prefix: Prefix untuk nama file output (opsional)
        output_dir: Direktori output untuk visualisasi (opsional)
        **kwargs: Parameter tambahan untuk visualisasi
        
    Returns:
        Dictionary berisi path ke file visualisasi yang dihasilkan
    """
```

Membuat visualisasi dari hasil evaluasi, seperti perbandingan mAP, F1-score, waktu inferensi, dan metrik lainnya.

## Adapters

### MetricsAdapter

```python
class MetricsAdapter:
    """
    Adapter untuk MetricsCalculator dari utils.metrics.
    Menyediakan antarmuka yang konsisten untuk menghitung dan mengelola metrik evaluasi.
    """
    
    def reset(self):
        """Reset metrics calculator untuk perhitungan baru."""
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray], 
        targets: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update metrik dengan batch prediksi dan target baru.
        
        Args:
            predictions: Tensor prediksi dari model
            targets: Tensor target ground truth
        """
    
    def compute(self) -> Dict[str, Any]:
        """
        Hitung metrik berdasarkan semua batch yang telah diupdate.
        
        Returns:
            Dictionary berisi metrik evaluasi
        """
```

### ModelManagerAdapter

```python
class ModelManagerAdapter:
    """
    Adapter untuk ModelManager.
    Menyediakan antarmuka untuk loading dan persiapan model untuk evaluasi.
    """
    
    def load_model(
        self,
        model_path: str,
        backbone: Optional[str] = None,
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Load model dari checkpoint.
        
        Args:
            model_path: Path ke checkpoint model
            backbone: Jenis backbone ('efficientnet', 'cspdarknet')
            device: Device untuk model ('cuda', 'cpu')
            force_reload: Paksa reload meskipun ada di cache
            
        Returns:
            Model yang sudah dimuat
        """
    
    def prepare_model_for_evaluation(
        self, 
        model: torch.nn.Module,
        half_precision: Optional[bool] = None
    ) -> torch.nn.Module:
        """
        Siapkan model untuk evaluasi.
        
        Args:
            model: Model PyTorch
            half_precision: Gunakan half precision (FP16)
            
        Returns:
            Model yang siap untuk evaluasi
        """
```

### DatasetAdapter

```python
class DatasetAdapter:
    """
    Adapter untuk DatasetManager.
    Menyediakan antarmuka untuk akses dataset dan pembuatan dataloader untuk evaluasi.
    """
    
    def get_dataset(
        self,
        dataset_path: str,
        split: str = 'test'
    ) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dataset PyTorch
        """
    
    def get_eval_loader(
        self,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> torch.utils.data.DataLoader:
        """
        Buat dataloader untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset (opsional, gunakan test_dir dari config)
            batch_size: Ukuran batch (opsional)
            num_workers: Jumlah worker (opsional)
            
        Returns:
            DataLoader untuk evaluasi
        """
```

### CheckpointManagerAdapter

```python
class CheckpointManagerAdapter:
    """
    Adapter untuk CheckpointManager.
    Menyediakan antarmuka untuk pencarian dan validasi checkpoint model.
    """
    
    def get_best_checkpoint(self, metric: str = 'mAP') -> str:
        """
        Dapatkan checkpoint terbaik berdasarkan metrik.
        
        Args:
            metric: Metrik untuk pemilihan ('mAP', 'f1', dll)
            
        Returns:
            Path ke checkpoint terbaik
        """
    
    def list_checkpoints(self, pattern: str = None) -> List[str]:
        """
        Dapatkan daftar checkpoint yang tersedia.
        
        Args:
            pattern: Pola glob untuk memfilter file (opsional)
            
        Returns:
            List path checkpoint
        """
```

### VisualizationAdapter

```python
class VisualizationAdapter:
    """
    Adapter untuk integrasi visualisasi evaluasi.
    Menghubungkan pipeline evaluasi dengan komponen visualisasi.
    """
    
    def generate_batch_plots(
        self,
        batch_results: Dict[str, Any],
        prefix: str = "batch",
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate visualisasi untuk evaluasi batch.
        
        Args:
            batch_results: Hasil evaluasi batch
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Dictionary berisi paths ke plots yang dibuat
        """
```

## Observer Pattern

### ProgressObserver

```python
class ProgressObserver(BaseObserver):
    """
    Observer untuk monitoring progres evaluasi.
    Menampilkan progress bar dan informasi runtime.
    """
    
    def update(self, event: str, data: Dict[str, Any] = None):
        """
        Update dari pipeline evaluasi.
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
```

### MetricsObserver

```python
class MetricsObserver(BaseObserver):
    """
    Observer untuk monitoring dan pencatatan metrik evaluasi.
    Berguna untuk tracking eksperimen dan visualisasi hasil.
    """
    
    def update(self, event: str, data: Dict[str, Any] = None):
        """
        Update dari pipeline evaluasi.
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
```

## Format Hasil

### Hasil Evaluasi Model Tunggal

```python
{
    'pipeline_name': 'EvaluationPipeline',
    'model_path': '/path/to/model.pt',
    'dataset_path': '/path/to/dataset',
    'execution_time': 45.23,  # seconds
    'metrics': {
        'accuracy': 0.92,
        'precision': 0.88,
        'recall': 0.89,
        'f1': 0.885,
        'mAP': 0.91,
        'inference_time': 0.023,  # seconds per image
        'class_metrics': {
            # Metrik per kelas
            0: {'precision': 0.95, 'recall': 0.92, 'f1': 0.93, 'ap': 0.94},
            # ...
        }
    },
    'model_info': {
        'filename': 'model.pt',
        'backbone': 'efficientnet',
        'size': 42500000,  # bytes
        'last_modified': 1645678900,  # timestamp
        'epoch': 50
    },
    'dataset_info': {
        'num_samples': 1000,
        'num_classes': 7,
        'class_distribution': {
            # Distribusi kelas dalam dataset
            0: 150,  # Jumlah sampel kelas 0
            # ...
        }
    }
}
```

### Hasil Evaluasi Batch

```python
{
    'pipeline_name': 'BatchEvaluationPipeline',
    'dataset_path': '/path/to/dataset',
    'num_models': 5,
    'execution_time': 120.5,  # seconds
    'model_results': {
        'model1': {
            # Hasil evaluasi model1, sama format dengan hasil evaluasi model tunggal
        },
        'model2': {
            # Hasil evaluasi model2
        },
        # ...
    },
    'summary': {
        'num_models': 5,
        'successful_models': 5,
        'failed_models': 0,
        'best_model': 'efficientnet_model',
        'best_map': 0.93,
        'average_map': 0.89,
        'metrics_table': pd.DataFrame,  # DataFrame metrik perbandingan
        'performance_comparison': {
            'mAP': {
                'best_model': 'efficientnet_model',
                'best_value': 0.93,
                'average': 0.89
            },
            # Perbandingan untuk metrik lain
        }
    },
    'plots': {
        'map_comparison': '/path/to/map_comparison.png',
        'inference_time': '/path/to/inference_time.png',
        # Path ke berbagai plot visualisasi
    }
}
```

### Hasil Evaluasi Skenario Penelitian

```python
{
    'pipeline_name': 'ResearchPipeline',
    'num_scenarios': 4,
    'execution_time': 350.2,  # seconds
    'scenario_results': {
        'Skenario-1': {
            'config': {
                'desc': 'YOLOv5 Default (CSPDarknet) - Posisi Bervariasi',
                'model': 'cspdarknet_position_varied.pt',
                'data': 'test_position_varied'
            },
            'results': {
                'run_results': [
                    # Hasil dari setiap run
                ],
                'avg_metrics': {
                    'mAP': 0.88,
                    'f1': 0.87,
                    'inference_time': 0.022
                },
                'std_metrics': {
                    'mAP': 0.02,
                    'f1': 0.01,
                    'inference_time': 0.001
                },
                'num_successful_runs': 3
            },
            'model_path': '/path/to/cspdarknet_model.pt',
            'dataset_path': '/path/to/dataset'
        },
        # Hasil untuk skenario lainnya
    },
    'summary': {
        'num_scenarios': 4,
        'successful_scenarios': 4,
        'failed_scenarios': 0,
        'best_scenario': 'Skenario-3',
        'best_map': 0.92,
        'metrics_table': pd.DataFrame,  # DataFrame metrik perbandingan
        'backbone_comparison': {
            'efficientnet': {
                'mAP': 0.91,
                'F1': 0.90,
                'inference_time': 0.024,
                'count': 2
            },
            'cspdarknet': {
                'mAP': 0.85,
                'F1': 0.86,
                'inference_time': 0.018,
                'count': 2
            }
        },
        'condition_comparison': {
            'Posisi Bervariasi': {
                'mAP': 0.89,
                'F1': 0.88,
                'count': 2
            },
            'Pencahayaan Bervariasi': {
                'mAP': 0.87,
                'F1': 0.85,
                'count': 2
            }
        }
    },
    'plots': {
        'backbone_comparison': '/path/to/backbone_comparison.png',
        'condition_comparison': '/path/to/condition_comparison.png',
        # Path ke berbagai plot visualisasi
    }
}
```

## Konfigurasi

EvaluationManager menggunakan bagian `evaluation` dari konfigurasi utama:

```python
config = {
    # Konfigurasi umum (diambil dari base_config.yaml)
    'app_name': "SmartCash",
    'version': "1.0.0",
    'data_dir': "data",
    'output_dir': "runs/evaluation",
    
    # Konfigurasi evaluasi
    'evaluation': {
        # Parameter deteksi
        'conf_threshold': 0.25,
        'iou_threshold': 0.45,
        
        # Konfigurasi metrics
        'metrics': {
            'primary_metric': 'mAP',  # Untuk pemilihan model terbaik
            'iou_threshold': 0.5,
            'conf_threshold': 0.25
        },
        
        # Konfigurasi batch evaluation
        'batch': {
            'max_workers': 4,  # Jumlah thread untuk evaluasi paralel
            'timeout': 3600,   # Timeout dalam detik
            'compare_metrics': ['mAP', 'f1', 'inference_time']
        },
        
        # Konfigurasi research scenarios
        'research': {
            'num_runs': 3,  # Jumlah run per skenario untuk stabilitas hasil
            'generate_plots': true,
            'parallel_scenarios': true,
            'max_workers': 2  # Jumlah thread untuk evaluasi paralel
        },
        
        # Konfigurasi visualisasi
        'visualization': {
            'plot_types': ['bar', 'heatmap', 'line'],
            'figsize': [10, 8],
            'dpi': 150,
            'colors': ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01']
        },
        
        # Konfigurasi reporting
        'reporting': {
            'default_format': 'markdown',
            'include_plots': true,
            'output_dir': 'reports/evaluation'
        }
    }
}
```

## Integrasi dengan Google Colab

EvaluationManager mendukung integrasi dengan Google Colab melalui parameter `colab_mode`:

```python
# Inisialisasi di Colab
evaluator = EvaluationManager(config, colab_mode=True)
```

Dalam mode Colab, EvaluationManager:

1. Menggunakan progress bar yang compatible dengan notebook
2. Mendukung integrasi dengan widget interaktif
3. Menyimpan hasil ke Google Drive jika tersedia
4. Menyesuaikan visualisasi untuk ditampilkan di notebook

## Kesimpulan

EvaluationManager SmartCash menawarkan:

1. **Fleksibilitas Evaluasi**: Mendukung evaluasi model tunggal, batch, dan skenario penelitian
2. **Modularitas**: Berbagai komponen dapat digunakan secara independen atau bersama-sama
3. **Optimasi Performa**: Paralelisasi, caching, dan optimasi memory
4. **Visualisasi Komprehensif**: Berbagai jenis plot dan format laporan
5. **Analisis Mendalam**: Perbandingan backbone, kondisi pengujian, dan performa model
6. **Integrasi Mulus**: Dengan komponen lain di SmartCash melalui adapter pattern
7. **Kemudahan Penggunaan**: Antarmuka facade yang sederhana namun powerful