# Rencana Restrukturisasi dan Optimasi Evaluation Manager SmartCash

## Tujuan

Restrukturisasi `evaluator.py` dan `base_evaluation_handler.py` menjadi komponen-komponen atomic yang mengikuti prinsip Single Responsibility dengan menerapkan pola desain Factory, Strategy, dan Observer. Integrasi dengan manager dan utilitas yang sudah ada untuk menghindari duplikasi.

## ⚠️ Peringatan Duplikasi

**HINDARI menduplikasi implementasi yang sudah ada di folder `utils` dan manager lain:**
- `MetricsCalculator` - Gunakan dari `utils.evaluation_metrics`
- Visualizer - Gunakan dari `utils.visualization` 
- Komponen dari `preprocessing_manager` dan `dataset_manager`
- Gunakan `CheckpointManager` dan `ModelManager` yang sudah ada

## Struktur Folder dan File

```
smartcash/handlers/evaluation/
├── __init__.py                          # Export komponen utama
├── evaluation_manager.py                # Entry point minimal (facade)

├── core/                                # Komponen inti evaluation
│   ├── evaluation_component.py          # Komponen dasar untuk evaluasi
│   ├── model_evaluator.py               # Evaluasi model dasar
│   ├── metrics_processor.py             # Prosesor metrik evaluasi
│   └── report_generator.py              # Generator laporan evaluasi

├── pipeline/                            # Pipeline dan workflow
│   ├── evaluation_pipeline.py           # Pipeline evaluasi dasar
│   ├── batch_evaluation_pipeline.py     # Pipeline evaluasi batch
│   └── research_pipeline.py             # Pipeline untuk skenario penelitian

├── strategies/                          # Strategi-strategi evaluasi
│   ├── base_strategy.py                 # Strategi dasar
│   ├── model_evaluation/                # Strategi evaluasi model
│   │   ├── standard_evaluation.py       # Evaluasi standar
│   │   ├── multilayer_evaluation.py     # Evaluasi multilayer
│   │   └── research_evaluation.py       # Evaluasi skenario penelitian
│   └── reporting/                       # Strategi reporting
│       ├── json_reporter.py             # Reporter format JSON
│       ├── csv_reporter.py              # Reporter format CSV
│       └── markdown_reporter.py         # Reporter format Markdown

├── integration/                         # Adapter untuk integrasi
│   ├── metrics_adapter.py               # Adapter untuk MetricsCalculator
│   ├── dataset_adapter.py               # Adapter untuk DatasetManager
│   ├── model_manager_adapter.py         # Adapter untuk ModelManager
│   ├── checkpoint_manager_adapter.py    # Adapter untuk CheckpointManager
│   └── colab_adapter.py                 # Adapter untuk Google Colab

└── observers/                           # Observer pattern untuk monitoring
    ├── base_observer.py                 # Observer dasar
    ├── progress_observer.py             # Monitoring progres evaluasi
    └── metrics_observer.py              # Monitoring metrik evaluasi
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `evaluation_manager.py` sebagai entry point tunggal
   - Menyederhanakan antarmuka evaluasi kompleks

2. **Strategy Pattern**: 
   - Strategi untuk evaluasi dan pelaporan yang dapat diganti
   - Fleksibilitas dalam pemilihan metode evaluasi

3. **Adapter Pattern**: 
   - Adapter untuk `MetricsCalculator` dari `utils.evaluation_metrics`
   - Adapter untuk `DatasetManager` untuk akses dataset
   - Adapter untuk `CheckpointManager` untuk pengelolaan checkpoint
   - Adapter untuk `ModelManager` untuk loading dan inisialisasi model

4. **Observer Pattern**: 
   - Monitoring progres evaluasi tanpa mengubah logika inti
   - Dukungan untuk integrasi dengan UI Colab

5. **Factory Pattern**: 
   - Pembuatan komponen evaluasi yang terpisah dari penggunaannya
   - Dukungan untuk dependency injection

6. **Pipeline Pattern**: 
   - Evaluasi sebagai serangkaian langkah yang dapat dikonfigurasi
   - Kemampuan untuk menyesuaikan langkah-langkah evaluasi

## Kelas Kunci

### EvaluationManager

```python
class EvaluationManager:
    """
    Manager utama evaluasi sebagai facade.
    Menyederhanakan antarmuka untuk evaluasi model.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: bool = False
    ):
        """Inisialisasi evaluation manager."""
        # Inisialisasi adapter
        self.metrics_adapter = MetricsAdapter(config, logger)
        self.dataset_adapter = DatasetAdapter(config, logger)
        self.model_adapter = ModelManagerAdapter(config, logger)
        self.checkpoint_adapter = CheckpointManagerAdapter(config, logger)
        
        # Setup pipeline evaluasi
        self.standard_pipeline = EvaluationPipeline(config, logger)
        self.research_pipeline = ResearchPipeline(config, logger)
        self.batch_pipeline = BatchEvaluationPipeline(config, logger)
        
    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        **kwargs
    ) -> Dict:
        """Evaluasi model tunggal."""
        pass
        
    def evaluate_batch(
        self,
        model_paths: List[str],
        dataset_path: str,
        **kwargs
    ) -> Dict:
        """Evaluasi batch model."""
        pass
        
    def evaluate_research_scenarios(
        self,
        scenarios: Dict,
        **kwargs
    ) -> Dict:
        """Evaluasi skenario penelitian."""
        pass
        
    def generate_report(
        self,
        results: Dict,
        format: str = 'json',
        **kwargs
    ) -> str:
        """Buat laporan hasil evaluasi."""
        pass
```

### CheckpointManagerAdapter

```python
class CheckpointManagerAdapter:
    """
    Adapter untuk CheckpointManager.
    Menyediakan akses ke fungsi pengelolaan checkpoint.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi adapter CheckpointManager."""
        # Inisialisasi checkpoint manager dari existing component
        
    def get_latest_checkpoint(self, pattern: str = None) -> str:
        """Dapatkan checkpoint terbaru."""
        pass
        
    def get_best_checkpoint(self, metric: str = 'mAP') -> str:
        """Dapatkan checkpoint terbaik berdasarkan metrik."""
        pass
        
    def list_checkpoints(self, pattern: str = None) -> List[str]:
        """Dapatkan daftar checkpoint yang tersedia."""
        pass
```

### ModelManagerAdapter

```python
class ModelManagerAdapter:
    """
    Adapter untuk ModelManager.
    Menyediakan akses ke fungsi pengelolaan model.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi adapter ModelManager."""
        # Inisialisasi model manager dari existing component
        
    def load_model(
        self,
        model_path: str,
        backbone: str = None,
        device: str = None
    ) -> torch.nn.Module:
        """Load model dari checkpoint."""
        pass
        
    def get_model_info(self, model_path: str) -> Dict:
        """Dapatkan informasi model."""
        pass
```

### EvaluationPipeline

```python
class EvaluationPipeline:
    """
    Pipeline evaluasi dengan tahap yang dapat dikonfigurasi.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi evaluation pipeline."""
        self.config = config
        self.logger = logger or get_logger("evaluation_pipeline")
        self.observers = []
        self.components = []
        
    def add_component(self, component: EvaluationComponent) -> 'EvaluationPipeline':
        """Tambahkan komponen ke pipeline."""
        self.components.append(component)
        return self
        
    def add_observer(self, observer: BaseObserver) -> 'EvaluationPipeline':
        """Tambahkan observer untuk monitoring pipeline."""
        self.observers.append(observer)
        return self
        
    def run(
        self,
        model_path: str,
        dataset_path: str,
        **kwargs
    ) -> Dict:
        """Jalankan pipeline evaluasi."""
        # Notifikasi observer tentang start
        # Jalankan setiap komponen dalam urutan
        # Kumpulkan dan gabungkan hasil
        # Notifikasi observer tentang completion
        pass
```

### ModelEvaluator

```python
class ModelEvaluator(EvaluationComponent):
    """
    Komponen untuk evaluasi model dengan berbagai strategi.
    """
    
    def __init__(
        self,
        config: Dict,
        metrics_adapter: Optional[MetricsAdapter] = None,
        model_adapter: Optional[ModelManagerAdapter] = None,
        dataset_adapter: Optional[DatasetAdapter] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi model evaluator."""
        super().__init__(config, logger)
        self.metrics_adapter = metrics_adapter or MetricsAdapter(config, logger)
        self.model_adapter = model_adapter or ModelManagerAdapter(config, logger)
        self.dataset_adapter = dataset_adapter or DatasetAdapter(config, logger)
        
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = None,
        **kwargs
    ) -> Dict:
        """Evaluasi model dengan dataloader tertentu."""
        # Jalankan evaluasi batch-by-batch
        # Gunakan metrics adapter untuk menghitung metrik
        # Kembalikan dict hasil evaluasi
        pass
```

## Integrasi dengan CheckpointManager dan ModelManager

### CheckpointManagerAdapter

Adapter untuk `CheckpointManager` menyediakan akses ke fungsi pengelolaan checkpoint:

1. **Pencarian Checkpoint**:
   - `get_latest_checkpoint()`: Dapatkan checkpoint terbaru
   - `get_best_checkpoint()`: Dapatkan checkpoint terbaik berdasarkan metrik
   - `list_checkpoints()`: Dapatkan daftar checkpoint tersedia

2. **Validasi Checkpoint**:
   - `validate_checkpoint()`: Validasi integritas checkpoint
   - `get_checkpoint_info()`: Dapatkan metadata checkpoint

### ModelManagerAdapter

Adapter untuk `ModelManager` menyediakan akses ke fungsi pengelolaan model:

1. **Model Loading**:
   - `load_model()`: Load model dari checkpoint dengan backbone yang benar
   - `get_model_info()`: Dapatkan informasi model (arsitektur, params, dll)

2. **Model Operations**:
   - `prepare_model_for_evaluation()`: Siapkan model untuk evaluasi
   - `get_model_requirements()`: Dapatkan requirements model (input size, dll)

## Optimasi Performa

1. **Paralelisasi**:
   - Evaluasi batch model secara paralel dengan multiprocessing
   - Pemrosesan dataset dengan num_workers optimal

2. **Caching**:
   - Caching hasil loading model untuk evaluasi batch
   - Caching hasil perhitungan untuk dataset yang sama

3. **Batching dan Prefetching**:
   - Optimalisasi batch size untuk performa terbaik
   - Prefetching data untuk mengurangi waktu tunggu

4. **Progress Tracking**:
   - Monitoring progres evaluasi dengan observer pattern
   - Tampilan tqdm progress bar untuk evaluasi jangka panjang

## Contoh Penggunaan Setelah Restrukturisasi

```python
# Inisialisasi evaluator manager
from smartcash.handlers.evaluation import EvaluationManager
from smartcash.config import get_config_manager

# Dapatkan konfigurasi
config_manager = get_config_manager("configs/base_config.yaml")
config = config_manager.get_config()

# Inisialisasi evaluation manager
evaluator = EvaluationManager(config)

# Evaluasi model tunggal
results = evaluator.evaluate_model(
    model_path="checkpoints/best.pt",
    dataset_path="data/test",
    batch_size=32,
    num_workers=4
)

# Evaluasi skenario penelitian
research_results = evaluator.evaluate_research_scenarios(
    scenarios={
        "Skenario-1": {
            "desc": "YOLOv5 Default (CSPDarknet) - Posisi Bervariasi",
            "model": "cspdarknet_position_varied.pth",
            "data": "test_position_varied"
        },
        "Skenario-2": {
            "desc": "YOLOv5 EfficientNet-B4 - Posisi Bervariasi",
            "model": "efficientnet_position_varied.pth",
            "data": "test_position_varied"
        }
    },
    num_runs=3
)

# Buat laporan evaluasi
report_path = evaluator.generate_report(
    results=research_results,
    format="markdown",
    output_path="reports/research_evaluation.md"
)
```