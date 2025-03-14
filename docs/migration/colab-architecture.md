# CB01 - SmartCash Colab Cell Template Architecture Guide

## Struktur Direktori

```
smartcash/colab/
│
├── __init__.py             # Ekspor komponen publik untuk colab
├── notebook_builder.py     # Builder untuk mengatur cells secara terstruktur
│
├── cells/                  # Template cells dasar
│   ├── __init__.py
│   ├── setup_cell.py           # Cell setup dasar (install dependencies)
│   ├── import_cell.py          # Cell import library
│   ├── config_cell.py          # Cell konfigurasi project
│   ├── data_loading_cell.py    # Cell untuk loading data
│   ├── training_cell.py        # Cell untuk training
│   ├── evaluation_cell.py      # Cell untuk evaluasi
│   ├── inference_cell.py       # Cell untuk inferensi
│   ├── visualization_cell.py   # Cell untuk visualisasi
│   └── export_cell.py          # Cell untuk export model
│
├── components/             # Komponen UI untuk notebook
│   ├── __init__.py
│   ├── image_display.py        # Komponen display gambar 
│   ├── progress_display.py     # Komponen display progress
│   ├── metrics_chart.py        # Komponen chart untuk metrik
│   ├── file_browser.py         # Komponen browser file
│   ├── detection_visualizer.py # Komponen visualisasi deteksi
│   └── control_widget.py       # Komponen widget kontrol
│
├── handlers/               # Handler untuk interaksi UI
│   ├── __init__.py
│   ├── dataset_handler.py      # Handler untuk dataset
│   ├── model_handler.py        # Handler untuk model
│   ├── training_handler.py     # Handler untuk training
│   ├── inference_handler.py    # Handler untuk inferensi
│   ├── upload_handler.py       # Handler untuk upload file
│   └── event_handler.py        # Handler untuk event UI
│
├── utils/                  # Utilitas khusus Colab
│   ├── __init__.py
│   ├── drive_helper.py         # Helper untuk Google Drive
│   ├── display_helper.py       # Helper untuk display format
│   ├── widget_helper.py        # Helper untuk widget
│   ├── chart_helper.py         # Helper untuk grafik
│   └── notebook_helper.py      # Helper untuk fungsi notebook
│
└── templates/              # Template notebook lengkap
    ├── __init__.py
    ├── training_notebook.py    # Template notebook training
    ├── inference_notebook.py   # Template notebook inferensi
    ├── evaluation_notebook.py  # Template notebook evaluasi
    └── research_notebook.py    # Template notebook penelitian
```

## Konsep Arsitektur

Arsitektur untuk Colab Cell Template dirancang untuk memberikan struktur yang konsisten untuk notebook Jupyter/Colab sambil mempertahankan fleksibilitas dan kemudahan penggunaan. Ini melibatkan beberapa konsep utama:

1. **Cell as Component**: Setiap cell diimplementasikan sebagai komponen mandiri yang dapat digunakan kembali
2. **Separation of Concerns**: Pemisahan antara logika UI (components), logika bisnis (handlers), dan notebook assembly (templates)
3. **Builder Pattern**: Menggunakan builder pattern untuk mengonstruksi notebook secara terstruktur
4. **Event-Driven Interactions**: Menggunakan event-driven architecture untuk interaksi UI yang responsif

### NotebookBuilder

Class `NotebookBuilder` bertindak sebagai koordinator untuk menyusun cell dan menghasilkan notebook yang terstruktur:

```python
# notebook_builder.py
class NotebookBuilder:
    """
    Builder untuk mengatur cells secara terstruktur untuk notebook Colab.
    Memfasilitasi pembuatan notebook dengan struktur yang konsisten.
    """
    
    def __init__(self, title="SmartCash Notebook", author="Alfrida Sabar", logger=None):
        """Inisialisasi NotebookBuilder."""
        self.title = title
        self.author = author
        self.cells = []
        self.logger = logger or get_logger("notebook_builder")
        self.context = {}  # Shared context untuk cell-cell
    
    def add_cell(self, cell):
        """
        Tambahkan cell ke notebook.
        
        Args:
            cell: Cell untuk ditambahkan
        
        Returns:
            Self untuk method chaining
        """
        self.cells.append(cell)
        return self
    
    def add_header(self, subtitle=None, description=None):
        """
        Tambahkan header cell ke notebook.
        
        Returns:
            Self untuk method chaining
        """
        from smartcash.colab.cells import create_header_cell
        header = create_header_cell(
            title=self.title,
            subtitle=subtitle,
            author=self.author,
            description=description
        )
        return self.add_cell(header)
    
    def add_setup(self, install_dependencies=True, mount_drive=True):
        """
        Tambahkan setup cell ke notebook.
        
        Returns:
            Self untuk method chaining
        """
        from smartcash.colab.cells import create_setup_cell
        setup = create_setup_cell(
            install_dependencies=install_dependencies,
            mount_drive=mount_drive
        )
        return self.add_cell(setup)
    
    def build(self):
        """
        Build dan return notebook cells.
        
        Returns:
            List cells yang siap untuk dijalankan di Colab
        """
        return self.cells
```

### Cell Template

Setiap cell diimplementasikan sebagai fungsi yang menghasilkan kode Colab yang siap dijalankan:

```python
# cells/setup_cell.py
def create_setup_cell(install_dependencies=True, mount_drive=True, extra_packages=None):
    """
    Buat cell untuk setup lingkungan Colab.
    
    Args:
        install_dependencies: Flag untuk install dependencies
        mount_drive: Flag untuk mount Google Drive
        extra_packages: List package tambahan yang akan diinstall
        
    Returns:
        String kode Python untuk cell
    """
    code = [
        "# Setup Lingkungan SmartCash",
        "import os, sys",
        "from tqdm.auto import tqdm",
        "from pathlib import Path"
    ]
    
    if install_dependencies:
        code.append("\n# Install dependencies")
        code.append("!pip install -q torch torchvision torchaudio")
        code.append("!pip install -q pyyaml tqdm matplotlib seaborn opencv-python ipywidgets")
        
        if extra_packages:
            code.append("# Install extra packages")
            for package in extra_packages:
                code.append(f"!pip install -q {package}")
    
    if mount_drive:
        code.append("\n# Mount Google Drive")
        code.append("from google.colab import drive")
        code.append("drive.mount('/content/drive')")
        code.append("print('✅ Google Drive berhasil dimount di /content/drive')")
    
    code.append("\n# Setup path untuk SmartCash")
    code.append("# Asumsi folder project ada di Drive atau akan dibuat di Colab local")
    code.append("DRIVE_PATH = '/content/drive/MyDrive/SmartCash'")
    code.append("LOCAL_PATH = '/content/SmartCash'")
    code.append("")
    code.append("# Gunakan Drive jika ada atau local jika tidak")
    code.append("if os.path.exists(DRIVE_PATH):")
    code.append("    PROJECT_PATH = DRIVE_PATH")
    code.append("    print(f'🔍 Menggunakan project di Google Drive: {PROJECT_PATH}')")
    code.append("else:")
    code.append("    PROJECT_PATH = LOCAL_PATH")
    code.append("    os.makedirs(PROJECT_PATH, exist_ok=True)")
    code.append("    print(f'🔍 Menggunakan project di Colab local: {PROJECT_PATH}')")
    code.append("")
    code.append("# Tambahkan project path ke sys.path")
    code.append("if PROJECT_PATH not in sys.path:")
    code.append("    sys.path.append(PROJECT_PATH)")
    code.append("    print(f'✅ Path {PROJECT_PATH} ditambahkan ke sys.path')")
    
    return "\n".join(code)
```

### Komponen UI

Komponen UI bertanggung jawab untuk tampilan dan interaksi pengguna:

```python
# components/detection_visualizer.py
class DetectionVisualizer:
    """
    Komponen untuk visualisasi hasil deteksi dengan interface interaktif.
    """
    
    def __init__(self, class_names=None, colors=None):
        """
        Inisialisasi visualizer.
        
        Args:
            class_names: Dict atau list nama kelas
            colors: Dict atau list warna untuk kelas
        """
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        
        self.class_names = class_names or []
        self.colors = colors or self._generate_colors(len(self.class_names))
        
        # Widgets untuk kontrol visualisasi
        self.confidence_slider = widgets.FloatSlider(
            value=0.25, min=0.0, max=1.0, step=0.05,
            description='Confidence:',
            continuous_update=False
        )
        
        self.layer_selector = widgets.Dropdown(
            options=['all', 'banknote', 'nominal', 'security'],
            value='all',
            description='Layer:'
        )
        
        # Output widget untuk display hasil
        self.output = widgets.Output()
        
        # Layout utama
        self.layout = widgets.VBox([
            widgets.HBox([self.confidence_slider, self.layer_selector]),
            self.output
        ])
    
    def display(self, image, detections):
        """
        Tampilkan deteksi dengan widgets kontrol.
        
        Args:
            image: Gambar original
            detections: Dict hasil deteksi per layer
        """
        self.image = image
        self.detections = detections
        
        # Update ketika slider atau dropdown berubah
        self.confidence_slider.observe(self._update_display, names='value')
        self.layer_selector.observe(self._update_display, names='value')
        
        # Initial display
        self._update_display(None)
        
        # Return layout untuk ditampilkan
        return self.layout
    
    def _update_display(self, change):
        """Update tampilan berdasarkan kontrol."""
        from IPython.display import clear_output
        import matplotlib.pyplot as plt
        
        # Clear current output
        with self.output:
            clear_output(wait=True)
            
            # Filter deteksi berdasarkan confidence dan layer
            conf_thresh = self.confidence_slider.value
            selected_layer = self.layer_selector.value
            
            filtered_detections = {}
            for layer, dets in self.detections.items():
                if selected_layer == 'all' or layer == selected_layer:
                    filtered_detections[layer] = [
                        d for d in dets if d['confidence'] >= conf_thresh
                    ]
            
            # Draw detections
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(self.image)
            
            # Draw each detection
            for layer, dets in filtered_detections.items():
                for det in dets:
                    box = det['bbox']
                    cls_id = det['class_id']
                    conf = det['confidence']
                    color = self.colors[cls_id % len(self.colors)]
                    
                    # Draw rectangle
                    rect = plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class-{cls_id}"
                    label = f"{class_name} ({layer}) {conf:.2f}"
                    ax.text(
                        box[0], box[1] - 5, label,
                        color='white', fontsize=10, backgroundcolor=color
                    )
            
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Print summary
            total_dets = sum(len(dets) for dets in filtered_detections.values())
            print(f"📊 Menampilkan {total_dets} deteksi dengan confidence ≥ {conf_thresh:.2f}")
            for layer, dets in filtered_detections.items():
                if dets:
                    print(f"  • Layer {layer}: {len(dets)} deteksi")
```

### Handler

Handler mengelola logika bisnis dan interaksi dengan komponen UI:

```python
# handlers/inference_handler.py
class InferenceHandler:
    """
    Handler untuk proses inferensi dan visualisasi hasil.
    Berfungsi sebagai penghubung antara model detector dan komponen UI.
    """
    
    def __init__(self, detector=None, logger=None):
        """
        Inisialisasi handler inferensi.
        
        Args:
            detector: Instance Detector untuk inferensi
            logger: Logger untuk mencatat aktivitas
        """
        import ipywidgets as widgets
        from IPython.display import display
        
        self.detector = detector
        self.logger = logger or get_logger("inference_handler")
        
        # Widgets untuk input
        self.file_upload = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='Upload Gambar:'
        )
        
        self.url_input = widgets.Text(
            placeholder='Masukkan URL gambar...',
            description='URL Gambar:',
            disabled=False
        )
        
        self.inference_button = widgets.Button(
            description='Deteksi',
            button_style='primary',
            icon='search'
        )
        
        # Setup event handler
        self.file_upload.observe(self._on_file_change, names='value')
        self.inference_button.on_click(self._on_inference_click)
        
        # Output area
        self.output = widgets.Output()
        
        # Main layout
        self.layout = widgets.VBox([
            widgets.HBox([self.file_upload, self.url_input, self.inference_button]),
            self.output
        ])
        
        # Internal state
        self.current_image = None
        self.visualizer = None
    
    def display(self):
        """Tampilkan UI handler."""
        from IPython.display import display
        display(self.layout)
    
    def _on_file_change(self, change):
        """Handle ketika file diupload."""
        import cv2
        import numpy as np
        
        if not change.new:
            return
            
        # Get uploaded file content
        file_data = next(iter(change.new.values()))
        content = file_data['content']
        
        # Convert content to image
        try:
            import io
            image_stream = io.BytesIO(content)
            image_array = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            self.current_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Clear URL input
            self.url_input.value = ""
            
            # Log success
            self.logger.info(f"✅ Gambar berhasil diupload: {file_data['metadata']['name']}")
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat gambar: {str(e)}")
    
    def _on_inference_click(self, button):
        """Handle ketika tombol inference diklik."""
        if self.current_image is None and not self.url_input.value:
            with self.output:
                print("⚠️ Silakan upload gambar atau masukkan URL terlebih dahulu")
            return
            
        # Get image from URL if provided
        if self.url_input.value:
            try:
                import cv2
                import numpy as np
                import urllib.request
                
                req = urllib.request.urlopen(self.url_input.value)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                self.current_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                # Convert BGR to RGB
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                
                self.logger.info(f"✅ Gambar berhasil dimuat dari URL: {self.url_input.value}")
            except Exception as e:
                with self.output:
                    print(f"❌ Gagal memuat gambar dari URL: {str(e)}")
                return
        
        # Run inference
        try:
            with self.output:
                from IPython.display import clear_output
                clear_output(wait=True)
                print("🔍 Menjalankan deteksi...")
                
            # Run inference
            results = self.detector.detect_multilayer(self.current_image)
            
            # Display results
            from smartcash.colab.components import DetectionVisualizer
            if self.visualizer is None:
                self.visualizer = DetectionVisualizer(
                    class_names=self.detector.get_class_names()
                )
            
            with self.output:
                clear_output(wait=True)
                display(self.visualizer.display(self.current_image, results))
                
        except Exception as e:
            with self.output:
                from IPython.display import clear_output
                clear_output(wait=True)
                print(f"❌ Gagal menjalankan inferensi: {str(e)}")
                import traceback
                traceback.print_exc()
```

### Template Notebook

Template notebook menyediakan notebook lengkap yang siap digunakan:

```python
# templates/inference_notebook.py
def create_inference_notebook():
    """
    Buat notebook inferensi yang lengkap.
    
    Returns:
        List cells untuk notebook inferensi
    """
    from smartcash.colab.notebook_builder import NotebookBuilder
    
    # Buat builder dengan judul yang sesuai
    builder = NotebookBuilder(
        title="SmartCash - Deteksi Mata Uang Rupiah",
        author="Alfrida Sabar"
    )
    
    # Tambahkan header
    builder.add_header(
        subtitle="Notebook Inferensi",
        description="Notebook ini untuk melakukan inferensi deteksi mata uang rupiah dengan model SmartCash."
    )
    
    # Tambahkan setup cell
    builder.add_setup(
        install_dependencies=True,
        mount_drive=True,
        extra_packages=["gdown"]
    )
    
    # Tambahkan cell untuk import library
    from smartcash.colab.cells import create_import_cell
    builder.add_cell(create_import_cell())
    
    # Tambahkan cell untuk loading model
    builder.add_cell("""
# Download model dari source yang tersedia atau gunakan model yang sudah ada
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_ID"
MODEL_PATH = os.path.join(PROJECT_PATH, "models/smartcash_model.pt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"📥 Mendownload model ke {MODEL_PATH}...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Model berhasil didownload")
else:
    print(f"✅ Model sudah ada di {MODEL_PATH}")
    """)
    
    # Tambahkan cell untuk inisialisasi detector
    builder.add_cell("""
# Inisialisasi detector
from smartcash.detection.detector import Detector

# Konfigurasi untuk detector
config = {
    'img_size': [640, 640],
    'layers': ['banknote', 'nominal', 'security'],
    'banknote_threshold': 0.25,
    'nominal_threshold': 0.3,
    'security_threshold': 0.35
}

# Buat instance detector
detector = Detector(MODEL_PATH, config=config)
print("✅ Detector berhasil diinisialisasi")
    """)
    
    # Tambahkan cell untuk UI inferensi
    builder.add_cell("""
# UI untuk inferensi
from smartcash.colab.handlers import InferenceHandler

# Buat handler inferensi
handler = InferenceHandler(detector=detector)

# Tampilkan UI
print("🔍 Silakan upload gambar atau masukkan URL untuk mulai deteksi")
handler.display()
    """)
    
    # Build notebook
    return builder.build()
```

## Integrasi dengan Utils dan Facades

Komponen dalam modul `colab` dirancang untuk berintegrasi erat dengan layer lain pada arsitektur SmartCash:

1. **Integrasi dengan Dataset Facade**:

```python
# handlers/dataset_handler.py
class DatasetHandler:
    """Handler untuk operasi dataset di notebook."""
    
    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger("dataset_handler")
        
        # Buat dataset facade
        from smartcash.dataset.manager import DatasetManager
        self.dataset_manager = DatasetManager(config, logger=self.logger)
        
        # Setup UI components
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup komponen UI."""
        import ipywidgets as widgets
        
        # UI untuk download dataset
        self.download_button = widgets.Button(
            description='Download Dataset',
            button_style='primary',
            icon='download'
        )
        self.download_button.on_click(self._on_download_click)
        
        # UI untuk validasi dataset
        self.validate_button = widgets.Button(
            description='Validasi Dataset',
            button_style='info',
            icon='check'
        )
        self.validate_button.on_click(self._on_validate_click)
        
        # Output area
        self.output = widgets.Output()
        
        # Main layout
        self.layout = widgets.VBox([
            widgets.HBox([self.download_button, self.validate_button]),
            self.output
        ])
    
    def display(self):
        """Tampilkan UI handler."""
        from IPython.display import display
        display(self.layout)
    
    def _on_download_click(self, button):
        """Handler untuk download dataset."""
        with self.output:
            from IPython.display import clear_output
            clear_output(wait=True)
            print("📥 Mendownload dataset...")
            
            try:
                # Delegate ke dataset manager
                result = self.dataset_manager.pull_dataset(show_progress=True)
                print(f"✅ Dataset berhasil didownload: {result}")
            except Exception as e:
                print(f"❌ Gagal mendownload dataset: {str(e)}")
```

2. **Integrasi dengan Detection Module**:

```python
# handlers/model_handler.py
class ModelHandler:
    """Handler untuk operasi model di notebook."""
    
    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger("model_handler")
        
        # Setup UI components
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup komponen UI."""
        import ipywidgets as widgets
        
        # UI untuk loading model
        self.model_path = widgets.Text(
            placeholder='/path/to/model.pt',
            description='Model Path:',
            disabled=False
        )
        
        self.load_button = widgets.Button(
            description='Load Model',
            button_style='primary',
            icon='upload'
        )
        self.load_button.on_click(self._on_load_click)
        
        # Output area
        self.output = widgets.Output()
        
        # Main layout
        self.layout = widgets.VBox([
            widgets.HBox([self.model_path, self.load_button]),
            self.output
        ])
        
        # Internal state
        self.detector = None
    
    def display(self):
        """Tampilkan UI handler."""
        from IPython.display import display
        display(self.layout)
    
    def _on_load_click(self, button):
        """Handler untuk loading model."""
        if not self.model_path.value:
            with self.output:
                print("⚠️ Silakan tentukan path model terlebih dahulu")
            return
            
        with self.output:
            from IPython.display import clear_output
            clear_output(wait=True)
            print(f"📂 Loading model dari {self.model_path.value}...")
            
            try:
                # Import detector
                from smartcash.detection.detector import Detector
                
                # Load model
                self.detector = Detector(self.model_path.value, config=self.config)
                print(f"✅ Model berhasil dimuat dari {self.model_path.value}")
                
                # Print model info
                print("📊 Informasi Model:")
                print(f"  • Layers: {self.detector.layers}")
                print(f"  • Input Size: {self.detector.get_input_size()}")
                
                return self.detector
            except Exception as e:
                print(f"❌ Gagal memuat model: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
```

## Pembuatan Notebook Programatik

Untuk memudahkan pembuatan dan pengelolaan notebook, arsitektur ini mendukung pembuatan notebook secara programatik:

```python
# utils/notebook_helper.py
def convert_cells_to_notebook(cells, output_path=None):
    """
    Konversi list cells ke file notebook .ipynb.
    
    Args:
        cells: List string kode cells
        output_path: Path output untuk menyimpan notebook
        
    Returns:
        Dict notebook dalam format nbformat
    """
    import nbformat as nbf
    
    # Buat notebook baru
    notebook = nbf.v4.new_notebook()
    
    # Tambahkan cells
    notebook.cells = [nbf.v4.new_code_cell(cell) for cell in cells]
    
    # Simpan jika output_path disediakan
    if output_path:
        with open(output_path, 'w') as f:
            nbf.write(notebook, f)
        print(f"✅ Notebook berhasil disimpan ke {output_path}")
    
    return notebook

def get_notebook_url(notebook_path, github_repo="username/smartcash", branch="main"):
    """
    Dapatkan URL Colab untuk notebook.
    
    Args:
        notebook_path: Path relatif ke notebook dalam repo
        github_repo: Nama repo GitHub
        branch: Branch repo
        
    Returns:
        URL Colab untuk notebook
    """
    github_url = f"https://github.com/{github_repo}/blob/{branch}/{notebook_path}"
    colab_url = f"https://colab.research.google.com/github/{github_repo}/blob/{branch}/{notebook_path}"
    
    return {
        'github': github_url,
        'colab': colab_url
    }
```

## Integrasi dengan Google Drive

Untuk memudahkan integrasi dengan Google Drive, tersedia helper khusus:

```python
# utils/drive_helper.py
class DriveHelper:
    """Helper untuk operasi Google Drive."""
    
    @staticmethod
    def mount_drive():
        """Mount Google Drive dan return path."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive berhasil dimount di /content/drive")
            return "/content/drive/MyDrive"
        except Exception as e:
            print(f"❌ Gagal mount Google Drive: {str(e)}")
            return None
    
    @staticmethod
    def ensure_project_dir(project_name="SmartCash"):
        """Pastikan direktori project ada di Drive."""
        drive_path = "/content/drive/MyDrive"
        project_path = f"{drive_path}/{project_name}"
        
        import os
        if not os.path.exists(project_path):
            os.makedirs(project_path, exist_ok=True)
            print(f"✅ Direktori project dibuat di {project_path}")
        else:
            print(f"✅ Direktori project sudah ada di {project_path}")
            
        return project_path
    
    @staticmethod
    def sync_to_drive(source_path, target_path=None, project_name="SmartCash"):
        """
        Sinkronisasi direktori atau file ke Google Drive.
        
        Args:
            source_path: Path sumber yang akan disinkronkan
            target_path: Path target di Drive (opsional)
            project_name: Nama direktori project di Drive
            
        Returns:
            Path target di Drive
        """
        import os
        import shutil
        from pathlib import Path
        
        # Ensure drive is mounted
        drive_path = "/content/drive/MyDrive"
        if not os.path.exists(drive_path):
            DriveHelper.mount_drive()
        
        # Set target path jika tidak disediakan
        if target_path is None:
            source = Path(source_path)
            if source.is_file():
                target_path = f"{drive_path}/{project_name}/{source.name}"
            else:
                target_path = f"{drive_path}/{project_name}/{source.name}"
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Copy file or directory
        try:
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
                print(f"✅ File disalin ke Drive: {target_path}")
            else:
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"✅ Direktori disalin ke Drive: {target_path}")
            
            return target_path
        except Exception as e:
            print(f"❌ Gagal menyalin ke Drive: {str(e)}")
            return None
```

## Contoh Penggunaan

### Membuat dan Menjalankan Notebook Inferensi

```python
from smartcash.colab.templates import create_inference_notebook
from smartcash.colab.utils.notebook_helper import convert_cells_to_notebook

# Buat notebook inferensi
cells = create_inference_notebook()

# Konversi ke file .ipynb
notebook = convert_cells_to_notebook(cells, output_path="smartcash_inference.ipynb")

# Cetak pesan dengan URL
from IPython.display import HTML
display(HTML(
    '<a href="https://colab.research.google.com/github/username/smartcash/blob/main/notebooks/smartcash_inference.ipynb" target="_blank">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
))
```

### Contoh Cell untuk Upload dan Deteksi

```python
# Setup SmartCash
!pip install -q torch torchvision albumentations>=1.3.0
!git clone https://github.com/username/smartcash.git
%cd smartcash

# Import libraries
import sys
sys.path.append('.')
from smartcash.detection.detector import Detector
from smartcash.colab.handlers.inference_handler import InferenceHandler

# Load model
detector = Detector('models/smartcash_yolov5.pt')

# Setup UI untuk deteksi
handler = InferenceHandler(detector=detector)
handler.display()
```

### Contoh Cell untuk Training

```python
# Setup dan download dataset
from smartcash.colab.handlers.dataset_handler import DatasetHandler
from smartcash.colab.handlers.training_handler import TrainingHandler

# Inisialisasi handlers
dataset_handler = DatasetHandler()
training_handler = TrainingHandler()

# Download dataset
dataset_handler.display()

# Setup dan mulai training
training_handler.display()
```

## Kesimpulan

Arsitektur SmartCash Colab Cell Template menyediakan:

1. **Struktur Konsisten**: Template cells yang konsisten untuk memudahkan pengembangan notebook
2. **Komponen UI Interaktif**: Widget dan komponen visualisasi yang meningkatkan pengalaman pengguna
3. **Integrasi Seamless**: Integrasi yang mulus dengan komponen dataset dan detection
4. **Kemudahan Penggunaan**: API yang sederhana dan intuitif untuk membuat dan mengelola notebook
5. **Fleksibilitas Deployment**: Support untuk deployment di Colab atau lokal

Dengan arsitektur ini, pengguna dapat dengan mudah membuat dan menggunakan notebook untuk berbagai kebutuhan penelitian dan produksi terkait deteksi mata uang Rupiah.
