# Struktur Project SmartCash untuk PyPI

Berikut adalah struktur yang direkomendasikan untuk packaging project SmartCash ke PyPI:

```
smartcash/
├── .github/                   # GitHub workflows untuk CI/CD
│   └── workflows/
│       ├── build.yml          # Build workflow
│       ├── publish.yml        # Publish ke PyPI workflow
│       └── test.yml           # Testing workflow
│
├── docs/                      # Dokumentasi
│   ├── conf.py                # Konfigurasi Sphinx
│   ├── index.rst              # Halaman utama dokumentasi
│   ├── installation.rst       # Panduan instalasi
│   ├── api/                   # Dokumentasi API
│   └── examples/              # Contoh penggunaan
│
├── examples/                  # Contoh script dan notebooks
│   ├── colab/                 # Notebook untuk Google Colab
│   └── scripts/               # Script contoh penggunaan
│
├── src/                       # Source code dalam folder src (best practice untuk PyPI)
│   └── smartcash/             # Package utama
│       ├── __init__.py        # Package initialization dengan __version__
│       ├── api/               # API untuk akses dari luar
│       │   ├── __init__.py    
│       │   ├── dataset.py     # API untuk dataset operations
│       │   ├── model.py       # API untuk model operations
│       │   └── detection.py   # API untuk detection operations
│       │
│       ├── core/              # Core functionality
│       │   ├── __init__.py
│       │   ├── dataset/       # Dataset handling
│       │   │   ├── __init__.py
│       │   │   ├── manager.py         # High-level dataset operations
│       │   │   ├── multilayer.py      # Multilayer dataset implementation
│       │   │   ├── loader.py          # Dataset loading operations
│       │   │   ├── validator.py       # Dataset validation
│       │   │   └── augmentor.py       # Dataset augmentation
│       │   │
│       │   ├── model/         # Model implementation
│       │   │   ├── __init__.py
│       │   │   ├── yolov5.py          # YOLOv5 model implementation
│       │   │   ├── backbones/         # Backbone implementations
│       │   │   │   ├── __init__.py
│       │   │   │   ├── base.py        # Base backbone interface
│       │   │   │   ├── efficientnet.py # EfficientNet backbone
│       │   │   │   └── cspdarknet.py  # CSPDarknet backbone
│       │   │   ├── necks/             # Feature processing networks
│       │   │   │   ├── __init__.py
│       │   │   │   └── fpn_pan.py     # FPN and PAN implementation
│       │   │   └── losses.py          # Custom loss functions
│       │   │
│       │   ├── detection/     # Detection logic
│       │   │   ├── __init__.py
│       │   │   ├── detector.py        # Core detection functionality
│       │   │   ├── processor.py       # Pre/post processing
│       │   │   └── visualizer.py      # Detection visualization
│       │   │
│       │   └── evaluation/    # Evaluation functionality
│       │       ├── __init__.py
│       │       ├── evaluator.py       # Model evaluation
│       │       ├── metrics.py         # Evaluation metrics
│       │       └── comparison.py      # Model comparison utilities
│       │
│       ├── utils/             # Utilities
│       │   ├── __init__.py
│       │   ├── logger.py              # Logging utilities
│       │   ├── config.py              # Configuration management
│       │   ├── checkpoint.py          # Checkpoint handling
│       │   ├── metrics.py             # Metrics calculation
│       │   ├── visualization.py       # Visualization utilities
│       │   ├── colab.py               # Google Colab integration
│       │   ├── coordinate.py          # Coordinate handling utilities
│       │   ├── early_stopping.py      # Early stopping implementation
│       │   └── cache.py               # Caching utilities
│       │
│       ├── cli/               # Command-line interface
│       │   ├── __init__.py
│       │   ├── commands.py            # CLI commands
│       │   ├── parsers.py             # Argument parsers
│       │   └── output.py              # CLI output formatting
│       │
│       └── ui/                # Optional UI components
│           ├── __init__.py
│           ├── app.py                 # Gradio web app
│           ├── components/            # UI components
│           │   ├── __init__.py
│           │   ├── dataset.py         # Dataset UI components
│           │   ├── model.py           # Model UI components
│           │   ├── training.py        # Training UI components
│           │   └── detection.py       # Detection UI components
│           └── handlers/              # UI event handlers
│               ├── __init__.py
│               ├── dataset_handlers.py
│               ├── model_handlers.py
│               ├── training_handlers.py
│               └── detection_handlers.py
│
├── tests/                     # Tests directory
│   ├── conftest.py            # Pytest configuration
│   ├── unit/                  # Unit tests
│   │   ├── test_dataset.py
│   │   ├── test_model.py
│   │   ├── test_detection.py
│   │   └── test_utils.py
│   └── integration/           # Integration tests
│       ├── test_api.py
│       ├── test_end_to_end.py
│       └── test_cli.py
│
├── .gitignore                 # Git ignore file
├── pyproject.toml             # Modern Python packaging (PEP 517/518)
├── setup.cfg                  # Package metadata
├── LICENSE                    # License file
├── README.md                  # Project readme
├── CHANGELOG.md               # Version changelog
└── CONTRIBUTING.md            # Contribution guidelines
```

## File Konfigurasi

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[tool.setuptools_scm]
write_to = "src/smartcash/_version.py"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'

[tool.isort]
profile = "black"
```

### setup.cfg

```ini
[metadata]
name = smartcash
description = Deteksi nilai mata uang Rupiah menggunakan YOLOv5 dengan EfficientNet-B4
long_description = file: README.md
long_description_content_type = text/markdown
author = Alfrida Sabar
author_email = info@example.com
license = MIT
license_file = LICENSE
classifiers =
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent
    Topic :: Scientific/Engineering :: Artificial Intelligence
    Topic :: Scientific/Engineering :: Image Recognition
project_urls =
    Bug Tracker = https://github.com/username/smartcash/issues
    Documentation = https://smartcash.readthedocs.io
    Source Code = https://github.com/username/smartcash

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.8
install_requires =
    torch>=1.9.0
    torchvision>=0.10.0
    numpy>=1.20.0
    Pillow>=8.2.0
    tqdm>=4.61.0
    timm>=0.5.4
    PyYAML>=5.4.1

[options.packages.find]
where = src

[options.extras_require]
dev =
    pytest>=6.0
    pytest-cov>=2.12
    black>=21.6b0
    isort>=5.9.1
    pre-commit>=2.13.0
ui =
    gradio>=2.4.0
colab =
    ipywidgets>=7.6.3
    matplotlib>=3.4.2
docs =
    sphinx>=4.0.2
    sphinx-rtd-theme>=0.5.2
all =
    %(dev)s
    %(ui)s
    %(colab)s
    %(docs)s

[options.entry_points]
console_scripts =
    smartcash = smartcash.cli.commands:main
```

## Contoh Implementasi __init__.py

### src/smartcash/__init__.py

```python
"""
SmartCash: Deteksi Nilai Mata Uang Rupiah

Sistem deteksi nilai mata uang Rupiah menggunakan YOLOv5 dengan backbone EfficientNet-B4.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("smartcash")
except PackageNotFoundError:
    # Package belum diinstal, coba gunakan setuptools_scm
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0+unknown"

# API utama untuk pengguna
from smartcash.api.dataset import DatasetAPI
from smartcash.api.model import ModelAPI
from smartcash.api.detection import DetectionAPI

# Convenience import (clean API)
dataset = DatasetAPI()
model = ModelAPI()
detection = DetectionAPI()

__all__ = [
    "dataset",
    "model",
    "detection",
    "DatasetAPI",
    "ModelAPI", 
    "DetectionAPI",
]
```

## Cara Penggunaan

### Instalasi

```bash
# Instalasi dasar
pip install smartcash

# Instalasi dengan UI components (Gradio)
pip install smartcash[ui]

# Instalasi dengan dukungan Colab
pip install smartcash[colab]

# Instalasi untuk pengembangan
pip install -e .[dev]

# Instalasi semua components
pip install smartcash[all]
```

### Contoh Penggunaan

```python
# Penggunaan API
import smartcash

# Setup dataset
smartcash.dataset.download(source="roboflow", version="1")
smartcash.dataset.validate(split="train", fix_issues=True)
smartcash.dataset.augment(split="train", types=["combined"], variations=3)

# Setup model dan training
model = smartcash.model.create(backbone="efficientnet_b4", pretrained=True)
results = smartcash.model.train(
    model=model,
    dataset="path/to/dataset",
    epochs=50,
    batch_size=16
)

# Deteksi
detections = smartcash.detection.detect(
    model_path="best.pt",
    image_path="test.jpg",
    confidence=0.25,
    visualize=True
)
```

### Penggunaan CLI

```bash
# Lihat bantuan
smartcash --help

# Training model
smartcash train --config config.yaml --epochs 50

# Evaluasi model
smartcash eval --model best.pt --data test_data/

# Deteksi
smartcash detect --model best.pt --image test.jpg --conf 0.25

# Running web UI
smartcash ui
```

## Keuntungan Struktur Ini

1. **Pemisahan Concerns yang Jelas**:
   - API layer yang stabil untuk pengguna
   - Core functionality terpisah dari interface

2. **Dependency Management yang Lebih Baik**:
   - Dependency optional melalui extras_require
   - Versioning yang jelas

3. **Modularitas**:
   - Pengguna dapat menggunakan bagian tertentu tanpa menginstal semua

4. **Maintainability**:
   - Struktur src/ menghindari import confusion
   - Package structure yang konsisten

5. **Usability**:
   - Clean API dengan convenience imports
   - CLI commands yang intuitif

6. **Testability**:
   - Struktur tests yang terorganisir
   - Separation antara unit dan integration tests

7. **Documentation**:
   - Dokumentasi yang terstruktur dengan Sphinx
   - Examples untuk use case umum
