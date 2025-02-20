# 🎨 Code Style Guide

## 📋 Overview

Panduan style code untuk project SmartCash.

## 🐍 Python Style

### 1. PEP 8
```python
# Good
def process_image(image: np.ndarray) -> np.ndarray:
    """Process image for model input."""
    return cv2.resize(image, (640, 640))

# Bad
def processImage(img):
    return cv2.resize(img,(640,640))
```

### 2. Type Hints
```python
from typing import List, Tuple, Optional

def detect_objects(
    image: np.ndarray,
    conf_thres: float = 0.25
) -> List[Dict[str, float]]:
    """
    Detect objects in image.
    
    Args:
        image: Input image
        conf_thres: Confidence threshold
        
    Returns:
        List of detections
    """
    return []
```

### 3. Docstrings
```python
def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int = 100
) -> Dict[str, float]:
    """
    Train model on dataset.
    
    Args:
        model: Model to train
        dataset: Training dataset
        epochs: Number of epochs
        
    Returns:
        Dictionary of metrics
        
    Raises:
        ValueError: If epochs < 1
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    return {"loss": 0.0}
```

## 🏗️ Code Organization

### 1. Project Structure
```
smartcash/
├── smartcash/
│   ├── __init__.py
│   ├── models/
│   ├── data/
│   └── utils/
├── tests/
├── docs/
└── setup.py
```

### 2. Module Organization
```python
"""Module docstring."""

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import torch

# Local imports
from .utils import process_image
from .models import SmartCash

# Constants
MAX_SIZE = 640
NUM_CLASSES = 7

# Classes
class Detector:
    """Detector class docstring."""
    
# Functions
def detect():
    """Function docstring."""
```

## 📝 Naming Conventions

### 1. Variables
```python
# Good
image_size = 640
num_classes = 7
is_training = True

# Bad
imgsize = 640
n_classes = 7
training = True
```

### 2. Functions
```python
# Good
def process_image():
    pass
    
def load_model():
    pass
    
# Bad
def processImage():
    pass
    
def loadmodel():
    pass
```

### 3. Classes
```python
# Good
class ImageProcessor:
    pass
    
class ModelTrainer:
    pass
    
# Bad
class imageprocessor:
    pass
    
class model_trainer:
    pass
```

## 🔧 Code Formatting

### 1. Line Length
```python
# Good
def long_function_name(
    parameter_1: int,
    parameter_2: str,
    parameter_3: float
) -> None:
    pass

# Bad
def long_function_name(parameter_1: int, parameter_2: str, parameter_3: float) -> None:
    pass
```

### 2. Imports
```python
# Good
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import process_image

# Bad
from torch.utils.data import *
import torch, numpy
```

### 3. Whitespace
```python
# Good
x = 1
y = 2
long_variable = 3

def func():
    return True

class MyClass:
    def method(self):
        pass

# Bad
x=1
y=2
long_variable=3

def func():
    return True
class MyClass:
    def method(self):
        pass
```

## 🔍 Code Quality

### 1. Error Handling
```python
# Good
try:
    image = load_image(path)
except FileNotFoundError:
    logger.error(f"Image not found: {path}")
    raise
except Exception as e:
    logger.error(f"Error loading image: {e}")
    raise

# Bad
try:
    image = load_image(path)
except:
    print("Error")
```

### 2. Logging
```python
# Good
import logging

logger = logging.getLogger(__name__)

def process():
    logger.info("Starting processing")
    try:
        result = do_work()
        logger.debug(f"Work result: {result}")
    except Exception:
        logger.exception("Processing failed")

# Bad
def process():
    print("Starting")
    try:
        result = do_work()
        print(f"Result: {result}")
    except:
        print("Error")
```

### 3. Testing
```python
# Good
def test_detection():
    """Test object detection."""
    model = SmartCash()
    image = load_test_image()
    result = model.detect(image)
    
    assert len(result) > 0
    assert all(0 <= conf <= 1 for conf in result["conf"])

# Bad
def test():
    model = SmartCash()
    assert model.detect(None) is not None
```

## 🛠️ Tools

### 1. Black
```bash
# Format code
black .

# Check formatting
black --check .
```

### 2. Flake8
```bash
# Check style
flake8 .

# Configuration
[flake8]
max-line-length = 88
extend-ignore = E203
```

### 3. MyPy
```bash
# Check types
mypy .

# Configuration
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
```

## 📊 Best Practices

### 1. Comments
```python
# Good
# Calculate confidence score
conf = box_score * class_score

# Bad
# Get score
s = a * b
```

### 2. Constants
```python
# Good
MAX_BATCH_SIZE = 32
DEFAULT_CONFIDENCE = 0.25
MODEL_CONFIG = {
    "backbone": "efficientnet_b4",
    "num_classes": 7
}

# Bad
batch_size = 32
conf = 0.25
```

### 3. Function Arguments
```python
# Good
def train(
    model: nn.Module,
    *,
    epochs: int = 100,
    batch_size: int = 32
) -> None:
    pass

# Bad
def train(model, epochs=100, batch_size=32):
    pass
```

## 🔄 Version Control

### 1. Commit Messages
```
# Good
feat: add EfficientNet backbone
fix: resolve detection confidence issue
docs: update API documentation

# Bad
update code
fix bug
add feature
```

### 2. Branch Names
```
# Good
feature/add-efficientnet
bugfix/detection-confidence
docs/api-documentation

# Bad
new-feature
fix
update
```

## 📚 Documentation

### 1. Code Documentation
```python
def process_batch(
    batch: torch.Tensor,
    *,
    augment: bool = False
) -> torch.Tensor:
    """
    Process batch of images.
    
    Args:
        batch: Batch of images (B, C, H, W)
        augment: Whether to apply augmentation
        
    Returns:
        Processed batch
        
    Raises:
        ValueError: If batch is empty
    """
    if batch.size(0) == 0:
        raise ValueError("Empty batch")
    return batch
```

### 2. README Documentation
```markdown
# Component Name

## Overview
Brief description

## Usage
```python
from component import func
result = func()
```

## Parameters
- param1: description
- param2: description

## Returns
Description of return value
```

## 🚀 Next Steps

1. [Testing Guide](TESTING.md)
2. [Git Workflow](GIT_WORKFLOW.md)
3. [Contributing](CONTRIBUTING.md)
