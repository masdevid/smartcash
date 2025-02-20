# ✅ Testing Guide

## 📋 Overview

Panduan lengkap untuk testing SmartCash project.

## 🔧 Setup Testing Environment

### 1. Install Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Test Configuration
```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture
def model():
    """Create model instance."""
    return SmartCash(
        backbone="efficientnet_b4",
        pretrained=False
    )

@pytest.fixture
def sample_image():
    """Load sample image."""
    return torch.randn(1, 3, 640, 640)
```

## 🧪 Test Types

### 1. Unit Tests
```python
# tests/test_model.py
def test_forward_pass(model, sample_image):
    """Test model forward pass."""
    output = model(sample_image)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 7, 80, 80)
    
def test_detection(model):
    """Test object detection."""
    image = load_test_image()
    boxes = model.detect(image)
    
    assert len(boxes) > 0
    assert all(0 <= conf <= 1 for conf in boxes[:, 4])
```

### 2. Integration Tests
```python
def test_training_pipeline():
    """Test complete training pipeline."""
    # Setup
    model = SmartCash()
    dataset = RupiahDataset("data/test")
    trainer = Trainer(model)
    
    # Train
    metrics = trainer.fit(dataset)
    
    # Validate
    assert metrics["map50"] > 0.5
    assert metrics["precision"] > 0.7
    assert metrics["recall"] > 0.7
```

### 3. Performance Tests
```python
@pytest.mark.benchmark
def test_inference_speed(model, benchmark):
    """Benchmark inference speed."""
    image = load_test_image()
    
    # Run benchmark
    result = benchmark(model.detect, image)
    
    assert result.stats.mean < 0.1  # 100ms max
```

## 📊 Test Coverage

### 1. Coverage Configuration
```ini
# .coveragerc
[run]
source = smartcash
omit = tests/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

### 2. Run Coverage
```bash
# Generate coverage report
pytest --cov=smartcash --cov-report=html

# View report
open htmlcov/index.html
```

## 🔄 Continuous Testing

### 1. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        types: [python]
        pass_filenames: false
```

### 2. GitHub Actions
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=smartcash
```

## 🔍 Test Categories

### 1. Model Tests
```python
class TestModel:
    def test_initialization(self):
        """Test model initialization."""
        model = SmartCash()
        assert isinstance(model.backbone, EfficientNet)
        
    def test_forward(self):
        """Test forward pass."""
        model = SmartCash()
        x = torch.randn(1, 3, 640, 640)
        y = model(x)
        assert y.shape == (1, 7, 80, 80)
        
    def test_inference(self):
        """Test inference."""
        model = SmartCash()
        image = load_test_image()
        boxes = model.detect(image)
        assert len(boxes) > 0
```

### 2. Data Tests
```python
class TestDataset:
    def test_loading(self):
        """Test dataset loading."""
        dataset = RupiahDataset("data/test")
        assert len(dataset) > 0
        
    def test_augmentation(self):
        """Test data augmentation."""
        transform = get_train_transform()
        image = load_test_image()
        augmented = transform(image=image)["image"]
        assert augmented.shape == image.shape
        
    def test_preprocessing(self):
        """Test preprocessing."""
        processor = Preprocessor()
        image = load_test_image()
        processed = processor(image)
        assert processed.shape == (640, 640, 3)
```

### 3. Training Tests
```python
class TestTraining:
    def test_loss_computation(self):
        """Test loss computation."""
        criterion = YOLOLoss()
        pred = torch.randn(1, 7, 80, 80)
        target = torch.randn(1, 7, 80, 80)
        loss = criterion(pred, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
    def test_optimization(self):
        """Test optimization step."""
        model = SmartCash()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Forward pass
        x = torch.randn(1, 3, 640, 640)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert all(
            param.grad is not None 
            for param in model.parameters()
        )
```

## 📈 Performance Testing

### 1. Speed Tests
```python
@pytest.mark.benchmark
def test_batch_inference(benchmark):
    """Test batch inference speed."""
    model = SmartCash()
    batch = torch.randn(16, 3, 640, 640)
    
    result = benchmark(model, batch)
    
    assert result.stats.mean < 1.0  # 1s max
```

### 2. Memory Tests
```python
def test_memory_usage():
    """Test memory usage."""
    import psutil
    
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    model = SmartCash()
    _ = model(torch.randn(1, 3, 640, 640))
    
    memory_after = process.memory_info().rss
    memory_used = memory_after - memory_before
    
    assert memory_used < 2e9  # 2GB max
```

## 🔍 Test Utilities

### 1. Test Helpers
```python
# tests/utils.py
def load_test_image():
    """Load test image."""
    return cv2.imread("tests/data/test.jpg")
    
def compare_outputs(out1, out2, rtol=1e-5):
    """Compare two outputs."""
    return torch.allclose(out1, out2, rtol=rtol)
```

### 2. Mock Data
```python
class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 640, 640),
            "boxes": torch.randn(10, 4),
            "labels": torch.randint(0, 7, (10,))
        }
```

## 📊 Test Reporting

### 1. JUnit Report
```bash
pytest --junitxml=report.xml
```

### 2. HTML Report
```bash
pytest --html=report.html --self-contained-html
```

## 🐛 Debugging Tests

### 1. Debug Mode
```bash
pytest -vv --pdb
```

### 2. Logging
```python
import logging

def test_with_logging():
    """Test with logging."""
    logging.debug("Starting test")
    assert True
    logging.debug("Test completed")
```

## 🚀 Best Practices

1. Write tests first
2. Keep tests simple
3. Use meaningful names
4. Test edge cases
5. Regular testing
6. Monitor coverage
7. Document tests
8. Review test code

## 📈 Next Steps

1. [Code Style](CODE_STYLE.md)
2. [Documentation](../README.md)
3. [Deployment](../technical/DEPLOYMENT.md)
