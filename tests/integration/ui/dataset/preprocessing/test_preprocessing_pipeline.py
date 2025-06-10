"""
File: tests/integration/ui/dataset/preprocessing/test_preprocessing_pipeline.py
Deskripsi: End-to-end integration test untuk preprocessing pipeline
"""
import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers
from smartcash.ui.dataset.preprocessing.handlers.operation_handlers import execute_operation
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config

# Test data
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
SOURCE_DIR = os.path.join(TEST_DATA_DIR, "source")
TEMP_DIR = tempfile.mkdtemp()
OUTPUT_DIR = os.path.join(TEMP_DIR, "output")

# Sample dataset structure
def create_test_dataset():
    """Buat test dataset sementara"""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SOURCE_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(SOURCE_DIR, "labels"), exist_ok=True)
    
    # Buat beberapa file gambar dan label dummy
    for i in range(5):
        # Buat file gambar dummy (1x1 pixel)
        with open(os.path.join(SOURCE_DIR, "images", f"img_{i}.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\t\x08\x0b\x0c\x0c\x0b\x08\t\n\n")
        
        # Buat file label dummy (format YOLO)
        with open(os.path.join(SOURCE_DIR, "labels", f"img_{i}.txt"), "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")  # class_id x_center y_center width height

@pytest.fixture(scope="module")
def test_data():
    """Setup test data"""
    # Buat direktori test data
    os.makedirs(SOURCE_DIR, exist_ok=True)
    create_test_dataset()
    
    yield {
        "source_dir": SOURCE_DIR,
        "output_dir": OUTPUT_DIR
    }
    
    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

def create_mock_ui_components(config):
    """Buat mock UI components untuk testing"""
    components = {
        # Input fields
        'source_dir_input': type('', (), {'value': config['source_dir']}),
        'output_dir_input': type('', (), {'value': config['output_dir']}),
        'resolution_dropdown': type('', (), {'value': '640x640'}),
        'normalization_dropdown': type('', (), {'value': 'minmax'}),
        'batch_size_input': type('', (), {'value': 32}),
        'preserve_aspect_checkbox': type('', (), {'value': True}),
        'validation_checkbox': type('', (), {'value': True}),
        'target_splits_select': type('', (), {'value': ['train', 'valid']}),
        
        # Buttons
        'preprocess_button': type('', (), {'disabled': False}),
        'check_button': type('', (), {'disabled': False}),
        'cleanup_button': type('', (), {'disabled': False}),
        
        # Output areas
        'output_area': type('', (), {'clear_output': lambda *args, **kwargs: None}),
        'log_area': type('', (), {'clear_output': lambda *args, **kwargs: None}),
        
        # Callbacks
        'progress_callback': MagicMock(),
        'setup_dual_progress_tracker': MagicMock(),
        'complete_progress_tracker': MagicMock(),
        'error_progress_tracker': MagicMock(),
        'show_ui_success': MagicMock(),
        'handle_ui_error': MagicMock(),
        'clear_outputs': MagicMock(),
        'log_to_accordion': MagicMock(),
        
        # Backend services
        'validate_dataset_ready': lambda _: (True, "Valid"),
        'check_preprocessed_exists': lambda _: (False, "No existing data"),
        '_convert_ui_to_backend_config': lambda _: {},
        
        # Logger
        'logger': MagicMock()
    }
    
    # Setup button handlers with default config
    default_config = {
        'source_dir': config.get('source_dir', ''),
        'output_dir': config.get('output_dir', ''),
        'target_size': (640, 640),
        'normalization': 'minmax',
        'batch_size': 32,
        'preserve_aspect_ratio': True,
        'validation_enabled': True,
        'target_splits': ['train', 'valid']
    }
    setup_preprocessing_handlers(components, default_config)
    
    return components

class MockPreprocessor:
    """Mock preprocessor untuk testing"""
    def preprocess(self, config, progress_callback=None):
        """Simulasi preprocessing"""
        print(f"DEBUG - Preprocess config: {config}")
        
        # Extract output_dir from the correct path in the config structure
        output_dir = config.get('preprocessing', {}).get('output_dir', 
                     config.get('output_dir', 'data/preprocessed'))
        
        print(f"DEBUG - Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if progress_callback:
            # Call with correct signature: (step, current, total, message)
            progress_callback('preprocess', 50, 100, "Sedang memproses...")
        
        # Return success with expected format
        return {
            'success': True,
            'message': 'Preprocessing berhasil',
            'summary': {
                'total_images': 100,
                'valid_images': 100,
                'invalid_images': 0,
                'processing_time': 5.2
            },
            'output_dir': output_dir
        }
        
    def validate_dataset(self, config, split=None, progress_callback=None):
        """Simulate dataset validation"""
        if progress_callback:
            # Call with correct signature: (step, current, total, message)
            progress_callback('validate', 50, 100, f"Memvalidasi dataset {split}...")
            progress_callback('validate', 100, 100, "Validasi selesai")
            
        # Return format expected by validate_dataset_ready
        return {
            'success': True,
            'message': f'Validasi berhasil untuk {split}',
            'summary': {
                'total_images': 100,
                'valid_images': 100,
                'invalid_images': 0,
                'invalid_samples': []
            }
        }

class MockChecker:
    """Mock dataset checker untuk testing"""
    def validate(self):
        return True, "Dataset valid"
    def check(self, config, progress_callback=None):
        """Simulasi pengecekan dataset"""
        if progress_callback:
            progress_callback(1.0, "Checking complete")
        return True, "Dataset is valid"

class MockCleanupService:
    """Mock cleanup service untuk testing"""
    def cleanup(self, config):
        """Simulasi cleanup"""
        output_dir = config.get('preprocessing', {}).get('output_dir', 
                      config.get('output_dir', 'data/preprocessed'))
        
        # Return format expected by the cleanup handler
        return {
            'success': True,
            'message': 'Cleanup berhasil',
            'summary': {
                'deleted_files': 0,
                'freed_space': '0B',
                'output_dir': output_dir
            }
        }

def test_preprocessing_pipeline(test_data, monkeypatch):
    """Test end-to-end preprocessing pipeline"""
    # Setup mock for validate_dataset_ready
    def mock_validate_dataset_ready(config, logger=None):
        return True, "Dataset valid"
        
    # Setup mock for check_preprocessed_exists
    def mock_check_preprocessed_exists(config):
        return False, "No existing preprocessed data found"
    
    # Setup mock for _convert_ui_to_backend_config
    def mock_convert_ui_to_backend_config(ui_components):
        return {
            'preprocessing': {
                'enabled': True,
                'source_dir': test_data['source_dir'],
                'output_dir': test_data['output_dir'],
                'target_splits': ['train', 'valid']
            },
            'validation': {
                'enabled': True,
                'validate_images': True,
                'validate_labels': True
            }
        }

    # Setup
    config = {
        'source_dir': test_data['source_dir'],
        'output_dir': test_data['output_dir'],
        'preprocessing': {
            'enabled': True,
            'target_splits': ['train', 'valid']
        }
    }

    # Buat mock UI components
    ui_components = create_mock_ui_components(config)
    
    # Add mock functions to ui_components
    ui_components['validate_dataset_ready'] = mock_validate_dataset_ready
    ui_components['check_preprocessed_exists'] = mock_check_preprocessed_exists
    ui_components['_convert_ui_to_backend_config'] = mock_convert_ui_to_backend_config

    # Setup mock backend services
    ui_components['create_backend_preprocessor'] = lambda _: MockPreprocessor()
    ui_components['create_backend_checker'] = lambda _: MockChecker()
    ui_components['create_backend_cleanup_service'] = lambda _: MockCleanupService()

    # Test 1: Check dataset
    success, message = execute_operation(ui_components, 'check', config)
    assert success is True, f"Check dataset failed: {message}"
    assert "valid" in message.lower()

    # Test 2: Run preprocessing
    success, message = execute_operation(ui_components, 'preprocess', config)
    assert success is True, f"Preprocessing failed: {message}"
    assert "berhasil" in message.lower()

    # Test 3: Check if output directory was created
    assert os.path.exists(test_data['output_dir'])
    
    # Test 4: Run cleanup
    success, message = execute_operation(ui_components, 'cleanup', config)
    assert success is True, f"Cleanup failed: {message}"
    assert "berhasil" in message.lower(), f"Expected success message to contain 'berhasil', got: {message}"

def test_config_extraction():
    """Test konfigurasi ekstraksi dari UI components"""
    # Setup mock UI components sesuai dengan yang diharapkan extract_preprocessing_config
    ui_components = {
        # Input fields
        'source_dir': type('', (), {'value': SOURCE_DIR}),
        'output_dir': type('', (), {'value': OUTPUT_DIR}),
        
        # Resolution (format: 'WxH')
        'resolution_width': type('', (), {'value': '640'}),
        'resolution_height': type('', (), {'value': '640'}),
        'preserve_aspect_ratio': type('', (), {'value': True}),
        
        # Normalization
        'normalization_method': type('', (), {'value': 'minmax'}),
        
        # Batch size
        'batch_size': type('', (), {'value': 32}),
        
        # Validation
        'enable_validation': type('', (), {'value': True}),
        'move_invalid': type('', (), {'value': True}),
        'invalid_dir': type('', (), {'value': 'data/invalid'}),
        
        # Target splits
        'target_splits': type('', (), {'value': ['train', 'valid']}),
        
        # Logger
        'logger': MagicMock()
    }
    
    # Extract config
    config = extract_preprocessing_config(ui_components)
    
    # Verifikasi struktur config
    assert isinstance(config, dict), "Config should be a dictionary"
    
    # Verifikasi nilai preprocessing
    preprocessing = config.get('preprocessing', {})
    assert isinstance(preprocessing, dict), "Preprocessing config should be a dictionary"
    
    # Output dir bisa di-root atau di dalam preprocessing
    output_dir = config.get('output_dir') or preprocessing.get('output_dir')
    assert output_dir, "Output directory should be set"
    
    # Verifikasi performance
    performance = config.get('performance', {})
    assert isinstance(performance, dict), "Performance config should be a dictionary"
    assert performance.get('batch_size') == 32, "Batch size should be 32"
    
    # Verifikasi normalisasi
    normalization = preprocessing.get('normalization', {})
    assert isinstance(normalization, dict), "Normalization config should be a dictionary"
    assert normalization.get('method') == 'minmax', "Normalization method should be minmax"
    
    # Verifikasi validasi
    validation = preprocessing.get('validation', {})
    assert isinstance(validation, dict), "Validation config should be a dictionary"
    assert validation.get('enabled', False) is True, "Validation should be enabled"

if __name__ == "__main__":
    # Buat test data
    create_test_dataset()
    
    # Jalankan test
    pytest.main(["-v", __file__])
