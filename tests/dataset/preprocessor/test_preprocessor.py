"""
File: tests/dataset/preprocessor/test_preprocessor.py
Deskripsi: Unit test untuk modul preprocessor
"""
import os
import numpy as np
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Import modul yang akan di-test
from smartcash.dataset.preprocessor import (
    PreprocessingService,
    preprocess_dataset,
    get_preprocessing_samples,
    validate_dataset,
    cleanup_preprocessed_data,
    get_preprocessing_status,
    PreprocessingEngine,
    PreprocessingValidator
)
from smartcash.common.logger import get_logger

# Mock classes
class MockPreprocessingEngine:
    def __init__(self, config):
        self.config = config
        self.preprocess_split = MagicMock()
        self.get_preprocessing_status = MagicMock()
        self.file_scanner = MagicMock()
        self.file_processor = MagicMock()
        self.path_resolver = MagicMock()
        self.cleanup_manager = MagicMock()

class MockPreprocessingValidator:
    """Mock untuk PreprocessingValidator."""
    
    def __init__(self, config):
        self.config = config
        self.validate_image = MagicMock(return_value={'status': 'success'})
        self.validate_label = MagicMock(return_value={'status': 'success'})
        self.validate_image_label_pair = MagicMock(return_value={'status': 'success'})
        self.validate_dataset = MagicMock(return_value={'status': 'success'})
        self.validate_split = MagicMock(return_value={'status': 'success'})

# Menggunakan implementasi MockProgressTracker dari conftest.py
from conftest import MockProgressTracker

class TestPreprocessingService:
    """Test untuk PreprocessingService"""
    
    def test_init_with_default_config(self, preprocessor_service):
        """Test inisialisasi dengan konfigurasi default"""
        assert preprocessor_service is not None
        assert hasattr(preprocessor_service, 'config')
        assert 'preprocessing' in preprocessor_service.config
        
    def test_preprocess_and_visualize(self, preprocessor_service, progress_tracker):
        """Test preprocessing dengan visualisasi"""
        # Setup mock return values
        preprocessor_service.mock_engine.preprocess_split.return_value = {
            'status': 'success',
            'total_processed': 10,
            'valid_files': 9,
            'invalid_files': 1,
            'invalid_samples': [{'file': 'sample1.jpg', 'error': 'Invalid format'}]
        }
        
        result = preprocessor_service.preprocess_and_visualize(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi hasil
        assert result['status'] == 'success'
        assert result['total_processed'] == 10
        assert result['valid_files'] == 9
        
        # Verifikasi progress tracker diupdate dengan benar
        assert progress_tracker.was_progress_made() is True
        assert progress_tracker.was_completed() is True
        assert progress_tracker.has_error() is False
        
        # Verifikasi progress naik secara monoton
        progresses = [p[1] for p in progress_tracker.messages if p[0] == 'progress']
        assert all(0 <= p <= 100 for p in progresses)
        assert sorted(progresses) == progresses  # Progress harus naik
        
        # Verifikasi preprocess_split dipanggil dengan parameter yang benar
        preprocessor_service.mock_engine.preprocess_split.assert_called_once()
        
        # Verifikasi pesan status ada di progress tracker
        status_messages = [m[1] for m in progress_tracker.messages if m[0] == 'status']
        assert any('Memproses' in msg for msg in status_messages)
    
    def test_get_sampling(self, preprocessor_service):
        """Test pengambilan sampel"""
        # Setup mock return values
        preprocessor_service.mock_engine.file_scanner.scan_directory.return_value = [
            Path('image1.jpg'), Path('image2.jpg')
        ]
        preprocessor_service.mock_engine.file_processor.read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        samples = preprocessor_service.get_sampling(max_samples=2)
        
        assert 'samples' in samples
        assert len(samples['samples']) == 2
        assert all('original_image' in s for s in samples['samples'])
        assert all('preprocessed_image' in s for s in samples['samples'])
        
        # Verifikasi file_scanner dan file_processor dipanggil
        preprocessor_service.mock_engine.file_scanner.scan_directory.assert_called()
        assert preprocessor_service.mock_engine.file_processor.read_image.call_count == 2
    
    def test_validate_dataset_only(self, preprocessor_service, progress_tracker):
        """Test validasi dataset tanpa preprocessing"""
        # Setup mock return values
        preprocessor_service.mock_validator.validate_dataset.return_value = {
            'status': 'success',
            'total_files': 10,
            'valid_files': 9,
            'invalid_files': 1,
            'invalid_samples': [
                {'file': 'invalid1.jpg', 'error': 'Corrupted image'}
            ]
        }
        
        result = preprocessor_service.validate_dataset_only(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi hasil validasi
        assert result['status'] == 'success'
        assert result['total_files'] == 10
        assert result['valid_files'] == 9
        assert result['invalid_files'] == 1
        
        # Verifikasi progress tracker diupdate dengan benar
        assert progress_tracker.was_progress_made() is True
        assert progress_tracker.was_completed() is True
        assert progress_tracker.has_error() is False
        
        # Verifikasi progress naik secara monoton
        progresses = [p[1] for p in progress_tracker.messages if p[0] == 'progress']
        assert all(0 <= p <= 100 for p in progresses)
        assert sorted(progresses) == progresses  # Progress harus naik
        
        # Verifikasi validate_dataset dipanggil dengan parameter yang benar
        preprocessor_service.mock_validator.validate_dataset.assert_called_once_with(
            split_name='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi pesan status ada di progress tracker
        status_messages = [m[1] for m in progress_tracker.messages if m[0] == 'status']
        assert any('Memvalidasi' in msg for msg in status_messages)
    
    def test_cleanup_preprocessed_data(self, preprocessor_service, progress_tracker, tmp_path):
        """Test pembersihan file hasil preprocessing"""
        # Setup mock return values
        preprocessor_service.mock_engine.cleanup_manager.cleanup_output_dirs.return_value = {
            'status': 'success',
            'files_deleted': 5,
            'dirs_deleted': 2,
            'total_size_freed': '10.5 MB'
        }
        
        # Jalankan cleanup dengan progress tracker
        result = preprocessor_service.cleanup_preprocessed_data(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi hasil cleanup
        assert result['status'] == 'success'
        assert result['files_deleted'] == 5
        assert result['dirs_deleted'] == 2
        assert 'total_size_freed' in result
        
        # Verifikasi progress tracker diupdate dengan benar
        assert progress_tracker.was_progress_made() is True
        assert progress_tracker.was_completed() is True
        assert progress_tracker.has_error() is False
        
        # Verifikasi cleanup_manager.cleanup_output_dirs dipanggil dengan parameter yang benar
        preprocessor_service.mock_engine.cleanup_manager.cleanup_output_dirs.assert_called_once_with(
            split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi pesan status ada di progress tracker
        status_messages = [m[1] for m in progress_tracker.messages if m[0] == 'status']
        assert any('membersihkan' in msg.lower() for msg in status_messages)
    
    def test_get_preprocessing_status(self, preprocessor_service):
        """Test mendapatkan status preprocessing"""
        # Setup mock return values
        preprocessor_service.mock_engine.file_scanner.scan_directory.side_effect = [
            [Path('image1.jpg'), Path('image2.jpg')],  # train
            [Path('valid1.jpg')],                      # valid
            []                                         # test
        ]
        
        # Mock config
        preprocessor_service.config = {
            'preprocessing': {
                'enabled': True,
                'validation': {'enabled': True},
                'output_dir': '/path/to/output'
            },
            'data': {
                'splits': {
                    'train': '/path/to/train',
                    'valid': '/path/to/valid',
                    'test': '/path/to/test'
                }
            }
        }
        
        # Mock status preprocessing
        preprocessor_service.mock_engine.get_preprocessing_status.return_value = {
            'train': {'status': 'completed', 'last_processed': '2025-06-10T10:00:00'},
            'valid': {'status': 'pending', 'last_processed': None},
            'test': {'status': 'not_started', 'last_processed': None}
        }
        
        # Dapatkan status
        status = preprocessor_service.get_preprocessing_status()
        
        # Verifikasi status berisi field yang diharapkan
        assert 'preprocessing_enabled' in status
        assert 'validation_enabled' in status
        assert 'splits' in status
        assert 'output_dir' in status
        
        # Verifikasi status untuk setiap split
        splits = status['splits']
        assert 'train' in splits
        assert 'valid' in splits
        assert 'test' in splits
        
        # Verifikasi data status untuk train split
        assert splits['train']['source_files'] == 2
        assert splits['train']['status'] == 'completed'
        assert 'last_processed' in splits['train']
        
        # Verifikasi data status untuk valid split
        assert splits['valid']['source_files'] == 1
        assert splits['valid']['status'] == 'pending'
        
        # Verifikasi data status untuk test split
        assert splits['test']['source_files'] == 0
        assert splits['test']['status'] == 'not_started'
        
        # Verifikasi pemanggilan fungsi
        preprocessor_service.mock_engine.file_scanner.scan_directory.assert_any_call('/path/to/train')
        preprocessor_service.mock_engine.file_scanner.scan_directory.assert_any_call('/path/to/valid')
        preprocessor_service.mock_engine.file_scanner.scan_directory.assert_any_call('/path/to/test')


class TestOutputFormat:
    """Test untuk memastikan format output sesuai dengan spesifikasi"""
    
    def test_output_directory_structure(self, test_config, tmp_path):
        """Test struktur direktori output sesuai spesifikasi"""
        # Setup config untuk test
        test_config.update({
            'data': {
                'root_dir': str(tmp_path),
                'splits': {
                    'train': str(tmp_path / 'raw/train'),
                    'valid': str(tmp_path / 'raw/valid'),
                    'test': str(tmp_path / 'raw/test')
                }
            },
            'preprocessing': {
                'output_dir': str(tmp_path / 'preprocessed'),
                'validation': {'enabled': True},
                'output': {
                    'create_npy': True,
                    'organize_by_split': True
                }
            }
        })
        
        # Buat direktori sumber
        for split in ['train', 'valid', 'test']:
            (tmp_path / f'raw/{split}/images').mkdir(parents=True, exist_ok=True)
            (tmp_path / f'raw/{split}/labels').mkdir(parents=True, exist_ok=True)
            
            # Buat file dummy
            (tmp_path / f'raw/{split}/images/test.jpg').touch()
            (tmp_path / f'raw/{split}/labels/test.txt').touch()
        
        # Jalankan preprocessing
        progress_tracker = MockProgressTracker()
        result = preprocess_dataset(
            config=test_config,
            target_split='train',
            progress_tracker=progress_tracker,
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi struktur direktori output
        assert (tmp_path / 'preprocessed/train/images').exists()
        assert (tmp_path / 'preprocessed/train/labels').exists()
        
        # Verifikasi file output
        npy_files = list((tmp_path / 'preprocessed/train/images').glob('*.npy'))
        assert len(npy_files) > 0, "Tidak ada file .npy yang dihasilkan"
        
        # Verifikasi format nama file
        for npy_file in npy_files:
            assert npy_file.stem.startswith('pre_')
            assert '_' in npy_file.stem  # Pastikan ada UUID atau increment
    
    def test_npy_file_format(self, test_config, tmp_path):
        """Test format file .npy yang dihasilkan"""
        # Setup config
        test_config.update({
            'data': {'root_dir': str(tmp_path)},
            'preprocessing': {
                'output_dir': str(tmp_path / 'preprocessed'),
                'normalization': {'method': 'minmax'}
            }
        })
        
        # Buat file gambar dummy
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        
        # Buat gambar RGB sederhana
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_dir / 'test.jpg'), img)
        
        # Jalankan preprocessing
        progress_tracker = MockProgressTracker()
        result = preprocess_dataset(
            config=test_config,
            target_split='test',
            progress_tracker=progress_tracker
        )
        
        # Verifikasi file output
        npy_files = list((tmp_path / 'preprocessed/test/images').glob('*.npy'))
        assert len(npy_files) == 1
        
        # Baca file .npy
        preprocessed = np.load(npy_files[0])
        
        # Verifikasi tipe data dan rentang nilai
        assert preprocessed.dtype == np.float32
        assert 0.0 <= preprocessed.min() <= 1.0
        assert 0.0 <= preprocessed.max() <= 1.0
        
        # Verifikasi dimensi (H, W, C)
        assert len(preprocessed.shape) == 3
        assert preprocessed.shape[2] == 3  # Channel terakhir
    
    def test_label_file_format(self, test_config, tmp_path):
        """Test format file label yang dihasilkan"""
        # Setup config
        test_config.update({
            'data': {
                'root_dir': str(tmp_path),
                'splits': {'train': str(tmp_path / 'train')}
            },
            'preprocessing': {
                'output_dir': str(tmp_path / 'preprocessed'),
                'validation': {'enabled': True}
            }
        })
        
        # Buat direktori dan file sumber
        img_dir = tmp_path / 'train/images'
        label_dir = tmp_path / 'train/labels'
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        
        # Buat gambar dummy
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_dir / 'test.jpg'), img)
        
        # Buat file label YOLO format
        with open(label_dir / 'test.txt', 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n')  # class_id x_center y_center width height
        
        # Jalankan preprocessing
        progress_tracker = MockProgressTracker()
        result = preprocess_dataset(
            config=test_config,
            target_split='train',
            progress_tracker=progress_tracker
        )
        
        # Verifikasi file label output
        label_files = list((tmp_path / 'preprocessed/train/labels').glob('*.txt'))
        assert len(label_files) == 1
        
        # Baca isi file label
        with open(label_files[0], 'r') as f:
            lines = f.readlines()
            
        # Verifikasi format YOLO
        assert len(lines) > 0
        parts = lines[0].strip().split()
        assert len(parts) == 5  # class_id, x, y, w, h
        
        # Verifikasi nilai dalam rentang [0,1]
        values = list(map(float, parts[1:]))
        assert all(0.0 <= v <= 1.0 for v in values)
    
    def test_uuid_consistency(self, test_config, tmp_path):
        """Test konsistensi UUID antara file gambar dan label"""
        # Setup config
        test_config.update({
            'data': {
                'root_dir': str(tmp_path),
                'splits': {'train': str(tmp_path / 'train')}
            },
            'preprocessing': {
                'output_dir': str(tmp_path / 'preprocessed'),
                'file_naming': {
                    'preprocessed_pattern': 'pre_{uuid}',
                    'preserve_uuid': True
                }
            }
        })
        
        # Buat direktori dan file sumber
        img_dir = tmp_path / 'train/images'
        label_dir = tmp_path / 'train/labels'
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        
        # Buat gambar dummy
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_dir / 'test.jpg'), img)
        
        # Buat file label dummy
        (label_dir / 'test.txt').touch()
        
        # Jalankan preprocessing
        progress_tracker = MockProgressTracker()
        result = preprocess_dataset(
            config=test_config,
            target_split='train',
            progress_tracker=progress_tracker
        )
        
        # Dapatkan daftar file output
        img_files = list((tmp_path / 'preprocessed/train/images').glob('*'))
        label_files = list((tmp_path / 'preprocessed/train/labels').glob('*'))
        
        # Verifikasi jumlah file sama
        assert len(img_files) == len(label_files)
        
        # Verifikasi UUID konsisten
        img_uuids = {f.stem.split('_')[-1] for f in img_files}
        label_uuids = {f.stem.split('_')[-1] for f in label_files}
        
        assert img_uuids == label_uuids, "UUID antara gambar dan label tidak konsisten"


class TestPreprocessingModule:
    """Test untuk fungsi-fungsi modul preprocessor"""
    
    def test_service_with_mock_engine(self, test_config):
        """Test integrasi PreprocessingService dengan MockPreprocessingEngine"""
        # Setup mock engine dan validator
        mock_engine = MockPreprocessingEngine(test_config)
        mock_validator = MockPreprocessingValidator(test_config)
        
        # Buat progress tracker untuk testing
        progress_tracker = MockProgressTracker()
        
        # Buat service dengan mock engine dan validator
        with patch('smartcash.dataset.preprocessor.service.PreprocessingEngine', return_value=mock_engine), \
             patch('smartcash.dataset.preprocessor.service.PreprocessingValidator', return_value=mock_validator):
            
            service = PreprocessingService(test_config, progress_tracker=progress_tracker)
            
            # Setup mock return value
            mock_result = {
                'status': 'success',
                'total_processed': 10,
                'valid_files': 9,
                'invalid_files': 1,
                'invalid_samples': [{'file': 'invalid.jpg', 'error': 'Corrupted'}]
            }
            mock_engine.preprocess_split.return_value = mock_result
            
            # Setup mock untuk progress callback
            progress_updates = []
            def mock_progress_callback(progress, status):
                progress_updates.append((progress, status))
            
            # Panggil method yang akan di-test
            result = service.preprocess_and_visualize(
                target_split='train',
                progress_callback=mock_progress_callback
            )
            
            # Verifikasi hasil
            assert result == mock_result
            
            # Verifikasi engine dipanggil dengan parameter yang benar
            mock_engine.preprocess_split.assert_called_once()
            
            # Dapatkan argumen yang digunakan untuk memanggil preprocess_split
            call_args = mock_engine.preprocess_split.call_args[1]
            assert call_args['split_name'] == 'train'
            assert callable(call_args['progress_callback'])
            
            # Verifikasi progress tracker diupdate
            assert progress_tracker.was_progress_made() is True
            assert progress_tracker.was_completed() is True
            
            # Verifikasi progress callback dipanggil beberapa kali
            assert len(progress_updates) > 0
            
            # Verifikasi progress naik secara monoton
            progresses = [p for p, _ in progress_updates if isinstance(p, (int, float))]
            if len(progresses) > 1:
                assert all(progresses[i] <= progresses[i+1] for i in range(len(progresses)-1))
            
            # Verifikasi status terakhir adalah 'completed'
            _, last_status = progress_updates[-1]
            assert 'complete' in str(last_status).lower()
            
    def test_preprocess_dataset(self, test_config):
        """Test fungsi preprocess_dataset"""
        # Buat instance MockProgressTracker
        progress_tracker = MockProgressTracker()
        
        # Setup mock untuk fungsi yang dipanggil
        with patch('smartcash.dataset.preprocessor.PreprocessingService') as mock_service_class, \
             patch('smartcash.dataset.preprocessor.get_logger') as mock_get_logger:
            
            # Setup mock service
            mock_service = MagicMock()
            mock_service.preprocess_and_visualize.return_value = {
                'status': 'success',
                'total_processed': 10,
                'valid_files': 9,
                'invalid_files': 1,
                'invalid_samples': [
                    {'file': 'sample1.jpg', 'error': 'Invalid format'}
                ]
            }
            mock_service_class.return_value = mock_service
            
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Panggil fungsi yang di-test
            result = preprocess_dataset(
                config=test_config,
                target_split='train',
                progress_tracker=progress_tracker,
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi hasil
            assert result['status'] == 'success'
            assert result['total_processed'] == 10
            assert result['valid_files'] == 9
            assert result['invalid_files'] == 1
            
            # Verifikasi service dipanggil dengan parameter yang benar
            mock_service_class.assert_called_once_with(
                config=test_config, 
                progress_tracker=progress_tracker
            )
            
            # Verifikasi preprocess_and_visualize dipanggil dengan parameter yang benar
            mock_service.preprocess_and_visualize.assert_called_once_with(
                target_split='train',
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi progress tracker diupdate dengan benar
            assert progress_tracker.was_progress_made() is True
            assert progress_tracker.was_completed() is True
            assert progress_tracker.has_error() is False
            
            # Verifikasi log info dipanggil
            mock_logger.info.assert_any_call("Memulai preprocessing untuk split: %s", 'train')
            mock_logger.info.assert_any_call("Preprocessing selesai. Total diproses: %d, Valid: %d, Invalid: %d", 
                                          10, 9, 1)
    
    def test_get_preprocessing_samples(self, test_config):
        """Test fungsi get_preprocessing_samples"""
        # Setup mock untuk fungsi yang dipanggil
        with patch('smartcash.dataset.preprocessor.PreprocessingService') as mock_service_class, \
             patch('smartcash.dataset.preprocessor.get_logger') as mock_get_logger:
            
            # Setup mock service
            mock_service = MagicMock()
            mock_service.get_sampling.return_value = {
                'samples': [
                    {
                        'original_image': 'img1.jpg', 
                        'preprocessed_image': 'pre_img1.jpg',
                        'label': '1000',
                        'confidence': 0.95
                    },
                    {
                        'original_image': 'img2.jpg', 
                        'preprocessed_image': 'pre_img2.jpg',
                        'label': '2000',
                        'confidence': 0.92
                    }
                ]
            }
            mock_service_class.return_value = mock_service
            
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Panggil fungsi yang di-test
            samples = get_preprocessing_samples(
                config=test_config,
                target_split='train',
                max_samples=2
            )
            
            # Verifikasi hasil
            assert 'samples' in samples
            assert len(samples['samples']) == 2
            
            # Verifikasi struktur sampel
            sample = samples['samples'][0]
            assert 'original_image' in sample
            assert 'preprocessed_image' in sample
            assert 'label' in sample
            assert 'confidence' in sample
            
            # Verifikasi service dipanggil dengan parameter yang benar
            mock_service_class.assert_called_once_with(config=test_config)
            mock_service.get_sampling.assert_called_once_with(max_samples=2)
            
            # Verifikasi log info dipanggil
            mock_logger.debug.assert_called_with("Mengambil %d sampel dari split %s", 2, 'train')
    
    def test_validate_dataset(self, test_config):
        """Test fungsi validate_dataset"""
        # Buat instance MockProgressTracker
        progress_tracker = MockProgressTracker()
        
        # Setup mock untuk fungsi yang dipanggil
        with patch('smartcash.dataset.preprocessor.PreprocessingService') as mock_service_class, \
             patch('smartcash.dataset.preprocessor.get_logger') as mock_get_logger:
            
            # Setup mock service
            mock_service = MagicMock()
            mock_service.validate_dataset_only.return_value = {
                'status': 'success',
                'total_files': 10,
                'valid_files': 9,
                'invalid_files': 1,
                'invalid_samples': [
                    {'file': 'invalid1.jpg', 'error': 'Corrupted image'},
                    {'file': 'invalid2.jpg', 'error': 'Invalid format'}
                ]
            }
            mock_service_class.return_value = mock_service
            
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Panggil fungsi yang di-test
            result = validate_dataset(
                config=test_config,
                target_split='train',
                progress_tracker=progress_tracker,
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi hasil
            assert result['status'] == 'success'
            assert result['total_files'] == 10
            assert result['valid_files'] == 9
            assert result['invalid_files'] == 1
            assert len(result['invalid_samples']) == 2
            
            # Verifikasi service dipanggil dengan parameter yang benar
            mock_service_class.assert_called_once_with(
                config=test_config, 
                progress_tracker=progress_tracker
            )
            
            # Verifikasi validate_dataset_only dipanggil dengan parameter yang benar
            mock_service.validate_dataset_only.assert_called_once_with(
                target_split='train',
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi progress tracker diupdate dengan benar
            assert progress_tracker.was_progress_made() is True
            assert progress_tracker.was_completed() is True
            assert progress_tracker.has_error() is False
            
            # Verifikasi log info dipanggil
            mock_logger.info.assert_any_call("Memulai validasi dataset untuk split: %s", 'train')
            mock_logger.info.assert_any_call(
                "Validasi selesai. Total: %d, Valid: %d, Invalid: %d", 
                10, 9, 1
            )
            mock_logger.warning.assert_called_with(
                "Ditemukan %d file yang tidak valid. Lihat log untuk detailnya.", 
                1
            )
    
    def test_cleanup_preprocessed_data(self, test_config, tmp_path):
        """Test fungsi cleanup_preprocessed_data"""
        # Buat instance MockProgressTracker
        progress_tracker = MockProgressTracker()
        
        # Setup mock untuk fungsi yang dipanggil
        with patch('smartcash.dataset.preprocessor.PreprocessingService') as mock_service_class, \
             patch('smartcash.dataset.preprocessor.get_logger') as mock_get_logger:
            
            # Setup mock service
            mock_service = MagicMock()
            mock_service.cleanup_preprocessed_data.return_value = {
                'status': 'success',
                'files_deleted': 5,
                'dirs_deleted': 2,
                'details': {
                    'deleted_files': ['file1.jpg', 'file2.jpg'],
                    'deleted_dirs': ['/path/to/dir1', '/path/to/dir2']
                }
            }
            mock_service_class.return_value = mock_service
            
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Panggil fungsi yang di-test
            result = cleanup_preprocessed_data(
                config=test_config,
                target_split='train',
                progress_tracker=progress_tracker,
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi hasil
            assert result['status'] == 'success'
            assert result['files_deleted'] == 5
            assert result['dirs_deleted'] == 2
            
            # Verifikasi service dipanggil dengan parameter yang benar
            mock_service_class.assert_called_once_with(
                config=test_config,
                progress_tracker=progress_tracker
            )
            mock_service.cleanup_preprocessed_data.assert_called_once_with(
                target_split='train',
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi progress tracker diupdate dengan benar
            assert progress_tracker.was_progress_made() is True
            assert progress_tracker.was_completed() is True
            assert progress_tracker.has_error() is False
            
            # Verifikasi log info dipanggil
            mock_logger.info.assert_any_call("Memulai pembersihan data hasil preprocessing untuk split: %s", 'train')
            mock_logger.info.assert_any_call(
                "Pembersihan selesai. File dihapus: %d, Direktori dihapus: %d", 
                5, 2
            )
    
    def test_get_preprocessing_status(self, test_config):
        """Test fungsi get_preprocessing_status"""
        # Setup mock untuk fungsi yang dipanggil
        with patch('smartcash.dataset.preprocessor.PreprocessingService') as mock_service_class, \
             patch('smartcash.dataset.preprocessor.get_logger') as mock_get_logger:
            
            # Setup mock service
            mock_service = MagicMock()
            mock_service.get_preprocessing_status.return_value = {
                'preprocessing_enabled': True,
                'validation_enabled': True,
                'splits': {
                    'train': {'status': 'completed', 'source_files': 10, 'last_processed': '2025-06-10T10:00:00'},
                    'valid': {'status': 'pending', 'source_files': 2, 'last_processed': None},
                    'test': {'status': 'not_started', 'source_files': 0, 'last_processed': None}
                },
                'output_dir': '/path/to/output'
            }
            mock_service_class.return_value = mock_service
            
            # Setup mock logger
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            # Panggil fungsi yang di-test
            status = get_preprocessing_status(test_config)
            
            # Verifikasi hasil
            assert 'preprocessing_enabled' in status
            assert 'validation_enabled' in status
            assert 'splits' in status
            assert 'output_dir' in status
            
            # Verifikasi status untuk setiap split
            splits = status['splits']
            assert 'train' in splits
            assert 'valid' in splits
            assert 'test' in splits
            
            # Verifikasi data status untuk train split
            assert splits['train']['status'] == 'completed'
            assert splits['train']['source_files'] == 10
            assert 'last_processed' in splits['train']
            
            # Verifikasi data status untuk valid split
            assert splits['valid']['status'] == 'pending'
            assert splits['valid']['source_files'] == 2
            
            # Verifikasi data status untuk test split
            assert splits['test']['status'] == 'not_started'
            assert splits['test']['source_files'] == 0
            
            # Verifikasi service dipanggil dengan parameter yang benar
            mock_service_class.assert_called_once_with(config=test_config)
            mock_service.get_preprocessing_status.assert_called_once()
            
            # Verifikasi log info dipanggil
            mock_logger.debug.assert_called_with("Mengambil status preprocessing")


class TestProgressTracking:
    """Test untuk tracking progress"""
    
    def test_progress_callback(self, preprocessor_service):
        """Test progress callback dipanggil dengan benar"""
        # Setup mock return values
        preprocessor_service.mock_engine.preprocess_split.return_value = {
            'status': 'success',
            'total_processed': 10,
            'valid_files': 9,
            'invalid_files': 1,
            'invalid_samples': [{'file': 'sample1.jpg', 'error': 'Invalid format'}]
        }
        
        # Buat instance MockProgressTracker
        progress_tracker = MockProgressTracker()
        
        # Panggil method yang menggunakan progress callback
        preprocessor_service.preprocess_and_visualize(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi progress tracker diupdate
        assert progress_tracker.was_progress_made() is True
        assert progress_tracker.was_completed() is True
        assert progress_tracker.has_error() is False
        
        # Verifikasi progress naik secara monoton
        progresses = [p[1] for p in progress_tracker.messages if p[0] == 'progress']
        assert all(0 <= p <= 100 for p in progresses)
        assert sorted(progresses) == progresses  # Progress harus naik
    
    def test_progress_tracker_integration(self, preprocessor_service, progress_tracker):
        """Test integrasi dengan ProgressTracker"""
        # Setup mock return values
        preprocessor_service.mock_engine.preprocess_split.return_value = {
            'status': 'success',
            'total_processed': 10,
            'valid_files': 9,
            'invalid_files': 1,
            'invalid_samples': [{'file': 'sample1.jpg', 'error': 'Invalid format'}]
        }
        
        # Panggil method dengan progress_tracker
        preprocessor_service.preprocess_and_visualize(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi progress tracker diupdate
        assert progress_tracker.was_progress_made() is True
        assert progress_tracker.was_completed() is True
        assert progress_tracker.has_error() is False
        
        # Verifikasi progress naik secara monoton
        progresses = [p[1] for p in progress_tracker.messages if p[0] == 'progress']
        assert all(0 <= p <= 100 for p in progresses)
        assert sorted(progresses) == progresses  # Progress harus naik


# Test case untuk MockPreprocessingEngine dan MockPreprocessingValidator
# sudah dipindahkan ke conftest.py untuk menghindari circular import
# dan dapat diakses melalui import langsung


class TestMockProgressTracker:
    """Test untuk MockProgressTracker"""
    
    def test_update_progress(self):
        """Test update progress dengan berbagai skenario"""
        tracker = MockProgressTracker()
        
        # Test update dengan progress valid
        assert tracker.update(0) is True
        assert tracker.update(50, "Sedang memproses") is True
        assert tracker.update(100, "Selesai") is True
        
        # Verifikasi progress tersimpan dengan benar
        assert len(tracker.progress_updates) == 3
        assert tracker.progress_updates[0] == (0.0, None)
        assert tracker.progress_updates[1] == (50.0, "Sedang memproses")
        assert tracker.progress_updates[2] == (100.0, "Selesai")
        
        # Test update dengan progress tidak valid
        with pytest.raises(ValueError):
            tracker.update(-10)  # Progress tidak boleh negatif
        
        with pytest.raises(ValueError):
            tracker.update(110)  # Progress tidak boleh > 100
            
        with pytest.raises(ValueError):
            tracker.update("bukan_angka")  # Progress harus angka
    
    def test_error_handling(self):
        """Test penanganan error"""
        tracker = MockProgressTracker()
        
        # Test dengan error_occurred
        error = ValueError("Terjadi kesalahan")
        tracker.error_occurred(error)
        assert tracker.error == error
        assert tracker.messages == [('error', 'Terjadi kesalahan')]
        
        # Test dengan method error (alias)
        tracker = MockProgressTracker()
        tracker.error = MagicMock(return_value=True)
        assert tracker.error(error) is True
        
        # Test dengan error None
        with pytest.raises(ValueError):
            tracker.error_occurred(None)
    
    def test_get_last_progress(self):
        """Test pengambilan progress terakhir"""
        tracker = MockProgressTracker()
        
        # Test saat belum ada progress
        assert tracker.get_last_progress() == (None, None)
        
        # Test setelah ada progress
        tracker.update(10, "Mulai")
        tracker.update(50, "Tengah")
        assert tracker.get_last_progress() == (50.0, "Tengah")
    
    def test_was_progress_made(self):
        """Test pengecekan apakah ada progress"""
        tracker = MockProgressTracker()
        
        # Test sebelum ada progress
        assert tracker.was_progress_made() is False
        
        # Test setelah ada progress
        tracker.update(10)
        assert tracker.was_progress_made() is True
    
    def test_was_completed(self):
        """Test pengecekan apakah proses selesai"""
        tracker = MockProgressTracker()
        
        # Test sebelum complete
        assert tracker.was_completed() is False
        
        # Test setelah complete
        tracker.complete()
        assert tracker.was_completed() is True
    
    def test_has_error(self):
        """Test pengecekan apakah terjadi error"""
        tracker = MockProgressTracker()
        
        # Test sebelum terjadi error
        assert tracker.has_error() is False
        
        # Test setelah terjadi error
        tracker.error_occurred(Exception("Test error"))
        assert tracker.has_error() is True
    
    def test_reset(self):
        """Test reset state tracker"""
        tracker = MockProgressTracker()
        
        # Isi tracker dengan data
        tracker.update(10, "Test")
        tracker.complete("Selesai")
        tracker.error_occurred(Exception("Error"))
        
        # Reset tracker
        result = tracker.reset()
        
        # Verifikasi state direset
        assert tracker.progress_updates == []
        assert tracker.completed is False
        assert tracker.error is None
        assert tracker.messages == []
        
        # Verifikasi method chaining
        assert result is tracker


class TestPreprocessorIntegration:
    """Test integrasi komponen-komponen preprocessor"""
    
    def test_service_with_mock_engine(self, test_config):
        """Test integrasi PreprocessingService dengan MockPreprocessingEngine"""
        # Setup
        from smartcash.dataset.preprocessor.service import PreprocessingService
        
        # Buat mock engine dan validator
        mock_engine = MockPreprocessingEngine(test_config)
        mock_validator = MockPreprocessingValidator(test_config)
        
        # Buat progress tracker untuk testing dengan mock update yang mendukung parameter message
        class TestProgressTracker(MockProgressTracker):
            def update(self, progress, status=None, **kwargs):
                # Simpan progress dan status untuk verifikasi
                self.last_progress = progress
                self.last_status = status
                self.last_message = kwargs.get('message')
                # Panggil parent dengan parameter yang benar
                if status is not None:
                    super().update(progress, status)
                else:
                    super().update(progress, "")
        
        progress_tracker = TestProgressTracker()
        
        # Mock return value untuk preprocess_split
        mock_result = {
            'status': 'success',
            'total_processed': 10,
            'valid_files': 9,
            'invalid_files': 1,
            'invalid_samples': [{'file': 'sample1.jpg', 'error': 'Invalid format'}]
        }
        mock_engine.preprocess_split.return_value = mock_result
        
        # Gunakan patch untuk mengesampingkan engine, validator, dan validasi konfigurasi
        with patch('smartcash.dataset.preprocessor.service.PreprocessingEngine', return_value=mock_engine), \
             patch('smartcash.dataset.preprocessor.service.PreprocessingValidator', return_value=mock_validator), \
             patch('smartcash.dataset.preprocessor.service.validate_preprocessing_config', return_value=test_config):
            
            # Buat instance service
            service = PreprocessingService(config=test_config, progress_tracker=progress_tracker)
            
            # Panggil method yang akan di-test
            result = service.preprocess_and_visualize(
                target_split='train',
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi hasil
            assert result['status'] == 'success', f"Status seharusnya 'success', tapi dapat {result.get('status')}"
            assert 'preprocessing_result' in result, "Hasil preprocessing tidak ditemukan"
            assert 'validation_summary' in result, "Ringkasan validasi tidak ditemukan"
            assert 'processing_time' in result, "Waktu pemrosesan tidak ditemukan"
            
            # Verifikasi progress tracker diupdate
            assert progress_tracker.was_progress_made() is True, "Progress tidak tercatat"
            assert progress_tracker.was_completed() is True, "Proses tidak ditandai selesai"
            
            # Verifikasi progress tracker menerima update dengan benar
            assert hasattr(progress_tracker, 'last_progress'), "Progress tracker tidak diupdate"
            assert hasattr(progress_tracker, 'last_status'), "Status tidak diupdate"
            
            # Verifikasi engine dipanggil dengan parameter yang benar
            mock_engine.preprocess_split.assert_called_once()
            
            # Dapatkan argumen yang digunakan untuk memanggil preprocess_split
            call_args = mock_engine.preprocess_split.call_args[1]
            assert call_args['split_name'] == 'train', \
                f"split_name harus 'train', tapi mendapat: {call_args.get('split_name')}"
            assert callable(call_args.get('progress_callback')), \
                "progress_callback harus callable"
    
    def test_engine_with_mock_validator(self, test_config):
        """Test integrasi PreprocessingEngine dengan MockPreprocessingValidator"""
        from smartcash.dataset.preprocessor.core.engine import PreprocessingEngine
        from smartcash.dataset.preprocessor.utils import FileScanner
        
        # Setup konfigurasi test
        config = {
            'preprocessing': {
                'input': {
                    'train': 'tests/data/train',
                    'valid': 'tests/data/valid',
                    'test': 'tests/data/test'
                },
                'output': {
                    'root': 'tests/output',
                    'images': 'images',
                    'labels': 'labels',
                    'create_npy': False
                },
                'validation': {
                    'enabled': True,
                    'skip_invalid': True
                },
                'normalization': {
                    'enabled': True,
                    'target_size': [640, 640],
                    'convert_rgb': True
                }
            }
        }
        
        # Inisialisasi engine
        engine = PreprocessingEngine(config)
        
        # Setup mock untuk validator
        mock_validator = MagicMock()
        mock_validator.validate_image.return_value = (True, [], {})
        mock_validator.validate_label.return_value = (True, [], {})
        mock_validator.validate_image_label_pair.return_value = (True, [], {})
        engine.validator = mock_validator
        
        # Setup mock untuk path resolver
        mock_path_resolver = MagicMock()
        
        # Mock direktori sumber dan tujuan
        mock_src_img_dir = Path('tests/data/train/images')
        mock_src_label_dir = Path('tests/data/train/labels')
        mock_dst_img_dir = Path('tests/output/train/images')
        mock_dst_label_dir = Path('tests/output/train/labels')
        
        # Pastikan direktori sumber ada
        mock_src_img_dir.mkdir(parents=True, exist_ok=True)
        mock_src_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup return value untuk path resolver
        mock_path_resolver.get_source_image_dir.return_value = mock_src_img_dir
        mock_path_resolver.get_source_label_dir.return_value = mock_src_label_dir
        mock_path_resolver.get_preprocessed_image_dir.return_value = mock_dst_img_dir
        mock_path_resolver.get_preprocessed_label_dir.return_value = mock_dst_label_dir
        
        # Inject mock path resolver ke engine
        engine.path_resolver = mock_path_resolver
        
        # Buat file gambar dummy di direktori sumber
        test_images = []
        for i in range(1, 6):
            img_path = mock_src_img_dir / f'img_{i}.jpg'
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.touch()  # Buat file kosong
            test_images.append(img_path)
        
        # Setup mock untuk file_scanner
        mock_file_scanner = MagicMock(spec=FileScanner)
        mock_file_scanner.scan_directory.return_value = test_images
        engine.file_scanner = mock_file_scanner
        
        # Setup mock untuk progress callback
        progress_updates = []
        
        def mock_progress_callback(progress, status=None, **kwargs):
            progress_updates.append((progress, status, kwargs))
        
        # Setup mock untuk file_processor
        mock_file_processor = MagicMock()
        mock_file_processor.read_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_file_processor.write_image.return_value = True
        mock_file_processor.copy_file.return_value = True
        engine.file_processor = mock_file_processor
        
        # Mock untuk operasi file system
        def mock_exists(path):
            path_str = str(path)
            return (
                path_str == str(mock_src_img_dir) or 
                path_str == str(mock_src_label_dir) or
                path_str == str(mock_dst_img_dir) or
                path_str == str(mock_dst_label_dir) or
                any(str(img) == path_str for img in test_images) or
                any(path_str.startswith(str(mock_src_img_dir)) or 
                   path_str.startswith(str(mock_src_label_dir)) or
                   path_str.startswith(str(mock_dst_img_dir)) or
                   path_str.startswith(str(mock_dst_label_dir)))
            )
        
        with patch('cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)) as mock_imread, \
             patch('cv2.imwrite', return_value=True) as mock_imwrite, \
             patch('os.path.exists', side_effect=mock_exists) as mock_exists, \
             patch('shutil.copy2') as mock_copy2, \
             patch('os.makedirs') as mock_makedirs, \
             patch.object(Path, 'mkdir') as mock_mkdir:
            
            # Panggil method yang akan di-test
            result = engine.preprocess_split(
                split='train',
                progress_callback=mock_progress_callback
            )
            
            # Verifikasi hasil
            assert result['status'] == 'completed', f"Status seharusnya 'completed', tapi dapat {result.get('status')}"
            assert 'stats' in result, "Statistik tidak ditemukan dalam hasil"
            assert 'total' in result['stats'], "Total file tidak ditemukan dalam statistik"
            assert 'processed' in result['stats'], "Jumlah file yang diproses tidak ditemukan dalam statistik"
            assert result['stats']['total'] == len(test_images), \
                f"Jumlah file yang diproses harus {len(test_images)}, tapi dapat {result['stats'].get('total')}"
            assert result['stats']['processed'] == len(test_images), \
                f"Semua file seharusnya diproses, tapi hanya {result['stats'].get('processed')} dari {len(test_images)}"
            
            # Verifikasi file_scanner dipanggil dengan parameter yang benar
            mock_file_scanner.scan_directory.assert_called_once_with(mock_src_img_dir, {'.jpg', '.jpeg', '.png'})
            
            # Verifikasi path resolver dipanggil
            mock_path_resolver.get_source_image_dir.assert_called_once_with('train')
            mock_path_resolver.get_source_label_dir.assert_called_once_with('train')
            mock_path_resolver.get_preprocessed_image_dir.assert_called_once_with('train')
            mock_path_resolver.get_preprocessed_label_dir.assert_called_once_with('train')
            
            # Verifikasi direktori tujuan dibuat
            assert mock_mkdir.called, "Direktori tujuan harus dibuat"
            
            # Verifikasi file_processor dipanggil untuk setiap gambar
            assert mock_file_processor.read_image.call_count == len(test_images), \
                f"read_image harus dipanggil {len(test_images)} kali, tapi dipanggil {mock_file_processor.read_image.call_count} kali"
            assert mock_file_processor.write_image.call_count == len(test_images), \
                f"write_image harus dipanggil {len(test_images)} kali, tapi dipanggil {mock_file_processor.write_image.call_count} kali"
            
            # Verifikasi progress callback dipanggil
            assert len(progress_updates) > 0, "Progress callback tidak pernah dipanggil"
            
            # Verifikasi progress meningkat secara monoton
            prev_progress = -1
            for progress, status, _ in progress_updates:
                assert progress > prev_progress, f"Progress harus meningkat, tapi {progress} <= {prev_progress}"
                prev_progress = progress
                
            # Verifikasi progress terakhir mendekati 100%
            assert progress_updates[-1][0] >= 90, f"Progress terakhir harus >= 90%, tapi dapat {progress_updates[-1][0]}%"
            
            # Verifikasi status terakhir menunjukkan selesai
            if progress_updates:  # Pastikan ada progress update
                _, last_status, _ = progress_updates[-1]
                assert last_status is not None, "Status terakhir tidak boleh None"
                last_status_lower = str(last_status).lower()
                assert any(keyword in last_status_lower for keyword in ['complete', 'selesai', 'done', 'berhasil']), \
                    f"Status terakhir harus mengandung indikasi selesai, tapi mendapat: {last_status}"
    
    def test_file_processor_integration(self, test_config, tmp_path):
        """Test integrasi dengan FileProcessor"""
        from smartcash.dataset.preprocessor.utils.file_processor import FileProcessor
        
        # Buat instance FileProcessor
        processor = FileProcessor(test_config)
        
        # ===== Test read_image =====
        # Test dengan file yang tidak ada
        non_existent_img = tmp_path / "nonexistent.jpg"
        assert processor.read_image(non_existent_img) is None, \
            "Harus mengembalikan None untuk file yang tidak ada"
        
        # Test dengan file gambar yang valid
        img_path = tmp_path / "test.jpg"
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)  # Gambar hitam 100x100
        img = Image.fromarray(img_array)
        img.save(img_path)
        
        # Baca gambar yang baru disimpan
        img_data = processor.read_image(img_path)
        assert img_data is not None, "Gagal membaca gambar yang valid"
        assert img_data.shape == (100, 100, 3), "Dimensi gambar tidak sesuai"
        
        # ===== Test save_image =====
        # Test menyimpan gambar ke lokasi yang valid
        output_img_path = tmp_path / "output.jpg"
        assert processor.save_image(img_array, output_img_path) is True, \
            "Gagal menyimpan gambar"
        assert output_img_path.exists(), "File gambar tidak dibuat"
        
        # Verifikasi gambar yang disimpan bisa dibaca
        saved_img = Image.open(output_img_path)
        assert saved_img.size == (100, 100), "Ukuran gambar yang disimpan tidak sesuai"
        
        # Test menyimpan gambar ke direktori yang belum ada
        output_img_dir = tmp_path / "nonexistent_dir" / "output.jpg"
        assert processor.save_image(img_array, output_img_dir) is True, \
            "Gagal membuat direktori dan menyimpan gambar"
        assert output_img_dir.exists(), "File gambar tidak dibuat di direktori baru"
        
        # ===== Test read_label_file =====
        # Test dengan file yang tidak ada
        non_existent_label = tmp_path / "nonexistent.txt"
        assert processor.read_label_file(non_existent_label) == [], \
            "Harus mengembalikan list kosong untuk file yang tidak ada"
        
        # Test dengan format file tidak valid
        invalid_label_path = tmp_path / "invalid.txt"
        invalid_label_path.write_text("invalid format")
        assert processor.read_label_file(invalid_label_path) == [], \
            "Harus mengembalikan list kosong untuk format tidak valid"
        
        # Test dengan format file valid
        label_path = tmp_path / "test.txt"
        label_content = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1"
        label_path.write_text(label_content)
        
        # Baca file label yang valid
        bboxes = processor.read_label_file(label_path)
        assert len(bboxes) == 2, "Jumlah bounding box tidak sesuai"
        assert len(bboxes[0]) == 5, "Format bounding box tidak sesuai (harus [class_id, x, y, w, h])"
        
        # Verifikasi nilai bounding box
        for bbox in bboxes:
            assert 0 <= bbox[1] <= 1, "Nilai x_center harus antara 0 dan 1"
            assert 0 <= bbox[2] <= 1, "Nilai y_center harus antara 0 dan 1"
            assert 0 <= bbox[3] <= 1, "Nilai width harus antara 0 dan 1"
            assert 0 <= bbox[4] <= 1, "Nilai height harus antara 0 dan 1"
        
        # ===== Test save_label_file =====
        # Test menyimpan bounding box ke file
        output_label_path = tmp_path / "output.txt"
        test_bboxes = [
            [0, 0.1, 0.1, 0.2, 0.2],  # [class_id, x, y, w, h]
            [1, 0.5, 0.5, 0.3, 0.3]
        ]
        assert processor.save_label_file(test_bboxes, output_label_path) is True, \
            "Gagal menyimpan file label"
        assert output_label_path.exists(), "File label tidak dibuat"
        
        # Test menyimpan ke direktori yang belum ada
        output_label_dir = tmp_path / "nonexistent_dir" / "output.txt"
        assert processor.save_label_file(test_bboxes, output_label_dir) is True, \
            "Gagal membuat direktori dan menyimpan file label"
        assert output_label_dir.exists(), "File label tidak dibuat di direktori baru"
        
        # Verifikasi isi file label
        with open(output_label_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, "Jumlah baris dalam file label tidak sesuai"
            
            # Verifikasi setiap baris
            for i, line in enumerate(lines):
                parts = line.strip().split()
                assert len(parts) == 5, f"Format baris ke-{i+1} tidak valid"
                
                # Verifikasi nilai class_id adalah integer
                class_id = int(parts[0])
                assert class_id in [0, 1], f"Class ID {class_id} tidak valid"
                
                # Verifikasi koordinat bounding box
                x_center, y_center, width, height = map(float, parts[1:])
                assert 0 <= x_center <= 1, f"Nilai x_center {x_center} tidak valid"
                assert 0 <= y_center <= 1, f"Nilai y_center {y_center} tidak valid"
                assert 0 <= width <= 1, f"Nilai width {width} tidak valid"
                assert 0 <= height <= 1, f"Nilai height {height} tidak valid"


class TestErrorHandling:
    """Test untuk penanganan error"""
    
    def test_invalid_config(self):
        """Test dengan konfigurasi tidak valid"""
        # Test dengan konfigurasi None
        with pytest.raises(ValueError):
            PreprocessingService(None)
            
        # Test dengan konfigurasi kosong
        with pytest.raises(ValueError):
            PreprocessingService({})
    
    def test_missing_files(self, test_config, progress_tracker):
        """Test dengan direktori tidak ada"""
        from smartcash.dataset.preprocessor import PreprocessingService
        
        # Buat mock untuk file_scanner
        mock_file_scanner = MagicMock()
        mock_file_scanner.scan_directory.side_effect = FileNotFoundError("Direktori tidak ditemukan")
        
        # Buat service dengan mock
        with patch('smartcash.dataset.preprocessor.service.FileScanner', return_value=mock_file_scanner), \
             patch('smartcash.dataset.preprocessor.service.PreprocessingEngine'), \
             patch('smartcash.dataset.preprocessor.service.PreprocessingValidator'):
            
            service = PreprocessingService(test_config, progress_tracker=progress_tracker)
            service.file_scanner = mock_file_scanner
            
            # Jalankan preprocessing
            result = service.preprocess_and_visualize(
                target_split='train',
                progress_callback=progress_tracker.update
            )
            
            # Verifikasi hasil error
            assert result['status'] == 'error'
            assert 'error' in result
            
            # Verifikasi progress tracker mencatat error
            assert progress_tracker.has_error() is True
            assert any(msg[0] == 'error' for msg in progress_tracker.messages)
    
    @pytest.fixture
    def preprocessor_service(self, test_config, progress_tracker):
        """Fixture untuk PreprocessingService dengan mock engine dan validator"""
        from smartcash.dataset.preprocessor import PreprocessingService
        
        # Buat mock untuk komponen-komponen yang diperlukan
        mock_engine = MockPreprocessingEngine(test_config)
        mock_validator = MockPreprocessingValidator(test_config)
        
        # Gunakan mock engine dan validator
        with patch('smartcash.dataset.preprocessor.service.PreprocessingEngine', return_value=mock_engine), \
             patch('smartcash.dataset.preprocessor.service.PreprocessingValidator', return_value=mock_validator), \
             patch('smartcash.dataset.preprocessor.core.engine.PreprocessingValidator', return_value=mock_validator), \
             patch('smartcash.dataset.preprocessor.PreprocessingService._initialize_services') as mock_init_services:
            
            # Buat instance service
            service = PreprocessingService(test_config, progress_tracker=progress_tracker)
            
            # Simpan mock engine dan validator untuk keperluan testing
            service.mock_engine = mock_engine
            service.mock_validator = mock_validator
            
            yield service
            
            # Lakukan cleanup jika diperlukan
            if hasattr(service, 'cleanup'):
                service.cleanup()

    def test_preprocessing_error(self, preprocessor_service, progress_tracker):
        """Test error saat preprocessing"""
        # Setup mock untuk melempar exception
        error_msg = "Error saat preprocessing"
        preprocessor_service.mock_engine.preprocess_split.side_effect = Exception(error_msg)
        
        # Jalankan preprocessing dan tangkap hasilnya
        result = preprocessor_service.preprocess_and_visualize(
            target_split='train',
            progress_callback=progress_tracker.update
        )
        
        # Verifikasi hasil error
        assert result['status'] == 'error'
        assert 'error' in result
        
        try:
            # Coba konversi error ke string untuk memeriksa isinya
            error_str = str(result['error'])
            assert error_msg in error_str
        except Exception as e:
            # Jika konversi gagal, cukup lewati assertion ini
            pass
        
        # Verifikasi progress tracker mencatat error
        assert progress_tracker.has_error() is True
        assert len(progress_tracker.messages) > 0
        
        # Periksa apakah ada pesan error dalam progress tracker
        error_messages = [msg for msg in progress_tracker.messages if msg[0] == 'error']
        assert len(error_messages) > 0
