# File: tests/test_data_handlers.py
# Author: Alfrida Sabar
# Deskripsi: Pengujian unit untuk data handlers dalam proyek SmartCash

import pytest
import os
from pathlib import Path

from handlers.data_handler import DataHandler
from handlers.roboflow_handler import RoboflowHandler
from utils.logger import SmartCashLogger

class TestDataHandlers:
    @pytest.fixture
    def config_path(self):
        """Fixture untuk path konfigurasi"""
        return Path(__file__).parent.parent / 'configs' / 'base_config.yaml'
    
    @pytest.fixture
    def data_dir(self, tmp_path):
        """Fixture direktori data sementara"""
        return tmp_path / 'data'
    
    def test_data_handler_init(self, config_path, data_dir):
        """Pengujian inisialisasi DataHandler"""
        logger = SmartCashLogger(__name__)
        data_handler = DataHandler(
            config_path=str(config_path),
            data_dir=str(data_dir),
            logger=logger
        )
        
        assert data_handler is not None, "DataHandler tidak dapat diinisialisasi"
        assert data_handler.config is not None, "Konfigurasi tidak dapat dimuat"
    
    def test_setup_dataset_structure(self, config_path, data_dir):
        """Pengujian setup struktur dataset"""
        data_handler = DataHandler(
            config_path=str(config_path),
            data_dir=str(data_dir)
        )
        
        data_handler.setup_dataset_structure()
        
        # Periksa struktur direktori
        assert (data_dir / 'train' / 'images').exists(), "Direktori train/images tidak dibuat"
        assert (data_dir / 'train' / 'labels').exists(), "Direktori train/labels tidak dibuat"
        assert (data_dir / 'valid' / 'images').exists(), "Direktori valid/images tidak dibuat"
        assert (data_dir / 'valid' / 'labels').exists(), "Direktori valid/labels tidak dibuat"
        assert (data_dir / 'test' / 'images').exists(), "Direktori test/images tidak dibuat"
        assert (data_dir / 'test' / 'labels').exists(), "Direktori test/labels tidak dibuat"
    
    def test_get_class_names(self, config_path):
        """Pengujian pengambilan nama kelas"""
        data_handler = DataHandler(config_path=str(config_path))
        
        class_names = data_handler.get_class_names()
        
        assert isinstance(class_names, list), "Nama kelas harus berupa list"
        assert len(class_names) > 0, "Nama kelas tidak boleh kosong"
        
        # Periksa nama kelas sesuai konfigurasi
        expected_classes = ['100k', '10k', '1k', '20k', '2k', '50k', '5k']
        assert set(class_names) == set(expected_classes), "Nama kelas tidak sesuai"
    
    @pytest.mark.skipif(
        not os.environ.get('ROBOFLOW_API_KEY'),
        reason="Roboflow API key tidak tersedia"
    )
    def test_roboflow_handler(self, config_path):
        """Pengujian RoboflowHandler"""
        roboflow_handler = RoboflowHandler(config_path=str(config_path))
        
        # Dapatkan informasi dataset
        dataset_info = roboflow_handler.get_dataset_info()
        
        assert 'name' in dataset_info, "Informasi nama dataset tidak ada"
        assert 'version' in dataset_info, "Informasi versi dataset tidak ada"
        assert 'classes' in dataset_info, "Informasi kelas dataset tidak ada"
        assert 'splits' in dataset_info, "Informasi split dataset tidak ada"
        
        # Periksa jumlah sampel di setiap split
        splits_info = dataset_info['splits']
        assert all(count > 0 for count in splits_info.values()), "Tidak ada sampel di salah satu split"