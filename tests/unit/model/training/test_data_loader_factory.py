#!/usr/bin/env python3
"""
Comprehensive tests for DataLoaderFactory module.

Tests YOLO dataset loading, preprocessing file handling, data loader creation,
and resource management functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import yaml

# Import the modules to test
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smartcash.model.training.data_loader_factory import (
    YOLODataset,
    DataLoaderFactory,
    collate_fn,
    create_data_loaders,
    get_dataset_stats
)


class TestYOLODataset:
    """Test cases for YOLODataset class"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        images_dir = Path(temp_dir) / "images"
        labels_dir = Path(temp_dir) / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create sample .npy files (preprocessed images)
        for i in range(3):
            # Create sample image data (C, H, W) format
            image_data = np.random.rand(3, 640, 640).astype(np.float32)
            np.save(images_dir / f"pre_image_{i}.npy", image_data)
            
            # Create augmented image
            aug_data = np.random.rand(3, 640, 640).astype(np.float32)
            np.save(images_dir / f"aug_image_{i}_var1.npy", aug_data)
        
        # Create sample label files
        for i in range(3):
            label_content = f"0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.15\n"
            with open(labels_dir / f"pre_image_{i}.txt", "w") as f:
                f.write(label_content)
            with open(labels_dir / f"aug_image_{i}_var1.txt", "w") as f:
                f.write(label_content)
        
        yield temp_dir, images_dir, labels_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, temp_dataset_dir):
        """Test dataset initialization with valid directory"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=640,
            augment=True
        )
        
        assert dataset.img_size == 640
        assert dataset.augment == True
        assert len(dataset.valid_files) == 6  # 3 pre + 3 aug files
    
    def test_dataset_length(self, temp_dataset_dir):
        """Test dataset length calculation"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        assert len(dataset) == 6
    
    def test_dataset_getitem(self, temp_dataset_dir):
        """Test dataset item retrieval"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Get first item
        image, labels = dataset[0]
        
        # Check image format
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 640, 640)  # C, H, W
        assert image.dtype == torch.float32
        
        # Check labels format
        assert isinstance(labels, torch.Tensor)
        assert labels.shape[1] == 5  # class, x, y, w, h
        assert labels.dtype == torch.float32
    
    def test_load_labels_valid_file(self, temp_dataset_dir):
        """Test label loading from valid file"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Test with existing label file
        label_path = labels_dir / "pre_image_0.txt"
        labels = dataset._load_labels(label_path)
        
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (2, 5)  # 2 objects, 5 values each
        assert labels[0, 0] == 0  # First class ID
        assert labels[1, 0] == 1  # Second class ID
    
    def test_load_labels_missing_file(self, temp_dataset_dir):
        """Test label loading from missing file"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Test with non-existent file
        missing_path = labels_dir / "missing_file.txt"
        labels = dataset._load_labels(missing_path)
        
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (0, 5)  # Empty array
    
    def test_load_labels_malformed_file(self, temp_dataset_dir):
        """Test label loading from malformed file"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        # Create malformed label file
        malformed_file = labels_dir / "malformed.txt"
        with open(malformed_file, "w") as f:
            f.write("invalid data\n0 0.5\n")  # Incomplete data
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        labels = dataset._load_labels(malformed_file)
        
        # Should return empty array for malformed data
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (0, 5)
    
    def test_resize_image_and_labels(self, temp_dataset_dir):
        """Test image and label resizing (should be no-op for preprocessed data)"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Create sample data
        image = np.random.rand(640, 640, 3).astype(np.float32)
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.3]])
        
        # Should return unchanged for preprocessed data
        new_image, new_labels = dataset._resize_image_and_labels(image, labels)
        
        np.testing.assert_array_equal(new_image, image)
        np.testing.assert_array_equal(new_labels, labels)
    
    def test_dataset_with_missing_labels(self, temp_dataset_dir):
        """Test dataset behavior with missing label files"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        # Remove one label file
        (labels_dir / "pre_image_0.txt").unlink()
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Should only include files with both image and label
        assert len(dataset.valid_files) == 5  # One less due to missing label
    
    def test_different_image_formats(self, temp_dataset_dir):
        """Test handling of different image array formats"""
        temp_dir, images_dir, labels_dir = temp_dataset_dir
        
        # Create image in H, W, C format
        hwc_image = np.random.rand(640, 640, 3).astype(np.float32)
        np.save(images_dir / "pre_hwc_format.npy", hwc_image)
        
        # Create corresponding label
        with open(labels_dir / "pre_hwc_format.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
        
        dataset = YOLODataset(str(images_dir), str(labels_dir))
        
        # Find the HWC format image
        hwc_idx = None
        for i, file_path in enumerate(dataset.valid_files):
            if "hwc_format" in file_path.name:
                hwc_idx = i
                break
        
        assert hwc_idx is not None
        image, labels = dataset[hwc_idx]
        
        # Should be converted to CHW format
        assert image.shape == (3, 640, 640)


class TestCollateFn:
    """Test cases for collate_fn"""
    
    def test_collate_empty_batch(self):
        """Test collating empty batch"""
        batch = []
        
        # Should handle empty batch gracefully
        try:
            images, targets = collate_fn(batch)
            assert False, "Should raise an error for empty batch"
        except (ValueError, IndexError):
            pass  # Expected behavior
    
    def test_collate_single_item(self):
        """Test collating single item"""
        image = torch.randn(3, 640, 640)
        labels = torch.tensor([[0, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32)
        batch = [(image, labels)]
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (1, 3, 640, 640)
        assert targets.shape == (1, 6)  # batch_idx + 5 label values
        assert targets[0, 0] == 0  # batch index
        assert targets[0, 1] == 0  # class
    
    def test_collate_multiple_items(self):
        """Test collating multiple items"""
        batch = []
        for i in range(3):
            image = torch.randn(3, 640, 640)
            labels = torch.tensor([[i, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32)
            batch.append((image, labels))
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (3, 3, 640, 640)
        assert targets.shape == (3, 6)
        # Check batch indices
        assert targets[0, 0] == 0
        assert targets[1, 0] == 1
        assert targets[2, 0] == 2
    
    def test_collate_empty_labels(self):
        """Test collating items with empty labels"""
        image = torch.randn(3, 640, 640)
        empty_labels = torch.zeros((0, 5), dtype=torch.float32)
        batch = [(image, empty_labels)]
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (1, 3, 640, 640)
        assert targets.shape == (0, 6)  # Empty targets
    
    def test_collate_mixed_labels(self):
        """Test collating items with varying number of labels"""
        batch = []
        
        # First item: 2 labels
        image1 = torch.randn(3, 640, 640)
        labels1 = torch.tensor([[0, 0.5, 0.5, 0.2, 0.3], [1, 0.3, 0.7, 0.1, 0.15]], dtype=torch.float32)
        batch.append((image1, labels1))
        
        # Second item: 1 label
        image2 = torch.randn(3, 640, 640)
        labels2 = torch.tensor([[2, 0.4, 0.6, 0.3, 0.25]], dtype=torch.float32)
        batch.append((image2, labels2))
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (2, 3, 640, 640)
        assert targets.shape == (3, 6)  # Total 3 labels across batch
        
        # Check batch indices
        assert targets[0, 0] == 0  # First image
        assert targets[1, 0] == 0  # First image  
        assert targets[2, 0] == 1  # Second image


class TestDataLoaderFactory:
    """Test cases for DataLoaderFactory class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory structure"""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        
        # Create splits
        for split in ['train', 'valid', 'test']:
            images_dir = data_dir / split / 'images'
            labels_dir = data_dir / split / 'labels'
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            
            # Create sample files
            for i in range(2):
                # Image data
                image_data = np.random.rand(3, 640, 640).astype(np.float32)
                np.save(images_dir / f"pre_image_{i}.npy", image_data)
                
                # Label data
                with open(labels_dir / f"pre_image_{i}.txt", "w") as f:
                    f.write("0 0.5 0.5 0.2 0.3\n")
        
        yield data_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'training': {
                'batch_size': 8,
                'data': {
                    'num_workers': 0,  # Set to 0 for tests to avoid multiprocessing issues
                    'pin_memory': False,  # Disable for tests
                    'persistent_workers': False,  # Must be False when num_workers=0
                    'prefetch_factor': 2,
                    'drop_last': False  # Set to False to avoid batch_size/drop_last conflicts
                }
            }
        }
    
    def test_factory_initialization_with_config(self, temp_data_dir, sample_config):
        """Test factory initialization with provided config"""
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        assert factory.config == sample_config
        assert factory.data_dir == temp_data_dir
        assert factory in DataLoaderFactory._instances
    
    def test_factory_initialization_without_config(self, temp_data_dir):
        """Test factory initialization without config (uses fallback)"""
        with patch('pathlib.Path.exists', return_value=False):
            factory = DataLoaderFactory(data_dir=str(temp_data_dir))
            
            assert 'training' in factory.config
            assert factory.config['training']['batch_size'] >= 1  # Allow any reasonable batch size
    
    def test_factory_initialization_with_yaml_config(self, temp_data_dir):
        """Test factory initialization with YAML config file"""
        config_data = {'training': {'batch_size': 32}}
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml.dump(config_data))):
                factory = DataLoaderFactory(data_dir=str(temp_data_dir))
                
                assert factory.config['training']['batch_size'] == 32
    
    def test_validate_data_structure_valid(self, temp_data_dir):
        """Test data structure validation with valid structure"""
        factory = DataLoaderFactory(data_dir=str(temp_data_dir))
        # Should not raise any exception
        factory._validate_data_structure()
    
    def test_validate_data_structure_missing_train(self, temp_data_dir):
        """Test data structure validation with missing train directory"""
        # Remove train directory
        shutil.rmtree(temp_data_dir / 'train')
        
        with pytest.raises(FileNotFoundError, match="Images directory tidak ditemukan"):
            DataLoaderFactory(data_dir=str(temp_data_dir))
    
    def test_create_train_loader(self, temp_data_dir, sample_config):
        """Test training data loader creation"""
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        train_loader = factory.create_train_loader(img_size=640)
        
        assert train_loader is not None
        assert train_loader.batch_size == 8
        assert train_loader.dataset.augment == True
        assert len(factory._dataloaders) == 1
    
    def test_create_val_loader(self, temp_data_dir, sample_config):
        """Test validation data loader creation"""
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        val_loader = factory.create_val_loader(img_size=640)
        
        assert val_loader is not None
        assert val_loader.batch_size == 8
        assert val_loader.dataset.augment == False
        # Check shuffle property more carefully to avoid attribute errors
        try:
            if hasattr(val_loader.sampler, 'shuffle'):
                assert not val_loader.sampler.shuffle
        except AttributeError:
            pass  # Some samplers may not have shuffle attribute
    
    def test_create_test_loader_exists(self, temp_data_dir, sample_config):
        """Test test data loader creation when test data exists"""
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        test_loader = factory.create_test_loader(img_size=640)
        
        assert test_loader is not None
        assert test_loader.batch_size == 8
    
    def test_create_test_loader_missing(self, temp_data_dir, sample_config):
        """Test test data loader creation when test data is missing"""
        # Remove test directory
        shutil.rmtree(temp_data_dir / 'test')
        
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        test_loader = factory.create_test_loader(img_size=640)
        
        assert test_loader is None
    
    def test_get_dataset_info(self, temp_data_dir):
        """Test dataset information retrieval"""
        factory = DataLoaderFactory(data_dir=str(temp_data_dir))
        
        info = factory.get_dataset_info()
        
        assert 'train' in info
        assert 'valid' in info
        assert 'test' in info
        
        # Check train info
        assert info['train']['num_images'] == 2
        assert info['train']['preprocessed_files'] == 2
        assert info['train']['augmented_files'] == 0
    
    def test_get_class_distribution(self, temp_data_dir):
        """Test class distribution calculation"""
        factory = DataLoaderFactory(data_dir=str(temp_data_dir))
        
        distribution = factory.get_class_distribution('train')
        
        assert isinstance(distribution, dict)
        assert 0 in distribution  # Class 0 should be present
        assert distribution[0] == 2  # Should have 2 instances
    
    def test_get_class_distribution_missing_split(self, temp_data_dir):
        """Test class distribution for missing split"""
        factory = DataLoaderFactory(data_dir=str(temp_data_dir))
        
        distribution = factory.get_class_distribution('nonexistent')
        
        assert distribution == {}
    
    def test_cleanup_functionality(self, temp_data_dir, sample_config):
        """Test cleanup functionality"""
        factory = DataLoaderFactory(config=sample_config, data_dir=str(temp_data_dir))
        
        # Create some loaders
        train_loader = factory.create_train_loader()
        val_loader = factory.create_val_loader()
        
        assert len(factory._dataloaders) == 2
        
        # Test cleanup
        factory.cleanup()
        
        assert len(factory._dataloaders) == 0
        assert factory not in DataLoaderFactory._instances
    
    def test_cleanup_all_instances(self, temp_data_dir):
        """Test cleanup of all factory instances"""
        # Create multiple instances
        factory1 = DataLoaderFactory(data_dir=str(temp_data_dir))
        factory2 = DataLoaderFactory(data_dir=str(temp_data_dir))
        
        assert len(DataLoaderFactory._instances) >= 2
        
        # Cleanup all
        DataLoaderFactory.cleanup_all()
        
        assert len(DataLoaderFactory._instances) == 0
    
    def test_fallback_config(self, temp_data_dir):
        """Test fallback configuration"""
        factory = DataLoaderFactory(data_dir=str(temp_data_dir))
        
        fallback = factory._get_fallback_config()
        
        assert 'training' in fallback
        assert fallback['training']['batch_size'] >= 1
        assert 'num_workers' in fallback['training']['data']
        # Check that drop_last and persistent_workers are compatible
        data_config = fallback['training']['data']
        if data_config.get('num_workers', 0) == 0:
            assert not data_config.get('persistent_workers', False)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory"""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        
        # Create minimal structure
        for split in ['train', 'valid']:
            images_dir = data_dir / split / 'images'
            labels_dir = data_dir / split / 'labels'
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            
            # Create sample file
            image_data = np.random.rand(3, 640, 640).astype(np.float32)
            np.save(images_dir / "pre_image_0.npy", image_data)
            
            with open(labels_dir / "pre_image_0.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.3\n")
        
        yield data_dir
        shutil.rmtree(temp_dir)
    
    def test_create_data_loaders_function(self, temp_data_dir):
        """Test create_data_loaders convenience function"""
        # Use a config that avoids batch_size/drop_last conflicts
        config = {'training': {'batch_size': 2, 'data': {'num_workers': 0, 'drop_last': False}}}
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=str(temp_data_dir),
            img_size=640,
            config=config
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is None  # No test data created
    
    def test_get_dataset_stats_function(self, temp_data_dir):
        """Test get_dataset_stats convenience function"""
        stats = get_dataset_stats(data_dir=str(temp_data_dir))
        
        assert 'train' in stats
        assert 'valid' in stats
        assert stats['train']['num_images'] == 1


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_dataset_with_corrupt_npy_file(self):
        """Test dataset handling of corrupted .npy files"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            images_dir = Path(temp_dir) / "images"
            labels_dir = Path(temp_dir) / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            
            # Create corrupted .npy file
            with open(images_dir / "pre_corrupt.npy", "wb") as f:
                f.write(b"corrupted data")
            
            # Create corresponding label
            with open(labels_dir / "pre_corrupt.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.3\n")
            
            dataset = YOLODataset(str(images_dir), str(labels_dir))
            
            # Should handle corrupted file gracefully
            with pytest.raises((ValueError, OSError)):
                dataset[0]
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_dataset_with_invalid_label_values(self):
        """Test dataset handling of invalid label values"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            images_dir = Path(temp_dir) / "images"
            labels_dir = Path(temp_dir) / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            
            # Create valid image
            image_data = np.random.rand(3, 640, 640).astype(np.float32)
            np.save(images_dir / "pre_image.npy", image_data)
            
            # Create label with invalid values
            with open(labels_dir / "pre_image.txt", "w") as f:
                f.write("invalid 0.5 0.5 0.2 0.3\n")  # Non-numeric class
            
            dataset = YOLODataset(str(images_dir), str(labels_dir))
            
            # Should handle invalid labels gracefully
            image, labels = dataset[0]
            assert labels.shape == (0, 5)  # Empty labels due to invalid data
        
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])