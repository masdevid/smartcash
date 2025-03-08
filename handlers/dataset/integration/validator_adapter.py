# File: smartcash/handlers/dataset/integration/validator_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk mengintegrasikan EnhancedDatasetValidator dengan MultilayerDataset

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.handlers.dataset.multilayer_label_handler import MultilayerLabelHandler


class DatasetValidatorAdapter:
    """
    Adapter untuk mengintegrasikan EnhancedDatasetValidator dengan MultilayerDataset.
    
    Menerapkan Adapter Pattern untuk menghindari duplikasi kode validasi dataset
    antara handlers/dataset dan utils/dataset.
    """
    
    def __init__(
        self, 
        validator: EnhancedDatasetValidator,
        layers: Optional[List[str]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi adapter.
        
        Args:
            validator: EnhancedDatasetValidator yang akan digunakan
            layers: Daftar layer aktif (opsional)
            logger: Custom logger (opsional)
        """
        self.validator = validator
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup layer config
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Setup label handler
        self.label_handler = MultilayerLabelHandler(
            layers=self.layers,
            logger=self.logger
        )
    
    def validate_sample(
        self, 
        img_path: Path, 
        label_path: Path,
        require_all_layers: bool = False
    ) -> Dict:
        """
        Validasi satu sampel (gambar + label) untuk MultilayerDataset.
        
        Args:
            img_path: Path ke file gambar
            label_path: Path ke file label
            require_all_layers: Jika True, sampel harus memiliki semua layer
            
        Returns:
            Dict info sampel dengan status validasi
        """
        sample_info = {
            'image_path': img_path,
            'label_path': label_path,
            'available_layers': [],
            'is_valid': False
        }
        
        # Validasi gambar
        img_valid, img_info = self.validator.validate_image_file(str(img_path))
        if not img_valid:
            return sample_info
            
        # Tambahkan info gambar
        sample_info['image_size'] = img_info.get('size', (0, 0))
        
        # Validasi label
        if label_path.exists():
            try:
                # Validasi format label dengan validator
                label_valid, _ = self.validator.validate_label_file(
                    str(label_path), 
                    img_size=sample_info['image_size']
                )
                
                if label_valid:
                    # Parse layer yang tersedia dengan label handler
                    available_layers = self.label_handler.get_available_layers(label_path)
                    sample_info['available_layers'] = available_layers
                    
                    # Tentukan validitas sampel berdasarkan ketersediaan layer
                    if require_all_layers:
                        # Harus memiliki semua layer yang diminta
                        required_layers = set(self.layers)
                        available_layers_set = set(available_layers)
                        sample_info['is_valid'] = required_layers.issubset(available_layers_set)
                    else:
                        # Harus memiliki setidaknya satu layer yang diminta
                        sample_info['is_valid'] = any(layer in self.layers for layer in available_layers)
            except Exception as e:
                self.logger.debug(f"⚠️ Error validasi label {label_path}: {str(e)}")
        
        return sample_info
    
    def validate_dataset_for_multilayer(
        self, 
        data_path: Path,
        require_all_layers: bool = False
    ) -> List[Dict]:
        """
        Validasi dataset untuk MultilayerDataset.
        
        Args:
            data_path: Path ke direktori dataset
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            
        Returns:
            List info sampel valid
        """
        valid_samples = []
        
        # Cari file gambar
        images_dir = data_path / 'images'
        labels_dir = data_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Direktori dataset tidak lengkap: {data_path}")
            return valid_samples
            
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        self.logger.info(f"🔍 Memvalidasi {len(image_files)} file gambar di {images_dir}")
        
        # Validasi setiap sampel
        with tqdm(image_files, desc="Validasi Dataset") as pbar:
            for img_path in pbar:
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Validasi sampel
                sample_info = self.validate_sample(
                    img_path=img_path,
                    label_path=label_path,
                    require_all_layers=require_all_layers
                )
                
                if sample_info['is_valid']:
                    valid_samples.append(sample_info)
        
        # Log hasil validasi
        self.logger.info(f"✅ Dataset tervalidasi: {len(valid_samples)}/{len(image_files)} sampel valid")
        if require_all_layers:
            self.logger.info(f"ℹ️ Mode require_all_layers=True: Hanya menerima sampel dengan semua layer: {self.layers}")
        else:
            self.logger.info(f"ℹ️ Mode require_all_layers=False: Menerima sampel dengan minimal 1 layer dari: {self.layers}")
            
        # Log distribusi layer
        layer_counts = {layer: 0 for layer in self.layer_config_manager.get_layer_names()}
        for sample in valid_samples:
            for layer in sample['available_layers']:
                if layer in layer_counts:
                    layer_counts[layer] += 1
                
        for layer, count in layer_counts.items():
            if layer in self.layers and count > 0:
                self.logger.info(f"📊 Layer '{layer}': {count} sampel")
        
        return valid_samples