# File: smartcash/handlers/dataset/dataset_utils_adapter.py
# Deskripsi: Adapter untuk integrasi utils/dataset ke handlers/dataset dengan ObserverManager

from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import (
    EnhancedDatasetValidator, 
    DatasetAnalyzer,
    DatasetFixer
)
from smartcash.utils.dataset.dataset_utils import DatasetUtils
from smartcash.utils.augmentation import AugmentationManager

class DatasetUtilsAdapter:
    """Adapter untuk mengintegrasikan komponen utils/dataset dengan handlers/dataset."""
    
    def __init__(
        self, 
        config: Dict, 
        data_dir: Optional[str] = None, 
        logger: Optional[SmartCashLogger] = None,
        observer_manager: Optional[ObserverManager] = None
    ):
        """
        Inisialisasi adapter.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori data (opsional)
            logger: Logger kustom (opsional)
            observer_manager: ObserverManager untuk integrasi (opsional)
        """
        self.config = config
        self.data_dir = data_dir or config.get('data_dir', 'data')
        self.logger = logger or SmartCashLogger(__name__)
        self._components = {}  # Cache komponen untuk lazy loading
        
        # Setup ObserverManager
        self.observer_manager = observer_manager or ObserverManager(auto_register=True)
        
    
    def _get_component(self, name: str, factory_func):
        """Helper untuk lazy loading komponen."""
        if name not in self._components:
            self._components[name] = factory_func()
        return self._components[name]
        
    @property
    def validator(self) -> EnhancedDatasetValidator:
        """Lazy load validator."""
        return self._get_component('validator', lambda: EnhancedDatasetValidator(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def analyzer(self) -> DatasetAnalyzer:
        """Lazy load analyzer."""
        return self._get_component('analyzer', lambda: DatasetAnalyzer(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def fixer(self) -> DatasetFixer:
        """Lazy load fixer."""
        return self._get_component('fixer', lambda: DatasetFixer(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def augmentor(self) -> AugmentationManager:
        """Lazy load augmentor."""
        return self._get_component('augmentor', lambda: AugmentationManager(
            config=self.config,
            output_dir=self.data_dir,
            logger=self.logger,
            num_workers=self.config.get('model', {}).get('workers', 4)
        ))
    
    @property
    def utils(self) -> DatasetUtils:
        """Lazy load utils."""
        return self._get_component('utils', lambda: DatasetUtils(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Validasi dataset menggunakan EnhancedDatasetValidator.
        
        Args:
            split: Split dataset yang akan divalidasi
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil validasi
        """
        # Notifikasi event mulai validasi
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.VALIDATION_EVENT,
            callback=lambda event_type, sender, **params: self.logger.start(f"🔍 Validasi '{split}' dimulai"),
            name=f"ValidationStart_{split}"
        )
        
        # Setup progress observer jika diperlukan
        if kwargs.get('show_progress', True):
            # Dapatkan perkiraan total langkah
            split_dir = Path(self.data_dir) / split
            total_files = sum(1 for _ in split_dir.glob('**/*.jpg'))
            
            self.observer_manager.create_progress_observer(
                event_types=[EventTopics.VALIDATION_EVENT],
                total=total_files or 100,
                desc=f"Validasi {split}",
                name=f"ValidationProgress_{split}",
                group="validation_progress"
            )
        
        try:
            # Jalankan validasi
            result = self.validator.validate_dataset(split=split, **kwargs)
            
            # Notifikasi event selesai validasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.VALIDATION_EVENT,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Validasi '{split}' selesai: {result.get('valid_images', 0)}/{result.get('total_images', 0)} gambar valid"
                ),
                name=f"ValidationComplete_{split}"
            )
            
            return result
        except Exception as e:
            # Notifikasi event error validasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.VALIDATION_EVENT,
                callback=lambda event_type, sender, **params: self.logger.error(f"❌ Validasi '{split}' gagal: {str(e)}"),
                name=f"ValidationError_{split}"
            )
            
            self.logger.error(f"❌ Validasi dataset gagal: {str(e)}")
            raise
    
    def analyze_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Analisis dataset menggunakan DatasetAnalyzer.
        
        Args:
            split: Split dataset yang akan dianalisis
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil analisis
        """
        # Notifikasi event mulai analisis
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING,
            callback=lambda event_type, sender, **params: self.logger.start(f"🔍 Analisis '{split}' dimulai"),
            name=f"AnalysisStart_{split}"
        )
        
        try:
            # Jalankan analisis
            result = self.analyzer.analyze_dataset(split=split, **kwargs)
            
            # Notifikasi event selesai analisis
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.success(f"✅ Analisis '{split}' selesai"),
                name=f"AnalysisComplete_{split}"
            )
            
            return result
        except Exception as e:
            # Notifikasi event error analisis
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.error(f"❌ Analisis '{split}' gagal: {str(e)}"),
                name=f"AnalysisError_{split}"
            )
            
            self.logger.error(f"❌ Analisis dataset gagal: {str(e)}")
            raise
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Perbaiki masalah dataset menggunakan DatasetFixer.
        
        Args:
            split: Split dataset yang akan diperbaiki
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil perbaikan
        """
        # Notifikasi event mulai perbaikan
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING,
            callback=lambda event_type, sender, **params: self.logger.start(f"🔧 Perbaikan '{split}' dimulai"),
            name=f"FixStart_{split}"
        )
        
        try:
            # Jalankan perbaikan
            result = self.fixer.fix_dataset(split=split, **kwargs)
            
            # Notifikasi event selesai perbaikan
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Perbaikan '{split}' selesai: {result.get('fixed_labels', 0)} label diperbaiki"
                ),
                name=f"FixComplete_{split}"
            )
            
            return result
        except Exception as e:
            # Notifikasi event error perbaikan
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.error(f"❌ Perbaikan '{split}' gagal: {str(e)}"),
                name=f"FixError_{split}"
            )
            
            self.logger.error(f"❌ Perbaikan dataset gagal: {str(e)}")
            raise
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """
        Augmentasi dataset menggunakan AugmentationManager.
        
        Args:
            **kwargs: Parameter augmentasi
            
        Returns:
            Hasil augmentasi
        """
        split = kwargs.get('split', 'train')
        aug_types = kwargs.get('augmentation_types', ['combined'])
        
        # Notifikasi event mulai augmentasi
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.AUGMENTATION_EVENT,
            callback=lambda event_type, sender, **params: self.logger.start(
                f"🎨 Augmentasi '{split}' dengan tipe {aug_types} dimulai"
            ),
            name=f"AugmentationStart_{split}"
        )
        
        # Setup progress observer jika diperlukan
        if kwargs.get('show_progress', True):
            # Dapatkan perkiraan total langkah
            split_dir = Path(self.data_dir) / split
            total_files = sum(1 for _ in split_dir.glob('**/*.jpg'))
            
            # Perkirakan total berdasarkan jumlah variasi
            num_variations = kwargs.get('num_variations', 2)
            total_expected = total_files * num_variations
            
            self.observer_manager.create_progress_observer(
                event_types=[EventTopics.AUGMENTATION_EVENT],
                total=total_expected or 100,
                desc=f"Augmentasi {split}",
                name=f"AugmentationProgress_{split}",
                group="augmentation_progress"
            )
        
        try:
            # Jalankan augmentasi
            result = self.augmentor.augment_dataset(**kwargs)
            
            # Notifikasi event selesai augmentasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.AUGMENTATION_EVENT,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Augmentasi '{split}' selesai: {result.get('augmented', 0)} gambar dibuat"
                ),
                name=f"AugmentationComplete_{split}"
            )
            
            return result
        except Exception as e:
            # Notifikasi event error augmentasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.AUGMENTATION_EVENT,
                callback=lambda event_type, sender, **params: self.logger.error(f"❌ Augmentasi '{split}' gagal: {str(e)}"),
                name=f"AugmentationError_{split}"
            )
            
            self.logger.error(f"❌ Augmentasi dataset gagal: {str(e)}")
            raise
    
    def split_dataset(self, **kwargs) -> Dict:
        """
        Split dataset menggunakan DatasetUtils.
        
        Args:
            **kwargs: Parameter split
            
        Returns:
            Hasil split
        """
        # Notifikasi event mulai split
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING,
            callback=lambda event_type, sender, **params: self.logger.start(f"✂️ Split dataset dimulai"),
            name="SplitStart"
        )
        
        try:
            # Jalankan split
            result = self.utils.split_dataset(**kwargs)
            
            # Notifikasi event selesai split
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Split dataset selesai: {result.get('train', 0)} train, {result.get('valid', 0)} valid, {result.get('test', 0)} test"
                ),
                name="SplitComplete"
            )
            
            return result
        except Exception as e:
            # Notifikasi event error split
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING,
                callback=lambda event_type, sender, **params: self.logger.error(f"❌ Split dataset gagal: {str(e)}"),
                name="SplitError"
            )
            
            self.logger.error(f"❌ Split dataset gagal: {str(e)}")
            raise
            
    def unregister_observers(self):
        """Membatalkan registrasi semua observer."""
        self.observer_manager.unregister_all()
        
    def __del__(self):
        """Cleanup saat instance dihapus."""
        try:
            self.unregister_observers()
        except:
            pass