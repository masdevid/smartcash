# File: smartcash/handlers/dataset/facades/pipeline_facade.py
# Deskripsi: Facade untuk menjalankan pipeline dataset lengkap dengan ObserverManager

import time
from typing import Dict, Optional, Any

from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.facades.data_loading_facade import DataLoadingFacade
from smartcash.handlers.dataset.facades.data_processing_facade import DataProcessingFacade
from smartcash.handlers.dataset.facades.data_operations_facade import DataOperationsFacade
from smartcash.handlers.dataset.facades.visualization_facade import VisualizationFacade


class PipelineFacade(DatasetBaseFacade):
    """Facade untuk menjalankan pipeline dataset lengkap."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, 
                 cache_dir: Optional[str] = None, logger: Optional = None):
        """Inisialisasi PipelineFacade."""
        super().__init__(config, data_dir, cache_dir, logger)
        self.observer_manager = ObserverManager(auto_register=True)
        self.loading_facade = DataLoadingFacade(config, data_dir, cache_dir, logger)
        self.processing_facade = DataProcessingFacade(config, data_dir, cache_dir, logger)
        self.operations_facade = DataOperationsFacade(config, data_dir, cache_dir, logger)
        self.visualization_facade = VisualizationFacade(config, data_dir, cache_dir, logger)
        
        # Integrasi observer_manager dengan adapter
        if hasattr(self.processing_facade, 'utils_adapter'):
            self.processing_facade.utils_adapter.observer_manager = self.observer_manager
    
    def setup_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """Setup pipeline lengkap untuk dataset."""
        start_time = time.time()
        self.logger.info(f"🚀 Setup pipeline dataset dimulai")
        
        # Setup observer untuk pipeline
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END],
            name="PipelineLogger", group="pipeline"
        )
        
        # Setup progress observer jika diperlukan
        if kwargs.get('show_progress', True):
            self.observer_manager.create_progress_observer(
                event_types=[EventTopics.PREPROCESSING_PROGRESS],
                total=sum([1 if kwargs.get(k, False) else 0 for k in 
                       ['download_dataset', 'validate_dataset', 'augment_data', 
                        'balance_classes', 'visualize_results']]) or 1,
                desc="Dataset Pipeline", name="PipelineProgress", group="pipeline"
            )
        
        # Notifikasi mulai pipeline
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_START,
            callback=lambda *args, **kw: None,
            name="PipelineStart"
        )
        
        results = {
            'download': None, 'validation': {}, 'fixes': {},
            'augmentation': None, 'balancing': None, 'visualization': {},
            'splits': {}
        }
        
        try:
            # 1. Download dataset
            if kwargs.get('download_dataset', True):
                try:
                    paths = self.loading_facade.pull_dataset(show_progress=kwargs.get('show_progress', True))
                    results['download'] = {'train_path': paths[0], 'val_path': paths[1], 'test_path': paths[2]}
                except Exception as e:
                    self.logger.error(f"❌ Download gagal: {str(e)}")
                    results['download'] = {'error': str(e)}
            
            # 2. Validasi dan perbaiki dataset
            if kwargs.get('validate_dataset', True):
                for split in ['train', 'valid', 'test']:
                    try:
                        val_result = self.processing_facade.validate_dataset(
                            split=split, fix_issues=False, visualize=False)
                        results['validation'][split] = val_result
                        
                        # Perbaiki masalah jika perlu
                        if kwargs.get('fix_issues', True) and val_result.get('invalid_labels', 0) + val_result.get('missing_labels', 0) > 0:
                            results['fixes'][split] = self.processing_facade.fix_dataset(
                                split=split, fix_coordinates=True, fix_labels=True, backup=True)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Validasi '{split}' gagal: {str(e)}")
                        results['validation'][split] = {'error': str(e)}
            
            # 3. Augmentasi dataset
            if kwargs.get('augment_data', True):
                try:
                    results['augmentation'] = self.processing_facade.augment_dataset(
                        split='train', augmentation_types=['combined', 'lighting'], 
                        num_variations=2, resume=True, validate_results=True)
                except Exception as e:
                    results['augmentation'] = {'error': str(e)}
            
            # 4. Balancing dataset
            if kwargs.get('balance_classes', False):
                try:
                    results['balancing'] = self.processing_facade.balance_by_undersampling(
                        split='train', target_ratio=2.0)
                except Exception as e:
                    results['balancing'] = {'error': str(e)}
            
            # 5. Visualisasi hasil
            if kwargs.get('visualize_results', True):
                try:
                    results['visualization'] = self.visualization_facade.generate_dataset_report(
                        splits=['train', 'valid', 'test'])
                except Exception as e:
                    results['visualization'] = {'error': str(e)}
            
            # 6. Hitung statistik akhir
            results['splits'] = self.operations_facade.get_split_statistics()
        
        finally:
            # Hitung total waktu
            elapsed_time = time.time() - start_time
            results['duration'] = elapsed_time
            
            # Notifikasi selesai
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING_END,
                callback=lambda *args, **kw: None,
                name="PipelineEnd"
            )
            
            self.logger.success(f"✅ Pipeline selesai dalam {elapsed_time:.2f} detik")
        
        return results
    
    def unregister_observers(self):
        """Membatalkan registrasi semua observer."""
        self.observer_manager.unregister_all()
    
    def __del__(self):
        """Cleanup saat instance dihapus."""
        try:
            self.unregister_observers()
        except:
            pass
        
    # Delegasi metode ke facades
    def pull_dataset(self, **kwargs): return self.loading_facade.pull_dataset(**kwargs)
    def get_dataset(self, **kwargs): return self.loading_facade.get_dataset(**kwargs)
    def get_dataloader(self, **kwargs): return self.loading_facade.get_dataloader(**kwargs)
    def get_all_dataloaders(self, **kwargs): return self.loading_facade.get_all_dataloaders(**kwargs)
    def validate_dataset(self, **kwargs): return self.processing_facade.validate_dataset(**kwargs)
    def fix_dataset(self, **kwargs): return self.processing_facade.fix_dataset(**kwargs)
    def augment_dataset(self, **kwargs): return self.processing_facade.augment_dataset(**kwargs)
    def balance_by_undersampling(self, **kwargs): return self.processing_facade.balance_by_undersampling(**kwargs)
    def get_split_statistics(self, **kwargs): return self.operations_facade.get_split_statistics(**kwargs)
    def export_to_local(self, **kwargs): return self.loading_facade.export_to_local(**kwargs)
    def merge_datasets(self, **kwargs): return self.operations_facade.merge_datasets(**kwargs)
    def visualize_class_distribution(self, **kwargs): return self.visualization_facade.visualize_class_distribution(**kwargs)
    def generate_dataset_report(self, **kwargs): return self.visualization_facade.generate_dataset_report(**kwargs)