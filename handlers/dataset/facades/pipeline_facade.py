# File: smartcash/handlers/dataset/facades/pipeline_facade.py
# Deskripsi: Facade untuk menjalankan pipeline dataset lengkap

import time
from typing import Dict, List, Optional, Any

from smartcash.utils.observer import EventTopics, notify
from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.facades.data_loading_facade import DataLoadingFacade
from smartcash.handlers.dataset.facades.data_processing_facade import DataProcessingFacade
from smartcash.handlers.dataset.facades.data_operations_facade import DataOperationsFacade
from smartcash.handlers.dataset.facades.visualization_facade import VisualizationFacade


class PipelineFacade(DatasetBaseFacade):
    """Facade untuk menjalankan pipeline dataset lengkap."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, 
                 cache_dir: Optional[str] = None, logger: Optional = None):
        """Inisialisasi PipelineFacade.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori data (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger (opsional)
        """
        super().__init__(config, data_dir, cache_dir, logger)
        self.loading_facade = DataLoadingFacade(config, data_dir, cache_dir, logger)
        self.processing_facade = DataProcessingFacade(config, data_dir, cache_dir, logger)
        self.operations_facade = DataOperationsFacade(config, data_dir, cache_dir, logger)
        self.visualization_facade = VisualizationFacade(config, data_dir, cache_dir, logger)
    
    def setup_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """Setup pipeline lengkap untuk dataset."""
        start_time = time.time()
        
        # Konfigurasikan parameter pipeline
        download = kwargs.get('download_dataset', True)
        validate = kwargs.get('validate_dataset', True)
        fix = kwargs.get('fix_issues', True)
        augment = kwargs.get('augment_data', True)
        balance = kwargs.get('balance_classes', False)
        visualize = kwargs.get('visualize_results', True)
        show_progress = kwargs.get('show_progress', True)
        
        self.logger.info(f"🚀 Setup pipeline dataset: {download=}, {validate=}, {fix=}, {augment=}, {balance=}")
        
        # Notifikasi mulai pipeline
        notify(EventTopics.PREPROCESSING_START, self, pipeline_config=kwargs)
        
        results = {
            'download': None,
            'validation': {},
            'fixes': {},
            'augmentation': None,
            'balancing': None,
            'visualization': {},
            'splits': {}
        }
        
        try:
            # 1. Download dataset
            if download:
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="download", status="start")
                try:
                    paths = self.loading_facade.pull_dataset(show_progress=show_progress)
                    results['download'] = {
                        'train_path': paths[0],
                        'val_path': paths[1],
                        'test_path': paths[2]
                    }
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="download", status="complete")
                except Exception as e:
                    self.logger.error(f"❌ Download gagal: {str(e)}")
                    results['download'] = {'error': str(e)}
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="download", status="error", error=str(e))
            
            # 2. Validasi dan perbaiki dataset
            if validate:
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="validation", status="start")
                
                for split in ['train', 'valid', 'test']:
                    try:
                        val_result = self.processing_facade.validate_dataset(split=split, fix_issues=False, visualize=False)
                        results['validation'][split] = val_result
                        
                        # Perbaiki masalah jika perlu
                        if fix and val_result.get('invalid_labels', 0) + val_result.get('missing_labels', 0) > 0:
                            results['fixes'][split] = self.processing_facade.fix_dataset(
                                split=split, fix_coordinates=True, fix_labels=True, backup=True
                            )
                    except Exception as e:
                        self.logger.warning(f"⚠️ Validasi '{split}' gagal: {str(e)}")
                        results['validation'][split] = {'error': str(e)}
                
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="validation", status="complete")
            
            # 3. Augmentasi dataset
            if augment:
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="augmentation", status="start")
                try:
                    results['augmentation'] = self.processing_facade.augment_dataset(
                        split='train', augmentation_types=['combined', 'lighting'], 
                        num_variations=2, resume=True, validate_results=True
                    )
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="augmentation", status="complete")
                except Exception as e:
                    results['augmentation'] = {'error': str(e)}
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="augmentation", status="error", error=str(e))
            
            # 4. Balancing dataset
            if balance:
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="balancing", status="start")
                try:
                    results['balancing'] = self.processing_facade.balance_by_undersampling(split='train', target_ratio=2.0)
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="balancing", status="complete")
                except Exception as e:
                    results['balancing'] = {'error': str(e)}
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="balancing", status="error", error=str(e))
            
            # 5. Visualisasi hasil
            if visualize:
                notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="visualization", status="start")
                try:
                    results['visualization'] = self.visualization_facade.generate_dataset_report(splits=['train', 'valid', 'test'])
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="visualization", status="complete")
                except Exception as e:
                    results['visualization'] = {'error': str(e)}
                    notify(EventTopics.PREPROCESSING_PROGRESS, self, stage="visualization", status="error", error=str(e))
            
            # 6. Hitung statistik akhir
            results['splits'] = self.operations_facade.get_split_statistics()
            
        finally:
            # Hitung total waktu
            elapsed_time = time.time() - start_time
            results['duration'] = elapsed_time
            
            # Notifikasi selesai
            self.logger.success(f"✅ Pipeline selesai dalam {elapsed_time:.2f} detik")
            notify(EventTopics.PREPROCESSING_END, self, duration=elapsed_time, result=results)
        
        return results
        
    # Metode delegasi untuk mempertahankan API yang sama
    
    # Delegasi ke DataLoadingFacade
    def pull_dataset(self, **kwargs):
        return self.loading_facade.pull_dataset(**kwargs)
    
    def get_dataset(self, **kwargs):
        return self.loading_facade.get_dataset(**kwargs)
    
    def get_dataloader(self, **kwargs):
        return self.loading_facade.get_dataloader(**kwargs)
    
    def get_all_dataloaders(self, **kwargs):
        return self.loading_facade.get_all_dataloaders(**kwargs)
    
    # Delegasi ke DataProcessingFacade
    def validate_dataset(self, **kwargs):
        return self.processing_facade.validate_dataset(**kwargs)
    
    def fix_dataset(self, **kwargs):
        return self.processing_facade.fix_dataset(**kwargs)
    
    def augment_dataset(self, **kwargs):
        return self.processing_facade.augment_dataset(**kwargs)
    
    def balance_by_undersampling(self, **kwargs):
        return self.processing_facade.balance_by_undersampling(**kwargs)
    
    # Delegasi ke DataOperationsFacade
    def get_split_statistics(self, **kwargs):
        return self.operations_facade.get_split_statistics(**kwargs)
    
    def export_to_local(self, **kwargs):
        return self.operations_facade.export_to_local(**kwargs)
    
    def merge_datasets(self, **kwargs):
        return self.operations_facade.merge_datasets(**kwargs)
    
    # Delegasi ke VisualizationFacade
    def visualize_class_distribution(self, **kwargs):
        return self.visualization_facade.visualize_class_distribution(**kwargs)
    
    def generate_dataset_report(self, **kwargs):
        return self.visualization_facade.generate_dataset_report(**kwargs)