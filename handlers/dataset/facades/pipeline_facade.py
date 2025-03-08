# File: smartcash/handlers/dataset/facades/pipeline_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade untuk menjalankan pipeline dataset lengkap

import time
from typing import Dict, List, Optional, Any

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.facades.data_loading_facade import DataLoadingFacade
from smartcash.handlers.dataset.facades.data_processing_facade import DataProcessingFacade
from smartcash.handlers.dataset.facades.data_operations_facade import DataOperationsFacade
from smartcash.handlers.dataset.facades.visualization_facade import VisualizationFacade


class PipelineFacade(DatasetBaseFacade, DataLoadingFacade, DataProcessingFacade, 
                     DataOperationsFacade, VisualizationFacade):
    """
    Facade yang menyediakan akses ke semua operasi dataset dan 
    menjalankan pipeline dataset lengkap.
    """
    
    def setup_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Setup pipeline lengkap untuk dataset.
        
        Args:
            **kwargs: Parameter konfigurasi pipeline
                download_dataset: Jika True, download dataset dari Roboflow
                validate_dataset: Jika True, validasi dataset
                fix_issues: Jika True, perbaiki masalah dataset
                augment_data: Jika True, augmentasi dataset
                balance_classes: Jika True, seimbangkan distribusi kelas
                visualize_results: Jika True, buat visualisasi hasil
                show_progress: Jika True, tampilkan progress bar
            
        Returns:
            Dict berisi statistik setup
        """
        start_time = time.time()
        
        # Konfigurasikan parameter pipeline
        download_dataset = kwargs.get('download_dataset', True)
        validate_dataset = kwargs.get('validate_dataset', True)
        fix_issues = kwargs.get('fix_issues', True)
        augment_data = kwargs.get('augment_data', True)
        balance_classes = kwargs.get('balance_classes', False)
        visualize_results = kwargs.get('visualize_results', True)
        show_progress = kwargs.get('show_progress', True)
        
        self.logger.info(
            f"🚀 Memulai setup pipeline dataset lengkap:\n"
            f"   • Download: {download_dataset}\n"
            f"   • Validasi: {validate_dataset}\n"
            f"   • Perbaikan: {fix_issues}\n"
            f"   • Augmentasi: {augment_data}\n"
            f"   • Balancing: {balance_classes}\n"
            f"   • Visualisasi: {visualize_results}"
        )
        
        results = {
            'download': None,
            'validation': {},
            'fixes': {},
            'augmentation': None,
            'balancing': None,
            'visualization': {},
            'splits': {}
        }
        
        # 1. Download dataset jika diminta
        if download_dataset:
            try:
                self.logger.info("🔽 Langkah 1: Download dataset")
                paths = self.pull_dataset(show_progress=show_progress)
                results['download'] = {
                    'train_path': paths[0],
                    'val_path': paths[1],
                    'test_path': paths[2]
                }
            except Exception as e:
                self.logger.error(f"❌ Download dataset gagal: {str(e)}")
                results['download'] = {'error': str(e)}
        
        # 2. Validasi dan perbaiki dataset jika diminta
        if validate_dataset:
            for split in ['train', 'valid', 'test']:
                try:
                    self.logger.info(f"🔍 Validasi dataset split '{split}'")
                    val_result = self.validate_dataset(
                        split=split,
                        fix_issues=False,
                        visualize=False
                    )
                    results['validation'][split] = val_result
                    
                    # Perbaiki masalah jika diminta dan ada masalah
                    if fix_issues:
                        invalid_count = (
                            val_result.get('invalid_labels', 0) + 
                            val_result.get('missing_labels', 0) +
                            val_result.get('fixed_coordinates', 0)
                        )
                        
                        if invalid_count > 0:
                            self.logger.info(f"🔧 Perbaiki masalah di split '{split}'")
                            fix_result = self.fix_dataset(
                                split=split,
                                fix_coordinates=True,
                                fix_labels=True,
                                backup=True
                            )
                            results['fixes'][split] = fix_result
                except Exception as e:
                    self.logger.warning(f"⚠️ Validasi split '{split}' gagal: {str(e)}")
                    results['validation'][split] = {'error': str(e)}
        
        # 3. Augmentasi dataset train jika diminta
        if augment_data:
            try:
                self.logger.info("🎨 Langkah 3: Augmentasi dataset")
                aug_result = self.augment_dataset(
                    split='train',
                    augmentation_types=['combined', 'lighting'],
                    num_variations=2,
                    resume=True,
                    validate_results=True
                )
                results['augmentation'] = aug_result
            except Exception as e:
                self.logger.error(f"❌ Augmentasi dataset gagal: {str(e)}")
                results['augmentation'] = {'error': str(e)}
        
        # 4. Balancing dataset jika diminta
        if balance_classes:
            try:
                self.logger.info("⚖️ Langkah 4: Penyeimbangan dataset")
                balance_result = self.balance_by_undersampling(
                    split='train',
                    target_ratio=2.0  # Max 2x perbedaan antara kelas terbanyak dan tersedikit
                )
                results['balancing'] = balance_result
            except Exception as e:
                self.logger.error(f"❌ Penyeimbangan dataset gagal: {str(e)}")
                results['balancing'] = {'error': str(e)}
        
        # 5. Visualisasi hasil jika diminta
        if visualize_results:
            try:
                self.logger.info("🖼️ Langkah 5: Visualisasi dataset")
                report = self.generate_dataset_report(splits=['train', 'valid', 'test'])
                results['visualization'] = report
            except Exception as e:
                self.logger.error(f"❌ Visualisasi dataset gagal: {str(e)}")
                results['visualization'] = {'error': str(e)}
        
        # 6. Hitung statistik split dataset akhir
        results['splits'] = self.get_split_statistics()
        
        # Hitung total waktu
        elapsed_time = time.time() - start_time
        results['duration'] = elapsed_time
        
        self.logger.success(
            f"✅ Setup pipeline selesai dalam {elapsed_time:.2f} detik:\n"
            f"   • Train: {results['splits'].get('train', {}).get('images', 0)} gambar\n"
            f"   • Valid: {results['splits'].get('valid', {}).get('images', 0)} gambar\n"
            f"   • Test: {results['splits'].get('test', {}).get('images', 0)} gambar"
        )
        
        return results