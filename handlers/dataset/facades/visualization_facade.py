# File: smartcash/handlers/dataset/facades/visualization_facade.py
# Deskripsi: Facade untuk visualisasi dataset menggunakan utils/visualization dengan ObserverManager

from typing import Dict, List, Optional, Any, Tuple

from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.utils.visualization.data import DataVisualizationHelper


class VisualizationFacade(DatasetBaseFacade):
    """Facade untuk operasi visualisasi dataset menggunakan utils/visualization."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, 
                 cache_dir: Optional[str] = None, logger: Optional = None):
        """
        Inisialisasi VisualizationFacade.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori data (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger (opsional)
        """
        super().__init__(config, data_dir, cache_dir, logger)
        
        # Setup ObserverManager
        self.observer_manager = ObserverManager(auto_register=True)
    
    @property
    def visualizer(self) -> DataVisualizationHelper:
        """Akses ke visualizer."""
        return self._get_component('visualizer', lambda: DataVisualizationHelper(
            output_dir=str(self.data_dir / 'visualizations'),
            logger=self.logger
        ))
    
    def _visualize_with_notify(self, type_name, split, visualize_func, show_progress=True, **kwargs):
        """
        Helper untuk visualisasi dengan notifikasi.
        
        Args:
            type_name: Nama tipe visualisasi
            split: Split dataset
            visualize_func: Fungsi visualisasi
            show_progress: Tampilkan progress bar
            **kwargs: Parameter tambahan untuk fungsi visualisasi
            
        Returns:
            Hasil visualisasi
        """
        # Notifikasi mulai visualisasi
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.UI_UPDATE,
            callback=lambda event_type, sender, **params: self.logger.start(
                f"🎨 Visualisasi {type_name} untuk split '{split}' dimulai"
            ),
            name=f"Visualize{type_name}Start_{split}"
        )
        
        # Setup progress observer jika diperlukan
        if show_progress:
            self.observer_manager.create_progress_observer(
                event_types=[EventTopics.UI_UPDATE],
                total=100,  # Perkiraan sederhana
                desc=f"Visualisasi {type_name}",
                name=f"Visualize{type_name}Progress_{split}",
                group="visualization_progress"
            )
        
        try:
            # Jalankan visualisasi
            result = visualize_func(**kwargs)
            
            # Notifikasi selesai visualisasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.UI_UPDATE,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Visualisasi {type_name} untuk split '{split}' selesai: {result}"
                ),
                name=f"Visualize{type_name}Complete_{split}"
            )
            
            return result
        except Exception as e:
            # Notifikasi error visualisasi
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.UI_UPDATE,
                callback=lambda event_type, sender, **params: self.logger.error(
                    f"❌ Visualisasi {type_name} untuk split '{split}' gagal: {str(e)}"
                ),
                name=f"Visualize{type_name}Error_{split}"
            )
            
            self.logger.error(f"❌ Visualisasi {type_name} gagal: {str(e)}")
            raise
    
    def visualize_class_distribution(self, split: str = 'train', save_path: Optional[str] = None,
                               top_n: int = 10, figsize: Tuple[int, int] = (12, 8), 
                               show_progress: bool = True) -> str:
        """
        Visualisasikan distribusi kelas.
        
        Args:
            split: Split dataset
            save_path: Path untuk menyimpan hasil (opsional)
            top_n: Jumlah kelas teratas yang ditampilkan
            figsize: Ukuran gambar (width, height)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path hasil visualisasi
        """
        # Dapatkan statistik kelas
        from smartcash.handlers.dataset.facades.dataset_explorer_facade import DatasetExplorerFacade
        explorer = DatasetExplorerFacade(self.config, str(self.data_dir), self.logger)
        class_stats = explorer.get_class_statistics(split)
        
        return self._visualize_with_notify(
            'class_distribution', 
            split,
            self.visualizer.plot_class_distribution,
            show_progress=show_progress,
            class_stats=class_stats,
            title=f"Distribusi Kelas - Split {split}",
            save_path=save_path,
            top_n=top_n,
            figsize=figsize
        )
    
    def visualize_layer_distribution(self, split: str = 'train', save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6), show_progress: bool = True) -> str:
        """
        Visualisasikan distribusi layer.
        
        Args:
            split: Split dataset
            save_path: Path untuk menyimpan hasil (opsional)
            figsize: Ukuran gambar (width, height)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path hasil visualisasi
        """
        # Dapatkan statistik layer
        from smartcash.handlers.dataset.facades.dataset_explorer_facade import DatasetExplorerFacade
        explorer = DatasetExplorerFacade(self.config, str(self.data_dir), self.logger)
        layer_stats = explorer.get_layer_statistics(split)
        
        return self._visualize_with_notify(
            'layer_distribution', 
            split,
            self.visualizer.plot_layer_distribution,
            show_progress=show_progress,
            layer_stats=layer_stats,
            title=f"Distribusi Layer - Split {split}",
            save_path=save_path,
            figsize=figsize
        )
    
    def visualize_sample_images(self, split: str = 'train', num_samples: int = 9,
                          classes: Optional[List[str]] = None, random_seed: int = 42,
                          figsize: Tuple[int, int] = (15, 15), save_path: Optional[str] = None,
                          show_progress: bool = True) -> str:
        """
        Visualisasikan sampel gambar dengan bounding box.
        
        Args:
            split: Split dataset
            num_samples: Jumlah sampel
            classes: Filter kelas (opsional)
            random_seed: Seed untuk random sampling
            figsize: Ukuran gambar (width, height)
            save_path: Path untuk menyimpan hasil (opsional)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path hasil visualisasi
        """
        return self._visualize_with_notify(
            'sample_images', 
            split,
            self.visualizer.plot_sample_images,
            show_progress=show_progress,
            data_dir=str(self.data_dir / split),
            num_samples=num_samples,
            classes=classes,
            random_seed=random_seed,
            figsize=figsize,
            save_path=save_path
        )
        
    def visualize_augmentation_comparison(self, image_path: str,
                                   augmentation_types: List[str] = ['original', 'lighting', 'geometric', 'combined'],
                                   save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10),
                                   show_progress: bool = True) -> str:
        """
        Visualisasikan perbandingan berbagai jenis augmentasi pada gambar.
        
        Args:
            image_path: Path gambar
            augmentation_types: Tipe augmentasi yang dibandingkan
            save_path: Path untuk menyimpan hasil (opsional)
            figsize: Ukuran gambar (width, height)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path hasil visualisasi
        """
        return self._visualize_with_notify(
            'augmentation_comparison', 
            None,
            self.visualizer.plot_augmentation_comparison,
            show_progress=show_progress,
            image_path=image_path,
            augmentation_types=augmentation_types,
            save_path=save_path,
            figsize=figsize
        )
    
    def generate_dataset_report(self, splits: List[str] = ['train', 'valid', 'test'],
                           include_validation: bool = True, include_analysis: bool = True,
                           include_visualizations: bool = True, sample_count: int = 9,
                           save_format: str = 'json', show_progress: bool = True) -> Dict[str, Any]:
        """
        Membuat laporan komprehensif tentang dataset.
        
        Args:
            splits: List split dataset
            include_validation: Sertakan hasil validasi
            include_analysis: Sertakan hasil analisis
            include_visualizations: Sertakan visualisasi
            sample_count: Jumlah sampel gambar untuk visualisasi
            save_format: Format penyimpanan hasil ('json', 'markdown', 'both')
            show_progress: Tampilkan progress bar
            
        Returns:
            Hasil laporan
        """
        # Notifikasi mulai report
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.UI_UPDATE,
            callback=lambda event_type, sender, **params: self.logger.start(
                f"📊 Pembuatan laporan dataset dimulai untuk {len(splits)} split"
            ),
            name="ReportStart"
        )
        
        # Setup progress observer jika diperlukan
        if show_progress:
            # Perkiraan total langkah
            total_steps = len(splits) * (
                (1 if include_validation else 0) +
                (1 if include_analysis else 0) +
                (1 if include_visualizations else 0)
            )
            
            self.observer_manager.create_progress_observer(
                event_types=[EventTopics.UI_UPDATE],
                total=total_steps or 1,
                desc="Generate Report",
                name="ReportProgress",
                group="report_progress"
            )
        
        try:
            # Buat report
            from smartcash.handlers.dataset.operations.dataset_reporting_operation import DatasetReportingOperation
            
            reporter = DatasetReportingOperation(
                config=self.config,
                data_dir=str(self.data_dir),
                logger=self.logger
            )
            
            report = reporter.generate_dataset_report(
                splits=splits,
                include_validation=include_validation,
                include_analysis=include_analysis,
                include_visualizations=include_visualizations,
                sample_count=sample_count,
                save_format=save_format
            )
            
            # Notifikasi selesai report
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.UI_UPDATE,
                callback=lambda event_type, sender, **params: self.logger.success(
                    f"✅ Laporan dataset selesai: {len(splits)} split diproses"
                ),
                name="ReportComplete"
            )
            
            return report
        except Exception as e:
            # Notifikasi error report
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.UI_UPDATE,
                callback=lambda event_type, sender, **params: self.logger.error(
                    f"❌ Laporan dataset gagal: {str(e)}"
                ),
                name="ReportError"
            )
            
            self.logger.error(f"❌ Laporan dataset gagal: {str(e)}")
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