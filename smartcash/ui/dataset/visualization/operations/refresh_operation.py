"""
file_path: smartcash/ui/dataset/visualization/operations/refresh_operation.py

Operasi untuk memperbarui visualisasi dengan integrasi backend API.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.ui.dataset.visualization.operations.visualization_base_operation import VisualizationBaseOperation

class RefreshVisualizationOperation(VisualizationBaseOperation):
    """Operasi untuk memperbarui visualisasi dengan backend API integration."""
    
    def __init__(self, ui_module, config=None, callbacks=None):
        """Inisialisasi operasi refresh.
        
        Args:
            ui_module: Referensi ke modul UI yang memanggil operasi ini
            config: Konfigurasi untuk operasi
            callbacks: Callback untuk event operasi
        """
        super().__init__(
            ui_module=ui_module,
            config=config or {},
            callbacks=callbacks
        )
        self.name = "refresh_visualization"
        self.description = "Memperbarui tampilan visualisasi dengan data backend terbaru"
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the refresh visualization operation.
        
        Returns:
            Dict[str, Any]: Operation result with success status and details
        """
        try:
            # Start operation phase
            if hasattr(self, 'phase') and hasattr(self.__class__, '__module__'):
                from smartcash.ui.dataset.visualization.operations.visualization_base_operation import VisualizationOperationPhase
                self.phase = VisualizationOperationPhase.STARTED
            return self._execute_impl()
        except Exception as e:
            self.logger.error(f"Refresh visualization operation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Failed to refresh visualization: {str(e)}",
                "error": str(e)
            }
    
    def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementasi eksekusi operasi refresh dengan integrasi backend API.
        
        Returns:
            Dict berisi hasil operasi dengan comprehensive stats
        """
        self.log("📊 Memulai proses refresh visualisasi dengan backend integration...", 'info')
        
        try:
            # Get data directory from config
            config = getattr(self._ui_module, '_config', {})
            data_dir = config.get('data', {}).get('data_path', 'data')
            
            # Step 1: Fetch comprehensive dataset statistics
            self.log("🔄 Mengambil statistik dataset dari preprocessor API...", 'info')
            dataset_stats = self._fetch_dataset_stats(data_dir)
            
            # Step 2: Fetch augmentation status
            self.log("🔄 Mengambil status augmentasi dari augmentor API...", 'info')
            augmentation_stats = self._fetch_augmentation_stats(config)
            
            # Step 3: Fetch class distribution for charts
            self.log("🔄 Mengambil distribusi kelas untuk charts...", 'info')
            class_distribution = self._fetch_class_distribution(data_dir)
            
            # Step 4: Combine all stats
            comprehensive_stats = {
                'success': True,
                'dataset_stats': dataset_stats,
                'augmentation_stats': augmentation_stats,
                'class_distribution': class_distribution,
                'data_directory': str(data_dir),
                'last_updated': self._get_timestamp()
            }
            
            # Step 5: Update UI components with new data
            self.log("🔄 Memperbarui komponen UI dengan data terbaru...", 'info')
            if hasattr(self._ui_module, 'update_stats_cards'):
                self._ui_module.update_stats_cards(comprehensive_stats)
            if hasattr(self._ui_module, 'update_charts'):
                self._ui_module.update_charts(comprehensive_stats)
            
            self.log("✅ Visualisasi berhasil diperbarui dengan data backend", 'success')
            return comprehensive_stats
            
        except Exception as e:
            error_msg = f"Gagal memperbarui visualisasi: {str(e)}"
            self.log(f"❌ {error_msg}", 'error')
            
            # Return empty placeholder data instead of pure failure
            # This ensures cards are still displayed with placeholder values
            placeholder_stats = {
                "success": False,
                "message": error_msg,
                "error": str(e),
                "dataset_stats": {
                    'success': False,
                    'overview': {'total_files': 0},
                    'by_split': {
                        'train': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                        'valid': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                        'test': {'raw': 0, 'preprocessed': 0, 'augmented': 0}
                    },
                    'message': 'Backend API unavailable'
                },
                'augmentation_stats': {
                    'success': False,
                    'by_split': {
                        'train': {'file_count': 0},
                        'valid': {'file_count': 0},
                        'test': {'file_count': 0}
                    },
                    'message': 'Augmentation API unavailable'
                },
                'class_distribution': {
                    'success': False,
                    'total_classes': 0,
                    'total_objects': 0,
                    'message': 'Classification API unavailable'
                },
                'data_directory': 'Unknown - backend unavailable',
                'last_updated': 'Failed to update - backend error'
            }
            
            # Still try to update UI with placeholder data
            if hasattr(self._ui_module, 'update_stats_cards'):
                self._ui_module.update_stats_cards(placeholder_stats)
            if hasattr(self._ui_module, 'update_charts'):
                self._ui_module.update_charts(placeholder_stats)
            
            return placeholder_stats
    
    def _fetch_dataset_stats(self, data_dir: str) -> Dict[str, Any]:
        """
        Fetch comprehensive dataset statistics from preprocessor API.
        
        Args:
            data_dir: Data directory path
            
        Returns:
            Dataset statistics dictionary
        """
        try:
            from smartcash.dataset.preprocessor.api.stats_api import get_dataset_stats
            
            splits = ['train', 'valid', 'test']
            stats = get_dataset_stats(data_dir, splits=splits, include_details=True)
            
            if stats.get('success', False):
                self.log(f"📊 Dataset stats loaded: {stats['overview']['total_files']} total files", 'info')
                return stats
            else:
                self.log(f"⚠️ Dataset stats warning: {stats.get('message', 'Unknown error')}", 'warning')
                return self._get_empty_dataset_stats()
                
        except ImportError as e:
            self.log(f"⚠️ Preprocessor API not available: {e}", 'warning')
            return self._get_empty_dataset_stats()
        except Exception as e:
            self.log(f"❌ Error fetching dataset stats: {e}", 'error')
            return self._get_empty_dataset_stats()
    
    def _fetch_augmentation_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch augmentation statistics from augmentor API.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Augmentation statistics dictionary
        """
        try:
            from smartcash.dataset.augmentor import get_augmentation_status, get_sample_statistics
            
            # Get augmentation status
            aug_status = get_augmentation_status(config)
            
            # Get sample statistics for each split
            sample_stats = {}
            for split in ['train', 'valid', 'test']:
                try:
                    split_stats = get_sample_statistics(config, split)
                    sample_stats[split] = split_stats
                except Exception as e:
                    self.log(f"⚠️ Sample stats warning for {split}: {e}", 'warning')
                    sample_stats[split] = {'file_count': 0, 'total_size_mb': 0}
            
            combined_stats = {
                'success': True,
                'status': aug_status,
                'by_split': sample_stats
            }
            
            total_aug_files = sum(stats.get('file_count', 0) for stats in sample_stats.values())
            self.log(f"🎨 Augmentation stats loaded: {total_aug_files} augmented files", 'info')
            
            return combined_stats
            
        except ImportError as e:
            self.log(f"⚠️ Augmentor API not available: {e}", 'warning')
            return self._get_empty_augmentation_stats()
        except Exception as e:
            self.log(f"❌ Error fetching augmentation stats: {e}", 'error')
            return self._get_empty_augmentation_stats()
    
    def _fetch_class_distribution(self, data_dir: str) -> Dict[str, Any]:
        """
        Fetch class distribution statistics for charts.
        
        Args:
            data_dir: Data directory path
            
        Returns:
            Class distribution dictionary
        """
        try:
            from smartcash.dataset.preprocessor.api.stats_api import get_class_distribution_stats
            
            class_stats = get_class_distribution_stats(data_dir, splits=['train', 'valid', 'test'])
            
            if class_stats.get('success', False):
                total_classes = class_stats.get('total_classes', 0)
                total_objects = class_stats.get('total_objects', 0)
                self.log(f"🏷️ Class distribution loaded: {total_classes} classes, {total_objects} objects", 'info')
                return class_stats
            else:
                self.log(f"⚠️ Class distribution warning: {class_stats.get('message', 'Unknown error')}", 'warning')
                return self._get_empty_class_distribution()
                
        except ImportError as e:
            self.log(f"⚠️ Preprocessor API not available: {e}", 'warning')
            return self._get_empty_class_distribution()
        except Exception as e:
            self.log(f"❌ Error fetching class distribution: {e}", 'error')
            return self._get_empty_class_distribution()
    
    def _get_empty_dataset_stats(self) -> Dict[str, Any]:
        """Return empty dataset stats structure."""
        return {
            'success': False,
            'overview': {'total_files': 0, 'total_size_mb': 0},
            'file_types': {
                'raw_images': {'count': 0, 'total_size_mb': 0},
                'preprocessed_npy': {'count': 0, 'total_size_mb': 0},
                'augmented_npy': {'count': 0, 'total_size_mb': 0},
                'sample_images': {'count': 0, 'total_size_mb': 0}
            },
            'by_split': {
                'train': {'total_files': 0, 'raw': 0, 'preprocessed': 0, 'augmented': 0},
                'valid': {'total_files': 0, 'raw': 0, 'preprocessed': 0, 'augmented': 0},
                'test': {'total_files': 0, 'raw': 0, 'preprocessed': 0, 'augmented': 0}
            },
            'message': 'No data available'
        }
    
    def _get_empty_augmentation_stats(self) -> Dict[str, Any]:
        """Return empty augmentation stats structure."""
        return {
            'success': False,
            'status': {'file_count': 0, 'total_size_mb': 0},
            'by_split': {
                'train': {'file_count': 0, 'total_size_mb': 0},
                'valid': {'file_count': 0, 'total_size_mb': 0},
                'test': {'file_count': 0, 'total_size_mb': 0}
            },
            'message': 'No augmentation data available'
        }
    
    def _get_empty_class_distribution(self) -> Dict[str, Any]:
        """Return empty class distribution structure."""
        return {
            'success': False,
            'total_classes': 0,
            'total_objects': 0,
            'by_layer': {},
            'main_banknotes': {},
            'class_balance': {'balanced': True, 'imbalance_ratio': 1.0},
            'message': 'No class distribution data available'
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking updates."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')