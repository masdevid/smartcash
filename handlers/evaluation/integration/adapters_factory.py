# File: smartcash/handlers/evaluation/integration/adapters_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory pattern untuk pembuatan adapter evaluasi

from typing import Dict, Optional, Any, Type

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.integration.metrics_adapter import MetricsAdapter
from smartcash.handlers.evaluation.integration.dataset_adapter import DatasetAdapter
from smartcash.handlers.evaluation.integration.model_manager_adapter import ModelManagerAdapter
from smartcash.handlers.evaluation.integration.checkpoint_manager_adapter import CheckpointManagerAdapter
from smartcash.handlers.evaluation.integration.visualization_adapter import VisualizationAdapter

class AdaptersFactory:
    """
    Factory untuk membuat adapter dengan parameter standar.
    Memastikan reusability dan konsistensi adapter.
    """
    
    def __init__(self, config: Dict, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi factory adapter.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("adapters_factory")
        
        # Cache adapter untuk reuse
        self._cached_adapters = {}
    
    def get_metrics_adapter(self, **kwargs) -> MetricsAdapter:
        """
        Dapatkan MetricsAdapter.
        
        Args:
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance MetricsAdapter
        """
        return self._get_or_create_adapter("metrics", MetricsAdapter, **kwargs)
    
    def get_dataset_adapter(self, **kwargs) -> DatasetAdapter:
        """
        Dapatkan DatasetAdapter.
        
        Args:
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance DatasetAdapter
        """
        return self._get_or_create_adapter("dataset", DatasetAdapter, **kwargs)
    
    def get_model_adapter(self, **kwargs) -> ModelManagerAdapter:
        """
        Dapatkan ModelManagerAdapter.
        
        Args:
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance ModelManagerAdapter
        """
        return self._get_or_create_adapter("model", ModelManagerAdapter, **kwargs)
    
    def get_checkpoint_adapter(self, **kwargs) -> CheckpointManagerAdapter:
        """
        Dapatkan CheckpointManagerAdapter.
        
        Args:
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance CheckpointManagerAdapter
        """
        return self._get_or_create_adapter("checkpoint", CheckpointManagerAdapter, **kwargs)
    
    def get_visualization_adapter(self, output_dir: Optional[str] = None, **kwargs) -> VisualizationAdapter:
        """
        Dapatkan VisualizationAdapter.
        
        Args:
            output_dir: Direktori output untuk visualisasi (opsional)
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance VisualizationAdapter
        """
        # VisualizationAdapter disimpan terpisah karena memiliki parameter output_dir
        if output_dir:
            # Buat adapter dengan output_dir custom
            return VisualizationAdapter(
                config=self.config,
                output_dir=output_dir,
                logger=self.logger,
                **kwargs
            )
        else:
            # Gunakan cache jika output_dir standar
            return self._get_or_create_adapter("visualization", VisualizationAdapter, **kwargs)
    
    def _get_or_create_adapter(self, adapter_key: str, adapter_class: Type, **kwargs) -> Any:
        """
        Dapatkan adapter dari cache atau buat baru jika belum ada.
        
        Args:
            adapter_key: Key untuk menyimpan adapter di cache
            adapter_class: Kelas adapter yang akan dibuat
            **kwargs: Parameter tambahan untuk adapter
            
        Returns:
            Instance adapter
        """
        if adapter_key not in self._cached_adapters:
            self._cached_adapters[adapter_key] = adapter_class(
                config=self.config,
                logger=self.logger,
                **kwargs
            )
            self.logger.debug(f"🔧 Adapter {adapter_key} dibuat")
        
        return self._cached_adapters[adapter_key]
    
    def clear_cache(self):
        """Bersihkan cache adapter."""
        self._cached_adapters.clear()
        self.logger.debug("🧹 Cache adapter dibersihkan")