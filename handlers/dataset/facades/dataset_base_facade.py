# File: smartcash/handlers/dataset/facades/dataset_base_facade.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk facade dataset dengan integrasi environment_manager

from pathlib import Path
from typing import Dict, Optional, Any, Callable

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetBaseFacade:
    """Kelas dasar untuk facade dataset dengan environment manager."""
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger=None
    ):
        """Inisialisasi DatasetBaseFacade."""
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.env_manager = EnvironmentManager(logger=self.logger)
        
        # Setup paths
        self.data_dir = Path(data_dir) if data_dir else self.env_manager.get_path('data')
        cache_path = config.get('data', {}).get('preprocessing', {}).get('cache_dir', '.cache/smartcash')
        self.cache_dir = Path(cache_dir) if cache_dir else self.env_manager.get_path(cache_path)
        
        # Model dan training params
        self.img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        self.batch_size = config.get('model', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Layer config
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Lazy-loaded components
        self._components = {}
    
    def _get_component(self, component_id: str, factory_func: Callable) -> Any:
        """Lazy load komponnen."""
        if component_id not in self._components:
            self._components[component_id] = factory_func()
        return self._components[component_id]
    
    def _get_split_path(self, split: str) -> Path:
        """Dapatkan path untuk split dataset."""
        if split in ('val', 'validation'):
            split = 'valid'
            
        # Cek config atau gunakan default
        split_paths = self.config.get('data', {}).get('local', {})
        return Path(split_paths[split]) if split in split_paths else self.data_dir / split
        
    def get_colab_path(self, path: Path) -> str:
        """Dapatkan path user-friendly untuk Colab."""
        if not self.env_manager.is_colab:
            return str(path)
            
        abs_path = Path(path).absolute()
        if '/content/drive/' in str(abs_path):
            return f"📂 Drive: {str(abs_path).replace('/content/drive/MyDrive/', '')}"
        else:
            return f"📂 Colab: {str(abs_path).replace('/content/', '')}"