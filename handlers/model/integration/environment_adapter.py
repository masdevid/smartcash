from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.environment_manager import EnvironmentManager
from smartcash.handlers.model.integration.base_adapter import BaseAdapter

class EnvironmentAdapter(BaseAdapter):
    """
    Adapter untuk integrasi dengan EnvironmentManager.
    Menyediakan antarmuka yang konsisten untuk manajemen environment.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi environment adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        super().__init__(config, logger, "environment_adapter")
    
    def _initialize(self):
        """Inisialisasi environment manager."""
        # Inisialisasi environment manager
        self._env_manager = EnvironmentManager(logger=self.logger)
        
        self.logger.info(f"🔄 EnvironmentAdapter diinisialisasi (Colab: {self._env_manager.is_colab})")