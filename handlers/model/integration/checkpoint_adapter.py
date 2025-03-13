from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.integration.base_adapter import BaseAdapter

class CheckpointAdapter(BaseAdapter):
    """
    Adapter untuk integrasi dengan CheckpointManager.
    Menyediakan antarmuka yang konsisten untuk manajemen checkpoint.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        output_dir: Optional[str] = None
    ):
        """
        Inisialisasi checkpoint adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
            output_dir: Direktori output untuk checkpoint (opsional)
        """
        super().__init__(config, logger, "checkpoint_adapter")
        
        # Override output directory jika diberikan
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        
        # Import CheckpointManager (lazy import)
        self._checkpoint_manager = None
    
    def _initialize(self):
        """Inisialisasi output directory."""
        # Default output directory untuk checkpoint
        self.output_dir = Path(self.config.get('output_dir', 'runs/train')) / "weights"
        self.output_dir.mkdir(parents=True, exist_ok=True)