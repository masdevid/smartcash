from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.experiment_tracker import ExperimentTracker
from smartcash.handlers.model.integration.base_adapter import BaseAdapter

class ExperimentAdapter(BaseAdapter):
    """
    Adapter untuk integrasi dengan ExperimentTracker.
    Menyediakan antarmuka yang konsisten untuk tracking eksperimen.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi experiment adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        super().__init__(config, logger, "experiment_adapter")
        
        # Lazy initialization untuk experiment tracker
        self._experiment_tracker = None
    
    def _initialize(self):
        """Inisialisasi parameter eksperimen."""
        # Set default experiment name berdasarkan config
        self.experiment_name = self.config.get('experiment', {}).get('name', "default_experiment")
        
        # Set output directory berdasarkan config
        self.output_dir = Path(self.config.get('experiment', {}).get('output_dir', "runs/train/experiments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🧪 ExperimentAdapter diinisialisasi untuk {self.experiment_name}")