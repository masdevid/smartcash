from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.metrics import MetricsCalculator
from smartcash.handlers.model.integration.base_adapter import BaseAdapter

class MetricsAdapter(BaseAdapter):
    """
    Adapter untuk integrasi dengan MetricsCalculator dari utils.
    Menyediakan antarmuka yang konsisten untuk perhitungan metrik.
    """
    
    def __init__(
        self,
        logger: Optional[SmartCashLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Inisialisasi metrics adapter.
        
        Args:
            logger: Custom logger (opsional)
            config: Konfigurasi untuk metrics (opsional)
            output_dir: Direktori output untuk visualisasi (opsional)
        """
        self.config = config or {}
        super().__init__(self.config, logger, "metrics_adapter")
        
        # Override output directory jika diberikan
        if output_dir is not None:
            self.output_dir = Path(output_dir)
    
    def _initialize(self):
        """Inisialisasi metrics calculator."""
        # Metrik configuration
        self.metrics_config = self.config.get('evaluation', {}).get('metrics', {})
        
        # Default output directory untuk metrics
        self.output_dir = Path(self.config.get('output_dir', 'runs/eval')) / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi metrics calculator
        self._metrics_calculator = MetricsCalculator()
        
        self.logger.info(
            f"📊 MetricsAdapter diinisialisasi dengan MetricsCalculator"
        )