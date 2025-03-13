from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.integration.base_adapter import BaseAdapter

class ExporterAdapter(BaseAdapter):
    """
    Adapter untuk integrasi dengan ModelExporter.
    Menyediakan antarmuka yang konsisten untuk ekspor model ke berbagai format.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi exporter adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        super().__init__(config, logger, "exporter_adapter")
    
    def _initialize(self):
        """Inisialisasi output directory."""
        # Setup direktori output
        export_dir = self.config.get('model', {}).get('export_dir', "exports")
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📦 ExporterAdapter diinisialisasi (output: {self.export_dir})")