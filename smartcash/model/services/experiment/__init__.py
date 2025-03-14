"""
File: smartcash/model/services/experiment/__init__.py
Deskripsi: Package initialization untuk layanan eksperimen model individual
"""

from smartcash.model.services.experiment.experiment_service import ExperimentService
from smartcash.model.services.experiment.data_manager import ExperimentDataManager
from smartcash.model.services.experiment.metrics_tracker import ExperimentMetricsTracker

# Ekspor kelas-kelas utama
__all__ = [
    'ExperimentService',
    'ExperimentDataManager',
    'ExperimentMetricsTracker'
]

# Singleton/factory untuk mendapatkan layanan eksperimen
_experiment_services = {}

def get_experiment_service(experiment_dir="runs/experiment"):
    """
    Dapatkan instance ExperimentService.
    
    Args:
        experiment_dir: Direktori untuk eksperimen
        
    Returns:
        Instance ExperimentService
    """
    if experiment_dir not in _experiment_services:
        _experiment_services[experiment_dir] = ExperimentService(experiment_dir=experiment_dir)
    return _experiment_services[experiment_dir]

def get_data_manager(dataset_path="data", batch_size=16):
    """
    Dapatkan instance ExperimentDataManager.
    
    Args:
        dataset_path: Path ke dataset
        batch_size: Ukuran batch default
        
    Returns:
        Instance ExperimentDataManager
    """
    key = f"data_manager_{dataset_path}_{batch_size}"
    if key not in _experiment_services:
        _experiment_services[key] = ExperimentDataManager(
            dataset_path=dataset_path,
            batch_size=batch_size
        )
    return _experiment_services[key]

def create_experiment(name, config=None, experiment_dir=None):
    """
    Buat eksperimen baru.
    
    Args:
        name: Nama eksperimen
        config: Konfigurasi eksperimen (opsional)
        experiment_dir: Direktori eksperimen (opsional)
        
    Returns:
        Instance ExperimentService dengan eksperimen baru
    """
    experiment_service = get_experiment_service(experiment_dir) if experiment_dir else get_experiment_service()
    experiment_service.setup_experiment(name, config)
    return experiment_service