"""
File: smartcash/model/services/research/__init__.py
Deskripsi: Package initialization untuk layanan penelitian model SmartCash
"""

from smartcash.model.services.research.experiment_service import ExperimentService
from smartcash.model.services.research.scenario_service import ScenarioService
from smartcash.model.services.research.experiment_creator import ExperimentCreator
from smartcash.model.services.research.experiment_runner import ExperimentRunner
from smartcash.model.services.research.experiment_analyzer import ExperimentAnalyzer
from smartcash.model.services.research.parameter_tuner import ParameterTuner
from smartcash.model.services.research.comparison_runner import ComparisonRunner

# Ekspor kelas-kelas utama
__all__ = [
    'ExperimentService',
    'ScenarioService',
    'ExperimentCreator',
    'ExperimentRunner',
    'ExperimentAnalyzer',
    'ParameterTuner',
    'ComparisonRunner'
]

# Singleton/factory untuk mendapatkan layanan penelitian
_research_services = {}

def get_experiment_service(base_dir="runs/experiments"):
    """
    Dapatkan instance ExperimentService.
    
    Args:
        base_dir: Direktori dasar untuk eksperimen
        
    Returns:
        Instance ExperimentService
    """
    if "experiment" not in _research_services:
        _research_services["experiment"] = ExperimentService(base_dir=base_dir)
    return _research_services["experiment"]

def get_scenario_service(base_dir="runs/scenarios"):
    """
    Dapatkan instance ScenarioService.
    
    Args:
        base_dir: Direktori dasar untuk skenario
        
    Returns:
        Instance ScenarioService
    """
    if "scenario" not in _research_services:
        _research_services["scenario"] = ScenarioService(base_dir=base_dir)
    return _research_services["scenario"]

# Alias untuk backward compatibility
ResearchService = ExperimentService