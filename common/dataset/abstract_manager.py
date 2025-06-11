from abc import ABC, abstractmethod

class AbstractDatasetManager(ABC):
    """
    Abstract class untuk manajemen dataset
    """
    
    @abstractmethod
    def download_from_roboflow(self, project_name, version, api_key):
        """
        Download dataset dari Roboflow
        """
        pass
