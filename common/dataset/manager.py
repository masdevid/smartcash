from .abstract_manager import AbstractDatasetManager

class DatasetManager(AbstractDatasetManager):
    """
    Implementasi konkrit manajer dataset
    """
    
    def download_from_roboflow(self, project_name, version, api_key):
        """
        Implementasi download dari Roboflow
        """
        # Implementasi asli
        return True
