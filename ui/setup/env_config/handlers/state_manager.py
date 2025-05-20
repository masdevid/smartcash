"""
File: smartcash/ui/setup/env_config/handlers/state_manager.py
Deskripsi: State manager untuk environment config
"""

class StateManager:
    """
    State manager untuk environment config
    """
    
    def __init__(self):
        """
        Inisialisasi state manager
        """
        self._is_drive_connected = False
        self._is_directory_setup = False
        self._is_drive_synced = False
        self._progress = 0
    
    @property
    def is_drive_connected(self) -> bool:
        """
        Get drive connection state
        """
        return self._is_drive_connected
    
    def set_drive_connected(self, value: bool):
        """
        Set drive connection state
        """
        self._is_drive_connected = value
    
    @property
    def is_directory_setup(self) -> bool:
        """
        Get directory setup state
        """
        return self._is_directory_setup
    
    def set_directory_setup(self, value: bool):
        """
        Set directory setup state
        """
        self._is_directory_setup = value
    
    @property
    def is_drive_synced(self) -> bool:
        """
        Get drive sync state
        """
        return self._is_drive_synced
    
    def set_drive_synced(self, value: bool):
        """
        Set drive sync state
        """
        self._is_drive_synced = value
    
    @property
    def progress(self) -> float:
        """
        Get progress value
        """
        return self._progress
    
    def update_progress(self, value: float, message: str = ""):
        """
        Update progress value
        
        Args:
            value: Progress value
            message: Progress message (optional)
        """
        self._progress = value 