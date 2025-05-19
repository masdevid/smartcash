"""
File: smartcash/common/config/dependency_manager.py
Deskripsi: Pengelolaan dependency injection untuk ConfigManager
"""

from typing import Dict, Any, Type, TypeVar, Callable

# Type variable untuk dependency injection
T = TypeVar('T')

class DependencyManager:
    """
    Pengelolaan dependency injection untuk ConfigManager
    """
    
    def __init__(self):
        """
        Inisialisasi dependency manager
        """
        self._dependencies = {}
        self._factory_functions = {}
    
    def register(self, interface_type: Type[T], implementation: Type[T]) -> None:
        """
        Daftarkan implementasi untuk interface tertentu.
        
        Args:
            interface_type: Tipe interface
            implementation: Tipe implementasi
        """
        self._dependencies[interface_type] = implementation
    
    def register_instance(self, interface_type: Type[T], instance: T) -> None:
        """
        Daftarkan instance untuk interface tertentu (singleton).
        
        Args:
            interface_type: Tipe interface
            instance: Instance implementasi
        """
        self._dependencies[interface_type] = instance
    
    def register_factory(self, interface_type: Type[T], factory: Callable[..., T]) -> None:
        """
        Daftarkan factory function untuk membuat implementasi.
        
        Args:
            interface_type: Tipe interface
            factory: Factory function
        """
        self._factory_functions[interface_type] = factory
    
    def resolve(self, interface_type: Type[T], *args, **kwargs) -> T:
        """
        Resolve dependency untuk interface tertentu, dengan factory atau implementation class
        
        Args:
            interface_type: Tipe interface
            *args: Argumen untuk constructor
            **kwargs: Keyword arguments untuk constructor
            
        Returns:
            Instance implementasi
            
        Raises:
            KeyError: Jika interface tidak terdaftar
        """
        if interface_type in self._factory_functions:
            return self._factory_functions[interface_type](*args, **kwargs)
        
        impl = self._dependencies.get(interface_type)
        if impl is None:
            raise KeyError(f"No implementation registered for {interface_type}")
            
        if isinstance(impl, type):  # Class
            return impl(*args, **kwargs)
        else:  # Instance
            return impl
