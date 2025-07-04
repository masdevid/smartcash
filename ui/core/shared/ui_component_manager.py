"""
file_path: smartcash/ui/core/shared/ui_component_manager.py
Deskripsi: Stub minimal dari UIComponentManager untuk memenuhi dependensi impor
pada fase refaktor awal. Implementasi penuh akan ditambahkan saat modul terkait
sudah dipindahkan. Seluruh komponen dikemas seminimal mungkin agar tidak
tergantung pada library eksternal dan tetap kompatibel dengan test import.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, TypeVar, Generic

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Registri sederhana untuk menyimpan komponen berdasarkan nama."""

    def __init__(self) -> None:
        self._components: Dict[str, T] = {}

    def register(self, name: str, component: T) -> None:
        self._components[name] = component

    def get(self, name: str) -> Optional[T]:
        return self._components.get(name)

    def clear(self) -> None:
        self._components.clear()

    def all(self) -> Dict[str, T]:  # pragma: no cover
        return dict(self._components)


class UIComponentManager:
    """Manager sederhana untuk UI components.

    Dirancang sebagai placeholder agar modul lain yang bergantung
    dapat diimport tanpa error. Metode yang belum diimplementasikan
    akan menaikkan NotImplementedError untuk mencegah silent failure.
    """

    def __init__(self) -> None:
        self._registry: ComponentRegistry[Any] = ComponentRegistry()

    # ------------------------------------------------------------------
    # API publik - minimal
    # ------------------------------------------------------------------
    def register(self, name: str, component: Any) -> None:
        self._registry.register(name, component)

    def get(self, name: str) -> Any:
        component = self._registry.get(name)
        if component is None:
            raise KeyError(f"Komponen '{name}' tidak ditemukan")
        return component

    # ------------------------------------------------------------------
    # Placeholder untuk API lanjutan
    # ------------------------------------------------------------------
    def render(self, name: str) -> Any:  # pragma: no cover
        raise NotImplementedError("Method render belum diimplementasikan")


# === Singleton helper ===

_component_manager: Optional[UIComponentManager] = None


def get_component_manager() -> UIComponentManager:
    global _component_manager
    if _component_manager is None:
        _component_manager = UIComponentManager()
    return _component_manager


__all__ = [
    "ComponentRegistry",
    "UIComponentManager",
    "get_component_manager",
]
