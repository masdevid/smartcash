"""
File: smartcash/components/observer/__init__.py
Deskripsi: Package initialization untuk observer pattern terkonsolidasi di SmartCash
"""

from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.event_topics_observer import EventTopics
from smartcash.components.observer.priority_observer import ObserverPriority
from smartcash.components.observer.manager_observer import ObserverManager, get_observer_manager

# Import EventDispatcher after other modules to avoid circular imports
from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
from smartcash.components.observer.event_registry_observer import EventRegistry
from smartcash.components.observer.decorators_observer import observable, observe

__all__ = [
    'BaseObserver',
    'EventDispatcher',
    'EventRegistry',
    'ObserverManager',
    'get_observer_manager',
    'observable',
    'observe',
    'register',
    'unregister',
    'notify',
    'register_many',
    'unregister_many',
    'unregister_all',
    'EventTopics',
    'ObserverPriority'
]

# Shortcut functions (move these after __all__ to avoid circular import issues)
register = EventDispatcher.register
unregister = EventDispatcher.unregister
notify = EventDispatcher.notify
register_many = EventDispatcher.register_many
unregister_many = EventDispatcher.unregister_many
unregister_all = EventDispatcher.unregister_all