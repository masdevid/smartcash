"""
File: smartcash/components/observer/__init__.py
Deskripsi: Package initialization untuk observer pattern terkonsolidasi di SmartCash
"""

from smartcash.components.observer.event_dispatcher import EventDispatcher
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.event_registry import EventRegistry
from smartcash.components.observer.observer_manager import ObserverManager
from smartcash.components.observer.decorators import observable, observe
from smartcash.components.observer.event_topics import EventTopics
from smartcash.components.observer.observer_priority import ObserverPriority

# Shortcut functions
register = EventDispatcher.register
unregister = EventDispatcher.unregister
notify = EventDispatcher.notify
register_many = EventDispatcher.register_many
unregister_many = EventDispatcher.unregister_many
unregister_all = EventDispatcher.unregister_all

__all__ = [
    'BaseObserver',
    'EventDispatcher',
    'EventRegistry',
    'ObserverManager',
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