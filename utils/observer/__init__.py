# File: smartcash/utils/observer/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Package initialization untuk observer pattern terkonsolidasi di SmartCash

from .event_dispatcher import EventDispatcher
from .base_observer import BaseObserver
from .event_registry import EventRegistry
from .observer_manager import ObserverManager
from .decorators import observable, observe
from .event_topics import EventTopics
from .observer_priority import ObserverPriority

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