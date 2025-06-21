"""
File: smartcash/ui/components/progress_tracker/callback_manager.py
Deskripsi: Manager untuk handling callbacks dengan type safety dan automatic cleanup
"""

import time
from typing import Dict, List, Callable, Set

class CallbackManager:
    """Manager untuk handling callbacks dengan type safety"""
    
    def __init__(self):
        self.callbacks: Dict[str, List[tuple]] = {}
        self.one_time_callbacks: Set[str] = set()
    
    def register(self, event: str, callback: Callable, one_time: bool = False) -> str:
        """Register callback dengan automatic cleanup"""
        callback_id = f"{event}_{id(callback)}_{time.time()}"
        self.callbacks.setdefault(event, []).append((callback_id, callback))
        one_time and self.one_time_callbacks.add(callback_id)
        return callback_id
    
    def unregister(self, callback_id: str):
        """Unregister specific callback dengan cleanup"""
        for event, callback_list in self.callbacks.items():
            self.callbacks[event] = [cb for cb in callback_list if cb[0] != callback_id]
        self.one_time_callbacks.discard(callback_id)
    
    def trigger(self, event: str, *args, **kwargs):
        """Trigger callbacks dengan error handling"""
        if event not in self.callbacks:
            return
        
        callbacks_to_remove = []
        for callback_id, callback in self.callbacks[event][:]:
            try:
                callback(*args, **kwargs)
                if callback_id in self.one_time_callbacks:
                    callbacks_to_remove.append(callback_id)
            except Exception as e:
                print(f"🚨 Callback error for {event}: {e}")
                callbacks_to_remove.append(callback_id)
        
        for callback_id in callbacks_to_remove:
            self.unregister(callback_id)
    
    def clear_event(self, event: str):
        """Clear all callbacks untuk specific event"""
        if event in self.callbacks:
            callback_ids = [cb[0] for cb in self.callbacks[event]]
            for callback_id in callback_ids:
                self.one_time_callbacks.discard(callback_id)
            self.callbacks[event].clear()
    
    def clear_all(self):
        """Clear semua callbacks"""
        self.callbacks.clear()
        self.one_time_callbacks.clear()
    
    def get_event_count(self, event: str) -> int:
        """Get jumlah callbacks untuk event"""
        return len(self.callbacks.get(event, []))
    
    def has_callbacks(self, event: str) -> bool:
        """Check apakah event punya callbacks"""
        return bool(self.callbacks.get(event))