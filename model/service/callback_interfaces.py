"""
File: smartcash/model/service/callback_interfaces.py
Deskripsi: Interface untuk callback training dan progress tracking
"""

from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod

class TrainingCallback(ABC):
    """Interface untuk callback training yang akan digunakan oleh UI"""
    
    @abstractmethod
    def on_training_start(self, total_epochs: int, total_batches: int, config: Dict[str, Any]) -> None: pass
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None: pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, total_batches: int, metrics: Dict[str, float]) -> None: pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None: pass
    
    @abstractmethod
    def on_training_end(self, final_metrics: Dict[str, float], total_time: float) -> None: pass
    
    @abstractmethod
    def on_validation_start(self, epoch: int) -> None: pass
    
    @abstractmethod
    def on_validation_end(self, epoch: int, metrics: Dict[str, float]) -> None: pass
    
    @abstractmethod
    def on_training_error(self, error_message: str, phase: str) -> None: pass

class MetricsCallback(ABC):
    """Interface untuk callback metrics yang akan digunakan oleh UI"""
    
    @abstractmethod
    def update_metrics(self, metrics: Dict[str, float], phase: str = "train") -> None: pass
    
    @abstractmethod
    def update_learning_rate(self, lr: float) -> None: pass
    
    @abstractmethod
    def update_loss_breakdown(self, loss_components: Dict[str, float]) -> None: pass
    
    @abstractmethod
    def update_prediction_samples(self, samples: List[Dict[str, Any]]) -> None: pass
    
    @abstractmethod
    def update_inference_time(self, inference_time: float) -> None: pass

class ProgressCallback(ABC):
    """Interface untuk callback progress tracking yang akan digunakan oleh UI"""
    
    @abstractmethod
    def update_progress(self, current: int, total: int, message: str, phase: str = "general") -> None: pass
    
    @abstractmethod
    def update_status(self, status: str, phase: str = "general") -> None: pass
    
    @abstractmethod
    def update_stage(self, stage: str, substage: Optional[str] = None) -> None: pass
    
    @abstractmethod
    def on_complete(self, success: bool, message: str) -> None: pass
    
    @abstractmethod
    def on_error(self, error_message: str, phase: str) -> None: pass

# Tipe untuk callback function yang lebih sederhana
ProgressCallbackFn = Callable[[int, int, str, Optional[str]], None]
StatusCallbackFn = Callable[[str, Optional[str]], None]
MetricsCallbackFn = Callable[[Dict[str, float], Optional[str]], None]
ErrorCallbackFn = Callable[[str, Optional[str]], None]
CompleteCallbackFn = Callable[[bool, str], None]

# Tipe gabungan untuk callback
CallbackType = Union[
    TrainingCallback, 
    MetricsCallback, 
    ProgressCallback, 
    ProgressCallbackFn,
    StatusCallbackFn,
    MetricsCallbackFn,
    ErrorCallbackFn,
    CompleteCallbackFn,
    Dict[str, Callable]
]
