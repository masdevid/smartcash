# File: smartcash/handlers/model/observers/experiment_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer ringkas untuk monitoring eksperimen model

from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.model.observers.model_observer_interface import ModelObserverInterface

class ExperimentObserver(ModelObserverInterface):
    """
    Observer untuk monitoring eksperimen model - versi ringkas.
    Mencatat eksperimen dan menyimpan hasil ke disk.
    """
    
    def __init__(
        self, 
        output_dir: str = "runs/train/experiments",
        logger: Optional[SmartCashLogger] = None,
        save_results: bool = True
    ):
        """
        Inisialisasi experiment observer.
        
        Args:
            output_dir: Direktori output untuk hasil eksperimen
            logger: Custom logger (opsional)
            save_results: Flag untuk menyimpan hasil ke file
        """
        super().__init__(name="experiment_observer", logger=logger)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_results = save_results
        
        # Data tracking utama
        self.experiment_data = {
            'start_time': None,
            'end_time': None,
            'type': None,
            'results': None,
            'errors': []
        }
        
        # Register event handler utama
        self.register('experiment_start', self._handle_experiment_start)
        self.register('experiment_end', self._handle_experiment_end)
        self.register('experiment_error', self._handle_experiment_error)
        
        self.logger.info(f"🔍 ExperimentObserver diinisialisasi, output: {self.output_dir}")
    
    def _handle_experiment_start(self, data: Dict[str, Any]):
        """
        Handle experiment start event.
        
        Args:
            data: Data event
        """
        # Reset data untuk eksperimen baru
        self.experiment_data = {
            'start_time': time.time(),
            'type': data.get('type', 'unknown'),
            'parameters': {k: v for k, v in data.items() if k != 'type'},
            'results': None,
            'errors': []
        }
        
        # Log info
        self.logger.info(f"🧪 Eksperimen dimulai: {self.experiment_data['type']}")
        
        # Simpan file inisialisasi jika diminta
        if self.save_results:
            self._save_experiment_state("start")
    
    def _handle_experiment_end(self, data: Dict[str, Any]):
        """
        Handle experiment end event.
        
        Args:
            data: Data event
        """
        # Update data
        self.experiment_data['end_time'] = time.time()
        self.experiment_data['results'] = data.get('results')
        
        # Hitung durasi
        duration = self.experiment_data['end_time'] - self.experiment_data['start_time']
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
        
        # Log hasil
        self.logger.success(
            f"✅ Eksperimen selesai: {self.experiment_data['type']} - Durasi: {h}h {m}m {s}s"
        )
        
        # Save final results
        if self.save_results:
            self._save_experiment_state("end")
    
    def _handle_experiment_error(self, data: Dict[str, Any]):
        """
        Handle experiment error event.
        
        Args:
            data: Data event
        """
        # Catat error
        error_entry = {
            'timestamp': time.time(),
            'error': data.get('error', 'Unknown error')
        }
        
        self.experiment_data['errors'].append(error_entry)
        
        # Log error
        self.logger.error(f"❌ Error eksperimen: {error_entry['error']}")
        
        # Save error state
        if self.save_results:
            self._save_experiment_state("error")
    
    def _save_experiment_state(self, state: str):
        """
        Simpan state eksperimen ke file.
        
        Args:
            state: State eksperimen ('start', 'end', 'error')
        """
        try:
            # Buat filename
            experiment_type = self.experiment_data.get('type', 'unknown')
            timestamp = int(time.time())
            filename = f"{experiment_type}_{state}_{timestamp}.json"
            
            # Simpan ke file
            output_path = self.output_dir / filename
            
            # Convert data untuk JSON
            experiment_data_clean = self._clean_for_json(self.experiment_data)
            
            with open(output_path, 'w') as f:
                json.dump(experiment_data_clean, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyimpan state eksperimen: {str(e)}")
    
    def _clean_for_json(self, data):
        """
        Bersihkan data untuk disimpan ke JSON.
        
        Args:
            data: Data untuk dibersihkan
            
        Returns:
            Data yang sudah dibersihkan
        """
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()
                   if k not in ['model', 'train_loader', 'val_loader', 'test_loader']}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, (Path, set)):
            return str(data)
        elif hasattr(data, 'tolist'):
            return data.tolist()
        elif hasattr(data, 'item'):
            return data.item()
        else:
            return data
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Dapatkan ringkasan eksperimen.
        
        Returns:
            Dict ringkasan eksperimen
        """
        if not self.experiment_data:
            return {}
            
        # Buat ringkasan dasar
        summary = {
            'type': self.experiment_data.get('type', 'unknown'),
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                      time.localtime(self.experiment_data.get('start_time', 0))),
            'status': 'Running' if not self.experiment_data.get('end_time') else 'Completed'
        }
        
        # Tambahkan durasi jika sudah selesai
        if self.experiment_data.get('end_time'):
            duration = self.experiment_data['end_time'] - self.experiment_data['start_time']
            summary['duration_seconds'] = duration
            
        # Tambahkan error jika ada
        if self.experiment_data.get('errors'):
            summary['has_errors'] = True
            
        return summary