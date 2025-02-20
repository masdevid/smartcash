# File: utils/experiment_tracker.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk tracking dan logging eksperimen
# dengan dukungan visualisasi dan komparasi hasil

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger

class ExperimentTracker:
    """
    Tracker untuk eksperimen dengan dukungan:
    - Logging metrik dan parameter
    - Visualisasi hasil
    - Komparasi antar eksperimen
    - Export hasil
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments",
        logger: Optional[SmartCashLogger] = None
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.logger = logger or SmartCashLogger(__name__)
        
        # Buat direktori eksperimen
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tracking
        self.current_run = None
        self.metrics_history = []
        self.run_metadata = {}
        
        self.logger.info(
            f"🔬 Experiment tracker diinisialisasi:\n"
            f"   Nama: {experiment_name}\n"
            f"   Output: {self.experiment_dir}"
        )
        
    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> None:
        """
        Mulai run eksperimen baru
        Args:
            run_name: Nama run (opsional)
            config: Konfigurasi eksperimen
        """
        # Generate run name jika tidak dispesifikasi
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
            
        self.current_run = run_name
        
        # Setup direktori untuk run ini
        run_dir = self.experiment_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Save config
        if config:
            with open(run_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
                
        # Initialize tracking untuk run ini
        self.metrics_history = []
        self.run_metadata = {
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'config': config
        }
        
        self.logger.info(
            f"🏃 Memulai run: {run_name}\n"
            f"   Config: {config if config else 'default'}"
        )
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrik evaluasi
        Args:
            metrics: Dict metrik
            step: Step/epoch saat ini
        """
        if self.current_run is None:
            raise ValueError("Tidak ada run yang aktif")
            
        # Add step info
        metrics_entry = {
            'step': step or len(self.metrics_history),
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Save metrics
        run_dir = self.experiment_dir / self.current_run
        with open(run_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        # Log ke console
        metrics_str = ", ".join([
            f"{k}: {v:.4f}" for k, v in metrics.items()
        ])
        self.logger.metric(
            f"📊 Step {metrics_entry['step']}: {metrics_str}"
        )
        
    def end_run(
        self,
        status: str = 'completed'
    ) -> None:
        """
        Akhiri run eksperimen saat ini
        Args:
            status: Status akhir run
        """
        if self.current_run is None:
            return
            
        # Update metadata
        self.run_metadata.update({
            'end_time': datetime.now().isoformat(),
            'status': status,
            'n_steps': len(self.metrics_history)
        })
        
        # Save metadata
        run_dir = self.experiment_dir / self.current_run
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(self.run_metadata, f, indent=2)
            
        self.logger.success(
            f"✨ Run {self.current_run} selesai\n"
            f"   Status: {status}\n"
            f"   Steps: {self.run_metadata['n_steps']}"
        )
        
        self.current_run = None
        
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Plot metrik dari run saat ini
        Args:
            metrics: List metrik yang akan diplot
            save: Simpan plot ke file
            show: Tampilkan plot
        """
        if not self.metrics_history:
            self.logger.warning("⚠️ Tidak ada metrik untuk diplot")
            return
            
        # Convert ke DataFrame
        df = pd.DataFrame(self.metrics_history)
        
        if metrics is None:
            # Plot semua metrik kecuali step dan timestamp
            metrics = [col for col in df.columns 
                      if col not in ['step', 'timestamp']]
            
        # Setup plot
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            n_metrics, 1,
            figsize=(10, 4*n_metrics),
            squeeze=False
        )
        
        # Plot setiap metrik
        for i, metric in enumerate(metrics):
            ax = axes[i, 0]
            
            df.plot(
                x='step',
                y=metric,
                ax=ax,
                legend=True,
                marker='o'
            )
            
            ax.set_title(f'Evolution of {metric}')
            ax.grid(True)
            
        plt.tight_layout()
        
        # Save plot
        if save and self.current_run:
            run_dir = self.experiment_dir / self.current_run
            plt.savefig(run_dir / 'metrics.png')
            self.logger.info("💾 Plot metrik disimpan")
            
        if show:
            plt.show()
            
    def compare_runs(
        self,
        metric: str,
        runs: Optional[List[str]] = None,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Bandingkan metrik antar runs
        Args:
            metric: Metrik yang akan dibandingkan
            runs: List run yang akan dibandingkan
            save: Simpan plot ke file
            show: Tampilkan plot
        """
        if runs is None:
            # Compare semua run yang ada
            runs = [d.name for d in self.experiment_dir.iterdir() 
                   if d.is_dir()]
            
        # Collect data dari setiap run
        data = []
        for run in runs:
            run_dir = self.experiment_dir / run
            metrics_file = run_dir / 'metrics.json'
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    run_metrics = json.load(f)
                    
                df = pd.DataFrame(run_metrics)
                if metric in df.columns:
                    df['run'] = run
                    data.append(df)
                    
        if not data:
            self.logger.warning(
                f"⚠️ Tidak ada data untuk metrik {metric}"
            )
            return
            
        # Combine data
        df = pd.concat(data, ignore_index=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x='step',
            y=metric,
            hue='run',
            marker='o'
        )
        
        plt.title(f'Comparison of {metric} across runs')
        plt.grid(True)
        
        # Save plot
        if save:
            plt.savefig(self.experiment_dir / f'comparison_{metric}.png')
            self.logger.info("💾 Plot perbandingan disimpan")
            
        if show:
            plt.show()
            
    def export_results(
        self,
        format: str = 'csv'
    ) -> None:
        """
        Export hasil eksperimen
        Args:
            format: Format export ('csv' atau 'json')
        """
        # Collect results dari semua run
        results = []
        
        for run_dir in self.experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            run_results = {'run_name': run_dir.name}
            
            # Load metadata
            metadata_file = run_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    run_results.update(metadata)
                    
            # Load final metrics
            metrics_file = run_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metrics:
                        # Ambil metrik terakhir
                        final_metrics = metrics[-1]
                        run_results['final_metrics'] = final_metrics
                        
            results.append(run_results)
            
        if not results:
            self.logger.warning("⚠️ Tidak ada hasil untuk diexport")
            return
            
        # Export
        output_path = self.experiment_dir / f'results.{format}'
        
        if format == 'csv':
            pd.DataFrame(results).to_csv(output_path, index=False)
        else:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        self.logger.success(
            f"📤 Hasil eksperimen diexport ke {output_path}"
        )