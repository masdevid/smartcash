# File: models/experiments/backbone_experiment.py
# Author: Alfrida Sabar
# Deskripsi: Template untuk eksperimen perbandingan backbone

from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import torch
import yaml

from ..backbones.cspdarknet import CSPDarknet
from ..backbones.efficientnet import EfficientNetBackbone
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.experiment_tracker import ExperimentTracker

class BackboneExperiment:
    """Template untuk eksperimen backbone"""
    
    BACKBONE_REGISTRY = {
        'cspdarknet': CSPDarknet,
        'efficientnet': EfficientNetBackbone
    }
    
    def __init__(
        self,
        config_path: str,
        experiment_name: str,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Setup experiment tracking
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            logger=self.logger
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi eksperimen"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def run_backbone_test(
        self,
        backbone_name: str,
        **backbone_kwargs
    ) -> None:
        """
        Jalankan pengujian untuk satu backbone
        """
        if backbone_name not in self.BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {backbone_name} tidak terdaftar")
            
        # Log konfirmasi backbone yang dipilih
        self.logger.info(
            f"🔬 Menguji Backbone: {backbone_name}\n"
            f"   • Parameter Tambahan: {backbone_kwargs}"
        )
        # Inisialisasi backbone
        backbone_cls = self.BACKBONE_REGISTRY[backbone_name]
        backbone = backbone_cls(**backbone_kwargs)
        
        # Setup tracking
        self.tracker.start_run(
            run_name=f"{backbone_name}_test",
            config={
                'backbone': backbone_name,
                'params': backbone_kwargs
            }
        )
        
        try:
            # Test forward pass
            self.logger.info(f"🔄 Testing {backbone_name} forward pass...")
            dummy_input = torch.randn(1, 3, 640, 640)
            features = backbone(dummy_input)
            
            # Log feature info
            for i, feat in enumerate(features):
                self.logger.info(
                    f"Stage P{i+3}: "
                    f"shape={feat.shape}, "
                    f"mean={feat.mean():.3f}, "
                    f"std={feat.std():.3f}"
                )
                
            # Log channels
            self.logger.info(
                f"Output channels: {backbone.get_output_channels()}"
            )
            
            # Log output shapes
            shapes = backbone.get_output_shapes((640, 640))
            self.logger.info(f"Output shapes: {shapes}")
            
            self.tracker.end_run(status='completed')
            
        except Exception as e:
            self.logger.error(f"❌ Test gagal: {str(e)}")
            self.tracker.end_run(status='failed')
            raise e
            
    def compare_backbones(
        self,
        input_shape: Tuple[int, int] = (640, 640)
    ) -> None:
        """
        Bandingkan semua backbone yang terdaftar
        
        Args:
            input_shape: Ukuran input untuk testing
        """
        self.logger.start("🔄 Memulai perbandingan backbone...")
        
        results = {}
        
        # Test setiap backbone
        for backbone_name in self.BACKBONE_REGISTRY:
            self.logger.info(f"Testing {backbone_name}...")
            
            # Setup untuk pengukuran performa
            backbone = self.BACKBONE_REGISTRY[backbone_name]()
            dummy_input = torch.randn(1, 3, *input_shape)
            
            try:
                # Warmup
                for _ in range(10):
                    _ = backbone(dummy_input)
                
                # Pengukuran waktu inferensi
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                times = []
                for _ in range(100):  # 100 iterasi untuk rata-rata
                    start_time.record()
                    features = backbone(dummy_input)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                
                # Analisis hasil
                avg_time = sum(times) / len(times)
                results[backbone_name] = {
                    'inference_time': avg_time,
                    'feature_channels': backbone.get_output_channels(),
                    'output_shapes': backbone.get_output_shapes(input_shape),
                    'parameter_count': sum(p.numel() for p in backbone.parameters())
                }
                
                self.logger.success(
                    f"✅ {backbone_name}:\n"
                    f"   Waktu Inferensi: {avg_time:.2f}ms\n"
                    f"   Parameter: {results[backbone_name]['parameter_count']:,}"
                )
                
            except Exception as e:
                self.logger.error(f"❌ Gagal testing {backbone_name}: {str(e)}")
                results[backbone_name] = {'error': str(e)}
                
        # Simpan hasil perbandingan
        comparison_path = Path(self.config['results_dir']) / 'backbone_comparison.yaml'
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(comparison_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        self.logger.success(f"💾 Hasil perbandingan tersimpan di {comparison_path}")
        
    def run_experiment_scenario(
        self,
        scenario: str,
        backbone_name: str
    ) -> None:
        """
        Jalankan skenario eksperimen spesifik
        
        Args:
            scenario: Nama skenario ('position' atau 'lighting')
            backbone_name: Nama backbone yang digunakan
        """
        self.logger.start(f"🚀 Memulai skenario {scenario} dengan {backbone_name}")
        
        # Load konfigurasi skenario
        scenario_config = next(
            (s for s in self.config['experiment_scenarios'] 
             if s['conditions'] == scenario),
            None
        )
        
        if not scenario_config:
            raise ValueError(f"Skenario {scenario} tidak ditemukan")
            
        if backbone_name not in self.BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {backbone_name} tidak terdaftar")
            
        # Setup eksperimen
        backbone = self.BACKBONE_REGISTRY[backbone_name]()
        
        # Jalankan evaluasi
        try:
            self.tracker.start_run(
                run_name=f"{backbone_name}_{scenario}",
                config={
                    'backbone': backbone_name,
                    'scenario': scenario,
                    **scenario_config
                }
            )
            
            # TODO: Implementasi evaluasi sesuai skenario
            # Ini akan terintegrasi dengan EvaluationHandler
            
            self.tracker.end_run(status='completed')
            
        except Exception as e:
            self.logger.error(f"❌ Eksperimen gagal: {str(e)}")
            self.tracker.end_run(status='failed')
            raise e