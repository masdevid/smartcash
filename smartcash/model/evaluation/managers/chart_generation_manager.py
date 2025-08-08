"""
Chart generation manager for backbone comparison visualizations.
Handles generating comparison charts for different metrics across backbones and scenarios.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger


class ChartGenerationManager:
    """Manages chart generation for backbone comparison evaluations"""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'data/evaluation/charts'):
        self.logger = get_logger('chart_generation')
        self.config = config or {}
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart configuration
        self.metrics_to_chart = ['accuracy', 'precision', 'recall', 'f1', 'map50', 'inference_time_avg']
        self.metrics_display_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision', 
            'recall': 'Recall',
            'f1': 'F1-Score',
            'map50': 'mAP@0.5',
            'inference_time_avg': 'Inference Time (s)'
        }
    
    def generate_backbone_comparison_charts(self, comparison_result: Dict[str, Any]) -> List[str]:
        """Generate all comparison charts for backbone evaluation results"""
        
        self.logger.info("📊 Generating backbone comparison charts...")
        
        all_results = comparison_result.get('results', {})
        scenarios = comparison_result.get('scenarios_evaluated', [])
        backbones = comparison_result.get('backbones_evaluated', [])
        
        if not all_results or not scenarios or not backbones:
            self.logger.warning("⚠️ Insufficient data for chart generation")
            return []
        
        chart_files = []
        
        try:
            # Prepare comparison data
            comparison_data = self._prepare_comparison_data(all_results, scenarios, backbones)
            
            # Generate individual backbone charts (2 backbones x 2 scenarios = 4 charts)
            individual_charts = self._generate_individual_backbone_charts(comparison_data)
            chart_files.extend(individual_charts)
            
            # Generate side-by-side comparison charts (2 charts: one per backbone)  
            comparison_charts = self._generate_scenario_comparison_charts(comparison_data)
            chart_files.extend(comparison_charts)
            
            # Generate comprehensive overview chart
            overview_charts = self._generate_overview_charts(comparison_data)
            chart_files.extend(overview_charts)
            
            self.logger.info(f"✅ Generated {len(chart_files)} comparison charts in {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Chart generation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return chart_files
    
    def _prepare_comparison_data(self, all_results: Dict[str, Any], scenarios: List[str], 
                               backbones: List[str]) -> Dict[str, Any]:
        """Prepare data structure for chart generation"""
        
        comparison_data = {
            'backbones': backbones,
            'scenarios': scenarios,
            'metrics': self.metrics_to_chart,
            'results_by_scenario': {},
            'results_by_backbone': {},
            'empty_metrics': {metric: 0.0 for metric in self.metrics_to_chart}
        }
        
        # Organize data by scenario
        for scenario in scenarios:
            comparison_data['results_by_scenario'][scenario] = {}
            for backbone in backbones:
                backbone_result = all_results.get(backbone, {})
                scenario_result = backbone_result.get(scenario, {})
                
                if scenario_result.get('status') == 'success':
                    comparison_data['results_by_scenario'][scenario][backbone] = scenario_result.get('metrics', {})
                else:
                    # Empty metrics for missing/failed backbones
                    comparison_data['results_by_scenario'][scenario][backbone] = comparison_data['empty_metrics'].copy()
        
        # Organize data by backbone
        for backbone in backbones:
            comparison_data['results_by_backbone'][backbone] = {}
            for scenario in scenarios:
                backbone_result = all_results.get(backbone, {})
                scenario_result = backbone_result.get(scenario, {})
                
                if scenario_result.get('status') == 'success':
                    comparison_data['results_by_backbone'][backbone][scenario] = scenario_result.get('metrics', {})
                else:
                    comparison_data['results_by_backbone'][backbone][scenario] = comparison_data['empty_metrics'].copy()
        
        return comparison_data
    
    def _generate_individual_backbone_charts(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate individual charts for each backbone-scenario combination"""
        
        chart_files = []
        
        for backbone in comparison_data['backbones']:
            for scenario in comparison_data['scenarios']:
                try:
                    chart_file = self._generate_backbone_scenario_chart(backbone, scenario, comparison_data)
                    if chart_file:
                        chart_files.append(chart_file)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate chart for {backbone}-{scenario}: {e}")
        
        return chart_files
    
    def _generate_scenario_comparison_charts(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate side-by-side comparison charts for each backbone"""
        
        chart_files = []
        
        for backbone in comparison_data['backbones']:
            try:
                chart_file = self._generate_backbone_scenario_comparison_chart(backbone, comparison_data)
                if chart_file:
                    chart_files.append(chart_file)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate scenario comparison chart for {backbone}: {e}")
        
        return chart_files
    
    def _generate_overview_charts(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate comprehensive overview charts"""
        
        chart_files = []
        
        try:
            # Overall backbone comparison chart
            overview_chart = self._generate_comprehensive_overview_chart(comparison_data)
            if overview_chart:
                chart_files.append(overview_chart)
                
            # Metric-specific comparison charts  
            for metric in comparison_data['metrics']:
                metric_chart = self._generate_metric_specific_chart(metric, comparison_data)
                if metric_chart:
                    chart_files.append(metric_chart)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate overview charts: {e}")
        
        return chart_files
    
    def _generate_backbone_scenario_chart(self, backbone: str, scenario: str, 
                                        comparison_data: Dict[str, Any]) -> Optional[str]:
        """Generate chart for specific backbone-scenario combination"""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get data for this backbone-scenario
            scenario_data = comparison_data['results_by_scenario'].get(scenario, {})
            backbone_metrics = scenario_data.get(backbone, comparison_data['empty_metrics'])
            
            # Prepare data
            metrics = [self.metrics_display_names.get(m, m) for m in self.metrics_to_chart]
            values = [backbone_metrics.get(m, 0.0) for m in self.metrics_to_chart]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(metrics, values, color='steelblue', alpha=0.7)
            
            # Customize chart
            ax.set_title(f'{backbone.upper()} - {scenario.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f'{backbone}_{scenario}_metrics.png'
            chart_path = self.output_dir / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Generated chart: {chart_filename}")
            return str(chart_path)
            
        except ImportError:
            self.logger.warning("⚠️ Matplotlib not available for chart generation")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error generating {backbone}-{scenario} chart: {e}")
            return None
    
    def _generate_backbone_scenario_comparison_chart(self, backbone: str, 
                                                   comparison_data: Dict[str, Any]) -> Optional[str]:
        """Generate side-by-side scenario comparison chart for a backbone"""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            backbone_data = comparison_data['results_by_backbone'].get(backbone, {})
            scenarios = comparison_data['scenarios']
            
            # Prepare data
            x = np.arange(len(self.metrics_to_chart))
            width = 0.35
            
            scenario_1_values = [backbone_data.get(scenarios[0], {}).get(m, 0.0) for m in self.metrics_to_chart]
            scenario_2_values = [backbone_data.get(scenarios[1], {}).get(m, 0.0) for m in self.metrics_to_chart] if len(scenarios) > 1 else [0] * len(self.metrics_to_chart)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars1 = ax.bar(x - width/2, scenario_1_values, width, label=scenarios[0].replace('_', ' ').title(), alpha=0.8)
            bars2 = ax.bar(x + width/2, scenario_2_values, width, label=scenarios[1].replace('_', ' ').title() if len(scenarios) > 1 else 'N/A', alpha=0.8)
            
            # Customize chart
            ax.set_title(f'{backbone.upper()} - Scenario Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Metric Value', fontsize=12)
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([self.metrics_display_names.get(m, m) for m in self.metrics_to_chart], rotation=45)
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f'{backbone}_scenario_comparison.png'
            chart_path = self.output_dir / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Generated comparison chart: {chart_filename}")
            return str(chart_path)
            
        except ImportError:
            self.logger.warning("⚠️ Matplotlib not available for chart generation")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error generating {backbone} scenario comparison chart: {e}")
            return None
    
    def _generate_comprehensive_overview_chart(self, comparison_data: Dict[str, Any]) -> Optional[str]:
        """Generate comprehensive overview chart comparing all backbones across all scenarios"""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            backbones = comparison_data['backbones']
            scenarios = comparison_data['scenarios']
            
            # Create subplots for each metric
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(self.metrics_to_chart):
                ax = axes[i]
                
                # Prepare data for this metric
                x = np.arange(len(backbones))
                width = 0.35
                
                scenario_values = []
                for scenario in scenarios:
                    values = []
                    for backbone in backbones:
                        scenario_data = comparison_data['results_by_scenario'].get(scenario, {})
                        backbone_metrics = scenario_data.get(backbone, comparison_data['empty_metrics'])
                        values.append(backbone_metrics.get(metric, 0.0))
                    scenario_values.append(values)
                
                # Plot bars
                for j, (scenario, values) in enumerate(zip(scenarios, scenario_values)):
                    offset = (j - 0.5) * width
                    bars = ax.bar(x + offset, values, width, label=scenario.replace('_', ' ').title(), alpha=0.8)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
                
                # Customize subplot
                ax.set_title(self.metrics_display_names.get(metric, metric), fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(backbones)
                if i == 0:  # Only show legend on first subplot
                    ax.legend()
            
            plt.suptitle('Comprehensive Backbone Comparison Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save chart
            chart_filename = 'backbone_comparison_overview.png'
            chart_path = self.output_dir / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Generated overview chart: {chart_filename}")
            return str(chart_path)
            
        except ImportError:
            self.logger.warning("⚠️ Matplotlib not available for chart generation")
            return None  
        except Exception as e:
            self.logger.error(f"❌ Error generating overview chart: {e}")
            return None
    
    def _generate_metric_specific_chart(self, metric: str, comparison_data: Dict[str, Any]) -> Optional[str]:
        """Generate chart focusing on a specific metric across all backbones and scenarios"""
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            backbones = comparison_data['backbones']
            scenarios = comparison_data['scenarios']
            
            # Prepare data matrix
            data = []
            labels = []
            
            for backbone in backbones:
                for scenario in scenarios:
                    scenario_data = comparison_data['results_by_scenario'].get(scenario, {})
                    backbone_metrics = scenario_data.get(backbone, comparison_data['empty_metrics'])
                    value = backbone_metrics.get(metric, 0.0)
                    data.append(value)
                    labels.append(f'{backbone}\\n{scenario.replace("_", " ").title()}')
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(data)), data, alpha=0.7, 
                         color=['steelblue' if i % 2 == 0 else 'lightcoral' for i in range(len(data))])
            
            # Customize chart
            metric_display = self.metrics_display_names.get(metric, metric)
            ax.set_title(f'{metric_display} Comparison Across Backbones and Scenarios', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_display, fontsize=12)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f'{metric}_detailed_comparison.png'
            chart_path = self.output_dir / chart_filename
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Generated {metric} chart: {chart_filename}")
            return str(chart_path)
            
        except ImportError:
            self.logger.warning("⚠️ Matplotlib not available for chart generation")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error generating {metric} chart: {e}")
            return None
    
    def get_chart_summary(self) -> Dict[str, Any]:
        """Get summary of generated charts"""
        
        chart_files = list(self.output_dir.glob('*.png'))
        
        return {
            'output_directory': str(self.output_dir),
            'total_charts': len(chart_files),
            'chart_files': [str(f) for f in chart_files],
            'chart_types': {
                'individual_backbone_scenario': len([f for f in chart_files if '_metrics.png' in f.name]),
                'scenario_comparison': len([f for f in chart_files if '_scenario_comparison.png' in f.name]),
                'overview': len([f for f in chart_files if 'overview.png' in f.name]),
                'metric_specific': len([f for f in chart_files if '_detailed_comparison.png' in f.name])
            }
        }


# Factory functions
def create_chart_generation_manager(config: Dict[str, Any] = None, output_dir: str = 'data/evaluation/charts') -> ChartGenerationManager:
    """Factory function to create chart generation manager"""
    return ChartGenerationManager(config, output_dir)


def generate_backbone_comparison_charts(comparison_result: Dict[str, Any], output_dir: str = 'data/evaluation/charts') -> List[str]:
    """One-liner function to generate backbone comparison charts"""
    manager = create_chart_generation_manager(output_dir=output_dir)
    return manager.generate_backbone_comparison_charts(comparison_result)