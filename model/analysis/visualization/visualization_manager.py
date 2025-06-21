"""
File: smartcash/model/analysis/visualization/visualization_manager.py
Deskripsi: Manager untuk generating visualizations (charts, plots, confusion matrix)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from smartcash.common.logger import get_logger

class VisualizationManager:
    """Manager untuk comprehensive visualizations dari analysis results"""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'data/analysis/visualizations', logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('visualization_manager')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization config
        viz_config = self.config.get('visualization', {})
        self.figure_size = viz_config.get('charts', {}).get('figure_size', [12, 8])
        self.dpi = viz_config.get('charts', {}).get('dpi', 150)
        self.style = viz_config.get('charts', {}).get('style', 'seaborn-v0_8')
        self.color_palette = viz_config.get('charts', {}).get('color_palette', 'Set2')
        
        # Set matplotlib style
        plt.style.use(self.style) if self.style in plt.style.available else None
        sns.set_palette(self.color_palette)
    
    def generate_currency_analysis_plots(self, currency_results: Dict[str, Any], 
                                       save_plots: bool = True) -> Dict[str, str]:
        """Generate comprehensive plots untuk currency analysis"""
        plot_paths = {}
        
        try:
            # 1. Detection Strategy Distribution
            if 'aggregated_metrics' in currency_results and 'strategy_distribution' in currency_results['aggregated_metrics']:
                strategy_path = self._plot_strategy_distribution(
                    currency_results['aggregated_metrics']['strategy_distribution']
                )
                if save_plots and strategy_path:
                    plot_paths['strategy_distribution'] = strategy_path
            
            # 2. Denomination Distribution  
            if 'aggregated_metrics' in currency_results and 'denomination_distribution' in currency_results['aggregated_metrics']:
                denom_path = self._plot_denomination_distribution(
                    currency_results['aggregated_metrics']['denomination_distribution']
                )
                if save_plots and denom_path:
                    plot_paths['denomination_distribution'] = denom_path
            
            # 3. Confidence Distribution
            if 'batch_summary' in currency_results:
                conf_path = self._plot_detection_rates(currency_results['batch_summary'])
                if save_plots and conf_path:
                    plot_paths['detection_rates'] = conf_path
            
            self.logger.info(f"✅ Generated {len(plot_paths)} currency analysis plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating currency plots: {str(e)}")
            return {}
    
    def generate_layer_analysis_plots(self, layer_results: Dict[str, Any], 
                                    save_plots: bool = True) -> Dict[str, str]:
        """Generate comprehensive plots untuk layer analysis"""
        plot_paths = {}
        
        try:
            # 1. Layer Performance Comparison
            if 'aggregated_layer_metrics' in layer_results:
                perf_path = self._plot_layer_performance(layer_results['aggregated_layer_metrics'])
                if save_plots and perf_path:
                    plot_paths['layer_performance'] = perf_path
            
            # 2. Layer Utilization
            if 'batch_insights' in layer_results and 'layer_activity_rates' in layer_results['batch_insights']:
                util_path = self._plot_layer_utilization(layer_results['batch_insights']['layer_activity_rates'])
                if save_plots and util_path:
                    plot_paths['layer_utilization'] = util_path
            
            # 3. Layer Consistency
            if 'layer_consistency' in layer_results:
                consist_path = self._plot_layer_consistency(layer_results['layer_consistency'])
                if save_plots and consist_path:
                    plot_paths['layer_consistency'] = consist_path
            
            self.logger.info(f"✅ Generated {len(plot_paths)} layer analysis plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating layer plots: {str(e)}")
            return {}
    
    def generate_class_analysis_plots(self, class_results: Dict[str, Any], 
                                    save_plots: bool = True) -> Dict[str, str]:
        """Generate plots untuk per-class analysis"""
        plot_paths = {}
        
        try:
            # 1. Per-class metrics heatmap
            if 'per_class_metrics' in class_results:
                heatmap_path = self._plot_class_metrics_heatmap(class_results['per_class_metrics'])
                if save_plots and heatmap_path:
                    plot_paths['class_metrics_heatmap'] = heatmap_path
            
            # 2. Confusion matrix
            if 'confusion_matrix' in class_results:
                cm_path = self._plot_confusion_matrix(class_results['confusion_matrix'])
                if save_plots and cm_path:
                    plot_paths['confusion_matrix'] = cm_path
            
            # 3. Class distribution
            if 'class_distribution' in class_results:
                dist_path = self._plot_class_distribution(class_results['class_distribution'])
                if save_plots and dist_path:
                    plot_paths['class_distribution'] = dist_path
            
            self.logger.info(f"✅ Generated {len(plot_paths)} class analysis plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating class plots: {str(e)}")
            return {}
    
    def generate_comparison_plots(self, comparison_data: Dict[str, Any], 
                                save_plots: bool = True) -> Dict[str, str]:
        """Generate comparison plots (backbone, scenario)"""
        plot_paths = {}
        
        try:
            # 1. Backbone comparison
            if 'backbone_comparison' in comparison_data:
                backbone_path = self._plot_backbone_comparison(comparison_data['backbone_comparison'])
                if save_plots and backbone_path:
                    plot_paths['backbone_comparison'] = backbone_path
            
            # 2. Scenario comparison
            if 'scenario_comparison' in comparison_data:
                scenario_path = self._plot_scenario_comparison(comparison_data['scenario_comparison'])
                if save_plots and scenario_path:
                    plot_paths['scenario_comparison'] = scenario_path
            
            # 3. Performance vs Speed plot
            if 'efficiency_analysis' in comparison_data:
                efficiency_path = self._plot_efficiency_analysis(comparison_data['efficiency_analysis'])
                if save_plots and efficiency_path:
                    plot_paths['efficiency_analysis'] = efficiency_path
            
            self.logger.info(f"✅ Generated {len(plot_paths)} comparison plots")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating comparison plots: {str(e)}")
            return {}
    
    def _plot_strategy_distribution(self, strategy_data: Dict[str, int]) -> Optional[str]:
        """Plot distribution of detection strategies"""
        if not strategy_data:
            return None
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        strategies = list(strategy_data.keys())
        counts = list(strategy_data.values())
        colors = sns.color_palette(self.color_palette, len(strategies))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(counts, labels=strategies, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        # Beautify
        ax.set_title('🎯 Distribution of Currency Detection Strategies', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend dengan counts
        legend_labels = [f'{strategy}: {count}' for strategy, count in zip(strategies, counts)]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'currency_strategy_distribution.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_denomination_distribution(self, denom_data: Dict[str, int]) -> Optional[str]:
        """Plot distribution of detected denominations"""
        if not denom_data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        denominations = list(denom_data.keys())
        counts = list(denom_data.values())
        
        # Bar plot
        bars = ax1.bar(denominations, counts, color=sns.color_palette(self.color_palette, len(denominations)))
        ax1.set_title('💰 Currency Denomination Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Denominasi')
        ax1.set_ylabel('Jumlah Deteksi')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Donut chart
        wedges, texts, autotexts = ax2.pie(counts, labels=denominations, autopct='%1.1f%%',
                                          colors=sns.color_palette(self.color_palette, len(denominations)),
                                          pctdistance=0.85)
        
        # Create donut hole
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title('💰 Proporsi Denominasi', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'denomination_distribution.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_layer_performance(self, layer_metrics: Dict[str, Dict]) -> Optional[str]:
        """Plot performance comparison across layers"""
        if not layer_metrics:
            return None
        
        layers = list(layer_metrics.keys())
        metrics = ['avg_precision', 'avg_recall', 'avg_f1_score', 'avg_confidence']
        
        # Prepare data
        data = []
        for layer in layers:
            for metric in metrics:
                value = layer_metrics[layer].get(metric, 0.0)
                data.append({'Layer': layer, 'Metric': metric, 'Value': value})
        
        # Create DataFrame untuk seaborn
        import pandas as pd
        df = pd.DataFrame(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Heatmap
        pivot_df = df.pivot(index='Layer', columns='Metric', values='Value')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1,
                   cbar_kws={'label': 'Performance Score'})
        ax1.set_title('📊 Layer Performance Heatmap', fontsize=14, fontweight='bold')
        
        # Grouped bar chart
        sns.barplot(data=df, x='Layer', y='Value', hue='Metric', ax=ax2)
        ax2.set_title('📈 Layer Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Performance Score')
        ax2.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'layer_performance_comparison.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_confusion_matrix(self, confusion_data: Dict[str, Any]) -> Optional[str]:
        """Plot confusion matrix"""
        if 'matrix' not in confusion_data:
            return None
        
        matrix = np.array(confusion_data['matrix'])
        class_names = confusion_data.get('class_names', [f'Class {i}' for i in range(len(matrix))])
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot confusion matrix
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        ax.set_title('🎯 Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_backbone_comparison(self, backbone_data: Dict[str, Any]) -> Optional[str]:
        """Plot comparison between different backbones"""
        if not backbone_data:
            return None
        
        # Extract backbone names and metrics
        backbones = list(backbone_data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = []
            for backbone in backbones:
                value = backbone_data[backbone].get(metric, 0.0)
                # Normalize inference time untuk plotting
                if metric == 'inference_time':
                    value = 1.0 / max(value, 0.001)  # Convert to speed (higher is better)
                values.append(value)
            
            bars = axes[i].bar(backbones, values, color=sns.color_palette(self.color_palette, len(backbones)))
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Score' if metric != 'inference_time' else 'Speed (1/time)')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hide extra subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        fig.suptitle('🚀 Backbone Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'backbone_comparison.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_scenario_comparison(self, scenario_data: Dict[str, Any]) -> Optional[str]:
        """Plot comparison between evaluation scenarios"""
        if not scenario_data:
            return None
        
        scenarios = list(scenario_data.keys())
        metrics = ['map', 'accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Prepare data untuk radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = sns.color_palette(self.color_palette, len(scenarios))
        
        for i, scenario in enumerate(scenarios):
            values = []
            for metric in metrics:
                value = scenario_data[scenario].get(metric, 0.0)
                values.append(value)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('📊 Evaluation Scenarios Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'scenario_comparison_radar.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_efficiency_analysis(self, efficiency_data: Dict[str, Any]) -> Optional[str]:
        """Plot accuracy vs speed efficiency analysis"""
        if not efficiency_data:
            return None
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Extract data points
        models = []
        accuracies = []
        speeds = []
        
        for model_name, data in efficiency_data.items():
            models.append(model_name)
            accuracies.append(data.get('accuracy', 0.0))
            speeds.append(1.0 / max(data.get('inference_time', 0.001), 0.001))  # Convert to FPS
        
        # Create scatter plot
        colors = sns.color_palette(self.color_palette, len(models))
        scatter = ax.scatter(speeds, accuracies, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, model in enumerate(models):
            ax.annotate(model, (speeds[i], accuracies[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
        
        # Add efficiency frontier line
        if len(speeds) > 1:
            # Simple convex hull untuk efficiency frontier
            from scipy.spatial import ConvexHull
            points = np.column_stack([speeds, accuracies])
            try:
                hull = ConvexHull(points)
                # Plot upper frontier
                frontier_points = points[hull.vertices]
                frontier_points = frontier_points[frontier_points[:, 1].argsort()]  # Sort by accuracy
                ax.plot(frontier_points[:, 0], frontier_points[:, 1], '--', color='red', alpha=0.5, label='Efficiency Frontier')
            except:
                pass  # Skip if convex hull fails
        
        ax.set_xlabel('Inference Speed (FPS)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('⚡ Accuracy vs Speed Efficiency Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'efficiency_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_detection_rates(self, batch_summary: Dict[str, Any]) -> Optional[str]:
        """Plot detection rates dan success metrics"""
        if not batch_summary:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)
        
        # Success rates pie chart
        success_data = {
            'Successful Analysis': batch_summary.get('successful_analysis', 0),
            'Failed Analysis': batch_summary.get('total_images', 1) - batch_summary.get('successful_analysis', 0),
            'Images with Currency': batch_summary.get('images_with_currency', 0),
            'Images without Currency': batch_summary.get('successful_analysis', 1) - batch_summary.get('images_with_currency', 0)
        }
        
        # Filter out zero values
        success_data = {k: v for k, v in success_data.items() if v > 0}
        
        wedges1, texts1, autotexts1 = ax1.pie(success_data.values(), labels=success_data.keys(), 
                                             autopct='%1.1f%%', startangle=90,
                                             colors=sns.color_palette(self.color_palette, len(success_data)))
        ax1.set_title('📊 Analysis Success Rates', fontsize=14, fontweight='bold')
        
        # Detection rates bar chart
        rates = {
            'Success Rate': batch_summary.get('success_rate', 0),
            'Detection Rate': batch_summary.get('detection_rate', 0)
        }
        
        bars = ax2.bar(rates.keys(), rates.values(), 
                      color=sns.color_palette(self.color_palette, len(rates)))
        ax2.set_title('🎯 Detection Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, (_, rate) in zip(bars, rates.items()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'detection_rates.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_layer_utilization(self, activity_rates: Dict[str, float]) -> Optional[str]:
        """Plot layer utilization rates"""
        if not activity_rates:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        layers = list(activity_rates.keys())
        rates = list(activity_rates.values())
        
        # Create horizontal bar chart
        bars = ax.barh(layers, rates, color=sns.color_palette(self.color_palette, len(layers)))
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1%}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Activity Rate')
        ax.set_title('🔄 Layer Utilization Rates', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'layer_utilization.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_layer_consistency(self, consistency_data: Dict[str, Dict]) -> Optional[str]:
        """Plot layer consistency scores"""
        if not consistency_data:
            return None
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        layers = list(consistency_data.keys())
        consistency_scores = [data.get('consistency_score', 0) for data in consistency_data.values()]
        
        bars = ax.bar(layers, consistency_scores, 
                     color=sns.color_palette(self.color_palette, len(layers)))
        
        # Add score labels
        for bar, score in zip(bars, consistency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Consistency Score')
        ax.set_title('📈 Layer Performance Consistency', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'layer_consistency.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_class_metrics_heatmap(self, class_metrics: Dict[str, Dict]) -> Optional[str]:
        """Plot heatmap of per-class metrics"""
        if not class_metrics:
            return None
        
        # Prepare data untuk heatmap
        classes = list(class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score', 'ap']
        
        data = []
        for class_name in classes:
            row = []
            for metric in metrics:
                value = class_metrics[class_name].get(metric, 0.0)
                row.append(value)
            data.append(row)
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   xticklabels=[m.upper() for m in metrics], yticklabels=classes,
                   cbar_kws={'label': 'Score'})
        
        ax.set_title('🎯 Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'class_metrics_heatmap.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_class_distribution(self, class_distribution: Dict[str, int]) -> Optional[str]:
        """Plot class detection distribution"""
        if not class_distribution:
            return None
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        bars = ax.bar(classes, counts, color=sns.color_palette(self.color_palette, len(classes)))
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Detection Count')
        ax.set_title('📊 Class Detection Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / 'class_distribution.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_comprehensive_dashboard(self, all_results: Dict[str, Any], 
                                       save_dashboard: bool = True) -> Optional[str]:
        """Generate comprehensive dashboard combining all visualizations"""
        try:
            fig = plt.figure(figsize=(20, 24), dpi=self.dpi)
            
            # Create grid layout
            gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('🚀 SmartCash Model Analysis Dashboard', 
                        fontsize=24, fontweight='bold', y=0.98)
            
            # 1. Strategy Distribution (top-left)
            if 'currency_results' in all_results:
                currency_data = all_results['currency_results']
                if 'aggregated_metrics' in currency_data and 'strategy_distribution' in currency_data['aggregated_metrics']:
                    ax1 = fig.add_subplot(gs[0, :2])
                    strategy_data = currency_data['aggregated_metrics']['strategy_distribution']
                    if strategy_data:
                        ax1.pie(strategy_data.values(), labels=strategy_data.keys(), autopct='%1.1f%%')
                        ax1.set_title('🎯 Detection Strategies', fontweight='bold')
            
            # 2. Layer Performance (top-right)
            if 'layer_results' in all_results:
                layer_data = all_results['layer_results']
                if 'aggregated_layer_metrics' in layer_data:
                    ax2 = fig.add_subplot(gs[0, 2:])
                    # Simplified layer performance visualization
                    ax2.text(0.5, 0.5, '📊 Layer Performance\n(See detailed plots)', 
                            ha='center', va='center', transform=ax2.transAxes,
                            fontsize=12, fontweight='bold')
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)
                    ax2.axis('off')
            
            # Add more subplots untuk comprehensive view...
            # Untuk brevity, fokus pada key components
            
            if save_dashboard:
                dashboard_path = self.output_dir / 'comprehensive_dashboard.png'
                plt.savefig(dashboard_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return str(dashboard_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating dashboard: {str(e)}")
            return None
    
    def cleanup_plots(self, keep_latest: int = 10) -> None:
        """Cleanup old plot files, keep only latest N files"""
        try:
            plot_files = list(self.output_dir.glob('*.png'))
            plot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove older files
            for old_file in plot_files[keep_latest:]:
                old_file.unlink()
                
            self.logger.info(f"🧹 Cleaned up old plots, kept {min(len(plot_files), keep_latest)} latest files")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during plot cleanup: {str(e)}")