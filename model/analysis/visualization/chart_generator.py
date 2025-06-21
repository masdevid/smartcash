"""
File: smartcash/model/analysis/visualization/chart_generator.py
Deskripsi: Generator untuk charts dan plots dengan professional styling dan publication quality
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from smartcash.common.logger import get_logger

class ChartGenerator:
    """Generator untuk publication-quality charts dengan professional styling"""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'data/analysis/charts', logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('chart_generator')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart configuration
        chart_config = self.config.get('visualization', {}).get('charts', {})
        self.figure_size = chart_config.get('figure_size', [12, 8])
        self.dpi = chart_config.get('dpi', 150)
        self.color_palette = chart_config.get('color_palette', 'Set2')
        self.style = chart_config.get('style', 'seaborn-v0_8')
        
        # Professional styling setup
        self._setup_professional_style()
    
    def _setup_professional_style(self) -> None:
        """Setup professional matplotlib styling"""
        plt.style.use(self.style) if self.style in plt.style.available else plt.style.use('default')
        
        # Professional font settings
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        # Set color palette
        sns.set_palette(self.color_palette)
    
    def generate_performance_radar_chart(self, metrics_data: Dict[str, Dict[str, float]], 
                                       title: str = "Performance Comparison", 
                                       save_path: Optional[str] = None) -> Optional[str]:
        """Generate radar chart untuk performance comparison"""
        try:
            if not metrics_data:
                return None
            
            # Prepare data
            categories = ['Precision', 'Recall', 'F1-Score', 'mAP', 'Confidence']
            models = list(metrics_data.keys())
            
            # Create angles untuk radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi, subplot_kw=dict(projection='polar'))
            
            colors = sns.color_palette(self.color_palette, len(models))
            
            for i, (model, metrics) in enumerate(metrics_data.items()):
                values = [
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0),
                    metrics.get('map', 0),
                    metrics.get('avg_confidence', 0)
                ]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors[i], markersize=6)
                ax.fill(angles, values, alpha=0.15, color=colors[i])
            
            # Customize radar chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True, alpha=0.4)
            
            plt.title(title, fontsize=16, fontweight='bold', pad=30)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, 
                      fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_radar.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating radar chart: {str(e)}")
            return None
    
    def generate_metrics_heatmap(self, metrics_matrix: Dict[str, Dict[str, float]], 
                               title: str = "Metrics Heatmap", 
                               save_path: Optional[str] = None) -> Optional[str]:
        """Generate professional heatmap untuk metrics visualization"""
        try:
            if not metrics_matrix:
                return None
            
            # Convert to DataFrame-like structure
            import pandas as pd
            df = pd.DataFrame(metrics_matrix).T
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Create heatmap dengan custom styling
            heatmap = sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                                center=0.5, square=True, ax=ax,
                                cbar_kws={'label': 'Performance Score', 'shrink': 0.8},
                                linewidths=0.5, linecolor='white')
            
            # Customize heatmap
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Metrics', fontweight='bold')
            ax.set_ylabel('Models/Layers', fontweight='bold')
            
            # Rotate labels untuk better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save heatmap
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_heatmap.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating heatmap: {str(e)}")
            return None
    
    def generate_distribution_plots(self, distribution_data: Dict[str, Any], 
                                  chart_type: str = 'bar', 
                                  title: str = "Distribution Analysis", 
                                  save_path: Optional[str] = None) -> Optional[str]:
        """Generate distribution plots dengan multiple chart types"""
        try:
            if not distribution_data:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
            
            # Extract data
            labels = list(distribution_data.keys())
            values = list(distribution_data.values())
            colors = sns.color_palette(self.color_palette, len(labels))
            
            # Bar chart
            bars = axes[0].bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[0].set_title(f'{title} - Bar Chart', fontweight='bold', fontsize=14)
            axes[0].set_ylabel('Count', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                           f'{value:,}', ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            wedges, texts, autotexts = axes[1].pie(values, labels=labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90,
                                                  explode=[0.05] * len(labels))
            
            axes[1].set_title(f'{title} - Distribution', fontweight='bold', fontsize=14)
            
            # Enhance pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_distribution.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating distribution plots: {str(e)}")
            return None
    
    def generate_comparison_bar_chart(self, comparison_data: Dict[str, Dict[str, float]], 
                                    metric_name: str = 'Performance', 
                                    title: str = "Model Comparison", 
                                    save_path: Optional[str] = None) -> Optional[str]:
        """Generate grouped bar chart untuk model comparison"""
        try:
            if not comparison_data:
                return None
            
            # Prepare data
            models = list(comparison_data.keys())
            metrics = list(next(iter(comparison_data.values())).keys())
            
            x = np.arange(len(models))
            width = 0.8 / len(metrics)
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            colors = sns.color_palette(self.color_palette, len(metrics))
            
            for i, metric in enumerate(metrics):
                values = [comparison_data[model].get(metric, 0) for model in models]
                bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
                            color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Customize chart
            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(models)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_comparison.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating comparison chart: {str(e)}")
            return None
    
    def generate_efficiency_scatter_plot(self, efficiency_data: Dict[str, Dict[str, float]], 
                                       title: str = "Accuracy vs Speed Analysis", 
                                       save_path: Optional[str] = None) -> Optional[str]:
        """Generate scatter plot untuk accuracy vs speed efficiency"""
        try:
            if not efficiency_data:
                return None
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Extract data
            models = list(efficiency_data.keys())
            accuracies = [data.get('accuracy', 0) for data in efficiency_data.values()]
            speeds = [1.0 / max(data.get('inference_time', 0.001), 0.001) for data in efficiency_data.values()]
            
            colors = sns.color_palette(self.color_palette, len(models))
            
            # Create scatter plot dengan enhanced styling
            scatter = ax.scatter(speeds, accuracies, c=colors, s=200, alpha=0.7, 
                               edgecolors='black', linewidth=2, zorder=3)
            
            # Add model labels
            for i, model in enumerate(models):
                ax.annotate(model, (speeds[i], accuracies[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                                   alpha=0.7, edgecolor='black'))
            
            # Add efficiency frontier line (convex hull)
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([speeds, accuracies])
                hull = ConvexHull(points)
                
                # Get upper frontier points
                frontier_indices = []
                for simplex in hull.simplices:
                    for vertex in simplex:
                        if points[vertex][1] >= np.percentile(accuracies, 50):  # Upper half
                            frontier_indices.append(vertex)
                
                frontier_indices = sorted(set(frontier_indices), key=lambda x: points[x][0])
                
                if len(frontier_indices) > 1:
                    frontier_points = points[frontier_indices]
                    ax.plot(frontier_points[:, 0], frontier_points[:, 1], '--', 
                           color='red', alpha=0.6, linewidth=2, label='Efficiency Frontier', zorder=2)
            except Exception:
                pass  # Skip frontier if calculation fails
            
            # Customize plot
            ax.set_xlabel('Inference Speed (FPS)', fontweight='bold', fontsize=12)
            ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
            ax.grid(True, alpha=0.3, zorder=1)
            
            # Add quadrant labels
            mid_x, mid_y = np.median(speeds), np.median(accuracies)
            ax.axhline(y=mid_y, color='gray', linestyle=':', alpha=0.5, zorder=1)
            ax.axvline(x=mid_x, color='gray', linestyle=':', alpha=0.5, zorder=1)
            
            # Quadrant annotations
            ax.text(0.95, 0.95, 'High Accuracy\nHigh Speed', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=9, style='italic', alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
            
            if 'red' in [line.get_color() for line in ax.lines]:
                ax.legend(frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_scatter.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating scatter plot: {str(e)}")
            return None
    
    def generate_trend_line_chart(self, trend_data: Dict[str, List[float]], 
                                title: str = "Performance Trends", 
                                save_path: Optional[str] = None) -> Optional[str]:
        """Generate line chart untuk trend analysis"""
        try:
            if not trend_data:
                return None
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            colors = sns.color_palette(self.color_palette, len(trend_data))
            
            for i, (label, values) in enumerate(trend_data.items()):
                x_points = range(len(values))
                ax.plot(x_points, values, marker='o', linewidth=2.5, markersize=6,
                       label=label, color=colors[i], alpha=0.8)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(x_points, values, 1)
                    trend_line = np.poly1d(z)
                    ax.plot(x_points, trend_line(x_points), '--', 
                           color=colors[i], alpha=0.5, linewidth=1.5)
            
            # Customize chart
            ax.set_xlabel('Time Points', fontweight='bold')
            ax.set_ylabel('Performance Score', fontweight='bold')
            ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_trend.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trend chart: {str(e)}")
            return None
    
    def generate_multi_panel_dashboard(self, dashboard_data: Dict[str, Any], 
                                     title: str = "Analysis Dashboard", 
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Generate comprehensive multi-panel dashboard"""
        try:
            if not dashboard_data:
                return None
            
            fig = plt.figure(figsize=(20, 16), dpi=self.dpi)
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
            
            # Create grid layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                                top=0.93, bottom=0.05, left=0.05, right=0.95)
            
            # Panel 1: Performance Radar (top-left, span 2 columns)
            if 'performance_data' in dashboard_data:
                ax1 = fig.add_subplot(gs[0, :2], projection='polar')
                self._create_mini_radar_chart(ax1, dashboard_data['performance_data'])
            
            # Panel 2: Distribution Bar Chart (top-right, span 2 columns)
            if 'distribution_data' in dashboard_data:
                ax2 = fig.add_subplot(gs[0, 2:])
                self._create_mini_bar_chart(ax2, dashboard_data['distribution_data'])
            
            # Panel 3: Comparison Heatmap (middle, span 4 columns)
            if 'comparison_data' in dashboard_data:
                ax3 = fig.add_subplot(gs[1, :])
                self._create_mini_heatmap(ax3, dashboard_data['comparison_data'])
            
            # Panel 4: Efficiency Scatter (bottom-left, span 2 columns)
            if 'efficiency_data' in dashboard_data:
                ax4 = fig.add_subplot(gs[2, :2])
                self._create_mini_scatter(ax4, dashboard_data['efficiency_data'])
            
            # Panel 5: Summary Stats (bottom-right, span 2 columns)
            if 'summary_stats' in dashboard_data:
                ax5 = fig.add_subplot(gs[2, 2:])
                self._create_summary_table(ax5, dashboard_data['summary_stats'])
            
            # Save dashboard
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_dashboard.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating dashboard: {str(e)}")
            return None
    
    def _create_mini_radar_chart(self, ax, data: Dict) -> None:
        """Create mini radar chart untuk dashboard"""
        categories = list(next(iter(data.values())).keys())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for model, metrics in data.items():
            values = list(metrics.values()) + [list(metrics.values())[0]]
            ax.plot(angles, values, 'o-', label=model, linewidth=2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
    
    def _create_mini_bar_chart(self, ax, data: Dict) -> None:
        """Create mini bar chart untuk dashboard"""
        labels, values = list(data.keys()), list(data.values())
        bars = ax.bar(labels, values, color=sns.color_palette(self.color_palette, len(labels)))
        ax.set_title('Distribution', fontweight='bold', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value}', ha='center', va='bottom', fontsize=8)
    
    def _create_mini_heatmap(self, ax, data: Dict) -> None:
        """Create mini heatmap untuk dashboard"""
        import pandas as pd
        df = pd.DataFrame(data)
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, 
                   cbar_kws={'shrink': 0.6}, square=False)
        ax.set_title('Model Comparison Heatmap', fontweight='bold', fontsize=10)
    
    def _create_mini_scatter(self, ax, data: Dict) -> None:
        """Create mini scatter plot untuk dashboard"""
        models = list(data.keys())
        x_vals = [d.get('speed', 0) for d in data.values()]
        y_vals = [d.get('accuracy', 0) for d in data.values()]
        
        scatter = ax.scatter(x_vals, y_vals, s=100, alpha=0.7)
        
        for i, model in enumerate(models):
            ax.annotate(model, (x_vals[i], y_vals[i]), fontsize=8)
        
        ax.set_xlabel('Speed', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_title('Efficiency Analysis', fontweight='bold', fontsize=10)
    
    def _create_summary_table(self, ax, stats: Dict) -> None:
        """Create summary statistics table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = [[key.replace('_', ' ').title(), f"{value:.3f}" if isinstance(value, float) else str(value)] 
                     for key, value in stats.items()]
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center', 
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Summary Statistics', fontweight='bold', fontsize=10, pad=20)
    
    def cleanup_charts(self, keep_latest: int = 20) -> None:
        """Cleanup old chart files"""
        try:
            chart_files = list(self.output_dir.glob('*.png'))
            chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_file in chart_files[keep_latest:]:
                old_file.unlink()
            
            self.logger.info(f"🧹 Chart cleanup: kept {min(len(chart_files), keep_latest)} latest files")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during chart cleanup: {str(e)}")