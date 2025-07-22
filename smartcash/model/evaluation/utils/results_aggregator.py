"""
File: smartcash/model/evaluation/utils/results_aggregator.py
Deskripsi: Results compilation dan aggregation untuk evaluation scenarios
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np

from smartcash.common.logger import get_logger

class ResultsAggregator:
    """Aggregator untuk compile dan analyze evaluation results"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger('results_aggregator')
        
        # Results storage
        self.scenario_results = {}
        self.backbone_results = {}
        self.aggregated_metrics = {}
        
        # Output configuration
        self.results_dir = Path(self.config.get('evaluation', {}).get('data', {}).get('results_dir', 'data/evaluation/results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_scenario_results(self, scenario_name: str, backbone: str, checkpoint_info: Dict[str, Any], 
                           metrics: Dict[str, Any], additional_data: Dict[str, Any] = None) -> None:
        """➕ Add results untuk specific scenario dan backbone"""
        
        result_entry = {
            'scenario_name': scenario_name,
            'backbone': backbone,
            'checkpoint_info': checkpoint_info,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'additional_data': additional_data or {}
        }
        
        # Store by scenario
        if scenario_name not in self.scenario_results:
            self.scenario_results[scenario_name] = {}
        self.scenario_results[scenario_name][backbone] = result_entry
        
        # Store by backbone
        if backbone not in self.backbone_results:
            self.backbone_results[backbone] = {}
        self.backbone_results[backbone][scenario_name] = result_entry
        
        self.logger.info(f"➕ Added results: {scenario_name} + {backbone}")
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """📊 Aggregate metrics across scenarios dan backbones"""
        
        aggregated = {
            'overall_summary': self._create_overall_summary(),
            'scenario_comparison': self._compare_scenarios(),
            'backbone_comparison': self._compare_backbones(),
            'detailed_analysis': self._create_detailed_analysis(),
            'best_configurations': self._find_best_configurations()
        }
        
        self.aggregated_metrics = aggregated
        self.logger.info("📊 Metrics aggregation complete")
        return aggregated
    
    def compare_backbones(self) -> Dict[str, Any]:
        """🏗️ Compare performance antar backbones"""
        
        comparison = {
            'backbones': list(self.backbone_results.keys()),
            'metrics_comparison': {},
            'scenario_winners': {},
            'overall_winner': None,
            'performance_summary': {}
        }
        
        # Compare key metrics
        key_metrics = ['mAP', 'accuracy', 'f1_score', 'avg_inference_time']
        
        for metric in key_metrics:
            comparison['metrics_comparison'][metric] = self._compare_metric_across_backbones(metric)
        
        # Determine winners per scenario
        for scenario_name in self.scenario_results.keys():
            comparison['scenario_winners'][scenario_name] = self._find_scenario_winner(scenario_name)
        
        # Overall winner berdasarkan aggregate performance
        comparison['overall_winner'] = self._determine_overall_winner()
        
        # Performance summary
        for backbone in comparison['backbones']:
            comparison['performance_summary'][backbone] = self._create_backbone_summary(backbone)
        
        self.logger.info(f"🏗️ Backbone comparison: {len(comparison['backbones'])} backbones analyzed")
        return comparison
    
    def generate_summary(self) -> Dict[str, Any]:
        """📋 Generate comprehensive evaluation summary"""
        
        summary = {
            'evaluation_overview': {
                'total_scenarios': len(self.scenario_results),
                'total_backbones': len(self.backbone_results),
                'total_evaluations': sum(len(scenarios) for scenarios in self.scenario_results.values()),
                'evaluation_date': datetime.now().isoformat()
            },
            'scenario_results': self.scenario_results,
            'backbone_results': self.backbone_results,
            'aggregated_metrics': self.aggregated_metrics if self.aggregated_metrics else self.aggregate_metrics(),
            'key_findings': self._extract_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        self.logger.info("📋 Comprehensive summary generated")
        return summary
    
    def export_results(self, formats: List[str] = None, detailed: bool = True) -> Dict[str, str]:
        """📤 Export results dalam multiple formats"""
        
        if formats is None:
            formats = self.config.get('output', {}).get('export_formats', ['json', 'csv'])
        
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary untuk export
        summary = self.generate_summary()
        
        # Export JSON
        if 'json' in formats:
            json_file = self.results_dir / f"evaluation_results_{timestamp}.json"
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            exported_files['json'] = str(json_file)
            self.logger.info(f"📄 JSON exported: {json_file.name}")
        
        # Export CSV
        if 'csv' in formats:
            csv_file = self.results_dir / f"evaluation_metrics_{timestamp}.csv"
            self._export_metrics_csv(csv_file, summary)
            exported_files['csv'] = str(csv_file)
            self.logger.info(f"📊 CSV exported: {csv_file.name}")
        
        # Export detailed report
        if detailed and 'report' in formats:
            report_file = self.results_dir / f"evaluation_report_{timestamp}.md"
            self._export_detailed_report(report_file, summary)
            exported_files['report'] = str(report_file)
            self.logger.info(f"📑 Report exported: {report_file.name}")
        
        return exported_files
    
    def _create_overall_summary(self) -> Dict[str, Any]:
        """📊 Create overall evaluation summary"""
        
        total_evaluations = sum(len(scenarios) for scenarios in self.scenario_results.values())
        
        # Collect all metrics
        all_maps = []
        all_accuracies = []
        all_f1_scores = []
        all_inference_times = []
        
        for scenario_results in self.scenario_results.values():
            for backbone_result in scenario_results.values():
                metrics = backbone_result['metrics']
                
                if 'mAP' in metrics:
                    all_maps.append(metrics['mAP'])
                if 'accuracy' in metrics:
                    all_accuracies.append(metrics['accuracy'])
                if 'f1_score' in metrics:
                    all_f1_scores.append(metrics['f1_score'])
                if 'avg_inference_time' in metrics:
                    all_inference_times.append(metrics['avg_inference_time'])
        
        return {
            'total_evaluations': total_evaluations,
            'scenarios_tested': len(self.scenario_results),
            'backbones_tested': len(self.backbone_results),
            'average_metrics': {
                'avg_map': np.mean(all_maps) if all_maps else 0,
                'avg_accuracy': np.mean(all_accuracies) if all_accuracies else 0,
                'avg_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0,
                'avg_inference_time': np.mean(all_inference_times) if all_inference_times else 0
            },
            'best_metrics': {
                'best_map': np.max(all_maps) if all_maps else 0,
                'best_accuracy': np.max(all_accuracies) if all_accuracies else 0,
                'best_f1_score': np.max(all_f1_scores) if all_f1_scores else 0,
                'fastest_inference': np.min(all_inference_times) if all_inference_times else 0
            }
        }
    
    def _compare_scenarios(self) -> Dict[str, Any]:
        """🎯 Compare performance across scenarios"""
        
        scenario_comparison = {}
        
        for scenario_name, backbone_results in self.scenario_results.items():
            scenario_stats = {
                'scenario_name': scenario_name,
                'backbones_tested': len(backbone_results),
                'metrics_summary': {},
                'best_backbone': None,
                'performance_variance': {}
            }
            
            # Collect metrics dari semua backbones untuk scenario ini
            scenario_metrics = defaultdict(list)
            for backbone, result in backbone_results.items():
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        scenario_metrics[metric_name].append(metric_value)
            
            # Calculate statistics per metric
            for metric_name, values in scenario_metrics.items():
                if values:
                    scenario_stats['metrics_summary'][metric_name] = {
                        'avg': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            # Find best backbone untuk scenario ini
            if 'mAP' in scenario_metrics and scenario_metrics['mAP']:
                best_map_idx = np.argmax(scenario_metrics['mAP'])
                backbone_names = list(backbone_results.keys())
                if best_map_idx < len(backbone_names):
                    scenario_stats['best_backbone'] = backbone_names[best_map_idx]
            
            scenario_comparison[scenario_name] = scenario_stats
        
        return scenario_comparison
    
    def _compare_backbones(self) -> Dict[str, Any]:
        """🏗️ Compare performance across backbones"""
        
        backbone_comparison = {}
        
        for backbone_name, scenario_results in self.backbone_results.items():
            backbone_stats = {
                'backbone_name': backbone_name,
                'scenarios_tested': len(scenario_results),
                'average_metrics': {},
                'best_scenario': None,
                'consistency_score': 0
            }
            
            # Collect metrics dari semua scenarios untuk backbone ini
            backbone_metrics = defaultdict(list)
            for scenario, result in scenario_results.items():
                metrics = result['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        backbone_metrics[metric_name].append(metric_value)
            
            # Calculate average metrics
            for metric_name, values in backbone_metrics.items():
                if values:
                    backbone_stats['average_metrics'][metric_name] = {
                        'avg': np.mean(values),
                        'std': np.std(values),
                        'consistency': 1 - (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                    }
            
            # Find best scenario untuk backbone ini
            if 'mAP' in backbone_metrics and backbone_metrics['mAP']:
                best_map_idx = np.argmax(backbone_metrics['mAP'])
                scenario_names = list(scenario_results.keys())
                if best_map_idx < len(scenario_names):
                    backbone_stats['best_scenario'] = scenario_names[best_map_idx]
            
            # Calculate consistency score
            consistency_scores = []
            for metric_name, metric_data in backbone_stats['average_metrics'].items():
                if 'consistency' in metric_data:
                    consistency_scores.append(metric_data['consistency'])
            
            backbone_stats['consistency_score'] = np.mean(consistency_scores) if consistency_scores else 0
            
            backbone_comparison[backbone_name] = backbone_stats
        
        return backbone_comparison
    
    def _create_detailed_analysis(self) -> Dict[str, Any]:
        """🔍 Create detailed analysis dengan insights"""
        
        analysis = {
            'performance_patterns': {},
            'statistical_significance': {},
            'efficiency_analysis': {},
            'scenario_difficulty': {}
        }
        
        # Performance patterns
        for scenario_name in self.scenario_results.keys():
            analysis['performance_patterns'][scenario_name] = self._analyze_scenario_patterns(scenario_name)
        
        # Efficiency analysis (accuracy vs speed)
        analysis['efficiency_analysis'] = self._analyze_efficiency()
        
        # Scenario difficulty ranking
        analysis['scenario_difficulty'] = self._rank_scenario_difficulty()
        
        return analysis
    
    def _find_best_configurations(self) -> Dict[str, Any]:
        """🏆 Find best configurations untuk different criteria"""
        
        best_configs = {
            'best_overall': None,
            'best_accuracy': None,
            'best_speed': None,
            'best_balanced': None,
            'recommendations': {}
        }
        
        all_results = []
        for scenario_results in self.scenario_results.values():
            for backbone_result in scenario_results.values():
                all_results.append(backbone_result)
        
        if not all_results:
            return best_configs
        
        # Best overall (highest mAP)
        best_map_result = max(all_results, key=lambda x: x['metrics'].get('mAP', 0))
        best_configs['best_overall'] = {
            'scenario': best_map_result['scenario_name'],
            'backbone': best_map_result['backbone'],
            'map': best_map_result['metrics'].get('mAP', 0)
        }
        
        # Best accuracy
        best_acc_result = max(all_results, key=lambda x: x['metrics'].get('accuracy', 0))
        best_configs['best_accuracy'] = {
            'scenario': best_acc_result['scenario_name'],
            'backbone': best_acc_result['backbone'],
            'accuracy': best_acc_result['metrics'].get('accuracy', 0)
        }
        
        # Best speed (lowest inference time)
        speed_results = [r for r in all_results if 'avg_inference_time' in r['metrics']]
        if speed_results:
            best_speed_result = min(speed_results, key=lambda x: x['metrics']['avg_inference_time'])
            best_configs['best_speed'] = {
                'scenario': best_speed_result['scenario_name'],
                'backbone': best_speed_result['backbone'],
                'inference_time': best_speed_result['metrics']['avg_inference_time']
            }
        
        # Best balanced (mAP * FPS)
        balanced_scores = []
        for result in all_results:
            metrics = result['metrics']
            if 'mAP' in metrics and 'avg_inference_time' in metrics:
                fps = 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
                balance_score = metrics['mAP'] * fps
                balanced_scores.append((balance_score, result))
        
        if balanced_scores:
            best_balanced_result = max(balanced_scores, key=lambda x: x[0])[1]
            best_configs['best_balanced'] = {
                'scenario': best_balanced_result['scenario_name'],
                'backbone': best_balanced_result['backbone'],
                'balance_score': balanced_scores[0][0]
            }
        
        return best_configs
    
    def _compare_metric_across_backbones(self, metric_name: str) -> Dict[str, Any]:
        """📊 Compare specific metric across backbones"""
        
        backbone_metrics = {}
        
        for backbone_name, scenario_results in self.backbone_results.items():
            metric_values = []
            for scenario_result in scenario_results.values():
                if metric_name in scenario_result['metrics']:
                    metric_values.append(scenario_result['metrics'][metric_name])
            
            if metric_values:
                backbone_metrics[backbone_name] = {
                    'avg': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'samples': len(metric_values)
                }
        
        # Rank backbones untuk metric ini
        if backbone_metrics:
            # Higher is better untuk most metrics, except inference_time
            reverse_sort = metric_name != 'avg_inference_time'
            ranked_backbones = sorted(
                backbone_metrics.items(), 
                key=lambda x: x[1]['avg'], 
                reverse=reverse_sort
            )
            
            return {
                'metric_name': metric_name,
                'backbone_metrics': backbone_metrics,
                'ranking': [backbone for backbone, _ in ranked_backbones],
                'best_backbone': ranked_backbones[0][0] if ranked_backbones else None
            }
        
        return {'metric_name': metric_name, 'backbone_metrics': {}}
    
    def _find_scenario_winner(self, scenario_name: str) -> Optional[str]:
        """🏆 Find best backbone untuk specific scenario"""
        
        if scenario_name not in self.scenario_results:
            return None
        
        backbone_results = self.scenario_results[scenario_name]
        
        # Rank berdasarkan mAP
        best_backbone = None
        best_map = 0
        
        for backbone, result in backbone_results.items():
            current_map = result['metrics'].get('mAP', 0)
            if current_map > best_map:
                best_map = current_map
                best_backbone = backbone
        
        return best_backbone
    
    def _determine_overall_winner(self) -> Optional[str]:
        """🥇 Determine overall best backbone"""
        
        backbone_scores = {}
        
        for backbone_name in self.backbone_results.keys():
            # Calculate composite score
            scores = []
            
            for scenario_results in self.backbone_results[backbone_name].values():
                metrics = scenario_results['metrics']
                
                # Weighted scoring
                score = 0
                if 'mAP' in metrics:
                    score += metrics['mAP'] * 0.4  # 40% weight
                if 'accuracy' in metrics:
                    score += metrics['accuracy'] * 0.3  # 30% weight
                if 'f1_score' in metrics:
                    score += metrics['f1_score'] * 0.2  # 20% weight
                if 'avg_inference_time' in metrics:
                    # Inverse score untuk timing (faster is better)
                    fps = 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
                    normalized_fps = min(fps / 100, 1.0)  # Normalize to 0-1
                    score += normalized_fps * 0.1  # 10% weight
                
                scores.append(score)
            
            backbone_scores[backbone_name] = np.mean(scores) if scores else 0
        
        if backbone_scores:
            return max(backbone_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _create_backbone_summary(self, backbone_name: str) -> Dict[str, Any]:
        """📋 Create summary untuk specific backbone"""
        
        if backbone_name not in self.backbone_results:
            return {}
        
        scenario_results = self.backbone_results[backbone_name]
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for result in scenario_results.values():
            for metric_name, metric_value in result['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    all_metrics[metric_name].append(metric_value)
        
        # Calculate summary statistics
        summary = {
            'backbone_name': backbone_name,
            'scenarios_tested': len(scenario_results),
            'metrics_summary': {}
        }
        
        for metric_name, values in all_metrics.items():
            if values:
                summary['metrics_summary'][metric_name] = {
                    'avg': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'consistency': 1 - (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                }
        
        return summary
    
    def _analyze_scenario_patterns(self, scenario_name: str) -> Dict[str, Any]:
        """🔍 Analyze patterns untuk specific scenario"""
        
        if scenario_name not in self.scenario_results:
            return {}
        
        backbone_results = self.scenario_results[scenario_name]
        
        # Analyze metric distributions
        metrics_analysis = {}
        
        for backbone, result in backbone_results.items():
            metrics = result['metrics']
            
            # Extract key insights
            analysis = {
                'performance_tier': 'low',  # low, medium, high
                'strengths': [],
                'weaknesses': []
            }
            
            # Performance tier berdasarkan mAP
            map_value = metrics.get('mAP', 0)
            if map_value >= 0.8:
                analysis['performance_tier'] = 'high'
            elif map_value >= 0.6:
                analysis['performance_tier'] = 'medium'
            
            # Identify strengths dan weaknesses
            if metrics.get('accuracy', 0) > 0.8:
                analysis['strengths'].append('high_accuracy')
            if metrics.get('avg_inference_time', float('inf')) < 0.1:
                analysis['strengths'].append('fast_inference')
            if metrics.get('f1_score', 0) > 0.75:
                analysis['strengths'].append('balanced_precision_recall')
            
            metrics_analysis[backbone] = analysis
        
        return {
            'scenario_name': scenario_name,
            'backbone_analysis': metrics_analysis,
            'scenario_insights': self._extract_scenario_insights(scenario_name, backbone_results)
        }
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """⚡ Analyze efficiency (accuracy vs speed tradeoffs)"""
        
        efficiency_data = []
        
        for scenario_results in self.scenario_results.values():
            for backbone_result in scenario_results.values():
                metrics = backbone_result['metrics']
                
                if 'mAP' in metrics and 'avg_inference_time' in metrics:
                    efficiency_data.append({
                        'scenario': backbone_result['scenario_name'],
                        'backbone': backbone_result['backbone'],
                        'map': metrics['mAP'],
                        'inference_time': metrics['avg_inference_time'],
                        'fps': 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0,
                        'efficiency_score': metrics['mAP'] / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
                    })
        
        if not efficiency_data:
            return {}
        
        # Find efficiency champions
        best_efficiency = max(efficiency_data, key=lambda x: x['efficiency_score'])
        fastest = min(efficiency_data, key=lambda x: x['inference_time'])
        most_accurate = max(efficiency_data, key=lambda x: x['map'])
        
        return {
            'efficiency_champion': best_efficiency,
            'speed_champion': fastest,
            'accuracy_champion': most_accurate,
            'efficiency_analysis': efficiency_data
        }
    
    def _rank_scenario_difficulty(self) -> Dict[str, Any]:
        """📊 Rank scenarios berdasarkan difficulty"""
        
        scenario_difficulty = {}
        
        for scenario_name, backbone_results in self.scenario_results.items():
            # Calculate average mAP untuk scenario ini
            maps = [result['metrics'].get('mAP', 0) for result in backbone_results.values()]
            avg_map = np.mean(maps) if maps else 0
            
            # Lower mAP indicates higher difficulty
            difficulty_score = 1 - avg_map
            
            scenario_difficulty[scenario_name] = {
                'avg_map': avg_map,
                'difficulty_score': difficulty_score,
                'backbones_tested': len(backbone_results)
            }
        
        # Rank by difficulty
        ranked_scenarios = sorted(
            scenario_difficulty.items(),
            key=lambda x: x[1]['difficulty_score'],
            reverse=True
        )
        
        return {
            'scenario_difficulty': scenario_difficulty,
            'difficulty_ranking': [scenario for scenario, _ in ranked_scenarios],
            'hardest_scenario': ranked_scenarios[0][0] if ranked_scenarios else None,
            'easiest_scenario': ranked_scenarios[-1][0] if ranked_scenarios else None
        }
    
    def _extract_key_findings(self) -> List[str]:
        """🔍 Extract key findings dari evaluation"""
        
        findings = []
        
        if not self.scenario_results:
            return ["No evaluation results available"]
        
        # Overall performance finding
        overall_summary = self._create_overall_summary()
        best_map = overall_summary['best_metrics']['best_map']
        findings.append(f"Best mAP achieved: {best_map:.3f}")
        
        # Backbone comparison finding
        backbone_comparison = self._compare_backbones()
        if backbone_comparison:
            best_backbone = max(
                backbone_comparison.items(),
                key=lambda x: x[1]['average_metrics'].get('mAP', {}).get('avg', 0)
            )[0]
            findings.append(f"Best performing backbone: {best_backbone}")
        
        # Scenario difficulty finding
        scenario_difficulty = self._rank_scenario_difficulty()
        if scenario_difficulty['hardest_scenario']:
            findings.append(f"Most challenging scenario: {scenario_difficulty['hardest_scenario']}")
        
        # Efficiency finding
        efficiency_analysis = self._analyze_efficiency()
        if efficiency_analysis and 'efficiency_champion' in efficiency_analysis:
            champion = efficiency_analysis['efficiency_champion']
            findings.append(f"Best efficiency: {champion['backbone']} on {champion['scenario']}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """💡 Generate recommendations berdasarkan results"""
        
        recommendations = []
        
        if not self.scenario_results:
            return ["Complete evaluation first to get recommendations"]
        
        # Best configuration recommendation
        best_configs = self._find_best_configurations()
        if best_configs['best_overall']:
            best = best_configs['best_overall']
            recommendations.append(
                f"For best accuracy: Use {best['backbone']} (mAP: {best['map']:.3f})"
            )
        
        if best_configs['best_speed']:
            fastest = best_configs['best_speed']
            recommendations.append(
                f"For fastest inference: Use {fastest['backbone']} ({fastest['inference_time']:.3f}s)"
            )
        
        if best_configs['best_balanced']:
            balanced = best_configs['best_balanced']
            recommendations.append(
                f"For balanced performance: Use {balanced['backbone']} on {balanced['scenario']}"
            )
        
        # Scenario-specific recommendations
        scenario_patterns = self._compare_scenarios()
        for scenario_name, scenario_data in scenario_patterns.items():
            if scenario_data['best_backbone']:
                recommendations.append(
                    f"For {scenario_name}: {scenario_data['best_backbone']} performs best"
                )
        
        return recommendations
    
    def _extract_scenario_insights(self, scenario_name: str, backbone_results: Dict[str, Any]) -> List[str]:
        """🔍 Extract insights untuk specific scenario"""
        
        insights = []
        
        # Performance range insight
        maps = [result['metrics'].get('mAP', 0) for result in backbone_results.values()]
        if maps:
            map_range = max(maps) - min(maps)
            if map_range > 0.1:
                insights.append(f"High performance variance ({map_range:.3f}) across backbones")
            else:
                insights.append("Consistent performance across backbones")
        
        # Speed consistency insight
        inference_times = [result['metrics'].get('avg_inference_time', 0) 
                          for result in backbone_results.values() 
                          if 'avg_inference_time' in result['metrics']]
        
        if inference_times:
            time_std = np.std(inference_times)
            if time_std > 0.05:
                insights.append("Significant speed differences between backbones")
            else:
                insights.append("Similar inference speeds across backbones")
        
        return insights
    
    def _export_metrics_csv(self, csv_file: Path, summary: Dict[str, Any]) -> None:
        """📊 Export metrics to CSV format"""
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Scenario', 'Backbone', 'mAP', 'Accuracy', 'F1_Score', 
                'Inference_Time', 'FPS', 'Checkpoint'
            ])
            
            # Data rows
            for scenario_name, backbone_results in self.scenario_results.items():
                for backbone, result in backbone_results.items():
                    metrics = result['metrics']
                    checkpoint_name = result['checkpoint_info'].get('filename', 'N/A')
                    
                    writer.writerow([
                        scenario_name,
                        backbone,
                        metrics.get('mAP', 0),
                        metrics.get('accuracy', 0),
                        metrics.get('f1_score', 0),
                        metrics.get('avg_inference_time', 0),
                        1.0 / metrics.get('avg_inference_time', 1) if metrics.get('avg_inference_time', 0) > 0 else 0,
                        checkpoint_name
                    ])
    
    def _export_detailed_report(self, report_file: Path, summary: Dict[str, Any]) -> None:
        """📑 Export detailed markdown report"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# SmartCash Model Evaluation Report\n\n")
            
            # Overview
            overview = summary['evaluation_overview']
            f.write(f"## Overview\n\n")
            f.write(f"- **Total Scenarios**: {overview['total_scenarios']}\n")
            f.write(f"- **Total Backbones**: {overview['total_backbones']}\n")
            f.write(f"- **Total Evaluations**: {overview['total_evaluations']}\n")
            f.write(f"- **Evaluation Date**: {overview['evaluation_date']}\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            for finding in summary['key_findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for recommendation in summary['recommendations']:
                f.write(f"- {recommendation}\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            for scenario_name, backbone_results in self.scenario_results.items():
                f.write(f"### {scenario_name}\n\n")
                
                for backbone, result in backbone_results.items():
                    metrics = result['metrics']
                    f.write(f"**{backbone}**:\n")
                    f.write(f"- mAP: {metrics.get('mAP', 0):.3f}\n")
                    f.write(f"- Accuracy: {metrics.get('accuracy', 0):.3f}\n")
                    f.write(f"- F1 Score: {metrics.get('f1_score', 0):.3f}\n")
                    
                    if 'avg_inference_time' in metrics:
                        f.write(f"- Inference Time: {metrics['avg_inference_time']:.3f}s\n")
                    f.write("\n")


# Factory functions dan utilities
def create_results_aggregator(config: Dict[str, Any] = None) -> ResultsAggregator:
    """🏭 Factory untuk ResultsAggregator"""
    return ResultsAggregator(config)

def aggregate_evaluation_results(scenario_results: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 One-liner untuk aggregate evaluation results"""
    aggregator = create_results_aggregator(config)
    
    # Add results ke aggregator
    for scenario_name, backbone_results in scenario_results.items():
        for backbone, result_data in backbone_results.items():
            aggregator.add_scenario_results(
                scenario_name, backbone, 
                result_data.get('checkpoint_info', {}),
                result_data.get('metrics', {}),
                result_data.get('additional_data', {})
            )
    
    return aggregator.generate_summary()