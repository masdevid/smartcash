"""
File: smartcash/model/analysis/utils/data_aggregator.py
Deskripsi: Aggregator untuk combining dan consolidating analysis data dari multiple sources
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from smartcash.common.logger import get_logger

@dataclass
class AggregatedResult:
    """Container untuk aggregated analysis results"""
    source_count: int
    aggregation_method: str
    confidence_level: float
    metadata: Dict[str, Any]

class DataAggregator:
    """Aggregator untuk comprehensive data consolidation dari multiple analysis sources"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('data_aggregator')
        
    def aggregate_currency_results(self, currency_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple currency analysis results"""
        if not currency_results_list:
            return {'error': 'No currency results to aggregate'}
        
        try:
            # Filter valid results
            valid_results = [r for r in currency_results_list if 'error' not in r]
            if not valid_results:
                return {'error': 'No valid currency results found'}
            
            # Aggregate strategy distributions
            combined_strategies = {}
            combined_denominations = {}
            
            for result in valid_results:
                metrics = result.get('aggregated_metrics', {})
                
                # Aggregate strategies
                for strategy, count in metrics.get('strategy_distribution', {}).items():
                    combined_strategies[strategy] = combined_strategies.get(strategy, 0) + count
                
                # Aggregate denominations
                for denom, count in metrics.get('denomination_distribution', {}).items():
                    combined_denominations[denom] = combined_denominations.get(denom, 0) + count
            
            # Calculate aggregated rates
            validation_rates = [r.get('aggregated_metrics', {}).get('avg_validation_rate', 0) 
                              for r in valid_results]
            boost_rates = [r.get('aggregated_metrics', {}).get('avg_boost_rate', 0) 
                          for r in valid_results]
            
            return {
                'aggregation_type': 'currency_analysis',
                'source_count': len(valid_results),
                'combined_strategy_distribution': combined_strategies,
                'combined_denomination_distribution': combined_denominations,
                'aggregated_validation_rate': np.mean(validation_rates) if validation_rates else 0.0,
                'aggregated_boost_rate': np.mean(boost_rates) if boost_rates else 0.0,
                'total_currency_detections': sum(combined_strategies.values()),
                'confidence_metrics': self._calculate_aggregation_confidence(valid_results, 'currency')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating currency results: {str(e)}")
            return {'error': str(e)}
    
    def aggregate_layer_results(self, layer_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple layer analysis results"""
        if not layer_results_list:
            return {'error': 'No layer results to aggregate'}
        
        try:
            valid_results = [r for r in layer_results_list if 'error' not in r]
            if not valid_results:
                return {'error': 'No valid layer results found'}
            
            # Aggregate layer metrics
            all_layer_metrics = {}
            layer_names = set()
            
            for result in valid_results:
                metrics = result.get('aggregated_layer_metrics', {})
                for layer_name, layer_data in metrics.items():
                    layer_names.add(layer_name)
                    if layer_name not in all_layer_metrics:
                        all_layer_metrics[layer_name] = []
                    all_layer_metrics[layer_name].append(layer_data)
            
            # Calculate consolidated metrics per layer
            consolidated_metrics = {}
            for layer_name in layer_names:
                layer_data_list = all_layer_metrics[layer_name]
                
                consolidated_metrics[layer_name] = {
                    'consolidated_precision': np.mean([d.get('avg_precision', 0) for d in layer_data_list]),
                    'consolidated_recall': np.mean([d.get('avg_recall', 0) for d in layer_data_list]),
                    'consolidated_f1_score': np.mean([d.get('avg_f1_score', 0) for d in layer_data_list]),
                    'consolidated_confidence': np.mean([d.get('avg_confidence', 0) for d in layer_data_list]),
                    'consistency_score': 1.0 - np.std([d.get('avg_confidence', 0) for d in layer_data_list]),
                    'data_points': len(layer_data_list)
                }
            
            # Calculate overall layer insights
            layer_activity_aggregated = self._aggregate_layer_activity(valid_results)
            
            return {
                'aggregation_type': 'layer_analysis',
                'source_count': len(valid_results),
                'consolidated_layer_metrics': consolidated_metrics,
                'aggregated_layer_activity': layer_activity_aggregated,
                'layer_consistency_overview': self._calculate_layer_consistency_overview(consolidated_metrics),
                'confidence_metrics': self._calculate_aggregation_confidence(valid_results, 'layer')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating layer results: {str(e)}")
            return {'error': str(e)}
    
    def aggregate_class_results(self, class_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple class analysis results"""
        if not class_results_list:
            return {'error': 'No class results to aggregate'}
        
        try:
            valid_results = [r for r in class_results_list if 'error' not in r]
            if not valid_results:
                return {'error': 'No valid class results found'}
            
            # Aggregate per-class metrics
            all_class_metrics = {}
            class_names = set()
            
            for result in valid_results:
                per_class = result.get('per_class_metrics', {})
                for class_id, class_data in per_class.items():
                    class_name = getattr(class_data, 'class_name', f'Class_{class_id}')
                    class_names.add(class_name)
                    
                    if class_name not in all_class_metrics:
                        all_class_metrics[class_name] = []
                    
                    all_class_metrics[class_name].append({
                        'precision': getattr(class_data, 'precision', 0),
                        'recall': getattr(class_data, 'recall', 0),
                        'f1_score': getattr(class_data, 'f1_score', 0),
                        'total_detections': getattr(class_data, 'total_detections', 0)
                    })
            
            # Calculate consolidated class metrics
            consolidated_class_metrics = {}
            for class_name in class_names:
                class_data_list = all_class_metrics[class_name]
                
                consolidated_class_metrics[class_name] = {
                    'consolidated_precision': np.mean([d['precision'] for d in class_data_list]),
                    'consolidated_recall': np.mean([d['recall'] for d in class_data_list]),
                    'consolidated_f1_score': np.mean([d['f1_score'] for d in class_data_list]),
                    'total_detections_sum': sum([d['total_detections'] for d in class_data_list]),
                    'precision_std': np.std([d['precision'] for d in class_data_list]),
                    'data_points': len(class_data_list)
                }
            
            # Aggregate class distributions
            combined_class_distribution = self._aggregate_class_distributions(valid_results)
            
            return {
                'aggregation_type': 'class_analysis',
                'source_count': len(valid_results),
                'consolidated_class_metrics': consolidated_class_metrics,
                'combined_class_distribution': combined_class_distribution,
                'class_balance_overview': self._calculate_aggregated_balance_overview(valid_results),
                'confidence_metrics': self._calculate_aggregation_confidence(valid_results, 'class')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating class results: {str(e)}")
            return {'error': str(e)}
    
    def aggregate_comparative_results(self, comparative_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple comparative analysis results"""
        if not comparative_results_list:
            return {'error': 'No comparative results to aggregate'}
        
        try:
            valid_results = [r for r in comparative_results_list if 'error' not in r]
            if not valid_results:
                return {'error': 'No valid comparative results found'}
            
            # Aggregate backbone comparisons
            backbone_aggregated = self._aggregate_backbone_comparisons(valid_results)
            
            # Aggregate scenario comparisons
            scenario_aggregated = self._aggregate_scenario_comparisons(valid_results)
            
            # Aggregate efficiency analysis
            efficiency_aggregated = self._aggregate_efficiency_analysis(valid_results)
            
            return {
                'aggregation_type': 'comparative_analysis',
                'source_count': len(valid_results),
                'aggregated_backbone_comparison': backbone_aggregated,
                'aggregated_scenario_comparison': scenario_aggregated,
                'aggregated_efficiency_analysis': efficiency_aggregated,
                'meta_insights': self._generate_meta_insights(backbone_aggregated, scenario_aggregated),
                'confidence_metrics': self._calculate_aggregation_confidence(valid_results, 'comparative')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error aggregating comparative results: {str(e)}")
            return {'error': str(e)}
    
    def create_comprehensive_aggregation(self, all_analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive aggregation dari all analysis types"""
        try:
            # Separate results by type
            currency_results = [r.get('currency_analysis', {}) for r in all_analysis_results if r.get('currency_analysis')]
            layer_results = [r.get('layer_analysis', {}) for r in all_analysis_results if r.get('layer_analysis')]
            class_results = [r.get('class_analysis', {}) for r in all_analysis_results if r.get('class_analysis')]
            comparative_results = [r.get('comparative_analysis', {}) for r in all_analysis_results if r.get('comparative_analysis')]
            
            # Aggregate each type
            aggregated_currency = self.aggregate_currency_results(currency_results) if currency_results else {}
            aggregated_layer = self.aggregate_layer_results(layer_results) if layer_results else {}
            aggregated_class = self.aggregate_class_results(class_results) if class_results else {}
            aggregated_comparative = self.aggregate_comparative_results(comparative_results) if comparative_results else {}
            
            # Create meta-analysis
            meta_analysis = self._create_meta_analysis(
                aggregated_currency, aggregated_layer, aggregated_class, aggregated_comparative
            )
            
            return {
                'comprehensive_aggregation': {
                    'total_sources': len(all_analysis_results),
                    'aggregated_currency_analysis': aggregated_currency,
                    'aggregated_layer_analysis': aggregated_layer,
                    'aggregated_class_analysis': aggregated_class,
                    'aggregated_comparative_analysis': aggregated_comparative,
                    'meta_analysis': meta_analysis,
                    'aggregation_summary': self._generate_aggregation_summary(all_analysis_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating comprehensive aggregation: {str(e)}")
            return {'error': str(e)}
    
    def _aggregate_layer_activity(self, layer_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate layer activity rates across results"""
        all_activity_rates = {}
        
        for result in layer_results:
            batch_insights = result.get('batch_insights', {})
            activity_rates = batch_insights.get('layer_activity_rates', {})
            
            for layer, rate in activity_rates.items():
                if layer not in all_activity_rates:
                    all_activity_rates[layer] = []
                all_activity_rates[layer].append(rate)
        
        aggregated_activity = {}
        for layer, rates in all_activity_rates.items():
            aggregated_activity[layer] = {
                'mean_activity_rate': np.mean(rates),
                'activity_consistency': 1.0 - np.std(rates),
                'max_activity_rate': np.max(rates),
                'min_activity_rate': np.min(rates)
            }
        
        return aggregated_activity
    
    def _aggregate_class_distributions(self, class_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate class distributions across results"""
        combined_counts = {}
        
        for result in class_results:
            class_dist = result.get('class_distribution', {})
            class_counts = class_dist.get('class_counts', {})
            
            for class_name, count in class_counts.items():
                combined_counts[class_name] = combined_counts.get(class_name, 0) + count
        
        total_detections = sum(combined_counts.values())
        distribution_pct = {k: v/max(total_detections, 1) for k, v in combined_counts.items()}
        
        return {
            'combined_class_counts': combined_counts,
            'combined_distribution_percentage': distribution_pct,
            'total_combined_detections': total_detections
        }
    
    def _aggregate_backbone_comparisons(self, comparative_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate backbone comparison data"""
        backbone_data = {}
        
        for result in comparative_results:
            backbone_comp = result.get('backbone_comparison', {})
            for backbone, metrics in backbone_comp.items():
                if backbone not in backbone_data:
                    backbone_data[backbone] = []
                backbone_data[backbone].append(metrics)
        
        aggregated_backbones = {}
        for backbone, metrics_list in backbone_data.items():
            aggregated_backbones[backbone] = {
                'mean_accuracy': np.mean([m.get('accuracy', 0) for m in metrics_list]),
                'mean_precision': np.mean([m.get('precision', 0) for m in metrics_list]),
                'mean_recall': np.mean([m.get('recall', 0) for m in metrics_list]),
                'mean_f1_score': np.mean([m.get('f1_score', 0) for m in metrics_list]),
                'mean_inference_time': np.mean([m.get('inference_time', 0) for m in metrics_list]),
                'consistency_score': 1.0 - np.std([m.get('accuracy', 0) for m in metrics_list]),
                'data_points': len(metrics_list)
            }
        
        return aggregated_backbones
    
    def _aggregate_scenario_comparisons(self, comparative_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate scenario comparison data"""
        scenario_data = {}
        
        for result in comparative_results:
            scenario_comp = result.get('scenario_comparison', {})
            for scenario, metrics in scenario_comp.items():
                if scenario not in scenario_data:
                    scenario_data[scenario] = []
                scenario_data[scenario].append(metrics)
        
        aggregated_scenarios = {}
        for scenario, metrics_list in scenario_data.items():
            aggregated_scenarios[scenario] = {
                'mean_map': np.mean([m.get('map', 0) for m in metrics_list]),
                'mean_accuracy': np.mean([m.get('accuracy', 0) for m in metrics_list]),
                'mean_precision': np.mean([m.get('precision', 0) for m in metrics_list]),
                'mean_recall': np.mean([m.get('recall', 0) for m in metrics_list]),
                'performance_consistency': 1.0 - np.std([m.get('accuracy', 0) for m in metrics_list]),
                'data_points': len(metrics_list)
            }
        
        return aggregated_scenarios
    
    def _aggregate_efficiency_analysis(self, comparative_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate efficiency analysis data"""
        efficiency_data = {}
        
        for result in comparative_results:
            efficiency_comp = result.get('efficiency_analysis', {})
            for model, metrics in efficiency_comp.items():
                if model not in efficiency_data:
                    efficiency_data[model] = []
                efficiency_data[model].append(metrics)
        
        aggregated_efficiency = {}
        for model, metrics_list in efficiency_data.items():
            aggregated_efficiency[model] = {
                'mean_accuracy': np.mean([m.get('accuracy', 0) for m in metrics_list]),
                'mean_inference_time': np.mean([m.get('inference_time', 0) for m in metrics_list]),
                'mean_efficiency_score': np.mean([m.get('efficiency_score', 0) for m in metrics_list]),
                'mean_fps': np.mean([m.get('fps', 0) for m in metrics_list]),
                'efficiency_consistency': 1.0 - np.std([m.get('efficiency_score', 0) for m in metrics_list])
            }
        
        return aggregated_efficiency
    
    def _calculate_aggregation_confidence(self, results_list: List[Dict], analysis_type: str) -> Dict[str, float]:
        """Calculate confidence metrics untuk aggregation quality"""
        data_points = len(results_list)
        
        # Simple confidence based on sample size dan consistency
        sample_confidence = min(1.0, data_points / 10.0)  # Full confidence at 10+ samples
        
        return {
            'sample_size_confidence': sample_confidence,
            'data_points': data_points,
            'analysis_type': analysis_type,
            'aggregation_quality': 'high' if sample_confidence > 0.7 else 'medium' if sample_confidence > 0.3 else 'low'
        }
    
    def _calculate_layer_consistency_overview(self, consolidated_metrics: Dict) -> Dict[str, Any]:
        """Calculate overall layer consistency"""
        consistency_scores = [metrics.get('consistency_score', 0) for metrics in consolidated_metrics.values()]
        
        return {
            'overall_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'consistency_std': np.std(consistency_scores) if consistency_scores else 0.0,
            'most_consistent_layer': max(consolidated_metrics.items(), 
                                       key=lambda x: x[1].get('consistency_score', 0))[0] if consolidated_metrics else None
        }
    
    def _calculate_aggregated_balance_overview(self, class_results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated class balance overview"""
        balance_assessments = []
        
        for result in class_results:
            balance_analysis = result.get('balance_analysis', {})
            if 'assessment' in balance_analysis:
                balance_assessments.append(balance_analysis['assessment'])
        
        if not balance_assessments:
            return {'overall_balance': 'unknown'}
        
        # Count assessments
        assessment_counts = {assessment: balance_assessments.count(assessment) 
                           for assessment in set(balance_assessments)}
        
        # Determine overall assessment
        most_common = max(assessment_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'overall_balance_assessment': most_common,
            'assessment_distribution': assessment_counts,
            'consistency': len(set(balance_assessments)) == 1
        }
    
    def _generate_meta_insights(self, backbone_aggregated: Dict, scenario_aggregated: Dict) -> List[str]:
        """Generate meta-insights dari aggregated comparisons"""
        insights = []
        
        # Backbone insights
        if backbone_aggregated:
            best_backbone = max(backbone_aggregated.items(), 
                              key=lambda x: x[1].get('mean_accuracy', 0))
            insights.append(f"🚀 Best overall backbone: {best_backbone[0]} (accuracy: {best_backbone[1].get('mean_accuracy', 0):.3f})")
        
        # Scenario insights
        if scenario_aggregated:
            easiest_scenario = max(scenario_aggregated.items(), 
                                 key=lambda x: x[1].get('mean_accuracy', 0))
            insights.append(f"📊 Easiest scenario: {easiest_scenario[0]} (accuracy: {easiest_scenario[1].get('mean_accuracy', 0):.3f})")
        
        return insights
    
    def _create_meta_analysis(self, currency_agg: Dict, layer_agg: Dict, 
                            class_agg: Dict, comparative_agg: Dict) -> Dict[str, Any]:
        """Create meta-analysis across all aggregated results"""
        meta_insights = []
        
        # Currency meta-insights
        if currency_agg and currency_agg.get('total_currency_detections', 0) > 0:
            meta_insights.append(f"💰 Total currency detections across all sources: {currency_agg['total_currency_detections']:,}")
        
        # Layer meta-insights
        if layer_agg and layer_agg.get('consolidated_layer_metrics'):
            best_layer = max(layer_agg['consolidated_layer_metrics'].items(), 
                           key=lambda x: x[1].get('consolidated_f1_score', 0))
            meta_insights.append(f"📊 Best performing layer: {best_layer[0]} (F1: {best_layer[1].get('consolidated_f1_score', 0):.3f})")
        
        return {
            'meta_insights': meta_insights,
            'aggregation_summary': {
                'has_currency_data': bool(currency_agg and 'error' not in currency_agg),
                'has_layer_data': bool(layer_agg and 'error' not in layer_agg),
                'has_class_data': bool(class_agg and 'error' not in class_agg),
                'has_comparative_data': bool(comparative_agg and 'error' not in comparative_agg)
            }
        }
    
    def _generate_aggregation_summary(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistik dari aggregation process"""
        return {
            'total_analysis_sources': len(all_results),
            'valid_sources': len([r for r in all_results if 'error' not in r]),
            'analysis_types_present': list(set(
                key for result in all_results for key in result.keys() 
                if key.endswith('_analysis')
            )),
            'aggregation_timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get timestamp untuk metadata"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')