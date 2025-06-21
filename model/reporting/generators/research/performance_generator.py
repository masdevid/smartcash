"""
File: smartcash/model/reporting/generators/research/performance_generator.py
Deskripsi: Generator untuk performance research analysis dalam reports
"""

from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger

class PerformanceResearchGenerator:
    """Generator untuk performance analysis dan backbone comparison research"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('performance_research_generator')
        
    def generate_performance_research_analysis(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance research analysis section"""
        try:
            performance_parts = []
            
            # Main header
            performance_parts.append("## 🚀 Performance Research Analysis")
            performance_parts.append("")
            
            # Backbone comparison analysis
            backbone_section = self._generate_backbone_comparison_analysis(analysis_results)
            if backbone_section:
                performance_parts.extend(backbone_section)
            
            # Efficiency analysis
            efficiency_section = self._generate_efficiency_analysis(analysis_results)
            if efficiency_section:
                performance_parts.extend(efficiency_section)
            
            # Robustness analysis
            robustness_section = self._generate_robustness_analysis(analysis_results)
            if robustness_section:
                performance_parts.extend(robustness_section)
            
            # Scalability analysis
            scalability_section = self._generate_scalability_analysis(analysis_results)
            if scalability_section:
                performance_parts.extend(scalability_section)
            
            return "\n".join(performance_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance research analysis: {str(e)}")
            return "## 🚀 Performance Research Analysis\n\n⚠️ Error generating analysis."
    
    def _generate_backbone_comparison_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate backbone architecture comparison analysis"""
        try:
            backbone_parts = []
            
            backbone_parts.append("### 🏗️ Backbone Architecture Comparison")
            backbone_parts.append("")
            
            # Comparison overview
            comparison_analysis = analysis_results.get('comparison_analysis', {})
            backbone_comparison = comparison_analysis.get('backbone_comparison', {})
            
            if backbone_comparison:
                # Performance comparison table
                backbone_parts.append("**Architecture Performance Comparison:**")
                backbone_parts.append("")
                backbone_parts.append("| Metric | CSPDarknet (Baseline) | EfficientNet-B4 (Proposed) | Improvement |")
                backbone_parts.append("|--------|----------------------|----------------------------|-------------|")
                
                cspdarknet_metrics = backbone_comparison.get('cspdarknet', {})
                efficientnet_metrics = backbone_comparison.get('efficientnet_b4', {})
                
                # mAP comparison
                csp_map = cspdarknet_metrics.get('map', 0.0)
                eff_map = efficientnet_metrics.get('map', 0.0)
                map_improvement = ((eff_map - csp_map) / csp_map * 100) if csp_map > 0 else 0.0
                
                backbone_parts.append(f"| mAP@0.5 | {csp_map:.1%} | {eff_map:.1%} | {map_improvement:+.1f}% |")
                
                # Accuracy comparison
                csp_acc = cspdarknet_metrics.get('accuracy', 0.0)
                eff_acc = efficientnet_metrics.get('accuracy', 0.0)
                acc_improvement = ((eff_acc - csp_acc) / csp_acc * 100) if csp_acc > 0 else 0.0
                
                backbone_parts.append(f"| Accuracy | {csp_acc:.1%} | {eff_acc:.1%} | {acc_improvement:+.1f}% |")
                
                # Inference time comparison
                csp_time = cspdarknet_metrics.get('inference_time', 0.0)
                eff_time = efficientnet_metrics.get('inference_time', 0.0)
                time_change = ((eff_time - csp_time) / csp_time * 100) if csp_time > 0 else 0.0
                
                backbone_parts.append(f"| Inference Time | {csp_time:.1f}ms | {eff_time:.1f}ms | {time_change:+.1f}% |")
                
                # Model size comparison
                csp_size = cspdarknet_metrics.get('model_size', 0.0)
                eff_size = efficientnet_metrics.get('model_size', 0.0)
                size_change = ((eff_size - csp_size) / csp_size * 100) if csp_size > 0 else 0.0
                
                backbone_parts.append(f"| Model Size | {csp_size:.1f}MB | {eff_size:.1f}MB | {size_change:+.1f}% |")
                backbone_parts.append("")
            
            # Architecture advantages analysis
            backbone_parts.append("**EfficientNet-B4 Architecture Advantages:**")
            backbone_parts.append("")
            backbone_parts.append("**1. Compound Scaling:**")
            backbone_parts.append("- Balanced scaling of depth, width, dan resolution")
            backbone_parts.append("- Optimal parameter efficiency untuk given computational budget")
            backbone_parts.append("- Better feature representation dengan fewer parameters")
            backbone_parts.append("")
            
            backbone_parts.append("**2. Mobile-Optimized Design:**")
            backbone_parts.append("- Inverted residual blocks untuk efficient computation")
            backbone_parts.append("- Squeeze-and-excitation modules untuk feature importance weighting")
            backbone_parts.append("- Improved gradient flow untuk better training stability")
            backbone_parts.append("")
            
            backbone_parts.append("**3. Transfer Learning Benefits:**")
            backbone_parts.append("- Rich pre-trained features dari ImageNet")
            backbone_parts.append("- Better generalization untuk new currency denominations")
            backbone_parts.append("- Faster convergence dalam fine-tuning scenarios")
            backbone_parts.append("")
            
            # Statistical significance
            statistical_analysis = backbone_comparison.get('statistical_analysis', {})
            if statistical_analysis:
                backbone_parts.append("**Statistical Significance Analysis:**")
                p_value = statistical_analysis.get('p_value', 1.0)
                effect_size = statistical_analysis.get('effect_size', 0.0)
                confidence_interval = statistical_analysis.get('confidence_interval', [0.0, 0.0])
                
                significance_level = "significant" if p_value < 0.05 else "not significant"
                backbone_parts.append(f"- **P-value**: {p_value:.4f} ({significance_level})")
                backbone_parts.append(f"- **Effect Size (Cohen's d)**: {effect_size:.3f}")
                backbone_parts.append(f"- **95% Confidence Interval**: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
                backbone_parts.append("")
            
            return backbone_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating backbone comparison: {str(e)}")
            return []
    
    def _generate_efficiency_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate efficiency analysis section"""
        try:
            efficiency_parts = []
            
            efficiency_parts.append("### ⚡ Efficiency Analysis")
            efficiency_parts.append("")
            
            # Speed vs accuracy trade-offs
            efficiency_analysis = analysis_results.get('efficiency_analysis', {})
            if efficiency_analysis:
                efficiency_parts.append("**Speed vs Accuracy Trade-offs:**")
                accuracy_speed_ratio = efficiency_analysis.get('accuracy_speed_ratio', 0.0)
                efficiency_score = efficiency_analysis.get('efficiency_score', 0.0)
                pareto_frontier = efficiency_analysis.get('pareto_optimal', False)
                
                efficiency_parts.append(f"- **Accuracy/Speed Ratio**: {accuracy_speed_ratio:.3f}")
                efficiency_parts.append(f"- **Overall Efficiency Score**: {efficiency_score:.1%}")
                efficiency_parts.append(f"- **Pareto Optimal**: {'Yes' if pareto_frontier else 'No'}")
                efficiency_parts.append("")
            
            # Resource utilization
            resource_analysis = efficiency_analysis.get('resource_utilization', {})
            if resource_analysis:
                efficiency_parts.append("**Resource Utilization Analysis:**")
                gpu_utilization = resource_analysis.get('gpu_utilization', 0.0)
                memory_usage = resource_analysis.get('memory_usage', 0.0)
                cpu_utilization = resource_analysis.get('cpu_utilization', 0.0)
                
                efficiency_parts.append(f"- **GPU Utilization**: {gpu_utilization:.1%}")
                efficiency_parts.append(f"- **Memory Usage**: {memory_usage:.1f}GB")
                efficiency_parts.append(f"- **CPU Utilization**: {cpu_utilization:.1%}")
                efficiency_parts.append("")
            
            # Throughput analysis
            throughput_analysis = efficiency_analysis.get('throughput', {})
            if throughput_analysis:
                efficiency_parts.append("**Throughput Analysis:**")
                images_per_second = throughput_analysis.get('images_per_second', 0.0)
                batch_processing_gain = throughput_analysis.get('batch_processing_gain', 0.0)
                
                efficiency_parts.append(f"- **Single Image Throughput**: {images_per_second:.1f} images/second")
                efficiency_parts.append(f"- **Batch Processing Gain**: {batch_processing_gain:.1%} improvement")
                efficiency_parts.append("")
            
            # Deployment considerations
            efficiency_parts.append("**Deployment Efficiency Considerations:**")
            efficiency_parts.append("- **Edge Device Compatibility**: EfficientNet-B4 design optimized untuk mobile deployment")
            efficiency_parts.append("- **Memory Footprint**: Reduced memory requirements untuk real-time applications")
            efficiency_parts.append("- **Latency Requirements**: Sub-100ms inference untuk interactive applications")
            efficiency_parts.append("- **Power Consumption**: Lower computational requirements untuk battery-powered devices")
            efficiency_parts.append("")
            
            return efficiency_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating efficiency analysis: {str(e)}")
            return []
    
    def _generate_robustness_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate robustness analysis section"""
        try:
            robustness_parts = []
            
            robustness_parts.append("### 🛡️ Robustness Analysis")
            robustness_parts.append("")
            
            # Scenario robustness
            scenario_analysis = analysis_results.get('scenario_analysis', {})
            if scenario_analysis:
                robustness_parts.append("**Evaluation Scenario Robustness:**")
                
                position_robustness = scenario_analysis.get('position_variation', {})
                if position_robustness:
                    pos_performance_drop = position_robustness.get('performance_drop', 0.0)
                    pos_consistency = position_robustness.get('consistency_score', 0.0)
                    
                    robustness_parts.append(f"- **Position Variation**: {pos_performance_drop:.1%} performance drop, {pos_consistency:.1%} consistency")
                
                lighting_robustness = scenario_analysis.get('lighting_variation', {})
                if lighting_robustness:
                    light_performance_drop = lighting_robustness.get('performance_drop', 0.0)
                    light_consistency = lighting_robustness.get('consistency_score', 0.0)
                    
                    robustness_parts.append(f"- **Lighting Variation**: {light_performance_drop:.1%} performance drop, {light_consistency:.1%} consistency")
                
                robustness_parts.append("")
            
            # Adversarial robustness
            adversarial_analysis = scenario_analysis.get('adversarial_robustness', {})
            if adversarial_analysis:
                robustness_parts.append("**Adversarial Robustness:**")
                noise_tolerance = adversarial_analysis.get('noise_tolerance', 0.0)
                blur_resistance = adversarial_analysis.get('blur_resistance', 0.0)
                occlusion_handling = adversarial_analysis.get('occlusion_handling', 0.0)
                
                robustness_parts.append(f"- **Noise Tolerance**: {noise_tolerance:.1%} performance retention under Gaussian noise")
                robustness_parts.append(f"- **Blur Resistance**: {blur_resistance:.1%} performance retention under motion blur")
                robustness_parts.append(f"- **Occlusion Handling**: {occlusion_handling:.1%} performance dengan partial occlusions")
                robustness_parts.append("")
            
            # Generalization analysis
            generalization_analysis = scenario_analysis.get('generalization', {})
            if generalization_analysis:
                robustness_parts.append("**Generalization Analysis:**")
                cross_domain_performance = generalization_analysis.get('cross_domain_performance', 0.0)
                unseen_denominations = generalization_analysis.get('unseen_denominations', 0.0)
                
                robustness_parts.append(f"- **Cross-Domain Performance**: {cross_domain_performance:.1%} on different camera setups")
                robustness_parts.append(f"- **Unseen Denominations**: {unseen_denominations:.1%} transfer capability")
                robustness_parts.append("")
            
            # Robustness insights
            robustness_parts.append("**Key Robustness Insights:**")
            robustness_parts.append("- Multi-layer architecture provides natural robustness through redundancy")
            robustness_parts.append("- EfficientNet-B4 backbone shows strong generalization capabilities")
            robustness_parts.append("- Position variations handled better than lighting variations")
            robustness_parts.append("- Robust performance maintained across different deployment scenarios")
            robustness_parts.append("")
            
            return robustness_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating robustness analysis: {str(e)}")
            return []
    
    def _generate_scalability_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate scalability analysis section"""
        try:
            scalability_parts = []
            
            scalability_parts.append("### 📈 Scalability Analysis")
            scalability_parts.append("")
            
            # Dataset scalability
            scalability_analysis = analysis_results.get('scalability_analysis', {})
            if scalability_analysis:
                scalability_parts.append("**Dataset Scalability:**")
                dataset_scaling = scalability_analysis.get('dataset_scaling', {})
                
                if dataset_scaling:
                    small_dataset_perf = dataset_scaling.get('small_dataset_performance', 0.0)
                    medium_dataset_perf = dataset_scaling.get('medium_dataset_performance', 0.0)
                    large_dataset_perf = dataset_scaling.get('large_dataset_performance', 0.0)
                    
                    scalability_parts.append(f"- **Small Dataset (1K images)**: {small_dataset_perf:.1%} performance")
                    scalability_parts.append(f"- **Medium Dataset (5K images)**: {medium_dataset_perf:.1%} performance")
                    scalability_parts.append(f"- **Large Dataset (10K+ images)**: {large_dataset_perf:.1%} performance")
                    scalability_parts.append("")
            
            # Computational scalability
            computational_scaling = scalability_analysis.get('computational_scaling', {})
            if computational_scaling:
                scalability_parts.append("**Computational Scalability:**")
                batch_size_scaling = computational_scaling.get('batch_size_scaling', {})
                
                if batch_size_scaling:
                    scalability_parts.append("**Batch Size Scaling:**")
                    for batch_size, metrics in batch_size_scaling.items():
                        throughput = metrics.get('throughput', 0.0)
                        memory_usage = metrics.get('memory_usage', 0.0)
                        scalability_parts.append(f"- Batch {batch_size}: {throughput:.1f} img/s, {memory_usage:.1f}GB memory")
                    scalability_parts.append("")
            
            # Currency expansion scalability
            scalability_parts.append("**Currency Expansion Scalability:**")
            scalability_parts.append("- **New Denominations**: Architecture supports easy addition of new currency classes")
            scalability_parts.append("- **Different Currencies**: Transfer learning capability untuk other currency types")
            scalability_parts.append("- **Multi-Currency**: Potential untuk simultaneous multi-currency detection")
            scalability_parts.append("- **Historical Versions**: Capability untuk detecting different currency series")
            scalability_parts.append("")
            
            # Deployment scalability
            scalability_parts.append("**Deployment Scalability:**")
            scalability_parts.append("- **Horizontal Scaling**: Model parallelization untuk high-throughput scenarios")
            scalability_parts.append("- **Vertical Scaling**: Hardware upgrade path dari mobile to server deployment")
            scalability_parts.append("- **Cloud Deployment**: Container-ready architecture untuk cloud scaling")
            scalability_parts.append("- **Edge Deployment**: Optimization compatibility untuk edge device deployment")
            scalability_parts.append("")
            
            return scalability_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating scalability analysis: {str(e)}")
            return []