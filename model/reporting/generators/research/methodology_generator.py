"""
File: smartcash/model/reporting/generators/research/methodology_generator.py
Deskripsi: Generator untuk methodology section dalam research reports
"""

from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger

class MethodologyReportGenerator:
    """Generator untuk methodology dan experimental setup documentation"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('methodology_generator')
        
    def generate_methodology_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive methodology section"""
        try:
            methodology_parts = []
            
            # Main header
            methodology_parts.append("## 🔬 Methodology")
            methodology_parts.append("")
            
            # Model architecture
            architecture_section = self._generate_architecture_description(analysis_results)
            if architecture_section:
                methodology_parts.extend(architecture_section)
            
            # Dataset methodology
            dataset_section = self._generate_dataset_methodology(analysis_results)
            if dataset_section:
                methodology_parts.extend(dataset_section)
            
            # Evaluation methodology
            evaluation_section = self._generate_evaluation_methodology(analysis_results)
            if evaluation_section:
                methodology_parts.extend(evaluation_section)
            
            # Analysis methodology
            analysis_section = self._generate_analysis_methodology(analysis_results)
            if analysis_section:
                methodology_parts.extend(analysis_section)
            
            return "\n".join(methodology_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating methodology section: {str(e)}")
            return "## 🔬 Methodology\n\n⚠️ Error generating methodology section."
    
    def _generate_architecture_description(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate model architecture description"""
        try:
            arch_parts = []
            
            arch_parts.append("### 🏗️ Model Architecture")
            arch_parts.append("")
            
            # Base architecture
            model_info = analysis_results.get('model_info', {})
            backbone = model_info.get('backbone', 'EfficientNet-B4')
            base_model = model_info.get('base_model', 'YOLOv5s')
            
            arch_parts.append(f"**Base Model**: {base_model}")
            arch_parts.append(f"**Backbone**: {backbone} architecture")
            arch_parts.append("")
            
            # Architecture details
            arch_parts.append("**Architecture Components:**")
            arch_parts.append("- **Backbone**: EfficientNet-B4 untuk feature extraction dengan improved efficiency")
            arch_parts.append("- **Neck**: PANet untuk multi-scale feature fusion")
            arch_parts.append("- **Head**: YOLOv5 detection head dengan anchor-based detection")
            arch_parts.append("")
            
            # Multi-layer detection strategy
            arch_parts.append("**Multi-Layer Detection Strategy:**")
            arch_parts.append("- **Primary Layer (Banknote)**: Main currency detection dengan 7 denomination classes")
            arch_parts.append("- **Boost Layer (Nominal)**: Confidence enhancement untuk denomination validation")
            arch_parts.append("- **Validation Layer (Security)**: Security feature detection untuk authenticity verification")
            arch_parts.append("")
            
            # Model configuration
            config_info = self.config.get('model', {})
            if config_info:
                arch_parts.append("**Model Configuration:**")
                input_size = config_info.get('input_size', 640)
                batch_size = config_info.get('batch_size', 16)
                arch_parts.append(f"- Input Size: {input_size}x{input_size} pixels")
                arch_parts.append(f"- Batch Size: {batch_size}")
                arch_parts.append("")
            
            return arch_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating architecture description: {str(e)}")
            return []
    
    def _generate_dataset_methodology(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate dataset methodology description"""
        try:
            dataset_parts = []
            
            dataset_parts.append("### 📊 Dataset Methodology")
            dataset_parts.append("")
            
            # Dataset overview
            dataset_info = analysis_results.get('dataset_info', {})
            if dataset_info:
                total_images = dataset_info.get('total_images', 0)
                train_split = dataset_info.get('train_split', 0)
                val_split = dataset_info.get('val_split', 0)
                test_split = dataset_info.get('test_split', 0)
                
                dataset_parts.append("**Dataset Composition:**")
                dataset_parts.append(f"- Total Images: {total_images:,}")
                dataset_parts.append(f"- Training Set: {train_split:,} images")
                dataset_parts.append(f"- Validation Set: {val_split:,} images") 
                dataset_parts.append(f"- Test Set: {test_split:,} images")
                dataset_parts.append("")
            
            # Currency denominations
            dataset_parts.append("**Currency Denominations:**")
            denominations = [
                "Rp 1.000", "Rp 2.000", "Rp 5.000", "Rp 10.000",
                "Rp 20.000", "Rp 50.000", "Rp 100.000"
            ]
            for idx, denom in enumerate(denominations):
                dataset_parts.append(f"- Class {idx}: {denom}")
            dataset_parts.append("")
            
            # Data collection methodology
            dataset_parts.append("**Data Collection Methodology:**")
            dataset_parts.append("- Controlled lighting conditions untuk consistent image quality")
            dataset_parts.append("- Multiple viewing angles untuk robust detection")
            dataset_parts.append("- Various background conditions untuk generalization")
            dataset_parts.append("- High-resolution capture untuk detailed feature extraction")
            dataset_parts.append("")
            
            # Annotation methodology
            dataset_parts.append("**Annotation Methodology:**")
            dataset_parts.append("- Multi-layer annotation strategy:")
            dataset_parts.append("  - **Banknote boxes**: Full currency boundary detection")
            dataset_parts.append("  - **Nominal boxes**: Denomination value regions")
            dataset_parts.append("  - **Security boxes**: Security feature locations")
            dataset_parts.append("- Quality control dengan double annotation verification")
            dataset_parts.append("")
            
            return dataset_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating dataset methodology: {str(e)}")
            return []
    
    def _generate_evaluation_methodology(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate evaluation methodology description"""
        try:
            eval_parts = []
            
            eval_parts.append("### 🧪 Evaluation Methodology")
            eval_parts.append("")
            
            # Research scenarios
            eval_parts.append("**Research Evaluation Scenarios:**")
            eval_parts.append("")
            
            eval_parts.append("**1. Position Variation Scenario:**")
            eval_parts.append("- Rotation testing: -30° to +30° range")
            eval_parts.append("- Translation testing: ±20% horizontal/vertical displacement")
            eval_parts.append("- Scale testing: 0.8x to 1.2x size variation")
            eval_parts.append("- Objective: Evaluate geometric robustness")
            eval_parts.append("")
            
            eval_parts.append("**2. Lighting Variation Scenario:**")
            eval_parts.append("- Brightness adjustment: ±30% intensity variation")
            eval_parts.append("- Contrast modification: 0.7x to 1.3x contrast range")
            eval_parts.append("- Gamma correction: 0.7 to 1.3 gamma values")
            eval_parts.append("- Objective: Evaluate photometric robustness")
            eval_parts.append("")
            
            # Metrics methodology
            eval_parts.append("**Evaluation Metrics:**")
            eval_parts.append("- **mAP (mean Average Precision)**: Primary detection accuracy metric")
            eval_parts.append("- **Precision/Recall**: Per-class performance analysis")
            eval_parts.append("- **F1-Score**: Balanced performance assessment")
            eval_parts.append("- **Inference Time**: Computational efficiency measurement")
            eval_parts.append("- **Confusion Matrix**: Error pattern analysis")
            eval_parts.append("")
            
            # Backbone comparison methodology
            eval_parts.append("**Comparative Analysis Methodology:**")
            eval_parts.append("- **Baseline**: YOLOv5s dengan CSPDarknet backbone")
            eval_parts.append("- **Proposed**: YOLOv5s dengan EfficientNet-B4 backbone")
            eval_parts.append("- **Comparison Metrics**: Accuracy, speed, model size")
            eval_parts.append("- **Statistical Testing**: Paired t-test untuk significance")
            eval_parts.append("")
            
            return eval_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating evaluation methodology: {str(e)}")
            return []
    
    def _generate_analysis_methodology(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate analysis methodology description"""
        try:
            analysis_parts = []
            
            analysis_parts.append("### 📈 Analysis Methodology")
            analysis_parts.append("")
            
            # Multi-dimensional analysis
            analysis_parts.append("**Multi-Dimensional Analysis Framework:**")
            analysis_parts.append("")
            
            analysis_parts.append("**1. Currency Denomination Analysis:**")
            analysis_parts.append("- Multi-layer detection strategy validation")
            analysis_parts.append("- Primary-boost-validation layer collaboration")
            analysis_parts.append("- IoU-based spatial relationship analysis")
            analysis_parts.append("- Denomination-specific performance metrics")
            analysis_parts.append("")
            
            analysis_parts.append("**2. Layer Performance Analysis:**")
            analysis_parts.append("- Individual layer effectiveness assessment")
            analysis_parts.append("- Cross-layer collaboration scoring")
            analysis_parts.append("- Layer balance dan distribution analysis")
            analysis_parts.append("- Performance consistency evaluation")
            analysis_parts.append("")
            
            analysis_parts.append("**3. Comparative Analysis:**")
            analysis_parts.append("- Backbone architecture comparison")
            analysis_parts.append("- Evaluation scenario difficulty assessment")
            analysis_parts.append("- Efficiency trade-off analysis")
            analysis_parts.append("- Statistical significance testing")
            analysis_parts.append("")
            
            # Visualization methodology
            analysis_parts.append("**Visualization Methodology:**")
            analysis_parts.append("- **Confusion Matrices**: Normalized error pattern visualization")
            analysis_parts.append("- **Performance Radar Charts**: Multi-metric comparison plots")
            analysis_parts.append("- **Distribution Charts**: Strategy usage dan class balance")
            analysis_parts.append("- **Efficiency Plots**: Accuracy vs speed frontier analysis")
            analysis_parts.append("")
            
            # Statistical analysis
            analysis_parts.append("**Statistical Analysis:**")
            analysis_parts.append("- Confidence intervals untuk performance metrics")
            analysis_parts.append("- Paired statistical tests untuk backbone comparison")
            analysis_parts.append("- Effect size calculation untuk practical significance")
            analysis_parts.append("- Bootstrap sampling untuk robustness validation")
            analysis_parts.append("")
            
            return analysis_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating analysis methodology: {str(e)}")
            return []