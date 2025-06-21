"""
File: smartcash/model/reporting/__init__.py
Deskripsi: Reporting module exports dan factory functions untuk Fase 4
"""

from .report_service import ReportService
from .generators.summary_generator import SummaryGenerator
from .generators.comparison_generator import ComparisonGenerator
from .generators.research_generator import ResearchGenerator

# Factory functions untuk quick setup
def create_report_service(config=None, output_dir='data/analysis/reports'):
    """Create configured ReportService instance"""
    return ReportService(config=config, output_dir=output_dir)

def generate_quick_summary(analysis_results, config=None):
    """Generate quick summary dari analysis results"""
    summary_gen = SummaryGenerator(config)
    return summary_gen.generate_quick_summary(analysis_results)

def generate_research_report(analysis_results, config=None):
    """Generate comprehensive research report"""
    research_gen = ResearchGenerator(config)
    return research_gen.generate_comprehensive_research_report(analysis_results)

def generate_comparison_report(analysis_results, config=None):
    """Generate comparative analysis report"""
    comparison_gen = ComparisonGenerator(config)
    return comparison_gen.generate_comprehensive_comparison_report(analysis_results)

def run_comprehensive_reporting_pipeline(analysis_results, config=None, output_dir='data/analysis/reports'):
    """Run complete reporting pipeline dengan all formats"""
    report_service = create_report_service(config, output_dir)
    
    # Generate all reports
    report_paths = report_service.generate_comprehensive_report(analysis_results)
    
    # Generate quick summary untuk immediate review
    quick_summary = report_service.generate_quick_summary(analysis_results)
    
    return {
        'report_paths': report_paths,
        'quick_summary': quick_summary,
        'service': report_service
    }

__all__ = [
    'ReportService',
    'SummaryGenerator', 
    'ComparisonGenerator',
    'ResearchGenerator',
    'create_report_service',
    'generate_quick_summary',
    'generate_research_report', 
    'generate_comparison_report',
    'run_comprehensive_reporting_pipeline'
]