"""
File: smartcash/ui/core/utils/markdown_formatter.py
Description: Markdown to HTML formatter utility for summary containers across all UI modules.
"""

import markdown
from typing import Dict, Any, Optional, List
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from markdown.postprocessors import Postprocessor
import re


class SmartCashMarkdownExtension(Extension):
    """Custom markdown extension for SmartCash UI formatting."""
    
    def extendMarkdown(self, md):
        """Add custom processors to markdown."""
        # Add custom preprocessor for SmartCash-specific formatting
        md.preprocessors.register(
            SmartCashPreprocessor(md), 'smartcash', 25
        )
        # Add custom postprocessor for final HTML cleanup
        md.postprocessors.register(
            SmartCashPostprocessor(md), 'smartcash', 25
        )


class SmartCashPreprocessor(Preprocessor):
    """Preprocessor for SmartCash-specific markdown formatting."""
    
    def run(self, lines: List[str]) -> List[str]:
        """Process lines before markdown conversion."""
        processed_lines = []
        
        for line in lines:
            # Convert SmartCash-specific emojis and status indicators
            line = self._process_status_indicators(line)
            line = self._process_progress_indicators(line)
            line = self._process_model_info(line)
            processed_lines.append(line)
        
        return processed_lines
    
    def _process_status_indicators(self, line: str) -> str:
        """Process status indicators like ✅, ❌, ⚠️."""
        # Add CSS classes for better styling
        replacements = {
            '✅': '<span class="status-success">✅</span>',
            '❌': '<span class="status-error">❌</span>',
            '⚠️': '<span class="status-warning">⚠️</span>',
            '🔍': '<span class="status-info">🔍</span>',
            '📊': '<span class="status-stats">📊</span>',
            '🏗️': '<span class="status-build">🏗️</span>',
            '🚀': '<span class="status-launch">🚀</span>',
            '💾': '<span class="status-save">💾</span>',
        }
        
        for emoji, replacement in replacements.items():
            if emoji in line and not line.strip().startswith('#'):
                line = line.replace(emoji, replacement)
        
        return line
    
    def _process_progress_indicators(self, line: str) -> str:
        """Process progress indicators and statistics."""
        # Pattern for parameter counts: "1,234,567 total, 987,654 trainable"
        param_pattern = r'(\d{1,3}(?:,\d{3})*)\s+(total|trainable|parameters?)'
        line = re.sub(param_pattern, r'<span class="param-count">\1</span> \2', line)
        
        # Pattern for percentages: "85.3%", "mAP: 0.756"
        percent_pattern = r'(\d+\.?\d*)%'
        line = re.sub(percent_pattern, r'<span class="percentage">\1%</span>', line)
        
        # Pattern for metrics: "mAP: 0.756", "Loss: 2.345"
        metric_pattern = r'(mAP|Loss|Precision|Recall|F1):\s*(\d+\.?\d+)'
        line = re.sub(metric_pattern, r'<span class="metric-label">\1:</span> <span class="metric-value">\2</span>', line)
        
        return line
    
    def _process_model_info(self, line: str) -> str:
        """Process model information and file paths."""
        # Pattern for file paths
        if '/data/' in line or '/content/' in line:
            path_pattern = r'(/(?:data|content)/[^\s]+)'
            line = re.sub(path_pattern, r'<code class="file-path">\1</code>', line)
        
        # Pattern for model names
        model_pattern = r'(efficientnet_b4|cspdarknet|yolov5|smartcash)'
        line = re.sub(model_pattern, r'<span class="model-name">\1</span>', line, flags=re.IGNORECASE)
        
        return line


class SmartCashPostprocessor(Postprocessor):
    """Postprocessor for final HTML cleanup and enhancement."""
    
    def run(self, text: str) -> str:
        """Process HTML after markdown conversion."""
        # Add custom CSS classes to tables
        text = re.sub(r'<table>', '<table class="smartcash-table">', text)
        
        # Add wrapper div for styling
        text = f'<div class="smartcash-summary">{text}</div>'
        
        return text


class MarkdownHTMLFormatter:
    """
    Enhanced markdown to HTML formatter for SmartCash UI summary containers.
    
    Features:
    - Custom styling for status indicators and emojis
    - Enhanced formatting for metrics and statistics
    - Responsive table styling
    - Progress indicator formatting
    - File path and model name highlighting
    """
    
    def __init__(self):
        """Initialize the formatter with custom extensions."""
        self.markdown_processor = markdown.Markdown(
            extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.nl2br',
                SmartCashMarkdownExtension()
            ],
            extension_configs={
                'markdown.extensions.tables': {
                    'use_align_attribute': True
                }
            }
        )
    
    def format_to_html(self, markdown_content: str, title: str = None, module_name: str = None) -> str:
        """
        Convert markdown content to styled HTML.
        
        Args:
            markdown_content: The markdown content to convert
            title: Optional title for the summary
            module_name: Optional module name for specific styling
            
        Returns:
            Formatted HTML string with embedded CSS
        """
        try:
            # Convert markdown to HTML
            html_content = self.markdown_processor.convert(markdown_content)
            
            # Reset markdown processor for next use
            self.markdown_processor.reset()
            
            # Add title if provided
            if title:
                title_html = f'<h2 class="summary-title">{title}</h2>'
                html_content = title_html + html_content
            
            # Wrap in styled container
            styled_html = self._wrap_with_styles(html_content, module_name)
            
            return styled_html
            
        except Exception as e:
            # Fallback to plain text if markdown processing fails
            fallback_html = f"""
            <div class="smartcash-summary error">
                <h3>Summary (Markdown Processing Failed)</h3>
                <pre>{markdown_content}</pre>
                <p><small>Error: {str(e)}</small></p>
            </div>
            """
            return fallback_html
    
    def _wrap_with_styles(self, html_content: str, module_name: str = None) -> str:
        """Wrap HTML content with embedded CSS styles."""
        module_class = f"module-{module_name}" if module_name else "module-generic"
        
        css_styles = self._get_embedded_css()
        
        return f"""
        <div class="smartcash-formatter {module_class}">
            <style>{css_styles}</style>
            {html_content}
        </div>
        """
    
    def _get_embedded_css(self) -> str:
        """Get embedded CSS styles for SmartCash summaries."""
        return """
        .smartcash-formatter {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0 auto;
        }
        
        .smartcash-summary {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
        }
        
        .summary-title {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        /* Status indicators */
        .status-success { color: #28a745; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-info { color: #17a2b8; font-weight: bold; }
        .status-stats { color: #6f42c1; font-weight: bold; }
        .status-build { color: #fd7e14; font-weight: bold; }
        .status-launch { color: #20c997; font-weight: bold; }
        .status-save { color: #6c757d; font-weight: bold; }
        
        /* Parameter counts */
        .param-count {
            font-weight: bold;
            color: #495057;
            font-family: 'Courier New', monospace;
        }
        
        /* Percentages and metrics */
        .percentage {
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .metric-label {
            font-weight: bold;
            color: #6c757d;
        }
        
        .metric-value {
            font-weight: bold;
            color: #28a745;
            font-family: 'Courier New', monospace;
        }
        
        /* File paths */
        .file-path {
            background: #f1f3f4;
            color: #5f6368;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        /* Model names */
        .model-name {
            background: #fff3cd;
            color: #856404;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        /* Tables */
        .smartcash-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .smartcash-table th,
        .smartcash-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .smartcash-table th {
            background: #3498db;
            color: white;
            font-weight: bold;
        }
        
        .smartcash-table tr:hover {
            background: #f5f5f5;
        }
        
        /* Code blocks */
        pre {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }
        
        code {
            background: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        /* Module-specific styling */
        .module-backbone .summary-title { border-color: #fd7e14; }
        .module-training .summary-title { border-color: #28a745; }
        .module-evaluation .summary-title { border-color: #6f42c1; }
        .module-preprocessing .summary-title { border-color: #17a2b8; }
        .module-augmentation .summary-title { border-color: #ffc107; }
        .module-download .summary-title { border-color: #dc3545; }
        
        /* Error state */
        .smartcash-summary.error {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .smartcash-formatter {
                font-size: 14px;
            }
            
            .smartcash-summary {
                padding: 15px;
                margin: 5px 0;
            }
            
            .smartcash-table {
                font-size: 12px;
            }
            
            .smartcash-table th,
            .smartcash-table td {
                padding: 8px 10px;
            }
        }
        """


# Global formatter instance
_formatter_instance = None


def get_markdown_formatter() -> MarkdownHTMLFormatter:
    """Get singleton instance of the markdown formatter."""
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = MarkdownHTMLFormatter()
    return _formatter_instance


def format_summary_to_html(
    markdown_content: str, 
    title: str = None, 
    module_name: str = None
) -> str:
    """
    Convenience function to format markdown summary to HTML.
    
    Args:
        markdown_content: The markdown content to convert
        title: Optional title for the summary
        module_name: Optional module name for specific styling
        
    Returns:
        Formatted HTML string
    """
    formatter = get_markdown_formatter()
    return formatter.format_to_html(markdown_content, title, module_name)