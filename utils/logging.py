# File: src/utils/logging.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas logging dengan warna dan emoji

from termcolor import colored
import logging
import sys

class ColoredLogger:
    
    COLORS = {
        'INFO': 'cyan',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'DEBUG': 'blue',
        'METRIC': 'green'
    }

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def _log(self, level, msg, *args, **kwargs):
        color = self.COLORS.get(level, 'white')
        
        formatted_msg = f"{level}: {msg}"
        if kwargs.get('metrics'):
            metrics = kwargs['metrics']
            metrics_str = ', '.join(f"{k}: {colored(v, 'yellow')}" 
                                  for k, v in metrics.items())
            formatted_msg = f"{formatted_msg} [{metrics_str}]"
            
        self.logger.log(getattr(logging, level), 
                       colored(formatted_msg, color))

    def info(self, msg, *args, **kwargs):
        self._log('INFO', msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log('WARNING', msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log('ERROR', msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log('DEBUG', msg, *args, **kwargs)

    def metric(self, msg, metrics=None, *args, **kwargs):
        kwargs['metrics'] = metrics
        self._log('METRIC', msg, *args, **kwargs)

if __name__ == '__main__':
    logger = ColoredLogger('test')
    logger.info('Starting process')
    logger.metric('Training metrics', {'loss': 0.342, 'accuracy': 0.945})
    logger.warning('GPU memory low')
    logger.error('Process failed')