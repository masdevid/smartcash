# File: src/interfaces/base_interface.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk antarmuka menu SmartCash Detector

from termcolor import colored
from utils.logging import ColoredLogger
from termcolor import colored

class BaseInterface:
    def __init__(self):
        self.logger = ColoredLogger(self.__class__.__name__)

    def prompt(self, message: str, default=None, color='yellow'):
        """Tampilkan prompt dengan nilai default"""
        if default:
            message = f"{message} (default: {default}): "
        else:
            message = f"{message}: "
            
        value = input(colored(message, color)) or default
        return value

    def confirm(self, message: str) -> bool:
        """Tampilkan konfirmasi yes/no"""
        response = self.prompt(f"{message} (y/n)", default='n')
        return response.lower() == 'y'

    def show_error(self, message: str):
        """Tampilkan pesan error"""
        self.logger.error(message)

    def show_success(self, message: str):
        """Tampilkan pesan sukses"""
        self.logger.info(message)