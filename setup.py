"""
File: setup.py
Deskripsi: Setup file untuk project SmartCash
"""

from setuptools import setup, find_packages

setup(
    name="smartcash",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "albumentations",
        "ipywidgets",
        "ipython",
        "tqdm",
    ],
    python_requires=">=3.8",
) 