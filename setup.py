"""
File: setup.py
Deskripsi: Setup file untuk project SmartCash
"""

from setuptools import setup, find_packages

setup(
    name="smartcash",
    version="0.1.1",
    packages=find_packages(include=['smartcash', 'smartcash.*']),
    package_dir={'': '.'},
    include_package_data=True,
    install_requires=[
        "albumentations",
        "ipywidgets",
        "ipython",
        "tqdm",
    ],
    python_requires=">=3.8",
)