from setuptools import setup, find_packages

setup(
    name="smartcash",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",  # For EfficientNet-B4
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        'console_scripts': [
            'smartcash=smartcash.cli:main',
        ],
    },
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="SmartCash - Sistem Deteksi Mata Uang Rupiah",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smartcash",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
