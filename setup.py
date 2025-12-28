"""
PeakWeights: Find the weights that matter. Protect them. Quantize the rest.

Installation:
    pip install .

    # Or in development mode:
    pip install -e .
"""

import os
from setuptools import setup

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "One-pass, data-free discovery of critical LLM parameters"

setup(
    name="peakweights",
    version="0.3.0",
    author="Thiyagarajan Maruthavanan, Vamshi Ambati",
    author_email="thiyagarajan@kalmantic.com",
    description="One-pass, data-free discovery of critical LLM parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kalmantic/peakweights",
    project_urls={
        "Bug Tracker": "https://github.com/Kalmantic/peakweights/issues",
        "Documentation": "https://github.com/Kalmantic/peakweights#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["peakweights"],  # Package install
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "peakweights=peakweights.peakweights:main",
        ],
    },
    keywords=[
        "llm",
        "quantization",
        "deep-learning",
        "transformers",
        "model-compression",
        "critical-weights",
        "machine-learning",
    ],
)
