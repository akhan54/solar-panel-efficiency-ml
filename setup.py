from setuptools import setup, find_packages

setup(
    name="solar-efficiency-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "shap>=0.42.0",
        "optuna>=3.3.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning pipeline for solar panel efficiency prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/solar-efficiency-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)