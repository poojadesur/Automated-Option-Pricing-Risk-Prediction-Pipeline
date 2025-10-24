from setuptools import setup, find_packages

setup(
    name="option_pricing_pipeline",
    version="0.1.0",
    description="Automated Option Pricing and Risk Prediction Pipeline",
    author="Pooja Desur",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scipy>=1.11.1",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "torch>=2.0.1",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "mlflow>=2.5.0",
        "prefect>=2.11.0",
        "pydantic>=2.1.1",
        "pyyaml>=6.0.1",
        "python-dateutil>=2.8.2",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
