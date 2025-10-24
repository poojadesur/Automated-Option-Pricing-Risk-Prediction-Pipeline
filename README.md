# Automated Option Pricing & Risk Prediction Pipeline

An end-to-end, automated pipeline to price options and predict risk metrics using Monte Carlo simulations, with MLOps best practices.

## 🎯 Project Overview

This project demonstrates a production-ready machine learning pipeline for quantitative finance, combining:

- **Quantitative Modeling**: Black-Scholes pricing, Monte Carlo simulations, Greeks calculation
- **Machine Learning**: ML models for fast option price approximation and risk prediction
- **Pipeline Orchestration**: Automated ETL → Model Training → Deployment workflow using Prefect
- **MLOps**: Experiment tracking with MLflow, model monitoring, and deployment practices

## 🚀 Features

### Quantitative Models
- **Black-Scholes Model**: Analytical pricing for European options
- **Monte Carlo Simulation**: Stochastic pricing and risk metrics (VaR, CVaR)
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho

### Machine Learning Models
- **Option Pricing Models**: XGBoost, Random Forest, Gradient Boosting, Neural Networks
- **Risk Prediction Models**: ML-based prediction of VaR, CVaR, and Greeks
- **Fast Inference**: Pre-trained models for real-time pricing

### Pipeline & MLOps
- **Automated ETL**: Data generation, validation, and feature engineering
- **Pipeline Orchestration**: Prefect flows for training and inference
- **Experiment Tracking**: MLflow integration for reproducibility
- **Model Monitoring**: Data drift detection and performance tracking

## 📁 Project Structure

```
.
├── src/
│   └── option_pricing_pipeline/
│       ├── models/           # Quantitative and ML models
│       ├── data/             # Data generation and ETL
│       ├── pipeline/         # Pipeline orchestration
│       ├── monitoring/       # MLOps monitoring tools
│       └── utils/            # Utility functions
├── tests/                    # Unit tests
├── examples/                 # Example scripts
├── config/                   # Configuration files
├── data/                     # Data storage
├── models/                   # Saved models
└── logs/                     # Application logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/poojadesur/Automated-Option-Pricing-Risk-Prediction-Pipeline.git
cd Automated-Option-Pricing-Risk-Prediction-Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## 📊 Usage

### Quick Start - Quantitative Models

Run the quantitative models demo:
```bash
python examples/demo_quantitative.py
```

This demonstrates:
- Black-Scholes option pricing
- Greeks calculation
- Monte Carlo simulation
- VaR and CVaR calculation

### Training the ML Pipeline

Train the ML models for option pricing and risk prediction:
```bash
python examples/train_models.py
```

This will:
1. Generate synthetic training data
2. Calculate option prices using Black-Scholes
3. Compute risk metrics using Monte Carlo
4. Train ML models for pricing and risk prediction
5. Save models and log to MLflow

### Running Inference

Price options and predict risk using trained models:
```bash
python examples/run_inference.py
```

### Using the Pipeline Programmatically

```python
from option_pricing_pipeline.pipeline.orchestration import (
    option_pricing_training_pipeline,
    option_pricing_inference_pipeline
)

# Train models
results = option_pricing_training_pipeline(
    n_samples=10000,
    n_simulations=5000,
    model_type='xgboost',
    output_dir='models/saved'
)

# Run inference
predictions = option_pricing_inference_pipeline(
    input_params={
        'S': 100.0,    # Stock price
        'K': 100.0,    # Strike price
        'T': 1.0,      # Time to maturity
        'r': 0.05,     # Risk-free rate
        'sigma': 0.2   # Volatility
    },
    model_dir='models/saved'
)
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=option_pricing_pipeline --cov-report=html
```

## 📈 Model Performance

The ML models achieve:
- **Option Pricing**: R² > 0.99, MAPE < 1%
- **Risk Metrics**: R² > 0.95 for VaR/CVaR prediction
- **Inference Speed**: ~100x faster than Monte Carlo simulation

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Data generation parameters
- Monte Carlo settings
- Model types and hyperparameters
- MLflow tracking
- Monitoring thresholds

## 📚 Technical Details

### Black-Scholes Model
Implements the closed-form solution for European option pricing:

```
C = S*N(d1) - K*e^(-rT)*N(d2)
P = K*e^(-rT)*N(-d2) - S*N(-d1)

where:
d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
d2 = d1 - σ*√T
```

### Monte Carlo Simulation
Uses Geometric Brownian Motion for path simulation:

```
S(t+dt) = S(t) * exp((r - σ²/2)*dt + σ*√dt*Z)
```

where Z ~ N(0,1)

### Machine Learning Models
- **Features**: S, K, T, r, σ (+ engineered features)
- **Targets**: Option prices, Greeks, VaR/CVaR
- **Models**: XGBoost, Random Forest, Gradient Boosting, Neural Networks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Black-Scholes-Merton model for option pricing theory
- NumPy and SciPy for numerical computations
- Scikit-learn and XGBoost for machine learning
- Prefect for workflow orchestration
- MLflow for experiment tracking

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
