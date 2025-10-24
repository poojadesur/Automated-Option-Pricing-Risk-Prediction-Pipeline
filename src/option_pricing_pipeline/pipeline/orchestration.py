"""
Pipeline orchestration using Prefect for workflow management.
"""
import pandas as pd
import numpy as np
from prefect import flow, task
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import mlflow

from ..models.quantitative import BlackScholesModel, MonteCarloSimulation
from ..models.ml_models import OptionPricingMLModel, RiskPredictionModel
from ..data.generator import OptionDataGenerator, DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="generate_data", retries=2)
def generate_training_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic training data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Generated DataFrame
    """
    logger.info(f"Generating {n_samples} training samples...")
    generator = OptionDataGenerator()
    df = generator.generate_training_data(n_samples=n_samples)
    return df


@task(name="calculate_prices", retries=2)
def calculate_option_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Calculate option prices using Black-Scholes model.
    
    Args:
        df: DataFrame with option parameters
        
    Returns:
        Tuple of (df, call_prices, put_prices)
    """
    logger.info("Calculating option prices using Black-Scholes...")
    bs_model = BlackScholesModel()
    
    call_prices = []
    put_prices = []
    
    for _, row in df.iterrows():
        call_price = bs_model.price_call(
            row['S'], row['K'], row['T'], row['r'], row['sigma']
        )
        put_price = bs_model.price_put(
            row['S'], row['K'], row['T'], row['r'], row['sigma']
        )
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    return df, np.array(call_prices), np.array(put_prices)


@task(name="calculate_greeks", retries=2)
def calculate_greeks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Calculate option Greeks.
    
    Args:
        df: DataFrame with option parameters
        
    Returns:
        Dictionary of Greeks arrays
    """
    logger.info("Calculating Greeks...")
    bs_model = BlackScholesModel()
    
    greeks_data = {
        'delta': [],
        'gamma': [],
        'theta': [],
        'vega': [],
        'rho': []
    }
    
    for _, row in df.iterrows():
        greeks = bs_model.calculate_greeks(
            row['S'], row['K'], row['T'], row['r'], row['sigma'], 'call'
        )
        for key in greeks_data.keys():
            greeks_data[key].append(greeks[key])
    
    return {key: np.array(values) for key, values in greeks_data.items()}


@task(name="calculate_risk_metrics", retries=2)
def calculate_risk_metrics(df: pd.DataFrame, n_simulations: int = 5000) -> Dict[str, np.ndarray]:
    """
    Calculate risk metrics using Monte Carlo.
    
    Args:
        df: DataFrame with option parameters
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary of risk metrics
    """
    logger.info(f"Calculating risk metrics with {n_simulations} simulations...")
    mc_sim = MonteCarloSimulation(n_simulations=n_simulations)
    
    var_values = []
    cvar_values = []
    
    for _, row in df.iterrows():
        risk = mc_sim.calculate_var(
            row['S'], row['T'], row['r'], row['sigma']
        )
        var_values.append(risk['var'])
        cvar_values.append(risk['cvar'])
    
    return {
        'var': np.array(var_values),
        'cvar': np.array(cvar_values)
    }


@task(name="train_pricing_model", retries=2)
def train_pricing_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = 'xgboost'
) -> Tuple[OptionPricingMLModel, Dict[str, float]]:
    """
    Train ML model for option pricing.
    
    Args:
        X: Features
        y: Target prices
        model_type: Type of ML model
        
    Returns:
        Tuple of (trained model, metrics)
    """
    logger.info(f"Training {model_type} pricing model...")
    model = OptionPricingMLModel(model_type=model_type)
    metrics = model.train(X, y)
    return model, metrics


@task(name="train_risk_model", retries=2)
def train_risk_model(
    X: pd.DataFrame,
    y: Dict[str, np.ndarray],
    model_type: str = 'xgboost'
) -> Tuple[RiskPredictionModel, Dict[str, Dict[str, float]]]:
    """
    Train ML models for risk prediction.
    
    Args:
        X: Features
        y: Dictionary of risk metrics
        model_type: Type of ML model
        
    Returns:
        Tuple of (trained model, metrics)
    """
    logger.info(f"Training {model_type} risk models...")
    model = RiskPredictionModel(model_type=model_type)
    metrics = model.train(X, y)
    return model, metrics


@task(name="save_models", retries=2)
def save_models(
    pricing_model: OptionPricingMLModel,
    risk_model: RiskPredictionModel,
    output_dir: str
):
    """
    Save trained models to disk.
    
    Args:
        pricing_model: Trained pricing model
        risk_model: Trained risk model
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pricing_path = Path(output_dir) / "pricing_model.joblib"
    risk_path = Path(output_dir) / "risk_model.joblib"
    
    pricing_model.save(str(pricing_path))
    risk_model.save(str(risk_path))
    
    logger.info(f"Models saved to {output_dir}")


@flow(name="option_pricing_pipeline")
def option_pricing_training_pipeline(
    n_samples: int = 10000,
    n_simulations: int = 5000,
    model_type: str = 'xgboost',
    output_dir: str = "models/saved",
    track_mlflow: bool = True
) -> Dict:
    """
    Complete pipeline for training option pricing and risk models.
    
    Args:
        n_samples: Number of training samples
        n_simulations: Number of Monte Carlo simulations
        model_type: Type of ML model
        output_dir: Directory to save models
        track_mlflow: Whether to track with MLflow
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info("Starting option pricing pipeline...")
    
    if track_mlflow:
        mlflow.set_experiment("option_pricing_pipeline")
        mlflow.start_run()
        mlflow.log_params({
            'n_samples': n_samples,
            'n_simulations': n_simulations,
            'model_type': model_type
        })
    
    # Generate data
    df = generate_training_data(n_samples)
    
    # Calculate option prices
    df, call_prices, put_prices = calculate_option_prices(df)
    
    # Calculate Greeks
    greeks = calculate_greeks(df)
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(df, n_simulations)
    
    # Combine risk metrics with Greeks
    all_risk_metrics = {**greeks, **risk_metrics}
    
    # Train pricing model (using call prices)
    pricing_model, pricing_metrics = train_pricing_model(df, call_prices, model_type)
    
    # Train risk model
    risk_model, risk_metrics_dict = train_risk_model(df, all_risk_metrics, model_type)
    
    # Save models
    save_models(pricing_model, risk_model, output_dir)
    
    if track_mlflow:
        mlflow.log_metrics(pricing_metrics)
        for risk_type, metrics in risk_metrics_dict.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{risk_type}_{metric_name}", value)
        mlflow.end_run()
    
    results = {
        'pricing_metrics': pricing_metrics,
        'risk_metrics': risk_metrics_dict,
        'n_samples': n_samples
    }
    
    logger.info("Pipeline completed successfully!")
    return results


@flow(name="option_pricing_inference")
def option_pricing_inference_pipeline(
    input_params: Dict,
    model_dir: str = "models/saved"
) -> Dict:
    """
    Inference pipeline for option pricing and risk prediction.
    
    Args:
        input_params: Dictionary with option parameters (S, K, T, r, sigma)
        model_dir: Directory containing saved models
        
    Returns:
        Dictionary with predictions
    """
    logger.info("Starting inference pipeline...")
    
    # Load models
    pricing_model = OptionPricingMLModel()
    risk_model = RiskPredictionModel()
    
    pricing_path = Path(model_dir) / "pricing_model.joblib"
    risk_path = Path(model_dir) / "risk_model.joblib"
    
    pricing_model.load(str(pricing_path))
    risk_model.load(str(risk_path))
    
    # Prepare input
    df = pd.DataFrame([input_params])
    
    # Predict
    predicted_price = pricing_model.predict(df)[0]
    predicted_risks = risk_model.predict(df)
    
    # Also calculate using Black-Scholes for comparison
    bs_model = BlackScholesModel()
    bs_call = bs_model.price_call(
        input_params['S'],
        input_params['K'],
        input_params['T'],
        input_params['r'],
        input_params['sigma']
    )
    bs_greeks = bs_model.calculate_greeks(
        input_params['S'],
        input_params['K'],
        input_params['T'],
        input_params['r'],
        input_params['sigma']
    )
    
    results = {
        'ml_predicted_price': float(predicted_price),
        'bs_price': float(bs_call),
        'predicted_risks': {k: float(v[0]) for k, v in predicted_risks.items()},
        'bs_greeks': {k: float(v) for k, v in bs_greeks.items()}
    }
    
    logger.info("Inference completed successfully!")
    return results
