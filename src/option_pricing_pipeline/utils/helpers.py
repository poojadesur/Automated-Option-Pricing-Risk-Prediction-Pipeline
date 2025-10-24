"""
Utility functions for the option pricing pipeline.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import yaml
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def format_price(price: float, decimals: int = 2) -> str:
    """
    Format price for display.
    
    Args:
        price: Price value
        decimals: Number of decimal places
        
    Returns:
        Formatted price string
    """
    return f"${price:.{decimals}f}"


def calculate_percentage_error(actual: float, predicted: float) -> float:
    """
    Calculate percentage error.
    
    Args:
        actual: Actual value
        predicted: Predicted value
        
    Returns:
        Percentage error
    """
    if actual == 0:
        return 0.0
    return abs((actual - predicted) / actual) * 100


def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table from results.
    
    Args:
        results: Results dictionary
        
    Returns:
        Summary DataFrame
    """
    data = []
    for key, value in results.items():
        if isinstance(value, (int, float)):
            data.append({'Metric': key, 'Value': value})
    
    return pd.DataFrame(data)


def validate_option_params(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> bool:
    """
    Validate option parameters.
    
    Args:
        S: Stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        True if valid
    """
    if S <= 0:
        raise ValueError("Stock price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T < 0:
        raise ValueError("Time to maturity must be non-negative")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    
    return True


def interpolate_volatility(
    strikes: List[float],
    vols: List[float],
    target_strike: float
) -> float:
    """
    Interpolate volatility for a given strike.
    
    Args:
        strikes: List of strikes
        vols: List of corresponding volatilities
        target_strike: Target strike to interpolate
        
    Returns:
        Interpolated volatility
    """
    if len(strikes) != len(vols):
        raise ValueError("Strikes and vols must have same length")
    
    return np.interp(target_strike, strikes, vols)
