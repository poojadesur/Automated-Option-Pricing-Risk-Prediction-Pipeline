#!/usr/bin/env python3
"""
Example script for inference - pricing options and predicting risk.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from option_pricing_pipeline.pipeline.orchestration import (
    option_pricing_inference_pipeline
)


def main():
    """Run inference pipeline."""
    print("=" * 60)
    print("Option Pricing and Risk Prediction - Inference")
    print("=" * 60)
    
    # Example option parameters
    input_params = {
        'S': 100.0,     # Current stock price
        'K': 100.0,     # Strike price
        'T': 1.0,       # Time to maturity (1 year)
        'r': 0.05,      # Risk-free rate (5%)
        'sigma': 0.2    # Volatility (20%)
    }
    
    print("\nOption Parameters:")
    print(f"  Stock Price (S): ${input_params['S']:.2f}")
    print(f"  Strike Price (K): ${input_params['K']:.2f}")
    print(f"  Time to Maturity (T): {input_params['T']:.2f} years")
    print(f"  Risk-free Rate (r): {input_params['r']*100:.2f}%")
    print(f"  Volatility (σ): {input_params['sigma']*100:.2f}%")
    
    print("\nRunning inference...")
    
    # Run inference
    results = option_pricing_inference_pipeline(
        input_params=input_params,
        model_dir='models/saved'
    )
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print("\nOption Pricing:")
    print(f"  ML Model Price: ${results['ml_predicted_price']:.4f}")
    print(f"  Black-Scholes Price: ${results['bs_price']:.4f}")
    print(f"  Difference: ${abs(results['ml_predicted_price'] - results['bs_price']):.4f}")
    
    print("\nRisk Metrics (ML Predictions):")
    for metric, value in results['predicted_risks'].items():
        print(f"  {metric.upper()}: {value:.6f}")
    
    print("\nGreeks (Black-Scholes):")
    for greek, value in results['bs_greeks'].items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
