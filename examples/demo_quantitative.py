#!/usr/bin/env python3
"""
Example script demonstrating Black-Scholes and Monte Carlo pricing.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from option_pricing_pipeline.models.quantitative import (
    BlackScholesModel,
    MonteCarloSimulation
)


def main():
    """Demonstrate quantitative models."""
    print("=" * 60)
    print("Quantitative Option Pricing Models")
    print("=" * 60)
    
    # Parameters
    S = 100.0  # Stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    print("\nParameters:")
    print(f"  S = ${S:.2f}")
    print(f"  K = ${K:.2f}")
    print(f"  T = {T:.2f} years")
    print(f"  r = {r*100:.2f}%")
    print(f"  σ = {sigma*100:.2f}%")
    
    # Black-Scholes Model
    print("\n" + "-" * 60)
    print("Black-Scholes Model")
    print("-" * 60)
    
    bs_model = BlackScholesModel()
    
    call_price = bs_model.price_call(S, K, T, r, sigma)
    put_price = bs_model.price_put(S, K, T, r, sigma)
    
    print(f"\nCall Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    
    # Greeks
    greeks = bs_model.calculate_greeks(S, K, T, r, sigma, 'call')
    print("\nGreeks (Call Option):")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():6s}: {value:10.6f}")
    
    # Monte Carlo Simulation
    print("\n" + "-" * 60)
    print("Monte Carlo Simulation")
    print("-" * 60)
    
    mc_sim = MonteCarloSimulation(n_simulations=10000, seed=42)
    
    # Price call option
    mc_call = mc_sim.price_european_option(S, K, T, r, sigma, 'call')
    print(f"\nCall Option Price: ${mc_call['price']:.4f}")
    print(f"Standard Error: ${mc_call['std_error']:.4f}")
    print(f"95% CI: [${mc_call['confidence_interval'][0]:.4f}, "
          f"${mc_call['confidence_interval'][1]:.4f}]")
    
    # Price put option
    mc_put = mc_sim.price_european_option(S, K, T, r, sigma, 'put')
    print(f"\nPut Option Price: ${mc_put['price']:.4f}")
    print(f"Standard Error: ${mc_put['std_error']:.4f}")
    
    # Compare prices
    print("\n" + "-" * 60)
    print("Comparison")
    print("-" * 60)
    print(f"\nCall Price Difference (MC - BS): ${mc_call['price'] - call_price:.4f}")
    print(f"Put Price Difference (MC - BS): ${mc_put['price'] - put_price:.4f}")
    
    # Risk Metrics
    print("\n" + "-" * 60)
    print("Risk Metrics (VaR and CVaR)")
    print("-" * 60)
    
    var_metrics = mc_sim.calculate_var(S, T, r, sigma, confidence_level=0.95)
    print(f"\nValue at Risk (95%): {var_metrics['var']*100:.2f}%")
    print(f"Conditional VaR (95%): {var_metrics['cvar']*100:.2f}%")
    print(f"VaR (absolute): ${var_metrics['var_absolute']:.2f}")
    print(f"CVaR (absolute): ${var_metrics['cvar_absolute']:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
