#!/usr/bin/env python3
"""
Example script demonstrating visualization capabilities.
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
from option_pricing_pipeline.utils.visualization import (
    plot_option_prices_vs_strike,
    plot_monte_carlo_paths,
    plot_var_distribution
)


def main():
    """Create visualizations."""
    print("=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    S = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    # 1. Option Prices vs Strike
    print("\n1. Creating option prices vs strike plot...")
    strikes = np.linspace(80, 120, 50)
    bs_model = BlackScholesModel()
    
    call_prices = [bs_model.price_call(S, K, T, r, sigma) for K in strikes]
    put_prices = [bs_model.price_put(S, K, T, r, sigma) for K in strikes]
    
    plot_option_prices_vs_strike(
        strikes,
        np.array(call_prices),
        np.array(put_prices),
        title=f"Option Prices (S=${S}, T={T}yr, σ={sigma*100}%)",
        save_path=str(output_dir / "option_prices_vs_strike.png")
    )
    
    # 2. Monte Carlo Paths
    print("2. Creating Monte Carlo simulation paths...")
    mc_sim = MonteCarloSimulation(n_simulations=1000, n_steps=252, seed=42)
    paths = mc_sim.simulate_paths(S, T, r, sigma)
    
    plot_monte_carlo_paths(
        paths,
        n_paths_to_plot=100,
        title=f"Monte Carlo Paths (S0=${S}, σ={sigma*100}%)",
        save_path=str(output_dir / "monte_carlo_paths.png")
    )
    
    # 3. VaR Distribution
    print("3. Creating VaR distribution plot...")
    terminal_prices = paths[:, -1]
    returns = (terminal_prices - S) / S
    
    var_metrics = mc_sim.calculate_var(S, T, r, sigma, confidence_level=0.95)
    
    plot_var_distribution(
        returns,
        var_metrics['var'],
        var_metrics['cvar'],
        confidence_level=0.95,
        title="Return Distribution with VaR and CVaR (95% Confidence)",
        save_path=str(output_dir / "var_distribution.png")
    )
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}/")
    print("=" * 60)
    print("\nGenerated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
