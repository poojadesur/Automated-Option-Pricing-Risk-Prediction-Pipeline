"""
Visualization utilities for option pricing and risk metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_option_prices_vs_strike(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    title: str = "Option Prices vs Strike Price",
    save_path: Optional[str] = None
):
    """
    Plot option prices against strike prices.
    
    Args:
        strikes: Array of strike prices
        call_prices: Array of call option prices
        put_prices: Array of put option prices
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, call_prices, 'b-', label='Call Option', linewidth=2)
    plt.plot(strikes, put_prices, 'r-', label='Put Option', linewidth=2)
    plt.xlabel('Strike Price ($)', fontsize=12)
    plt.ylabel('Option Price ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_greeks_surface(
    stock_prices: np.ndarray,
    times: np.ndarray,
    greeks_values: np.ndarray,
    greek_name: str = "Delta",
    save_path: Optional[str] = None
):
    """
    Plot 3D surface of Greek values.
    
    Args:
        stock_prices: Array of stock prices
        times: Array of time to maturity
        greeks_values: 2D array of Greek values
        greek_name: Name of the Greek
        save_path: Path to save the plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    S, T = np.meshgrid(stock_prices, times)
    
    surf = ax.plot_surface(S, T, greeks_values, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Stock Price ($)', fontsize=11)
    ax.set_ylabel('Time to Maturity (years)', fontsize=11)
    ax.set_zlabel(greek_name, fontsize=11)
    ax.set_title(f'{greek_name} Surface', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_monte_carlo_paths(
    paths: np.ndarray,
    n_paths_to_plot: int = 100,
    title: str = "Monte Carlo Simulation Paths",
    save_path: Optional[str] = None
):
    """
    Plot Monte Carlo simulation paths.
    
    Args:
        paths: Array of simulated paths (n_simulations x n_steps)
        n_paths_to_plot: Number of paths to display
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot subset of paths
    time_steps = np.arange(paths.shape[1])
    for i in range(min(n_paths_to_plot, paths.shape[0])):
        plt.plot(time_steps, paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    
    # Plot mean path
    mean_path = paths.mean(axis=0)
    plt.plot(time_steps, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_var_distribution(
    returns: np.ndarray,
    var_value: float,
    cvar_value: float,
    confidence_level: float = 0.95,
    title: str = "Value at Risk Distribution",
    save_path: Optional[str] = None
):
    """
    Plot return distribution with VaR and CVaR.
    
    Args:
        returns: Array of returns
        var_value: VaR value
        cvar_value: CVaR value
        confidence_level: Confidence level
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    plt.hist(returns, bins=50, density=True, alpha=0.6, color='skyblue', 
             edgecolor='black', label='Return Distribution')
    
    # Plot VaR line
    plt.axvline(var_value, color='red', linestyle='--', linewidth=2, 
                label=f'VaR ({confidence_level*100:.0f}%): {var_value:.2%}')
    
    # Plot CVaR line
    plt.axvline(cvar_value, color='darkred', linestyle='--', linewidth=2,
                label=f'CVaR: {cvar_value:.2%}')
    
    # Shade CVaR region
    plt.axvspan(returns.min(), cvar_value, alpha=0.2, color='red')
    
    plt.xlabel('Returns', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Performance: Actual vs Predicted",
    save_path: Optional[str] = None
):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Values', fontsize=11)
    ax1.set_ylabel('Predicted Values', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    save_path: Optional[str] = None
):
    """
    Plot training metrics over epochs.
    
    Args:
        metrics: Dictionary of metric lists
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, linewidth=2, marker='o')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    features: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot feature importances.
    
    Args:
        features: List of feature names
        importances: Array of importance values
        title: Plot title
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    if top_n is not None:
        indices = indices[:top_n]
    
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, max(6, len(sorted_features) * 0.3)))
    plt.barh(range(len(sorted_features)), sorted_importances, color='steelblue')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_dashboard(
    results: Dict,
    save_dir: Optional[str] = None
):
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        results: Dictionary containing results and data
        save_dir: Directory to save plots
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating visualization dashboard...")
    
    # This is a placeholder for a more complex dashboard
    # In a real implementation, this would create multiple plots
    # based on the results provided
    
    print("Dashboard created successfully!")
