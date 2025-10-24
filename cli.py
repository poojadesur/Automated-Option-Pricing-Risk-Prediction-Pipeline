#!/usr/bin/env python3
"""
Command-line interface for the option pricing pipeline.
"""
import argparse
import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from option_pricing_pipeline.models.quantitative import (
    BlackScholesModel,
    MonteCarloSimulation
)
from option_pricing_pipeline.pipeline.orchestration import (
    option_pricing_training_pipeline,
    option_pricing_inference_pipeline
)


def price_option(args):
    """Price an option using Black-Scholes."""
    bs_model = BlackScholesModel()
    
    if args.option_type == 'call':
        price = bs_model.price_call(
            args.stock_price,
            args.strike_price,
            args.time_to_maturity,
            args.risk_free_rate,
            args.volatility
        )
    else:
        price = bs_model.price_put(
            args.stock_price,
            args.strike_price,
            args.time_to_maturity,
            args.risk_free_rate,
            args.volatility
        )
    
    print(f"\n{'='*60}")
    print(f"Option Pricing Result")
    print(f"{'='*60}")
    print(f"Option Type: {args.option_type.upper()}")
    print(f"Stock Price: ${args.stock_price:.2f}")
    print(f"Strike Price: ${args.strike_price:.2f}")
    print(f"Time to Maturity: {args.time_to_maturity:.2f} years")
    print(f"Risk-free Rate: {args.risk_free_rate*100:.2f}%")
    print(f"Volatility: {args.volatility*100:.2f}%")
    print(f"\nOption Price: ${price:.4f}")
    print(f"{'='*60}\n")


def calculate_greeks(args):
    """Calculate option Greeks."""
    bs_model = BlackScholesModel()
    
    greeks = bs_model.calculate_greeks(
        args.stock_price,
        args.strike_price,
        args.time_to_maturity,
        args.risk_free_rate,
        args.volatility,
        args.option_type
    )
    
    print(f"\n{'='*60}")
    print(f"Option Greeks")
    print(f"{'='*60}")
    for greek, value in greeks.items():
        print(f"{greek.capitalize():8s}: {value:10.6f}")
    print(f"{'='*60}\n")


def monte_carlo_price(args):
    """Price option using Monte Carlo simulation."""
    mc_sim = MonteCarloSimulation(
        n_simulations=args.n_simulations,
        n_steps=args.n_steps,
        seed=args.seed
    )
    
    result = mc_sim.price_european_option(
        args.stock_price,
        args.strike_price,
        args.time_to_maturity,
        args.risk_free_rate,
        args.volatility,
        args.option_type
    )
    
    print(f"\n{'='*60}")
    print(f"Monte Carlo Pricing Result")
    print(f"{'='*60}")
    print(f"Simulations: {args.n_simulations}")
    print(f"Option Price: ${result['price']:.4f}")
    print(f"Standard Error: ${result['std_error']:.4f}")
    print(f"95% CI: [${result['confidence_interval'][0]:.4f}, ${result['confidence_interval'][1]:.4f}]")
    print(f"{'='*60}\n")


def train_models(args):
    """Train ML models."""
    print(f"\n{'='*60}")
    print(f"Training Models")
    print(f"{'='*60}\n")
    
    results = option_pricing_training_pipeline(
        n_samples=args.n_samples,
        n_simulations=args.n_simulations,
        model_type=args.model_type,
        output_dir=args.output_dir,
        track_mlflow=not args.no_mlflow
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"\nModels saved to: {args.output_dir}")
    print(f"Pricing Model R²: {results['pricing_metrics']['test_r2']:.4f}")
    print(f"{'='*60}\n")


def run_inference(args):
    """Run inference with trained models."""
    input_params = {
        'S': args.stock_price,
        'K': args.strike_price,
        'T': args.time_to_maturity,
        'r': args.risk_free_rate,
        'sigma': args.volatility
    }
    
    results = option_pricing_inference_pipeline(
        input_params=input_params,
        model_dir=args.model_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Inference Results")
    print(f"{'='*60}")
    print(f"\nML Predicted Price: ${results['ml_predicted_price']:.4f}")
    print(f"Black-Scholes Price: ${results['bs_price']:.4f}")
    print(f"\nPredicted Risk Metrics:")
    for metric, value in results['predicted_risks'].items():
        print(f"  {metric.upper():8s}: {value:.6f}")
    print(f"{'='*60}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Option Pricing and Risk Prediction Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Price command
    price_parser = subparsers.add_parser('price', help='Price an option using Black-Scholes')
    price_parser.add_argument('-S', '--stock-price', type=float, required=True, help='Current stock price')
    price_parser.add_argument('-K', '--strike-price', type=float, required=True, help='Strike price')
    price_parser.add_argument('-T', '--time-to-maturity', type=float, required=True, help='Time to maturity (years)')
    price_parser.add_argument('-r', '--risk-free-rate', type=float, required=True, help='Risk-free rate')
    price_parser.add_argument('-v', '--volatility', type=float, required=True, help='Volatility')
    price_parser.add_argument('-t', '--option-type', choices=['call', 'put'], default='call', help='Option type')
    price_parser.set_defaults(func=price_option)
    
    # Greeks command
    greeks_parser = subparsers.add_parser('greeks', help='Calculate option Greeks')
    greeks_parser.add_argument('-S', '--stock-price', type=float, required=True)
    greeks_parser.add_argument('-K', '--strike-price', type=float, required=True)
    greeks_parser.add_argument('-T', '--time-to-maturity', type=float, required=True)
    greeks_parser.add_argument('-r', '--risk-free-rate', type=float, required=True)
    greeks_parser.add_argument('-v', '--volatility', type=float, required=True)
    greeks_parser.add_argument('-t', '--option-type', choices=['call', 'put'], default='call')
    greeks_parser.set_defaults(func=calculate_greeks)
    
    # Monte Carlo command
    mc_parser = subparsers.add_parser('montecarlo', help='Price using Monte Carlo simulation')
    mc_parser.add_argument('-S', '--stock-price', type=float, required=True)
    mc_parser.add_argument('-K', '--strike-price', type=float, required=True)
    mc_parser.add_argument('-T', '--time-to-maturity', type=float, required=True)
    mc_parser.add_argument('-r', '--risk-free-rate', type=float, required=True)
    mc_parser.add_argument('-v', '--volatility', type=float, required=True)
    mc_parser.add_argument('-t', '--option-type', choices=['call', 'put'], default='call')
    mc_parser.add_argument('-n', '--n-simulations', type=int, default=10000, help='Number of simulations')
    mc_parser.add_argument('--n-steps', type=int, default=252, help='Number of time steps')
    mc_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    mc_parser.set_defaults(func=monte_carlo_price)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('-n', '--n-samples', type=int, default=10000, help='Number of training samples')
    train_parser.add_argument('-s', '--n-simulations', type=int, default=5000, help='MC simulations per sample')
    train_parser.add_argument('-m', '--model-type', default='xgboost', help='Model type')
    train_parser.add_argument('-o', '--output-dir', default='models/saved', help='Output directory')
    train_parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    train_parser.set_defaults(func=train_models)
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference with trained models')
    infer_parser.add_argument('-S', '--stock-price', type=float, required=True)
    infer_parser.add_argument('-K', '--strike-price', type=float, required=True)
    infer_parser.add_argument('-T', '--time-to-maturity', type=float, required=True)
    infer_parser.add_argument('-r', '--risk-free-rate', type=float, required=True)
    infer_parser.add_argument('-v', '--volatility', type=float, required=True)
    infer_parser.add_argument('-d', '--model-dir', default='models/saved', help='Model directory')
    infer_parser.set_defaults(func=run_inference)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
