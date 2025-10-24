#!/usr/bin/env python3
"""
Example script to train the option pricing and risk prediction models.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from option_pricing_pipeline.pipeline.orchestration import (
    option_pricing_training_pipeline
)


def main():
    """Run the training pipeline."""
    print("=" * 60)
    print("Option Pricing and Risk Prediction - Training Pipeline")
    print("=" * 60)
    
    # Configure pipeline
    config = {
        'n_samples': 10000,
        'n_simulations': 5000,
        'model_type': 'xgboost',
        'output_dir': 'models/saved',
        'track_mlflow': True
    }
    
    print("\nPipeline Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training pipeline...")
    
    # Run pipeline
    results = option_pricing_training_pipeline(
        n_samples=config['n_samples'],
        n_simulations=config['n_simulations'],
        model_type=config['model_type'],
        output_dir=config['output_dir'],
        track_mlflow=config['track_mlflow']
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    print("\nPricing Model Metrics:")
    for key, value in results['pricing_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nRisk Model Metrics:")
    for risk_type, metrics in results['risk_metrics'].items():
        print(f"\n  {risk_type.upper()}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    print(f"\nModels saved to: {config['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
