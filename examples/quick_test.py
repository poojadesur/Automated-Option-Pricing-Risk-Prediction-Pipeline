#!/usr/bin/env python3
"""
Quick test of the training pipeline with minimal samples.
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
    """Run quick training test."""
    print("=" * 60)
    print("Quick Training Pipeline Test")
    print("=" * 60)
    
    # Use minimal samples for fast testing
    config = {
        'n_samples': 1000,  # Much smaller for testing
        'n_simulations': 1000,  # Much smaller for testing
        'model_type': 'random_forest',  # Faster than xgboost
        'output_dir': 'models/test',
        'track_mlflow': False  # Skip MLflow for speed
    }
    
    print("\nTest Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting quick test...")
    
    # Run pipeline
    results = option_pricing_training_pipeline(
        n_samples=config['n_samples'],
        n_simulations=config['n_simulations'],
        model_type=config['model_type'],
        output_dir=config['output_dir'],
        track_mlflow=config['track_mlflow']
    )
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    print("\nPricing Model Metrics:")
    for key, value in results['pricing_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nRisk Model Metrics (sample):")
    for risk_type in list(results['risk_metrics'].keys())[:3]:
        print(f"\n  {risk_type.upper()}:")
        metrics = results['risk_metrics'][risk_type]
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    print(f"\nModels saved to: {config['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
