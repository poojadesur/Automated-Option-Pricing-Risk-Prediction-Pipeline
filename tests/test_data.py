"""
Tests for data generation and ETL pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from option_pricing_pipeline.data.generator import (
    OptionDataGenerator,
    DataProcessor,
    ETLPipeline
)


class TestOptionDataGenerator:
    """Test data generator."""
    
    def test_generate_training_data(self):
        """Test training data generation."""
        generator = OptionDataGenerator(seed=42)
        df = generator.generate_training_data(n_samples=100)
        
        # Check shape
        assert len(df) == 100
        assert set(df.columns) == {'S', 'K', 'T', 'r', 'sigma'}
        
        # Check ranges
        assert df['S'].min() >= 50
        assert df['S'].max() <= 150
        assert df['T'].min() >= 0.1
        assert df['T'].max() <= 2.0
        
        # No missing values
        assert not df.isnull().any().any()
    
    def test_generate_market_scenarios(self):
        """Test market scenario generation."""
        generator = OptionDataGenerator(seed=42)
        df = generator.generate_market_scenarios(n_scenarios=100)
        
        assert len(df) == 100
        assert set(df.columns) == {'S', 'K', 'T', 'r', 'sigma'}
        
        # All values should be positive
        assert (df > 0).all().all()


class TestDataProcessor:
    """Test data processor."""
    
    def test_add_features(self):
        """Test feature engineering."""
        processor = DataProcessor()
        
        # Create sample data
        df = pd.DataFrame({
            'S': [100, 110, 90],
            'K': [100, 100, 100],
            'T': [1.0, 1.0, 1.0],
            'r': [0.05, 0.05, 0.05],
            'sigma': [0.2, 0.2, 0.2]
        })
        
        df_with_features = processor.add_features(df)
        
        # Check new features
        assert 'moneyness' in df_with_features.columns
        assert 'log_moneyness' in df_with_features.columns
        assert 'vol_time' in df_with_features.columns
        assert 'r_time' in df_with_features.columns
        
        # Check moneyness calculation
        assert df_with_features.loc[0, 'moneyness'] == 1.0
        assert df_with_features.loc[1, 'moneyness'] == 1.1
        assert df_with_features.loc[2, 'moneyness'] == 0.9
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'S': [100, 110],
            'K': [100, 100],
            'T': [1.0, 1.0],
            'r': [0.05, 0.05],
            'sigma': [0.2, 0.2]
        })
        
        assert processor.validate_data(df) is True
    
    def test_validate_data_invalid(self):
        """Test data validation with invalid data."""
        processor = DataProcessor()
        
        # Negative stock price
        df = pd.DataFrame({
            'S': [-100, 110],
            'K': [100, 100],
            'T': [1.0, 1.0],
            'r': [0.05, 0.05],
            'sigma': [0.2, 0.2]
        })
        
        assert processor.validate_data(df) is False
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'S': [100, 110, 90],
            'K': [100, 100, 100],
            'T': [1.0, 1.0, 1.0]
        })
        
        stats = processor.calculate_statistics(df)
        
        assert 'n_samples' in stats
        assert stats['n_samples'] == 3
        assert 'feature_means' in stats
        assert 'feature_stds' in stats


class TestETLPipeline:
    """Test ETL pipeline."""
    
    def test_run_pipeline(self, tmp_path):
        """Test complete ETL pipeline."""
        pipeline = ETLPipeline(data_dir=str(tmp_path))
        
        df = pipeline.run_pipeline(
            n_samples=100,
            add_features=True,
            save_output=False
        )
        
        # Check output
        assert len(df) == 100
        assert 'S' in df.columns
        assert 'moneyness' in df.columns
