"""
Tests for ML models.
"""
import pytest
import pandas as pd
import numpy as np
from option_pricing_pipeline.models.ml_models import (
    OptionPricingMLModel,
    RiskPredictionModel
)


class TestOptionPricingMLModel:
    """Test option pricing ML model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = OptionPricingMLModel(model_type='xgboost')
        assert model.model_type == 'xgboost'
        assert model.is_fitted is False
    
    def test_prepare_features(self):
        """Test feature preparation."""
        model = OptionPricingMLModel()
        
        df = pd.DataFrame({
            'S': [100, 110],
            'K': [100, 100],
            'T': [1.0, 1.0],
            'r': [0.05, 0.05],
            'sigma': [0.2, 0.2]
        })
        
        features = model.prepare_features(df)
        assert features.shape == (2, 5)
    
    def test_train_predict(self):
        """Test model training and prediction."""
        model = OptionPricingMLModel(model_type='random_forest')
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            'S': np.random.uniform(80, 120, n_samples),
            'K': np.random.uniform(80, 120, n_samples),
            'T': np.random.uniform(0.5, 1.5, n_samples),
            'r': np.random.uniform(0.03, 0.07, n_samples),
            'sigma': np.random.uniform(0.15, 0.25, n_samples)
        })
        
        # Generate target (simple formula for testing)
        y = X['S'] - X['K'] + np.random.normal(0, 1, n_samples)
        
        # Train
        metrics = model.train(X, y, test_size=0.2, scale_features=False)
        
        assert model.is_fitted is True
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert 'mae' in metrics
        
        # Predict
        predictions = model.predict(X.head(10), scale_features=False)
        assert len(predictions) == 10
    
    def test_save_load(self, tmp_path):
        """Test model save and load."""
        model = OptionPricingMLModel(model_type='xgboost')
        
        # Create and train simple model
        X = pd.DataFrame({
            'S': [100, 110, 90],
            'K': [100, 100, 100],
            'T': [1.0, 1.0, 1.0],
            'r': [0.05, 0.05, 0.05],
            'sigma': [0.2, 0.2, 0.2]
        })
        y = np.array([10, 15, 5])
        
        model.train(X, y, test_size=0.3, scale_features=False)
        
        # Save
        model_path = tmp_path / "test_model.joblib"
        model.save(str(model_path))
        assert model_path.exists()
        
        # Load
        new_model = OptionPricingMLModel()
        new_model.load(str(model_path))
        assert new_model.is_fitted is True
        assert new_model.model_type == 'xgboost'


class TestRiskPredictionModel:
    """Test risk prediction model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = RiskPredictionModel(model_type='xgboost')
        assert model.model_type == 'xgboost'
        assert model.is_fitted is False
    
    def test_train_predict(self):
        """Test training and prediction of risk models."""
        model = RiskPredictionModel(model_type='random_forest')
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            'S': np.random.uniform(80, 120, n_samples),
            'K': np.random.uniform(80, 120, n_samples),
            'T': np.random.uniform(0.5, 1.5, n_samples),
            'r': np.random.uniform(0.03, 0.07, n_samples),
            'sigma': np.random.uniform(0.15, 0.25, n_samples)
        })
        
        # Generate risk metrics
        y = {
            'var': np.random.uniform(-0.2, -0.05, n_samples),
            'cvar': np.random.uniform(-0.3, -0.1, n_samples),
            'delta': np.random.uniform(0.3, 0.7, n_samples)
        }
        
        # Train
        metrics = model.train(X, y, test_size=0.2)
        
        assert model.is_fitted is True
        assert 'var' in metrics
        assert 'cvar' in metrics
        assert 'delta' in metrics
        
        # Predict
        predictions = model.predict(X.head(10))
        assert 'var' in predictions
        assert 'cvar' in predictions
        assert 'delta' in predictions
        assert len(predictions['var']) == 10
