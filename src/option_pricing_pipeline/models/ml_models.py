"""
Machine learning models for option pricing approximation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionPricingMLModel:
    """
    Machine learning models for fast option price approximation.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize ML model.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'gradient_boosting', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['S', 'K', 'T', 'r', 'sigma']
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model training/prediction.
        
        Args:
            data: DataFrame with columns S, K, T, r, sigma
            
        Returns:
            Feature array
        """
        features = data[self.feature_names].values
        return features
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """
        Train the ML model.
        
        Args:
            X: Feature DataFrame
            y: Target values (option prices)
            test_size: Proportion of data for testing
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X_array = self.prepare_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Training completed. Test R2: {test_score:.4f}, RMSE: {rmse:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame, scale_features: bool = True) -> np.ndarray:
        """
        Predict option prices.
        
        Args:
            X: Feature DataFrame
            scale_features: Whether to scale features
            
        Returns:
            Predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_array = self.prepare_features(X)
        
        if scale_features:
            X_array = self.scaler.transform(X_array)
        
        return self.model.predict(X_array)
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Model loaded from {filepath}")


class RiskPredictionModel:
    """
    ML models for predicting risk metrics.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize risk prediction model.
        
        Args:
            model_type: Type of model
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.risk_metrics = ['var', 'cvar', 'delta', 'gamma', 'vega']
        self.feature_names = ['S', 'K', 'T', 'r', 'sigma']
        self.is_fitted = False
    
    def train(
        self,
        X: pd.DataFrame,
        y: Dict[str, np.ndarray],
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Train models for each risk metric.
        
        Args:
            X: Feature DataFrame
            y: Dictionary of risk metric arrays
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with metrics for each risk type
        """
        all_metrics = {}
        X_array = X[self.feature_names].values
        
        for risk_metric in self.risk_metrics:
            if risk_metric not in y:
                continue
            
            logger.info(f"Training model for {risk_metric}...")
            
            # Initialize model and scaler
            if self.model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1
                )
            
            scaler = StandardScaler()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y[risk_metric], test_size=test_size, random_state=42
            )
            
            # Scale and train
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            y_pred = model.predict(X_test_scaled)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            
            all_metrics[risk_metric] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'mae': mae,
                'rmse': rmse
            }
            
            self.models[risk_metric] = model
            self.scalers[risk_metric] = scaler
            
            logger.info(f"{risk_metric} - Test R2: {test_score:.4f}, RMSE: {rmse:.6f}")
        
        self.is_fitted = True
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict risk metrics.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary of predicted risk metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_array = X[self.feature_names].values
        predictions = {}
        
        for risk_metric, model in self.models.items():
            X_scaled = self.scalers[risk_metric].transform(X_array)
            predictions[risk_metric] = model.predict(X_scaled)
        
        return predictions
    
    def save(self, filepath: str):
        """Save models to disk."""
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'model_type': self.model_type,
            'risk_metrics': self.risk_metrics,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Risk models saved to {filepath}")
    
    def load(self, filepath: str):
        """Load models from disk."""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scalers = data['scalers']
        self.model_type = data['model_type']
        self.risk_metrics = data['risk_metrics']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        logger.info(f"Risk models loaded from {filepath}")
