"""
Data generation and ETL pipeline for option pricing.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionDataGenerator:
    """
    Generate synthetic option data for training and testing.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_training_data(
        self,
        n_samples: int = 10000,
        S_range: tuple = (50, 150),
        K_range: tuple = (50, 150),
        T_range: tuple = (0.1, 2.0),
        r_range: tuple = (0.01, 0.10),
        sigma_range: tuple = (0.10, 0.50)
    ) -> pd.DataFrame:
        """
        Generate training data with random parameters.
        
        Args:
            n_samples: Number of samples to generate
            S_range: Range for stock price (min, max)
            K_range: Range for strike price (min, max)
            T_range: Range for time to maturity (min, max)
            r_range: Range for risk-free rate (min, max)
            sigma_range: Range for volatility (min, max)
            
        Returns:
            DataFrame with generated parameters
        """
        data = {
            'S': np.random.uniform(S_range[0], S_range[1], n_samples),
            'K': np.random.uniform(K_range[0], K_range[1], n_samples),
            'T': np.random.uniform(T_range[0], T_range[1], n_samples),
            'r': np.random.uniform(r_range[0], r_range[1], n_samples),
            'sigma': np.random.uniform(sigma_range[0], sigma_range[1], n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {n_samples} training samples")
        
        return df
    
    def generate_market_scenarios(
        self,
        n_scenarios: int = 1000,
        base_price: float = 100.0,
        price_volatility: float = 0.20
    ) -> pd.DataFrame:
        """
        Generate market scenarios with realistic correlations.
        
        Args:
            n_scenarios: Number of scenarios
            base_price: Base stock price
            price_volatility: Volatility for price generation
            
        Returns:
            DataFrame with market scenarios
        """
        # Generate correlated random variables
        mean = [0, 0, 0]
        cov = [[1, 0.3, -0.2],
               [0.3, 1, 0.1],
               [-0.2, 0.1, 1]]
        
        random_vars = np.random.multivariate_normal(mean, cov, n_scenarios)
        
        # Transform to desired ranges
        S = base_price * np.exp(random_vars[:, 0] * price_volatility)
        K = base_price * (1 + 0.1 * random_vars[:, 1])  # Strike around ATM
        sigma = 0.20 + 0.10 * random_vars[:, 2]  # Volatility around 20%
        
        # Generate other parameters
        T = np.random.uniform(0.1, 2.0, n_scenarios)
        r = np.random.uniform(0.02, 0.06, n_scenarios)
        
        df = pd.DataFrame({
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': np.clip(sigma, 0.05, 0.60)
        })
        
        logger.info(f"Generated {n_scenarios} market scenarios")
        return df


class DataProcessor:
    """
    Process and transform option data.
    """
    
    def __init__(self):
        self.statistics = {}
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Moneyness
        df['moneyness'] = df['S'] / df['K']
        df['log_moneyness'] = np.log(df['moneyness'])
        
        # Time-volatility product
        df['vol_time'] = df['sigma'] * np.sqrt(df['T'])
        
        # Interest rate impact
        df['r_time'] = df['r'] * df['T']
        
        # Categories
        df['option_category'] = pd.cut(
            df['moneyness'],
            bins=[0, 0.95, 1.05, np.inf],
            labels=['OTM', 'ATM', 'ITM']
        )
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate dataset statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_samples': len(df),
            'feature_means': df.select_dtypes(include=[np.number]).mean().to_dict(),
            'feature_stds': df.select_dtypes(include=[np.number]).std().to_dict(),
            'feature_mins': df.select_dtypes(include=[np.number]).min().to_dict(),
            'feature_maxs': df.select_dtypes(include=[np.number]).max().to_dict()
        }
        
        self.statistics = stats
        return stats
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if data is valid
        """
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Data contains missing values")
            return False
        
        # Check for negative values where they shouldn't exist
        positive_columns = ['S', 'K', 'T', 'sigma']
        for col in positive_columns:
            if col in df.columns and (df[col] <= 0).any():
                logger.warning(f"Column {col} contains non-positive values")
                return False
        
        # Check for reasonable ranges
        if 'r' in df.columns:
            if (df['r'] < -0.1).any() or (df['r'] > 0.5).any():
                logger.warning("Interest rate out of reasonable range")
                return False
        
        logger.info("Data validation passed")
        return True
    
    def save_data(self, df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """
        Save processed data to disk.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: File format ('csv' or 'parquet')
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from disk.
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded DataFrame
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Data loaded from {filepath}")
        return df


class ETLPipeline:
    """
    Complete ETL pipeline for option data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize ETL pipeline.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.generator = OptionDataGenerator()
        self.processor = DataProcessor()
    
    def run_pipeline(
        self,
        n_samples: int = 10000,
        add_features: bool = True,
        save_output: bool = True
    ) -> pd.DataFrame:
        """
        Run complete ETL pipeline.
        
        Args:
            n_samples: Number of samples to generate
            add_features: Whether to add derived features
            save_output: Whether to save processed data
            
        Returns:
            Processed DataFrame
        """
        logger.info("Starting ETL pipeline...")
        
        # Extract: Generate data
        df = self.generator.generate_training_data(n_samples=n_samples)
        
        # Transform: Process data
        if add_features:
            df = self.processor.add_features(df)
        
        # Validate
        if not self.processor.validate_data(df):
            raise ValueError("Data validation failed")
        
        # Calculate statistics
        stats = self.processor.calculate_statistics(df)
        logger.info(f"Dataset statistics: {stats['n_samples']} samples")
        
        # Load: Save processed data
        if save_output:
            output_path = self.data_dir / "processed" / "training_data.csv"
            self.processor.save_data(df, str(output_path))
        
        logger.info("ETL pipeline completed successfully")
        return df
