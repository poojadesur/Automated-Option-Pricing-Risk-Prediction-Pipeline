"""
Monitoring and logging utilities for MLOps.
"""
import logging
import mlflow
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitor model performance and data drift.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize model monitor.
        
        Args:
            log_dir: Directory for logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.performance_log = []
    
    def log_prediction(
        self,
        input_data: Dict,
        prediction: float,
        actual: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a prediction for monitoring.
        
        Args:
            input_data: Input features
            prediction: Model prediction
            actual: Actual value (if available)
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }
        
        if actual is not None:
            log_entry['error'] = abs(prediction - actual)
            log_entry['percentage_error'] = abs((prediction - actual) / actual * 100)
        
        self.performance_log.append(log_entry)
        
        # Save to file periodically
        if len(self.performance_log) % 100 == 0:
            self._save_logs()
    
    def _save_logs(self):
        """Save logs to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"predictions_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        logger.info(f"Saved {len(self.performance_log)} predictions to {log_file}")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics from logged predictions.
        
        Returns:
            Dictionary of metrics
        """
        if not self.performance_log:
            return {}
        
        errors = [entry['error'] for entry in self.performance_log if 'error' in entry]
        
        if not errors:
            return {'n_predictions': len(self.performance_log)}
        
        metrics = {
            'n_predictions': len(self.performance_log),
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
            'max_error': max(errors),
            'min_error': min(errors)
        }
        
        return metrics
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (production data)
            threshold: Drift threshold
            
        Returns:
            Dictionary with drift detection results
        """
        drift_detected = {}
        
        for column in reference_data.columns:
            if column not in current_data.columns:
                continue
            
            ref_mean = reference_data[column].mean()
            ref_std = reference_data[column].std()
            
            curr_mean = current_data[column].mean()
            curr_std = current_data[column].std()
            
            # Calculate relative change
            mean_change = abs((curr_mean - ref_mean) / ref_mean) if ref_mean != 0 else 0
            std_change = abs((curr_std - ref_std) / ref_std) if ref_std != 0 else 0
            
            drift_detected[column] = {
                'mean_change': mean_change,
                'std_change': std_change,
                'drift_detected': mean_change > threshold or std_change > threshold
            }
        
        return drift_detected


class MLflowTracker:
    """
    MLflow experiment tracking utilities.
    """
    
    def __init__(self, experiment_name: str = "option_pricing"):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None
    ):
        """
        Log a training run to MLflow.
        
        Args:
            model_name: Name of the model
            params: Training parameters
            metrics: Performance metrics
            artifacts: Paths to artifacts to log
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_name)
            
            logger.info(f"Logged training run for {model_name}")
    
    def log_model(self, model: Any, model_name: str):
        """
        Log model to MLflow.
        
        Args:
            model: Model object
            model_name: Name for the model
        """
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged model {model_name}")


class PerformanceLogger:
    """
    Log and track system performance.
    """
    
    def __init__(self, log_file: str = "logs/performance.log"):
        """
        Initialize performance logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure file logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def log_pipeline_execution(
        self,
        pipeline_name: str,
        execution_time: float,
        status: str,
        metrics: Optional[Dict] = None
    ):
        """
        Log pipeline execution.
        
        Args:
            pipeline_name: Name of the pipeline
            execution_time: Execution time in seconds
            status: Status (success/failure)
            metrics: Additional metrics
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'pipeline': pipeline_name,
            'execution_time': execution_time,
            'status': status,
            'metrics': metrics or {}
        }
        
        logger.info(f"Pipeline execution: {json.dumps(log_data)}")
    
    def log_model_inference(
        self,
        model_name: str,
        inference_time: float,
        batch_size: int
    ):
        """
        Log model inference performance.
        
        Args:
            model_name: Name of the model
            inference_time: Inference time in seconds
            batch_size: Number of samples processed
        """
        throughput = batch_size / inference_time if inference_time > 0 else 0
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'inference_time': inference_time,
            'batch_size': batch_size,
            'throughput': throughput
        }
        
        logger.info(f"Model inference: {json.dumps(log_data)}")
