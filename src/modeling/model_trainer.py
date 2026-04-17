import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.modeling.xgboost_model import XGBoostModel
from src.modeling.random_forest_model import RandomForestModel
from src.modeling.time_series_cross_validator import TimeSeriesCrossValidator
from src.modeling.model_evaluator import ModelEvaluator
from src.utils.logger import get_logger
from src.utils.constants import CV_FOLDS

logger = get_logger(__name__)


class ModelTrainerError(Exception):
    """Custom exception for ModelTrainer errors."""
    pass


class ModelTrainer:
    
    def __init__(
        self,
        n_cv_folds: int = CV_FOLDS,
        random_state: int = 42
    ):
        
        if n_cv_folds < 1:
            raise ModelTrainerError("n_cv_folds must be >= 1")
        
        self.models = {}
        self.cv_validator = TimeSeriesCrossValidator(n_splits=n_cv_folds)
        self.evaluator = ModelEvaluator()
        self.training_history = []
        self.random_state = random_state
        
        logger.info(
            f"ModelTrainer initialized with {n_cv_folds} CV folds, "
            f"random_state={random_state}"
        )
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None,
        use_cv: bool = True,
        verbose: bool = True
    ) -> Dict:
        
        try:
            if verbose:
                logger.info("Starting XGBoost model training")
            
            # Create model
            model = XGBoostModel(
                hyperparameters=hyperparameters,
                random_state=self.random_state
            )
            
            # Get CV splits if using cross-validation
            cv_splits = None
            if use_cv:
                cv_splits = self.cv_validator.split(X_train, y_train)
            
            # Train model
            cv_results = model.train(
                X_train,
                y_train,
                cv_splits=cv_splits,
                verbose=verbose
            )
            
            # Store model
            self.models['xgboost'] = model
            
            # Record training
            training_record = {
                'timestamp': datetime.now(),
                'model_type': 'XGBoost',
                'n_samples': len(X_train),
                'n_features': len(X_train.columns),
                'use_cv': use_cv,
                'cv_results': cv_results
            }
            self.training_history.append(training_record)
            
            if verbose:
                logger.info("XGBoost model training completed")
            
            return {
                'model_type': 'XGBoost',
                'cv_results': cv_results,
                'model': model
            }
        
        except Exception as e:
            raise ModelTrainerError(f"XGBoost training failed: {str(e)}")
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None,
        use_cv: bool = True,
        verbose: bool = True
    ) -> Dict:
        
        try:
            if verbose:
                logger.info("Starting Random Forest model training")
            
            # Create model
            model = RandomForestModel(
                hyperparameters=hyperparameters,
                random_state=self.random_state
            )
            
            # Get CV splits if using cross-validation
            cv_splits = None
            if use_cv:
                cv_splits = self.cv_validator.split(X_train, y_train)
            
            # Train model
            cv_results = model.train(
                X_train,
                y_train,
                cv_splits=cv_splits,
                verbose=verbose
            )
            
            # Store model
            self.models['random_forest'] = model
            
            # Record training
            training_record = {
                'timestamp': datetime.now(),
                'model_type': 'Random Forest',
                'n_samples': len(X_train),
                'n_features': len(X_train.columns),
                'use_cv': use_cv,
                'cv_results': cv_results
            }
            self.training_history.append(training_record)
            
            if verbose:
                logger.info("Random Forest model training completed")
            
            return {
                'model_type': 'Random Forest',
                'cv_results': cv_results,
                'model': model
            }
        
        except Exception as e:
            raise ModelTrainerError(f"Random Forest training failed: {str(e)}")
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        xgboost_params: Optional[Dict] = None,
        random_forest_params: Optional[Dict] = None,
        use_cv: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        
        results = {}
        
        try:
            # Train XGBoost
            results['xgboost'] = self.train_xgboost(
                X_train,
                y_train,
                hyperparameters=xgboost_params,
                use_cv=use_cv,
                verbose=verbose
            )
            
            # Train Random Forest
            results['random_forest'] = self.train_random_forest(
                X_train,
                y_train,
                hyperparameters=random_forest_params,
                use_cv=use_cv,
                verbose=verbose
            )
            
            logger.info("All models trained successfully")
            
            return results
        
        except Exception as e:
            raise ModelTrainerError(f"Training all models failed: {str(e)}")
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict:
        
        if model_name not in self.models:
            raise ModelTrainerError(f"Model '{model_name}' not found")
        
        try:
            model = self.models[model_name]
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            metrics = self.evaluator.evaluate(y_test, y_pred)
            
            if verbose:
                logger.info(
                    f"{model_name} evaluation: "
                    f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                    f"R²={metrics['r2']:.4f}"
                )
            
            return metrics
        
        except Exception as e:
            raise ModelTrainerError(f"Model evaluation failed: {str(e)}")
    
    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, Dict]:
       
        results = {}
        
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(
                model_name,
                X_test,
                y_test,
                verbose=verbose
            )
        
        return results
    
    def get_best_model(
        self,
        metric: str = 'r2'
    ) -> Tuple[str, object]:
        
        if not self.models:
            raise ModelTrainerError("No models trained")
        
        if metric not in ['r2', 'rmse', 'mae']:
            raise ModelTrainerError(f"Invalid metric: {metric}")
        
        best_model_name = None
        best_score = None
        
        for model_name, model in self.models.items():
            if not model.cv_results:
                continue
            
            if metric == 'r2':
                score = model.cv_results.get('r2_mean', 0)
                if best_score is None or score > best_score:
                    best_score = score
                    best_model_name = model_name
            
            elif metric == 'rmse':
                score = model.cv_results.get('rmse_mean', float('inf'))
                if best_score is None or score < best_score:
                    best_score = score
                    best_model_name = model_name
            
            elif metric == 'mae':
                score = model.cv_results.get('mae_mean', float('inf'))
                if best_score is None or score < best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ModelTrainerError("No valid models found")
        
        logger.info(
            f"Best model: {best_model_name} ({metric}={best_score:.4f})"
        )
        
        return best_model_name, self.models[best_model_name]
    
    def get_model(self, model_name: str) -> object:
        
        if model_name not in self.models:
            raise ModelTrainerError(f"Model '{model_name}' not found")
        
        return self.models[model_name]
    
    def get_training_summary(self) -> Dict:
      
        summary = {
            'n_training_runs': len(self.training_history),
            'models_trained': list(self.models.keys()),
            'training_history': self.training_history
        }
        
        return summary
