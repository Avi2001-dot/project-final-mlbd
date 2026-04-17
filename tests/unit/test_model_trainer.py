import pytest
import numpy as np
import pandas as pd

from src.modeling.model_trainer import ModelTrainer, ModelTrainerError


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    data = {
        'timestamp': dates,
        'aqi_lag_1h': np.random.uniform(50, 150, n_samples),
        'aqi_lag_3h': np.random.uniform(50, 150, n_samples),
        'aqi_lag_6h': np.random.uniform(50, 150, n_samples),
        'aqi_lag_12h': np.random.uniform(50, 150, n_samples),
        'aqi_lag_24h': np.random.uniform(50, 150, n_samples),
        'aqi_mean_3h': np.random.uniform(50, 150, n_samples),
        'aqi_std_3h': np.random.uniform(10, 50, n_samples),
        'aqi_min_3h': np.random.uniform(30, 100, n_samples),
        'aqi_max_3h': np.random.uniform(100, 200, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
    }
    
    X = pd.DataFrame(data)
    y = (
        0.5 * X['aqi_lag_1h'] +
        0.3 * X['aqi_mean_3h'] +
        0.1 * X['hour_of_day'] +
        np.random.normal(0, 5, n_samples)
    )
    y = pd.Series(y, name='aqi')
    
    return X, y


def test_trainer_initialization():
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer(n_cv_folds=3)
    
    assert trainer.models == {}
    assert trainer.cv_validator is not None
    assert trainer.evaluator is not None
    assert trainer.training_history == []


def test_trainer_initialization_invalid_folds():
    """Test initialization with invalid n_cv_folds."""
    with pytest.raises(ModelTrainerError):
        ModelTrainer(n_cv_folds=0)
    
    with pytest.raises(ModelTrainerError):
        ModelTrainer(n_cv_folds=-1)


def test_train_xgboost(sample_data):
    """Test training XGBoost model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    result = trainer.train_xgboost(X, y, use_cv=False, verbose=False)
    
    assert result['model_type'] == 'XGBoost'
    assert 'model' in result
    assert 'xgboost' in trainer.models


def test_train_xgboost_with_cv(sample_data):
    """Test training XGBoost with cross-validation."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    result = trainer.train_xgboost(X, y, use_cv=True, verbose=False)
    
    assert result['model_type'] == 'XGBoost'
    assert 'cv_results' in result
    assert 'rmse_mean' in result['cv_results']


def test_train_random_forest(sample_data):
    """Test training Random Forest model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    result = trainer.train_random_forest(X, y, use_cv=False, verbose=False)
    
    assert result['model_type'] == 'Random Forest'
    assert 'model' in result
    assert 'random_forest' in trainer.models


def test_train_random_forest_with_cv(sample_data):
    """Test training Random Forest with cross-validation."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    result = trainer.train_random_forest(X, y, use_cv=True, verbose=False)
    
    assert result['model_type'] == 'Random Forest'
    assert 'cv_results' in result
    assert 'rmse_mean' in result['cv_results']


def test_train_all_models(sample_data):
    """Test training all models."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    results = trainer.train_all_models(X, y, use_cv=False, verbose=False)
    
    assert 'xgboost' in results
    assert 'random_forest' in results
    assert len(trainer.models) == 2


def test_evaluate_model(sample_data):
    """Test evaluating a single model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    # Split data
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Train
    trainer.train_xgboost(X_train, y_train, use_cv=False, verbose=False)
    
    # Evaluate
    metrics = trainer.evaluate_model('xgboost', X_test, y_test, verbose=False)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_evaluate_model_not_found(sample_data):
    """Test evaluating non-existent model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    with pytest.raises(ModelTrainerError):
        trainer.evaluate_model('nonexistent', X, y)


def test_evaluate_all_models(sample_data):
    """Test evaluating all models."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    # Split data
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Train all models
    trainer.train_all_models(X_train, y_train, use_cv=False, verbose=False)
    
    # Evaluate all
    results = trainer.evaluate_all_models(X_test, y_test, verbose=False)
    
    assert 'xgboost' in results
    assert 'random_forest' in results


def test_get_best_model(sample_data):
    """Test getting best model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    # Train models with CV to get results
    trainer.train_xgboost(X, y, use_cv=True, verbose=False)
    trainer.train_random_forest(X, y, use_cv=True, verbose=False)
    
    # Get best model
    best_name, best_model = trainer.get_best_model(metric='r2')
    
    assert best_name in ['xgboost', 'random_forest']
    assert best_model is not None


def test_get_best_model_no_models():
    """Test getting best model with no trained models."""
    trainer = ModelTrainer(n_cv_folds=2)
    
    with pytest.raises(ModelTrainerError):
        trainer.get_best_model()


def test_get_model(sample_data):
    """Test getting a specific model."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    trainer.train_xgboost(X, y, use_cv=False, verbose=False)
    
    model = trainer.get_model('xgboost')
    
    assert model is not None


def test_get_model_not_found():
    """Test getting non-existent model."""
    trainer = ModelTrainer(n_cv_folds=2)
    
    with pytest.raises(ModelTrainerError):
        trainer.get_model('nonexistent')


def test_get_training_summary(sample_data):
    """Test getting training summary."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    trainer.train_xgboost(X, y, use_cv=False, verbose=False)
    trainer.train_random_forest(X, y, use_cv=False, verbose=False)
    
    summary = trainer.get_training_summary()
    
    assert summary['n_training_runs'] == 2
    assert 'xgboost' in summary['models_trained']
    assert 'random_forest' in summary['models_trained']


def test_custom_hyperparameters(sample_data):
    """Test training with custom hyperparameters."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    custom_xgb_params = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 50
    }
    
    trainer.train_xgboost(
        X, y,
        hyperparameters=custom_xgb_params,
        use_cv=False,
        verbose=False
    )
    
    model = trainer.get_model('xgboost')
    assert model.hyperparameters['max_depth'] == 8
    assert model.hyperparameters['learning_rate'] == 0.05


def test_training_history(sample_data):
    """Test training history tracking."""
    X, y = sample_data
    trainer = ModelTrainer(n_cv_folds=2)
    
    trainer.train_xgboost(X, y, use_cv=False, verbose=False)
    trainer.train_random_forest(X, y, use_cv=False, verbose=False)
    
    assert len(trainer.training_history) == 2
    assert trainer.training_history[0]['model_type'] == 'XGBoost'
    assert trainer.training_history[1]['model_type'] == 'Random Forest'
