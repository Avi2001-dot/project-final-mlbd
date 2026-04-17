import pytest
import numpy as np
import pandas as pd
import os

from src.modeling.random_forest_model import RandomForestModel, RandomForestModelError


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
    # Target is a function of features
    y = (
        0.5 * X['aqi_lag_1h'] +
        0.3 * X['aqi_mean_3h'] +
        0.1 * X['hour_of_day'] +
        np.random.normal(0, 5, n_samples)
    )
    y = pd.Series(y, name='aqi')
    
    return X, y


def test_random_forest_initialization():
    """Test RandomForestModel initialization."""
    model = RandomForestModel()
    
    assert model.model is None
    assert model.feature_columns is None
    assert model.cv_results == {}
    assert model.feature_importance is None


def test_random_forest_initialization_with_custom_params():
    """Test RandomForestModel initialization with custom hyperparameters."""
    custom_params = {
        'n_estimators': 50,
        'max_depth': 10,
        'min_samples_split': 10
    }
    
    model = RandomForestModel(hyperparameters=custom_params)
    
    assert model.hyperparameters['n_estimators'] == 50
    assert model.hyperparameters['max_depth'] == 10
    assert model.hyperparameters['min_samples_split'] == 10


def test_random_forest_train_without_cv(sample_data):
    """Test training without cross-validation."""
    X, y = sample_data
    model = RandomForestModel()
    
    result = model.train(X, y, cv_splits=None, verbose=False)
    
    assert model.model is not None
    assert model.feature_columns is not None
    assert len(model.feature_columns) == len(X.columns)
    assert result == {}


def test_random_forest_train_with_cv(sample_data):
    """Test training with cross-validation."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Create CV splits
    n_splits = 3
    fold_size = len(X) // (n_splits + 1)
    cv_splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = train_end + fold_size
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, len(X)))
        cv_splits.append((train_idx, test_idx))
    
    result = model.train(X, y, cv_splits=cv_splits, verbose=False)
    
    assert model.model is not None
    assert 'rmse_mean' in result
    assert 'mae_mean' in result
    assert 'r2_mean' in result
    assert result['rmse_mean'] > 0
    assert result['mae_mean'] > 0


def test_random_forest_train_invalid_input(sample_data):
    """Test training with invalid inputs."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Test with None X
    with pytest.raises(RandomForestModelError):
        model.train(None, y)
    
    # Test with empty X
    with pytest.raises(RandomForestModelError):
        model.train(pd.DataFrame(), y)
    
    # Test with mismatched lengths
    with pytest.raises(RandomForestModelError):
        model.train(X, y.iloc[:10])


def test_random_forest_predict(sample_data):
    """Test prediction."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Split data
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Train
    model.train(X_train, y_train, verbose=False)
    
    # Predict
    predictions = model.predict(X_test)
    
    assert predictions is not None
    assert len(predictions) == len(X_test)
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_random_forest_predict_not_trained():
    """Test prediction without training."""
    model = RandomForestModel()
    X = pd.DataFrame({'a': [1, 2, 3]})
    
    with pytest.raises(RandomForestModelError):
        model.predict(X)


def test_random_forest_predict_24h(sample_data):
    """Test 24-hour ahead prediction."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Train
    model.train(X, y, verbose=False)
    
    # Get current features
    X_current = X.iloc[[0]]
    
    # Predict 24 hours ahead
    predictions = model.predict_24h(X_current, X)
    
    assert predictions is not None
    assert len(predictions) == 24
    assert all(isinstance(p, (int, float, np.number)) for p in predictions)


def test_random_forest_feature_importance(sample_data):
    """Test feature importance extraction."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Train
    model.train(X, y, verbose=False)
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    assert importance is not None
    assert len(importance) == len(X.columns)
    assert all(isinstance(v, (int, float, np.number)) for v in importance.values())


def test_random_forest_feature_importance_top_n(sample_data):
    """Test feature importance with top N filtering."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Train
    model.train(X, y, verbose=False)
    
    # Get top 5 features
    importance = model.get_feature_importance(top_n=5)
    
    assert len(importance) == 5


def test_random_forest_evaluate(sample_data):
    """Test model evaluation."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Split data
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Train
    model.train(X_train, y_train, verbose=False)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0


def test_random_forest_save_load(sample_data, tmp_path):
    """Test model serialization and deserialization."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Train
    model.train(X, y, verbose=False)
    
    # Save
    model_path = str(tmp_path / "model.pkl")
    model.save(model_path)
    
    assert os.path.exists(model_path)
    
    # Load
    model2 = RandomForestModel()
    model2.load(model_path)
    
    # Verify predictions are identical
    pred1 = model.predict(X.iloc[:5])
    pred2 = model2.predict(X.iloc[:5])
    
    np.testing.assert_array_almost_equal(pred1, pred2)


def test_random_forest_get_model_info(sample_data):
    """Test getting model information."""
    X, y = sample_data
    model = RandomForestModel()
    
    # Train
    model.train(X, y, verbose=False)
    
    # Get info
    info = model.get_model_info()
    
    assert info['model_type'] == 'Random Forest'
    assert info['hyperparameters'] is not None
    assert info['feature_columns'] is not None
    assert info['is_trained'] is True
