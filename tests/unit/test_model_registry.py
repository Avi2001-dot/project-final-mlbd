import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from src.modeling.model_registry import ModelRegistry, ModelRegistryError
from src.modeling.xgboost_model import XGBoostModel


@pytest.fixture
def temp_registry():
    """Create a temporary registry directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    y = pd.Series(np.random.randn(n_samples))
    
    model = XGBoostModel()
    model.train(X, y, verbose=False)
    
    return model, X.columns.tolist()


def test_registry_initialization(temp_registry):
    """Test ModelRegistry initialization."""
    registry = ModelRegistry(registry_path=temp_registry)
    
    assert registry.registry_path == temp_registry
    assert registry.models == {}
    assert os.path.exists(temp_registry)


def test_register_model(temp_registry, sample_model):
    """Test registering a model."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    model_id = registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    assert model_id == 'xgboost_v1.0'
    assert model_id in registry.models
    assert os.path.exists(registry.models[model_id]['model_path'])
    assert os.path.exists(registry.models[model_id]['metadata_path'])


def test_get_model(temp_registry, sample_model):
    """Test retrieving a model."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    model_id = registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    # Retrieve model
    loaded_model = registry.get_model(model_id)
    
    assert loaded_model is not None


def test_get_model_not_found(temp_registry):
    """Test retrieving non-existent model."""
    registry = ModelRegistry(registry_path=temp_registry)
    
    with pytest.raises(ModelRegistryError):
        registry.get_model('nonexistent')


def test_get_model_info(temp_registry, sample_model):
    """Test getting model metadata."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    model_id = registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    info = registry.get_model_info(model_id)
    
    assert info['model_id'] == model_id
    assert info['model_name'] == 'xgboost'
    assert info['model_type'] == 'XGBoost'
    assert info['version'] == 'v1.0'


def test_list_models(temp_registry, sample_model):
    """Test listing models."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    # Register multiple models
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v2.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    models = registry.list_models()
    
    assert len(models) == 2


def test_list_models_filter_by_name(temp_registry, sample_model):
    """Test listing models with name filter."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    models = registry.list_models(model_name='xgboost')
    
    assert len(models) == 1
    assert models[0]['model_name'] == 'xgboost'


def test_get_best_model(temp_registry, sample_model):
    """Test getting best model."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    # Register models with different R² scores
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics={'rmse': 12.0, 'mae': 9.0, 'r2': 0.80},
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v2.0',
        metrics={'rmse': 10.5, 'mae': 8.2, 'r2': 0.85},
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    best = registry.get_best_model(metric='r2')
    
    assert best['version'] == 'v2.0'
    assert best['metrics']['r2'] == 0.85


def test_get_best_model_by_rmse(temp_registry, sample_model):
    """Test getting best model by RMSE."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics={'rmse': 12.0, 'mae': 9.0, 'r2': 0.80},
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v2.0',
        metrics={'rmse': 10.5, 'mae': 8.2, 'r2': 0.85},
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    best = registry.get_best_model(metric='rmse')
    
    assert best['version'] == 'v2.0'
    assert best['metrics']['rmse'] == 10.5


def test_delete_model(temp_registry, sample_model):
    """Test deleting a model."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    model_id = registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    assert model_id in registry.models
    
    registry.delete_model(model_id)
    
    assert model_id not in registry.models


def test_delete_model_not_found(temp_registry):
    """Test deleting non-existent model."""
    registry = ModelRegistry(registry_path=temp_registry)
    
    with pytest.raises(ModelRegistryError):
        registry.delete_model('nonexistent')


def test_load_registry(temp_registry, sample_model):
    """Test loading registry from disk."""
    registry1 = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    registry1.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    # Create new registry and load
    registry2 = ModelRegistry(registry_path=temp_registry)
    registry2.load_registry()
    
    assert len(registry2.models) == 1


def test_get_registry_stats(temp_registry, sample_model):
    """Test getting registry statistics."""
    registry = ModelRegistry(registry_path=temp_registry)
    model, feature_cols = sample_model
    
    metrics = {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85}
    hyperparameters = {'max_depth': 6, 'learning_rate': 0.1}
    
    registry.register_model(
        model=model,
        model_name='xgboost',
        model_type='XGBoost',
        version='v1.0',
        metrics=metrics,
        hyperparameters=hyperparameters,
        feature_columns=feature_cols
    )
    
    stats = registry.get_registry_stats()
    
    assert stats['total_models'] == 1
    assert 'XGBoost' in stats['models_by_type']
    assert 'xgboost' in stats['models_by_name']
