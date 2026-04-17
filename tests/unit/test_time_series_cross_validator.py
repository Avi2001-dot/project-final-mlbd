import pytest
import numpy as np
import pandas as pd

from src.modeling.time_series_cross_validator import (
    TimeSeriesCrossValidator,
    TimeSeriesCrossValidatorError
)


@pytest.fixture
def sample_data():
    """Create sample time-series data."""
    n_samples = 100
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    X = pd.DataFrame({
        'timestamp': dates,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    
    y = pd.Series(np.random.randn(n_samples), name='target')
    
    return X, y


def test_initialization():
    """Test TimeSeriesCrossValidator initialization."""
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    assert validator.n_splits == 3
    assert validator.test_size == 0.25  # 1/(3+1)


def test_initialization_with_custom_test_size():
    """Test initialization with custom test size."""
    validator = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
    
    assert validator.n_splits == 3
    assert validator.test_size == 0.2


def test_initialization_invalid_n_splits():
    """Test initialization with invalid n_splits."""
    with pytest.raises(TimeSeriesCrossValidatorError):
        TimeSeriesCrossValidator(n_splits=0)
    
    with pytest.raises(TimeSeriesCrossValidatorError):
        TimeSeriesCrossValidator(n_splits=-1)


def test_initialization_invalid_test_size():
    """Test initialization with invalid test_size."""
    with pytest.raises(TimeSeriesCrossValidatorError):
        TimeSeriesCrossValidator(n_splits=3, test_size=0)
    
    with pytest.raises(TimeSeriesCrossValidatorError):
        TimeSeriesCrossValidator(n_splits=3, test_size=1.0)
    
    with pytest.raises(TimeSeriesCrossValidatorError):
        TimeSeriesCrossValidator(n_splits=3, test_size=-0.1)


def test_split(sample_data):
    """Test generating cross-validation splits."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits = validator.split(X, y)
    
    assert len(splits) == 3
    
    # Verify temporal ordering
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert max(train_idx) < min(test_idx)


def test_split_no_leakage(sample_data):
    """Test that splits prevent data leakage."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits = validator.split(X, y)
    
    for train_idx, test_idx in splits:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        max_train_time = X_train['timestamp'].max()
        min_test_time = X_test['timestamp'].min()
        
        assert max_train_time < min_test_time


def test_split_invalid_input():
    """Test split with invalid inputs."""
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    # Test with None
    with pytest.raises(TimeSeriesCrossValidatorError):
        validator.split(None)
    
    # Test with empty DataFrame
    with pytest.raises(TimeSeriesCrossValidatorError):
        validator.split(pd.DataFrame())


def test_split_insufficient_samples():
    """Test split with insufficient samples."""
    validator = TimeSeriesCrossValidator(n_splits=100)
    X = pd.DataFrame({'a': range(10)})
    
    with pytest.raises(TimeSeriesCrossValidatorError):
        validator.split(X)


def test_get_train_test_split(sample_data):
    """Test getting a single train-test split."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
    
    X_train, X_test, y_train, y_test = validator.get_train_test_split(X, y)
    
    assert len(X_train) == int(len(X) * 0.8)
    assert len(X_test) == len(X) - len(X_train)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_get_train_test_split_no_target(sample_data):
    """Test getting train-test split without target."""
    X, _ = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
    
    X_train, X_test, y_train, y_test = validator.get_train_test_split(X)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert y_train is None
    assert y_test is None


def test_get_train_test_split_custom_test_size(sample_data):
    """Test train-test split with custom test size."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3, test_size=0.2)
    
    X_train, X_test, y_train, y_test = validator.get_train_test_split(
        X, y, test_size=0.3
    )
    
    assert len(X_test) == int(len(X) * 0.3)
    assert len(X_train) == len(X) - len(X_test)


def test_validate_no_leakage(sample_data):
    """Test data leakage validation."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits = validator.split(X, y)
    
    for train_idx, test_idx in splits:
        no_leakage = validator.validate_no_leakage(X, train_idx, test_idx)
        assert no_leakage is True


def test_validate_no_leakage_invalid_column():
    """Test leakage validation with invalid timestamp column."""
    X = pd.DataFrame({'a': range(10)})
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4, 5])
    
    with pytest.raises(TimeSeriesCrossValidatorError):
        validator.validate_no_leakage(X, train_idx, test_idx)


def test_get_n_splits():
    """Test getting number of splits."""
    validator = TimeSeriesCrossValidator(n_splits=5)
    
    assert validator.get_n_splits() == 5


def test_split_consistency(sample_data):
    """Test that splits are consistent across calls."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits1 = validator.split(X, y)
    splits2 = validator.split(X, y)
    
    assert len(splits1) == len(splits2)
    
    for (train1, test1), (train2, test2) in zip(splits1, splits2):
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)


def test_split_coverage(sample_data):
    """Test that splits cover all data."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits = validator.split(X, y)
    
    # Collect all test indices
    all_test_idx = []
    for train_idx, test_idx in splits:
        all_test_idx.extend(test_idx)
    
    # All test indices should be unique and cover a portion of data
    assert len(set(all_test_idx)) == len(all_test_idx)
    assert len(all_test_idx) > 0


def test_split_no_overlap(sample_data):
    """Test that train and test sets don't overlap."""
    X, y = sample_data
    validator = TimeSeriesCrossValidator(n_splits=3)
    
    splits = validator.split(X, y)
    
    for train_idx, test_idx in splits:
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0
