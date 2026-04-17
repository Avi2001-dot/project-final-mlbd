import pytest
import numpy as np
import pandas as pd

from src.modeling.model_evaluator import ModelEvaluator, ModelEvaluatorError


@pytest.fixture
def sample_predictions():
    """Create sample predictions and actuals."""
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.uniform(50, 150, n_samples)
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    
    return pd.Series(y_true), np.array(y_pred)


def test_evaluator_initialization():
    """Test ModelEvaluator initialization."""
    evaluator = ModelEvaluator()
    
    assert evaluator.predictions is None
    assert evaluator.actuals is None
    assert evaluator.residuals is None
    assert evaluator.metrics == {}


def test_evaluate(sample_predictions):
    """Test model evaluation."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate(y_true, y_pred)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0


def test_evaluate_with_residuals(sample_predictions):
    """Test evaluation with residual analysis."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate(y_true, y_pred, compute_residuals=True)
    
    assert 'residual_mean' in metrics
    assert 'residual_std' in metrics
    assert 'residual_skewness' in metrics
    assert 'residual_kurtosis' in metrics


def test_evaluate_invalid_input():
    """Test evaluation with invalid inputs."""
    evaluator = ModelEvaluator()
    y_true = pd.Series([1, 2, 3])
    y_pred = np.array([1, 2])
    
    # Mismatched lengths
    with pytest.raises(ModelEvaluatorError):
        evaluator.evaluate(y_true, y_pred)
    
    # None inputs
    with pytest.raises(ModelEvaluatorError):
        evaluator.evaluate(None, y_pred)


def test_analyze_residuals(sample_predictions):
    """Test residual analysis."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    analysis = evaluator.analyze_residuals()
    
    assert 'residual_mean' in analysis
    assert 'residual_std' in analysis
    assert 'residual_min' in analysis
    assert 'residual_max' in analysis
    assert 'residual_skewness' in analysis
    assert 'residual_kurtosis' in analysis


def test_get_metrics(sample_predictions):
    """Test getting computed metrics."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    metrics = evaluator.get_metrics()
    
    assert len(metrics) > 0
    assert 'rmse' in metrics


def test_get_metrics_not_evaluated():
    """Test getting metrics without evaluation."""
    evaluator = ModelEvaluator()
    
    with pytest.raises(ModelEvaluatorError):
        evaluator.get_metrics()


def test_get_residuals(sample_predictions):
    """Test getting residuals."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    residuals = evaluator.get_residuals()
    
    assert len(residuals) == len(y_true)
    np.testing.assert_array_almost_equal(
        residuals,
        y_true.values - y_pred
    )


def test_get_prediction_errors(sample_predictions):
    """Test getting prediction errors."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    errors = evaluator.get_prediction_errors()
    
    assert len(errors) == len(y_true)
    assert all(e >= 0 for e in errors)


def test_get_percentage_errors(sample_predictions):
    """Test getting percentage errors."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    pct_errors = evaluator.get_percentage_errors()
    
    assert len(pct_errors) == len(y_true)
    assert all(e >= 0 for e in pct_errors)


def test_get_error_statistics(sample_predictions):
    """Test getting error statistics."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    stats = evaluator.get_error_statistics()
    
    assert 'error_mean' in stats
    assert 'error_std' in stats
    assert 'error_min' in stats
    assert 'error_max' in stats
    assert 'error_median' in stats
    assert 'error_q25' in stats
    assert 'error_q75' in stats


def test_identify_outliers(sample_predictions):
    """Test outlier identification."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    outlier_idx, outlier_residuals = evaluator.identify_outliers(threshold_std=2.0)
    
    assert isinstance(outlier_idx, np.ndarray)
    assert isinstance(outlier_residuals, np.ndarray)
    assert len(outlier_idx) == len(outlier_residuals)


def test_identify_outliers_no_residuals():
    """Test outlier identification without residuals."""
    evaluator = ModelEvaluator()
    
    with pytest.raises(ModelEvaluatorError):
        evaluator.identify_outliers()


def test_get_evaluation_report(sample_predictions):
    """Test getting evaluation report."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    report = evaluator.get_evaluation_report()
    
    assert 'metrics' in report
    assert 'error_statistics' in report
    assert 'n_samples' in report
    assert 'n_outliers' in report
    assert report['n_samples'] == len(y_true)


def test_compare_models():
    """Test comparing multiple models."""
    evaluator = ModelEvaluator()
    
    model_results = {
        'model_a': {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85},
        'model_b': {'rmse': 12.3, 'mae': 9.1, 'r2': 0.80},
        'model_c': {'rmse': 9.8, 'mae': 7.5, 'r2': 0.88}
    }
    
    comparison = evaluator.compare_models(model_results)
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3
    # Should be sorted by r2 descending
    assert comparison.iloc[0]['r2'] >= comparison.iloc[1]['r2']


def test_reset(sample_predictions):
    """Test resetting evaluator state."""
    y_true, y_pred = sample_predictions
    evaluator = ModelEvaluator()
    
    evaluator.evaluate(y_true, y_pred)
    assert evaluator.predictions is not None
    
    evaluator.reset()
    
    assert evaluator.predictions is None
    assert evaluator.actuals is None
    assert evaluator.residuals is None
    assert evaluator.metrics == {}


def test_perfect_predictions():
    """Test evaluation with perfect predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    assert metrics['rmse'] == 0.0
    assert metrics['mae'] == 0.0
    assert metrics['r2'] == 1.0


def test_constant_predictions():
    """Test evaluation with constant predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert metrics['r2'] <= 0  # Worse than mean baseline
