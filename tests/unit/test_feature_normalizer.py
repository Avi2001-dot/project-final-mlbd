import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.etl_pipeline.feature_normalizer import FeatureNormalizer, FeatureNormalizerError


@pytest.fixture
def normalizer():
    """Create a Feature Normalizer instance."""
    return FeatureNormalizer()


def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'aqi': [100.0, 150.0, 120.0, 130.0, 140.0],
        'pm25': [40.0, 55.0, 45.0, 50.0, 52.0],
        'pm10': [80.0, 110.0, 90.0, 100.0, 105.0],
        'no2': [30.0, 35.0, 32.0, 33.0, 34.0],
        'city': ['Delhi', 'Delhi', 'Mumbai', 'Mumbai', 'Bangalore']
    })


class TestFitAndTransform:
    """Tests for fit and transform operations."""

    def test_fit_and_transform_returns_dataframe(self, normalizer):
        """Test that fit_and_transform returns a DataFrame."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        result = normalizer.fit_and_transform(df, feature_cols)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_fit_and_transform_normalizes_features(self, normalizer):
        """Test that features are normalized to zero mean and unit variance."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        result = normalizer.fit_and_transform(df, feature_cols)

        # Check that normalized features have approximately zero mean
        for col in feature_cols:
            mean = result[col].mean()
            assert abs(mean) < 0.2  # Allow small deviation due to sample size

        # Check that normalized features have approximately unit variance
        for col in feature_cols:
            std = result[col].std()
            assert abs(std - 1.0) < 0.2  # Allow small deviation due to sample size

    def test_fit_and_transform_preserves_non_feature_columns(self, normalizer):
        """Test that non-feature columns are preserved."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        result = normalizer.fit_and_transform(df, feature_cols)

        # City column should be preserved
        assert 'city' in result.columns
        assert list(result['city']) == list(df['city'])

    def test_fit_and_transform_with_default_features(self, normalizer):
        """Test fit_and_transform with default feature selection."""
        df = create_sample_dataframe()

        result = normalizer.fit_and_transform(df)

        # Should normalize numeric columns
        assert 'aqi' in result.columns
        assert 'pm25' in result.columns

    def test_fit_stores_feature_columns(self, normalizer):
        """Test that feature columns are stored after fitting."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)

        assert normalizer.feature_columns == feature_cols


class TestTransform:
    """Tests for transform operations."""

    def test_transform_requires_fitted_scaler(self, normalizer):
        """Test that transform requires a fitted scaler."""
        df = create_sample_dataframe()

        with pytest.raises(FeatureNormalizerError):
            normalizer.transform(df)

    def test_transform_uses_training_parameters(self, normalizer):
        """Test that transform uses training parameters."""
        df_train = create_sample_dataframe()
        df_test = create_sample_dataframe() * 2  # Different scale

        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        # Fit on training data
        normalizer.fit_and_transform(df_train, feature_cols)

        # Transform test data
        result = normalizer.transform(df_test)

        # Test data should be normalized using training parameters
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df_test)

    def test_transform_preserves_non_feature_columns(self, normalizer):
        """Test that non-feature columns are preserved during transform."""
        df_train = create_sample_dataframe()
        df_test = create_sample_dataframe()

        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df_train, feature_cols)
        result = normalizer.transform(df_test)

        # City column should be preserved
        assert 'city' in result.columns
        assert list(result['city']) == list(df_test['city'])


class TestFitTransform:
    """Tests for fit_transform alias."""

    def test_fit_transform_is_alias(self, normalizer):
        """Test that fit_transform is an alias for fit_and_transform."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        result = normalizer.fit_transform(df, feature_cols)

        assert isinstance(result, pd.DataFrame)
        assert normalizer.is_fitted()


class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_serialize_saves_scaler(self, normalizer):
        """Test that serialize saves the scaler to file."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'scaler.pkl')
            normalizer.serialize(path)

            assert os.path.exists(path)

    def test_serialize_requires_fitted_scaler(self, normalizer):
        """Test that serialize requires a fitted scaler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'scaler.pkl')

            with pytest.raises(FeatureNormalizerError):
                normalizer.serialize(path)

    def test_deserialize_loads_scaler(self, normalizer):
        """Test that deserialize loads the scaler from file."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'scaler.pkl')
            normalizer.serialize(path)

            # Create new normalizer and deserialize
            normalizer2 = FeatureNormalizer()
            normalizer2.deserialize(path)

            assert normalizer2.is_fitted()
            assert normalizer2.feature_columns == feature_cols

    def test_deserialize_nonexistent_file(self, normalizer):
        """Test that deserialize raises error for nonexistent file."""
        with pytest.raises(FeatureNormalizerError):
            normalizer.deserialize('/nonexistent/path/scaler.pkl')

    def test_serialization_preserves_parameters(self, normalizer):
        """Test that serialization preserves scaler parameters."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)
        params1 = normalizer.get_scaler_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'scaler.pkl')
            normalizer.serialize(path)

            normalizer2 = FeatureNormalizer()
            normalizer2.deserialize(path)
            params2 = normalizer2.get_scaler_params()

            # Parameters should be identical
            assert np.allclose(params1['mean'], params2['mean'])
            assert np.allclose(params1['scale'], params2['scale'])


class TestReproducibility:
    """Tests for reproducibility."""

    def test_fit_transform_reproducible(self, normalizer):
        """Test that fit_transform produces reproducible results."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        result1 = normalizer.fit_and_transform(df, feature_cols)

        normalizer2 = FeatureNormalizer()
        result2 = normalizer2.fit_and_transform(df, feature_cols)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_reproducible(self, normalizer):
        """Test that transform produces reproducible results."""
        df_train = create_sample_dataframe()
        df_test = create_sample_dataframe()

        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df_train, feature_cols)
        result1 = normalizer.transform(df_test)

        normalizer2 = FeatureNormalizer()
        normalizer2.fit_and_transform(df_train, feature_cols)
        result2 = normalizer2.transform(df_test)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestScalerParams:
    """Tests for scaler parameter inspection."""

    def test_get_scaler_params_returns_dict(self, normalizer):
        """Test that get_scaler_params returns a dictionary."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)
        params = normalizer.get_scaler_params()

        assert isinstance(params, dict)
        assert 'mean' in params
        assert 'scale' in params
        assert 'var' in params
        assert 'feature_columns' in params

    def test_get_scaler_params_requires_fitted_scaler(self, normalizer):
        """Test that get_scaler_params requires a fitted scaler."""
        with pytest.raises(FeatureNormalizerError):
            normalizer.get_scaler_params()

    def test_scaler_params_correct_shape(self, normalizer):
        """Test that scaler parameters have correct shape."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)
        params = normalizer.get_scaler_params()

        assert len(params['mean']) == len(feature_cols)
        assert len(params['scale']) == len(feature_cols)
        assert len(params['var']) == len(feature_cols)


class TestIsFitted:
    """Tests for is_fitted check."""

    def test_is_fitted_false_initially(self, normalizer):
        """Test that is_fitted returns False initially."""
        assert not normalizer.is_fitted()

    def test_is_fitted_true_after_fitting(self, normalizer):
        """Test that is_fitted returns True after fitting."""
        df = create_sample_dataframe()
        feature_cols = ['aqi', 'pm25', 'pm10', 'no2']

        normalizer.fit_and_transform(df, feature_cols)

        assert normalizer.is_fitted()
