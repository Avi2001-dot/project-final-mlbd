"""ETL pipeline module for Air Quality Prediction System."""

# Lazy imports – PySpark is only required when these classes are actually used.
# This prevents ImportError on machines without PySpark installed.

def __getattr__(name):
    _map = {
        'BronzeLayer': ('src.etl_pipeline.bronze_layer', 'BronzeLayer'),
        'BronzeLayerError': ('src.etl_pipeline.bronze_layer', 'BronzeLayerError'),
        'SilverLayer': ('src.etl_pipeline.silver_layer', 'SilverLayer'),
        'SilverLayerError': ('src.etl_pipeline.silver_layer', 'SilverLayerError'),
        'GoldLayer': ('src.etl_pipeline.gold_layer', 'GoldLayer'),
        'GoldLayerError': ('src.etl_pipeline.gold_layer', 'GoldLayerError'),
        'DataQualityValidator': ('src.etl_pipeline.data_validator', 'DataQualityValidator'),
        'DataQualityValidatorError': ('src.etl_pipeline.data_validator', 'DataQualityValidatorError'),
        'FeatureNormalizer': ('src.etl_pipeline.feature_normalizer', 'FeatureNormalizer'),
        'FeatureNormalizerError': ('src.etl_pipeline.feature_normalizer', 'FeatureNormalizerError'),
        'ETLPipeline': ('src.etl_pipeline.pipeline', 'ETLPipeline'),
        'ETLPipelineError': ('src.etl_pipeline.pipeline', 'ETLPipelineError'),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'src.etl_pipeline' has no attribute {name!r}")


__all__ = [
    'BronzeLayer', 'BronzeLayerError',
    'SilverLayer', 'SilverLayerError',
    'GoldLayer', 'GoldLayerError',
    'DataQualityValidator', 'DataQualityValidatorError',
    'FeatureNormalizer', 'FeatureNormalizerError',
    'ETLPipeline', 'ETLPipelineError',
]
