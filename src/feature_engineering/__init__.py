def __getattr__(name):
    _map = {
        'FeatureProcessor':       ('src.feature_engineering.feature_processor',    'FeatureProcessor'),
        'FeatureProcessorError':  ('src.feature_engineering.feature_processor',    'FeatureProcessorError'),
        'TimeSeriesSplitter':     ('src.feature_engineering.time_series_splitter', 'TimeSeriesSplitter'),
        'TimeSeriesSplitterError':('src.feature_engineering.time_series_splitter', 'TimeSeriesSplitterError'),
        'FeatureAnalyzer':        ('src.feature_engineering.feature_analyzer',     'FeatureAnalyzer'),
        'FeatureAnalyzerError':   ('src.feature_engineering.feature_analyzer',     'FeatureAnalyzerError'),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'src.feature_engineering' has no attribute {name!r}")

__all__ = [
    'FeatureProcessor', 'FeatureProcessorError',
    'TimeSeriesSplitter', 'TimeSeriesSplitterError',
    'FeatureAnalyzer', 'FeatureAnalyzerError',
]
