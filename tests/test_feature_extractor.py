import pandas as pd
import pytest
from unredactor_pipeline import FeatureExtractor



def test_feature_extractor():
    fe = FeatureExtractor()
    context = "This is a ████ test context"
    
    features = fe.extract_features(context)
    assert 'redacted_length' in features
    assert 'context_length' in features
    assert 'words_before_redaction' in features
    assert 'words_after_redaction' in features
    