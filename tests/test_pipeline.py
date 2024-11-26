import pandas as pd
import pytest
from unredactor_pipeline import UnredactorPipeline

@pytest.fixture
def sample_training_data():
    return pd.DataFrame({
        'split': ['training', 'training'],
        'name': ['John', 'Jane'],
        'context': ['This is ████ test.', 'Another ████ example.']
    })
    
@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        'id': ['1', '2'],
        'context': ['Predict ████ name.', 'Another ████ test.']
    })
    
    
def test_pipeline(sample_training_data, sample_test_data):
    pipeline = UnredactorPipeline()
    
    pipeline.train(sample_training_data)
    
    assert pipeline.model is not None
    
    prediction = pipeline.predict(sample_test_data)
    print("prediction",prediction)
    assert isinstance(prediction, pd.DataFrame)
    assert 'id' in prediction.columns
    assert 'name' in prediction.columns