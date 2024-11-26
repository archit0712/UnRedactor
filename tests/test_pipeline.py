import pandas as pd
import pytest
from unredactor_pipeline import UnredactorPipeline

@pytest.fixture
def sample_training_data():
    return pd.DataFrame({
        'split': ['training', 'training', 'training', 'training', 'training', 'training', 'training', 'training'],
        'name': [
            'Sterno', 
            'Stephen Williams', 
            'Stephen McNally', 
            'Stephen McNally', 
            'Stephen King', 
            'Stephen King', 
            'Stephen King', 
            'Stephen Garvey'
        ],
        'context': [
            '██████ says take a ride on Rocketship X-M.', 
            "I am glad Jack Bender is directing next week's episode and it'll be much better and I'm glad he got first Syaid episode to direct and I'm curious what he will pull off this time since ████████████████ had directed too many Syaid episodes before.", 
            'Riding into Dodge City with his trusty friend, Johnny Williams {Millard Mitchell}, Lin runs into Dutch Henry Brown {███████████████}, the man he wants.', 
            'Riding into Dodge City with his trusty friend, Johnny Williams {Millard Mitchell}, Lin runs into Dutch Henry Brown {███████████████}, the man he wants.', 
            'I love ████████████ - I read most of his books (not this one)and there are some good (and of course some bad)', 
            "I am probably one of the few who actually read ████████████'s book, the one this movie was based on.", 
            'As I read some of the user comments I see people blaming ████████████ for this piece of drivel.', 
            'This gem of a film showcases the brilliantly funny writing of ██████████████'
        ]
    })
    
@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        'id': ['1', '2'],
        'context': ["Would you like to know why French and Italians love/hate each others? Would you like to have a glimpse of history that drives our lifetime? So, go to watch Virzi's film (in original language, of course) and you can look at a wonderful Monica Bellucci who finally speaks her native language from Città di Castello (Umbria, just at the border with Tuscany). And the rest of the characters speaking Livornese (lovely Sabrina Impacciatore and all the others). Daniel ███████ definitely in his shoes with Napoleon. A lot of fun, a real fresco of the Elba Island landscape, and a picture about the political reasons to kill or leave alive a tyrant (good for all times).", 'But he must get past the princes and █████ first.']
    })
    
def test_pipeline(sample_training_data, sample_test_data):
    pipeline = UnredactorPipeline()
    
    pipeline.train(sample_training_data)
    
    assert pipeline.model is not None
    
    prediction = pipeline.predict(sample_test_data)
    
    assert isinstance(prediction, pd.DataFrame)
    assert 'id' in prediction.columns
    assert 'name' in prediction.columns