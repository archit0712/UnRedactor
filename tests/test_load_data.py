import pytest
import pandas as pd
from unredactor_pipeline import load_data



def test_load_data(tmp_path):
    train_file = tmp_path / 'test_train.tsv'
    train_file.write_text("train\tJohn\tThis is ████ test.\n", encoding='utf-8')
    
    test_file = tmp_path / "test_test.tsv"
    test_file.write_text("1\tPredict █ name.\n",encoding='utf-8')

    # Test training data loading
    df_train = load_data(str(train_file))
    assert isinstance(df_train, pd.DataFrame)
    assert 'split' in df_train.columns
    assert 'name' in df_train.columns
    assert 'context' in df_train.columns

    # Test test data loading
    df_test = load_data(str(test_file), is_test=True)
    assert isinstance(df_test, pd.DataFrame)
    assert 'id' in df_test.columns
    assert 'context' in df_test.columns
    
    