import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import joblib
import argparse

def load_data(file_path, is_test=False):
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split('\t')
        if is_test:
            # For test file: only id and context
            if len(parts) >= 2:
                data.append({
                    'id': parts[0],
                    'context': '\t'.join(parts[1:])
                })
        else:
            # For training file: split, name, context
            if len(parts) >= 3:
                data.append({
                    'split': parts[0],
                    'name': parts[1],
                    'context': '\t'.join(parts[2:])
                })
    
    return pd.DataFrame(data)

class FeatureExtractor:
    def __init__(self):
        resources = ['vader_lexicon', 'averaged_perceptron_tagger']
        for resource in resources:
            nltk.download(resource)
        
        self.sid = SentimentIntensityAnalyzer()
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            max_features=10,
            min_df=1,
            # stop_words='english'
        )
        self.vectorizer_fitted = False
        
    def fit_vectorizer(self, contexts):
            cleaned_contexts = [context.replace('█', ' ') for context in contexts]
            self.vectorizer.fit(cleaned_contexts)
            self.vectorizer_fitted = True

    def _get_ngram_features(self, context):
        if not self.vectorizer_fitted:
            return {'ngram_default': 0}
            
        cleaned_context = context.replace('█', ' ')
        try:
            ngrams = self.vectorizer.transform([cleaned_context])
            feature_dict = {
                f'ngram_{feat}': count 
                for feat, count in zip(self.vectorizer.get_feature_names_out(), 
                                     ngrams.toarray()[0])
            }
            return feature_dict if feature_dict else {'ngram_default': 0}
        except Exception as e:
            return {'ngram_default': 0}
    def extract_features(self, context, redacted_name=None):
        """Main feature extraction method"""
        features = {}
        
        # Add length features
        features.update(self._get_length_features(context, redacted_name))
        
        # Add context window features
        features.update(self._get_context_window_features(context))
        
        # Add n-gram features
        features.update(self._get_ngram_features(context))
        
        # Add sentiment features
        features.update(self._get_sentiment_features(context))
        
        # Add position features
        features.update(self._get_position_features(context))
        
        return features

    def _get_length_features(self, context, redacted_name):
        parts = context.split('█')
        redaction_length = len(''.join(['█' for _ in range(len(parts)-1)]))
        
        return {
            'redacted_length': len(redacted_name) if redacted_name else redaction_length,
            'context_length': len(context),
            'words_before_redaction': len(parts[0].strip().split()),
            'words_after_redaction': len(parts[-1].strip().split())
        }

    def _get_context_window_features(self, context):
        parts = context.split('█')
        before_words = parts[0].strip().split()[-5:] if parts[0] else []
        after_words = parts[-1].strip().split()[:5] if len(parts) > 1 else []
        
        return {
            'before_window_length': len(before_words),
            'after_window_length': len(after_words)
        }

    def _get_sentiment_features(self, context):
        return self.sid.polarity_scores(context.replace('█', ' '))

    def _get_position_features(self, context):
        parts = context.split('█')
        total_length = len(context)
        redaction_start = len(parts[0])
        
        return {
            'redaction_position_ratio': redaction_start / total_length,
            'is_name_at_start': len(parts[0].strip()) < 20,
            'is_name_at_end': len(parts[-1].strip()) < 20
        }
        
class UnredactorPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = None
    
    def _convert_features_to_vector(self, features_dict):
        """Convert dictionary of features to vector format"""
        vector = []
        feature_names = []
        
        # Handle numeric features
        numeric_features = ['redacted_length', 'context_length', 'words_before_redaction', 
                          'words_after_redaction', 'before_window_length', 'after_window_length',
                          'redaction_position_ratio']
        
        for feat in numeric_features:
            if feat in features_dict:
                vector.append(float(features_dict[feat]))
                feature_names.append(feat)
        
        # Handle boolean features
        bool_features = ['is_name_at_start', 'is_name_at_end']
        for feat in bool_features:
            if feat in features_dict:
                vector.append(1.0 if features_dict[feat] else 0.0)
                feature_names.append(feat)
        
        # Handle sentiment features
        sentiment_features = ['neg', 'neu', 'pos', 'compound']
        for feat in sentiment_features:
            if feat in features_dict:
                vector.append(float(features_dict[feat]))
                feature_names.append(feat)
        
        # Handle n-gram features
        ngram_features = [k for k in features_dict.keys() if k.startswith('ngram_')]
        for feat in ngram_features:
            vector.append(float(features_dict[feat]))
            feature_names.append(feat)
            
        return np.array(vector), feature_names
    
    def prepare_training_data(self, training_df):
        """Prepare training data from DataFrame"""
        X = []
        y = []
        first_features = None
        
        print("Extracting features from training data...")
        for _, row in training_df.iterrows():
            features = self.feature_extractor.extract_features(row['context'], row['name'])
            feature_vector, feature_names = self._convert_features_to_vector(features)
            
            if first_features is None:
                first_features = feature_names
                print(f"Number of features: {len(first_features)}")
            
            # Ensure all feature vectors have the same length
            if len(feature_vector) == len(first_features):
                X.append(feature_vector)
                y.append(row['name'])
        
        self.feature_names = first_features
        return np.array(X), np.array(y)
    
   

    def train(self, training_df):
        """Train the model"""
        print("Fitting vectorizer on all contexts...")
        self.feature_extractor.fit_vectorizer(training_df['context'].values)
        
        print("Preparing training data...")
        X, y = self.prepare_training_data(training_df)
        
        print(f"Training with {X.shape[1]} features...")
        
        # Create and train the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=3
            ))
        ])
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Save the trained model
        self.save_model('unredactor_model.joblib')
        print("Model saved to 'unredactor_model.joblib'")
        
        # Validate the model
        y_pred = self.model.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted',zero_division=1)
        print(f"Validation scores: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
        
        return self
    
    def save_model(self, filepath):
        """Save the trained model to file"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, filepath)
        
    def predict(self, test_df):
        """Generate predictions for test data"""
        predictions = []
        
        print("Generating predictions...")
        for _, row in test_df.iterrows():
            features = self.feature_extractor.extract_features(row['context'])
            feature_vector, _ = self._convert_features_to_vector(features)
            
            # Ensure feature vector has correct shape
            if len(feature_vector) == len(self.feature_names):
                pred = self.model.predict([feature_vector])[0]
                predictions.append({
                    'id': row['id'],
                    'name': pred
                })
            else:
                print(f"Warning: Skipping prediction for id {row['id']} due to feature mismatch")
        
        return pd.DataFrame(predictions)
    
    def load_model(self, filepath):
        """Load a trained model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print("Model loaded successfully from", filepath)
        
    # def evaluate_model(pipeline, X_test, y_test):
    #     y_pred = self.predict(X_test)
    #     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    #     return precision, recall, f1

def main():
    try:
        # Load training data with custom loader
        training_df = load_data('unredactor.tsv')
        print(f"Loaded {len(training_df)} training samples")
        
        # Load test data
        test_df = load_data('test.tsv', is_test=True)
        print(f"Loaded {len(test_df)} test samples")
        
        # Create and train the pipeline
        pipeline = UnredactorPipeline()
        pipeline.train(training_df)
        
        # Generate predictions
        print("Generating predictions...")
        predictions_df = pipeline.predict(test_df)
        
        # Save predictions
        predictions_df.to_csv('submission.tsv', sep='\t', index=False)
        print("\nPredictions saved to submission.tsv")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()