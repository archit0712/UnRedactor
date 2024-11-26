import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Download only the essential resources
        resources = ['vader_lexicon', 'averaged_perceptron_tagger']
        for resource in resources:
            nltk.download(resource)
        
        self.sid = SentimentIntensityAnalyzer()
    
    def extract_features(self, context, redacted_name):
        features = {}
        
        # 1. Basic Length Features
        features.update(self._get_length_features(context, redacted_name))
        
        # 2. Context Window Features
        features.update(self._get_context_window_features(context))
        
        # 3. N-gram Features
        features.update(self._get_ngram_features(context))
        
        # 4. Sentiment Features
        features.update(self._get_sentiment_features(context))
        
        return features
    
    def _get_length_features(self, context, redacted_name):
        return {
            'redacted_length': len(redacted_name),
            'context_length': len(context),
            'words_before_redaction': len(context.split('█')[0].strip().split()),
            'words_after_redaction': len(context.split('█')[-1].strip().split())
        }
    
    def _get_context_window_features(self, context):
        # Use simple string splitting instead of word_tokenize
        parts = context.split('█')
        before_words = parts[0].strip().split()[-5:] if parts[0] else []
        after_words = parts[-1].strip().split()[:5] if len(parts) > 1 else []
        
        return {
            'words_before': ' '.join(before_words),
            'words_after': ' '.join(after_words)
        }
    
    def _get_ngram_features(self, context):
        vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=100)
        ngrams = vectorizer.fit_transform([context])
        
        return {
            f'ngram_{feat}': count 
            for feat, count in zip(vectorizer.get_feature_names_out(), 
                                 ngrams.toarray()[0])
        }
    
    def _get_sentiment_features(self, context):
        return self.sid.polarity_scores(context)
# Example usage
def main():
    # Sample data
    context = "Lots of obvious symbolism about achieving manhood but mainly it's the acting by Stewart, his partner Millard Mitchell, Shelly Winters and the Waco Johnny Dean- ██████████."
    redacted_name = "Dan Duryea"
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(context, redacted_name)
    
    # Print some example features
    print("Length Features:", {k:v for k,v in features.items() if 'length' in k})
    print("\nSentiment Features:", {k:v for k,v in features.items() if k in ['neg', 'neu', 'pos', 'compound']})

if __name__ == "__main__":
    main()