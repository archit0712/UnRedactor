
# Unredactor Pipeline

This project implements an unredactor pipeline designed to predict redacted names in text using machine learning techniques.

## Overview

The unredactor pipeline uses a combination of natural language processing and machine learning to extract features from redacted text and predict the original names. It employs a Random Forest classifier as the core prediction model.

## Features

- **Data Loading**: Supports loading of both training and test data from TSV files.
- **Feature Extraction**: Extracts various features including length, context window, n-grams, sentiment, and position features.
- **Model Training**: Uses a scikit-learn pipeline with StandardScaler and RandomForestClassifier.
- **Model Evaluation**: Includes validation step with classification report.
- **Prediction**: Generates predictions for test data.
- **Model Persistence**: Saves and loads trained models using joblib.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- joblib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/archit0712/cis6930fa24-project2.git
   ```
   ```bash
   cd cis6930fa24-project2
   ```

2. Install the required packages:
   ```
   pipenv install
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('averaged_perceptron_tagger')
   ```

## Usage

1. Prepare your data:
   - Training data should be in a TSV file named `unredactor.tsv` with columns: split, name, context
   - Test data should be in a TSV file named `test.tsv` with columns: id, context

2. Run the pipeline:
   ```
   pipenv run python unredactor_pipeline.py
   ```

3. The script will:
   - Load the training and test data
   - Train the model
   - Generate predictions
   - Save the predictions to `submission.tsv`

## Design and Implementation Process

The unredactor pipeline was designed and implemented with a focus on modularity, efficiency, and extensibility. Here's an overview of the thought process behind the code:

### Data Handling
We started by creating a flexible `load_data` function that can handle both training and test data. This approach allows for easy adaptation to different data formats and sources.

### Feature Extraction
The `FeatureExtractor` class was designed to encapsulate all feature engineering logic. We chose to include a variety of features:

1. **Length features**: To capture the structural characteristics of the redacted text.
2. **Context window features**: To understand the immediate surroundings of the redaction.
3. **N-gram features**: To capture local patterns in the text.
4. **Sentiment features**: To incorporate the emotional context of the text.
5. **Position features**: To account for the location of the redaction within the text.

This diverse set of features aims to provide a comprehensive representation of the text for the machine learning model.

### Machine Learning Pipeline
We implemented the `UnredactorPipeline` class to orchestrate the entire process:

1. **Data preparation**: Converts extracted features into a format suitable for machine learning.
2. **Model training**: Uses a scikit-learn pipeline with StandardScaler and RandomForestClassifier.
   - StandardScaler ensures all features are on the same scale.
   - RandomForestClassifier was chosen for its ability to handle complex relationships and resistance to overfitting.
3. **Model evaluation**: Includes a validation step to assess model performance.
4. **Prediction**: Generates predictions for test data.
5. **Model persistence**: Saves and loads trained models for future use or deployment.

### Workflow
The `main` function ties everything together, providing a clear workflow from data loading to prediction generation. This structure allows for easy modification and extension of the pipeline.

### Design Choices
- **Modularity**: Each major component (data loading, feature extraction, model training) is encapsulated in its own function or class, promoting code reusability and maintainability.
- **Flexibility**: The pipeline can handle different data formats and can be easily extended with new features or models.
- **Error Handling**: Try-except blocks are used to gracefully handle potential errors during execution.
- **Performance Optimization**: We use efficient libraries like pandas for data handling and scikit-learn for machine learning tasks.

This thoughtful design process resulted in a robust and flexible unredactor pipeline capable of handling various text unredaction tasks while allowing for easy modifications and improvements.

## Customization

- To modify feature extraction, edit the `FeatureExtractor` class.
- To adjust the model, modify the `RandomForestClassifier` parameters in the `UnredactorPipeline` class.

## Output

The pipeline generates a `submission.tsv` file containing predictions for the test data. It also saves the trained model as `unredactor_model.joblib`.

## Troubleshooting

If you encounter any issues, ensure that:
- All required packages are installed
- NLTK resources are downloaded
- Input data files are in the correct format and location

For any persistent problems, please open an issue on the GitHub repository.
