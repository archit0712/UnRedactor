
# Unredactor Pipeline

This project implements an unredactor pipeline designed to predict redacted names in text using machine learning techniques.

## Overview

The unredactor pipeline uses a combination of natural language processing and machine learning to extract features from redacted text and predict the original names. It employs a Random Forest classifier as the core prediction model.

## Video Demo (Please refer to github repo for latest video)

[![Watch the video](https://lh3.googleusercontent.com/pw/AP1GczNlNM-FeNkXhuDQLX0aoj6SOHn5hwJVj3ufng5VCG_GyU-2LzzKP2JAE_Pf2T24LMBGYhPYfCO_ELt9aAupGMd8qDqsRVec8_XjsMP1EdWkdfk826RUagm9ac_DssHp79BiBWijyKSrkBKXJbAFGkbR0g=w1163-h653-s-no-gm?authuser=1)](https://youtu.be/U4ywFrcJFsI)


## Data Exploration and Analysis

### Dataset Overview
The project uses two main datasets:
- Training set (unredactor.tsv): 4,457 samples containing redacted names and their contexts
- Test set (test.tsv): 200 samples containing only redacted contexts

#### Name Distribution Analysis
- Total unique names: 3,020 out of 4,457 samples
- Most frequent name: "Sadako" (37 occurrences)
- Most names appear only once or twice, indicating high variability
- Common names include actors, directors, and film industry personalities

#### Context Characteristics
- Average context length: 204 characters
- Median context length: 135 characters
- Range: 6-1,845 characters
- Most contexts fall between 90-217 characters

#### Name Length Statistics
- Average length: 10.5 characters
- Median length: 11 characters
- Range: 2-27 characters
- Most names are between 7-13 characters

#### Feature Engineering Approach
The model extracts several types of features:

1. **Length Features**
   - Redacted name length
   - Context length
   - Words before/after redaction

2. **Context Window Features**
   - Words immediately surrounding redaction
   - Window size analysis

3. **N-gram Features**
   - Using CountVectorizer with parameters:
   - n-gram range: 1-3
   - max features: 100
   - minimum document frequency: 2

4. **Sentiment Features**
   - Using VADER sentiment analyzer
   - Captures emotional context

5. **Position Features**
   - Relative position of redaction
   - Start/end position indicators

## Libraries Used
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms and preprocessing
- NLTK: Natural language processing
- joblib: Model persistence



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

## Running Test Cases

To run the test cases for the Unredactor Pipeline, follow these steps:

1. Ensure you have pytest installed:
   ```bash
   pipenv install pytest
   ```

2. Navigate to the project directory containing the test files.

3. Run the tests using the following command:
   ```bash
   pipenv run python -m pytest
   ```

## Test Cases and Their Significance

### 1. Test Data Loading (`test_load_data`)

**Significance**: This test ensures that the `load_data` function correctly reads and processes both training and test data files.

**Explanation**: It verifies that:
- The function returns a pandas DataFrame.
- The DataFrame contains the expected columns for both training and test data.
- The data is loaded with the correct structure.

### 2. Test Feature Extraction (`test_feature_extractor`)

**Significance**: This test checks the functionality of the `FeatureExtractor` class, which is crucial for preparing input data for the model.

**Explanation**: It confirms that:
- The `extract_features` method returns a dictionary of features.
- All expected feature types (length, context window, n-grams, sentiment, position) are present.
- The feature extraction process handles redacted text correctly.

### 3. Test Unredactor Pipeline (`test_unredactor_pipeline`)

**Significance**: This test validates the end-to-end functionality of the `UnredactorPipeline` class, which is the core of the unredaction process.

**Explanation**: It verifies that:
- The pipeline can be trained on sample data without errors.
- The trained model can generate predictions for test data.
- The output predictions are in the correct format (DataFrame with 'id' and 'name' columns).

These test cases are designed to ensure the reliability and correctness of the Unredactor Pipeline at different stages of its operation. They help catch potential issues in data processing, feature extraction, and model training/prediction, which are critical for the overall performance of the unredaction task.


## Output

The pipeline generates a `submission.tsv` file containing predictions for the test data. It also saves the trained model as `unredactor_model.joblib`.

### Results
* Precision: 0.85
* Recall: 0.09
* F1 Score: 0.07

## Troubleshooting

If you encounter any issues, ensure that:
- All required packages are installed
- NLTK resources are downloaded
- Input data files are in the correct format and location

For any persistent problems, please open an issue on the GitHub repository.
