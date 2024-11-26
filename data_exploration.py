import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import io
# Load the data
def load_unredactor_data(file_path='unredactor.tsv'):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                split, name, context = parts[0], parts[1], '\t'.join(parts[2:])
                data.append({'split': split, 'name': name, 'context': context})
            else:
                print(f"Skipping invalid line {i}: {line.strip()}")
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows of data")
    return df

def analyze_context_lengths(df):
    df['context_length'] = df['context'].str.len()
    
    print("Context length statistics:")
    print(df['context_length'].describe())
    
    plt.figure(figsize=(10, 6))
    df['context_length'].hist(bins=20)
    plt.title("Distribution of Context Lengths")
    plt.xlabel("Context Length")
    plt.ylabel("Frequency")
    plt.savefig("context_length_distribution.png")
    plt.close()
# Analyze name lengths
def analyze_name_lengths(df):
    df['name_length'] = df['name'].str.len()
    
    print("Name length statistics:")
    print(df['name_length'].describe())
    
    plt.figure(figsize=(10, 6))
    df['name_length'].hist(bins=20)
    plt.title("Distribution of Name Lengths")
    plt.xlabel("Name Length")
    plt.ylabel("Frequency")
    plt.savefig("name_length_distribution.png")
    plt.close()
  
# Explore IMDB dataset structure
def explore_imdb_dataset(directory='aclImdb'):
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(directory, split, sentiment)
            num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
            print(f"Number of {sentiment} reviews in {split} set: {num_files}")
            
            
# Analyze the distribution of data
def analyze_data_distribution(df):
    print("Data distribution:")
    print(df['split'].value_counts())
    
    print("\nNumber of unique names:", df['name'].nunique())
    
    print("\nTop 10 most common names:")
    print(df['name'].value_counts().head(10))
    
def main():
    df = load_unredactor_data()
    analyze_data_distribution(df)
    analyze_context_lengths(df)
    explore_imdb_dataset()
    analyze_name_lengths(df)
    
    
if __name__ == '__main__':
    main()