
import os
import pandas as pd

# Make sure you have the pre-processed dataset in data/pisa2015/dataset.csv
dataset_filepath = 'data/pisa2015/dataset.csv'

if not os.path.exists(dataset_filepath):
    raise Exception(f"Make sure you have the pre-processed dataset in {dataset_filepath}")

print("Loading the dataset...")
df = pd.read_csv(dataset_filepath)

# The student id is unique across the entire table, so we can make this the index
# df.set_index(['CNTSTUID'], inplace=True)

print(df.head())