import pandas as pd

base_path = 'data/ml-1m/'
train_path = base_path + "train.csv"

# Load and preprocess rating data
train = pd.read_csv(train_path)
train = train.drop(['Gender', 'Age', 'Occupation'], axis=1)

# Print the first few rows of the dataset
print("Original Data:\n", train.head())

print("\nRating Distribution in Original Data (Sorted Descending):")
print(train['Rating'].value_counts(normalize=True).sort_index(ascending=False))

# Convert ratings for different thresholds and print class distribution
thresholds = [1, 2, 3]
for threshold in thresholds:
    temp_df = train.copy()
    temp_df['Rating'] = (temp_df['Rating'] > threshold).astype(int)
    print(f"\nClass Distribution with Threshold > {threshold}:")
    print(temp_df['Rating'].value_counts(normalize=True))
