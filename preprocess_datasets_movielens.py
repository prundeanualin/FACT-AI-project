# Make sure your data is in:
# data/ml-1m/movies.dat
# data/ml-1m/ratings.dat
# data/ml-1m/users.dat

import pandas as pd
import os
from sklearn.model_selection import train_test_split


base_path = 'data/ml-1m/'
movies_file = base_path + "movies.dat"
ratings_file = base_path + "ratings.dat"
users_file = base_path + "users.dat"


def load_dataset(path, names, encoding='utf-8'):
    if not os.path.exists(path):
        raise Exception(f"Filepath ({path}) does not exist. Make sure the datafile is in the right directory.")
    return pd.read_csv(path, sep='::', engine='python', header=None, names=names, encoding=encoding)


# Using 'latin1' or 'ISO-8859-1' encoding
movies = load_dataset(movies_file, names=['MovieID', 'Title', 'Genres'], encoding='latin1')
users = load_dataset(users_file, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin1')
ratings = pd.read_csv(ratings_file, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin1')

print("Movies:")
print(movies.head())
print()

print("Users:")
print(users.head())
print()

print("Ratings:")
print(ratings.head())
print()

# They use gender, age, and occupation as sensitive attributes from Users df where gender is a binary feature,
# occupation is a 21-class feature, and users are assigned to 7 groups based on age:
# 1: "Under 18"
# 18: "18-24"
# 25: "25-34"
# 35: "35-44"
# 45: "45-49"
# 50: "50-55"
# 56: "56+"

# Binary encoding for Gender
users['Gender'] = users['Gender'].map({'F': 0, 'M': 1})

# One-hot encoding for Age
users = pd.get_dummies(users, columns=['Age'], prefix='Age')

# Drop unnecessary cols
users = users.drop(['Zip-code'], axis=1)

# Merge ratings with users
merge_columns = ['UserID']
final_df = pd.merge(ratings, users, on=merge_columns)

print("\n\n\nFinal merged dataframe: ")
print(final_df.columns.tolist())
print(final_df.head())
print(final_df.tail())
print("Length final df: " + str(len(final_df)))
print("Length ratings df: " + str(len(ratings)))

print("Writing the final processed dataframe to csv...")
final_df.to_csv(base_path + "merged_dataset.csv")

# OLD CODE
# # Splitting the final df into train, test, valid
# train_data, temp_data = train_test_split(final_df, test_size=0.3, random_state=42)
#
# # Now split the temporary set into validation and test sets
# valid_data, test_data = train_test_split(temp_data, test_size=2/3, random_state=42)
#
# # Print the sizes of each set
# print(f"Train set: {len(train_data)} rows ({len(train_data)/len(final_df)*100:.2f}%)")
# print(f"Validation set: {len(valid_data)} rows ({len(valid_data)/len(final_df)*100:.2f}%)")
# print(f"Test set: {len(test_data)} rows ({len(test_data)/len(final_df)*100:.2f}%)")
#
# # Sort the sets by UserID because NCF requires this
# train_sorted = train_data.sort_values(by='UserID')
# validation_sorted = valid_data.sort_values(by='UserID')
# test_sorted = test_data.sort_values(by='UserID')
#
# train_sorted.to_csv(base_path + "train.csv")
# validation_sorted.to_csv(base_path + "validation.csv")
# test_sorted.to_csv(base_path + "test.csv")
# OLD CODE


# It seems that NCF (and PMF and LightGCN) cannot model/predict for users/movies it hasn't seen during training.
# So it seems we need to ensure that train, valid, test all contain at least 1 interaction for every user/movie comb
# We can use Timestamp to split by time so that for each user we include 70% in train (the first 70%) by sorting
# by Timestamp for that user. This is often done in recommender systems I think.

# Ensure each movie is represented at least once in the training set
unique_movies = final_df['MovieID'].unique()
initial_train_list = []

for movie in unique_movies:
    movie_data = final_df[final_df['MovieID'] == movie]
    initial_train_list.append(movie_data.iloc[0])

# Create an initial training DataFrame
initial_train_data = pd.DataFrame(initial_train_list)

# Remove these instances from final_df
remaining_df = final_df.drop(initial_train_data.index)

# Split the remaining data by user
def split_data_for_user(user_data):
    user_data = user_data.sort_values(by='Timestamp')
    n = len(user_data)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)
    train = user_data[:train_end]
    valid = user_data[train_end:valid_end]
    test = user_data[valid_end:]
    return train, valid, test

grouped = remaining_df.groupby('UserID')
train_list, valid_list, test_list = zip(*grouped.apply(split_data_for_user))

# Combine the lists into DataFrames
train_data = pd.concat([initial_train_data] + list(train_list)).reset_index(drop=True)
valid_data = pd.concat(valid_list).reset_index(drop=True)
test_data = pd.concat(test_list).reset_index(drop=True)

# Sort the sets by UserID
train_data = train_data.sort_values(by='UserID')
valid_data = valid_data.sort_values(by='UserID')
test_data = test_data.sort_values(by='UserID')

# Save the datasets
train_data.to_csv(base_path + "train.csv", index=False)
valid_data.to_csv(base_path + "validation.csv", index=False)
test_data.to_csv(base_path + "test.csv", index=False)

# Print the sizes of each set
print(f"Train set: {len(train_data)} rows ({len(train_data)/len(final_df)*100:.2f}%)")
print(f"Validation set: {len(valid_data)} rows ({len(valid_data)/len(final_df)*100:.2f}%)")
print(f"Test set: {len(test_data)} rows ({len(test_data)/len(final_df)*100:.2f}%)")

# Check if all unique movies from final_df are in train_data
unique_movies_final_df = set(final_df['MovieID'].unique())
unique_movies_train = set(train_data['MovieID'].unique())
assert unique_movies_final_df.issubset(unique_movies_train), "Not all movies in final_df are in train_data"

# Check if all unique user IDs from final_df are in train_data
unique_users_final_df = set(final_df['UserID'].unique())
unique_users_train = set(train_data['UserID'].unique())
assert unique_users_final_df.issubset(unique_users_train), "Not all user IDs in final_df are in train_data"

print("Assertion passed: All unique movies and user IDs in final_df are present in train_data.")
